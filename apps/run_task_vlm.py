from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_model(mujoco: Any, scene_path: Path) -> Any:
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except ValueError:
        model_xml = scene_path.read_text(encoding="utf-8")
        assets: dict[str, bytes] = {}
        for file_path in scene_path.parent.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(scene_path.parent).as_posix()
                assets[rel] = file_path.read_bytes()
        return mujoco.MjModel.from_xml_string(model_xml, assets=assets)


def _resolve_scene_path(scene: Path) -> Path:
    return scene if scene.is_absolute() else ROOT / scene


def _resolve_camera(camera: str) -> str | int:
    normalized = camera.strip()
    if normalized.startswith("-"):
        digits = normalized[1:]
        if digits.isdigit():
            return int(normalized)
        return normalized
    if normalized.isdigit():
        return int(normalized)
    return normalized


def _build_backend(args: argparse.Namespace) -> Any:
    from modules.vlm import LocalVLMBackend, MockVLMBackend, OpenAIVLMBackend

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if args.vlm_backend == "mock":
        return MockVLMBackend(model=args.vlm_model or "mock-vlm")
    if args.vlm_backend == "openai":
        if not api_key:
            raise ValueError("OpenAI backend requires --api-key or OPENAI_API_KEY")
        return OpenAIVLMBackend(
            model=args.vlm_model or "gpt-4.1-mini",
            api_key=api_key,
            base_url=args.vlm_base_url or "https://api.openai.com/v1",
        )
    return LocalVLMBackend(
        model=args.vlm_model or "llava",
        api_key=api_key,
        base_url=args.vlm_base_url or "http://localhost:8000/v1",
    )


def main() -> None:
    from interfaces.execution_trace import ExecutionTrace
    from modules.control import PlanExecutor
    from modules.grounding import SceneGrounder
    from modules.planner import TaskPlanner
    from modules.task_parser import TaskParser
    from modules.vlm import VLMService, capture_mujoco_rgb
    from modules.world_model import WorldModelStore

    parser = argparse.ArgumentParser(
        description="Run a task with VLM scene review and verification."
    )
    parser.add_argument("instruction", help="Natural language instruction to execute.")
    parser.add_argument(
        "--vlm-backend",
        choices=("openai", "local", "mock"),
        default="openai",
        help="Vision-language backend to use.",
    )
    parser.add_argument(
        "--vlm-model",
        help="Model identifier for the selected backend.",
    )
    parser.add_argument(
        "--vlm-base-url",
        help="Base URL for OpenAI-compatible backends.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the selected backend. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("sim/assets/franka_emika_panda/scene.xml"),
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--camera",
        default="wrist_rgbd",
        help="Camera name or index passed to MuJoCo renderer.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height in pixels.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=200,
        help="Simulation steps before each capture.",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        help="Optional path to write the execution trace as JSON.",
    )
    args = parser.parse_args()

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError("MuJoCo is required. Install it with `uv sync --group sim`.") from exc

    scene_path = _resolve_scene_path(args.scene)
    model = _load_model(mujoco, scene_path)
    data = mujoco.MjData(model)
    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data)

    task = TaskParser().parse(args.instruction)
    trace = ExecutionTrace(trace_id=f"trace-{uuid4().hex[:8]}", task_id=task.task_id)
    trace.add_event("task_parser", "task_parsed", "success", payload=task.to_dict())

    camera = _resolve_camera(args.camera)
    before_image = capture_mujoco_rgb(model, data, camera, args.width, args.height)
    backend = _build_backend(args)
    vlm_service = VLMService(backend=backend)

    scene_description = vlm_service.describe_scene(before_image, task)
    trace.add_event(
        "vlm_scene",
        "scene_described",
        "success",
        payload={
            "scene_path": str(scene_path),
            "camera": args.camera,
            "width": args.width,
            "height": args.height,
            "description": scene_description.to_dict(),
        },
    )

    grounder = SceneGrounder()
    world_model = WorldModelStore()
    planner = TaskPlanner()
    executor = PlanExecutor(world_model=world_model)

    world_state = world_model.update(grounder.ground(task))
    trace.add_event("grounding", "world_state_grounded", "success", payload=world_state.to_dict())

    plan = planner.build_plan(task, world_state)
    trace.add_event("planner", "plan_built", "success", payload=plan.to_dict())

    plan_review = vlm_service.review_plan(before_image, plan, task)
    trace.add_event(
        "vlm_plan_review",
        "plan_reviewed",
        "success" if plan_review.feasible else "warning",
        payload=plan_review.to_dict(),
    )

    result = executor.execute(plan, trace)

    trace.add_event(
        "simulation",
        "symbolic_execution_detached_from_mujoco",
        "warning",
        payload={
            "note": (
                "The current executor mutates symbolic world state only; "
                "MuJoCo scene remains unchanged unless synced externally."
            ),
            "execution_success": result.success,
        },
    )

    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data)
    after_image = capture_mujoco_rgb(model, data, camera, args.width, args.height)
    verification = vlm_service.verify_execution(before_image, after_image, task)
    trace.add_event(
        "vlm_verification",
        "execution_verified",
        "success" if verification.task_completed else "warning",
        payload=verification.to_dict(),
    )

    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    print(json.dumps(trace.to_dict(), indent=2))


if __name__ == "__main__":
    main()
