from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from interfaces.execution_trace import ExecutionTrace
from interfaces.skill_spec import SkillSpec
from interfaces.world_state import ObjectState, WorldState
from modules.control import MjctrlMPCBackend, MjctrlMPCConfig, PlanExecutor, SymbolicControlBackend
from modules.grounding import SceneGrounder
from modules.planner import PlanStep, TaskPlanner
from modules.task_parser import TaskParser
from modules.world_model import WorldModelStore

ROOT = Path(__file__).resolve().parents[1]


def test_task_parser_extracts_goal_and_entities() -> None:
    task = TaskParser().parse("place the bottle on the tray")
    assert task.goal == "place_object"
    assert task.target_object == "the bottle"
    assert task.target_location == "the tray"


def test_planner_builds_three_step_place_plan() -> None:
    task = TaskParser().parse("put the cup on the shelf")
    world_state = SceneGrounder().ground(task)
    plan = TaskPlanner().build_plan(task, world_state)
    assert [step.skill.name for step in plan.steps] == [
        "locate_object",
        "grasp_object",
        "place_object",
    ]


def test_executor_closes_loop_for_place_plan() -> None:
    task = TaskParser().parse("place the bottle on the tray")
    world_state = SceneGrounder().ground(task)
    plan = TaskPlanner().build_plan(task, world_state)
    world_model = WorldModelStore()
    world_model.update(world_state)
    trace = ExecutionTrace(trace_id="trace-place", task_id=task.task_id)

    result = PlanExecutor(world_model=world_model).execute(plan, trace)

    assert result.success
    assert result.completed_steps == 3
    bottle = _require_object(result.final_world_state, "the bottle")
    tray = _require_object(result.final_world_state, "the tray")
    assert bottle.pose == tray.pose
    assert "on:the-tray" in bottle.relations
    assert result.final_world_state.robot_mode == "ready"
    assert any(event.message == "step_3_feedback_observed" for event in trace.events)
    assert trace.events[-1].message == "execution_completed"


def test_executor_stops_on_missing_target_object() -> None:
    task = TaskParser().parse("place the bottle on the tray")
    plan = TaskPlanner().build_plan(task, WorldState(scene_id="scene-plan"))
    world_model = WorldModelStore()
    world_model.update(
        WorldState(
            scene_id="scene-missing-object",
            objects=[_make_object("the tray", [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
            robot_mode="ready",
        )
    )
    trace = ExecutionTrace(trace_id="trace-missing-object", task_id=task.task_id)

    result = PlanExecutor(world_model=world_model).execute(plan, trace)

    assert not result.success
    assert result.completed_steps == 0
    assert result.failed_step_index == 1
    assert result.failure_code == "object_not_found"
    assert any(event.message == "step_1_failed" for event in trace.events)
    assert trace.events[-1].message == "execution_failed"


def test_symbolic_backend_place_reports_missing_target_location() -> None:
    backend = SymbolicControlBackend()
    feedback = backend.execute_step(
        _build_step(
            "place_object",
            {"target_object": "the bottle", "target_location": "the tray"},
        ),
        WorldState(
            scene_id="scene-missing-location",
            objects=[
                _make_object(
                    "the bottle",
                    [0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                    relations=["held_by:gripper"],
                )
            ],
            robot_mode="holding",
        ),
        step_index=1,
    )

    assert not feedback.success
    assert feedback.failure_code == "target_location_not_found"


def test_symbolic_backend_place_requires_grasp() -> None:
    backend = SymbolicControlBackend()
    feedback = backend.execute_step(
        _build_step(
            "place_object",
            {"target_object": "the bottle", "target_location": "the tray"},
        ),
        WorldState(
            scene_id="scene-not-grasped",
            objects=[
                _make_object("the bottle", [0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
                _make_object("the tray", [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ],
            robot_mode="ready",
        ),
        step_index=1,
    )

    assert not feedback.success
    assert feedback.failure_code == "object_not_grasped"


def test_symbolic_backend_inspect_scene_is_non_mutating() -> None:
    backend = SymbolicControlBackend()
    world_state = WorldState(
        scene_id="scene-inspect",
        objects=[_make_object("the bottle", [0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])],
        robot_mode="ready",
    )

    feedback = backend.execute_step(
        _build_step("inspect_scene", {"scene_id": world_state.scene_id}),
        world_state,
        step_index=1,
    )

    assert feedback.success
    assert feedback.failure_code is None
    assert feedback.observed_world_state.to_dict() == world_state.to_dict()


def test_mjctrl_mpc_backend_closes_loop_for_place_step() -> None:
    backend = MjctrlMPCBackend()
    feedback = backend.execute_step(
        _build_step(
            "place_object",
            {"target_object": "the bottle", "target_location": "the tray"},
        ),
        WorldState(
            scene_id="scene-mpc-place",
            objects=[
                _make_object(
                    "the bottle",
                    [0.45, 0.12, 0.12, 0.0, 0.0, 0.0, 1.0],
                    relations=["held_by:gripper"],
                ),
                _make_object("the tray", [0.7, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ],
            robot_mode="holding",
        ),
        step_index=1,
    )

    assert feedback.success
    assert feedback.failure_code is None
    assert feedback.metrics["controller"] == "mjctrl_mpc"
    bottle = _require_object(feedback.observed_world_state, "the bottle")
    tray = _require_object(feedback.observed_world_state, "the tray")
    assert "on:the-tray" in bottle.relations
    assert abs(bottle.pose[0] - tray.pose[0]) < 0.03
    assert abs(bottle.pose[1] - tray.pose[1]) < 0.03
    assert abs(bottle.pose[2] - tray.pose[2]) < 0.03


def test_mjctrl_mpc_backend_reports_non_convergence() -> None:
    backend = MjctrlMPCBackend(
        MjctrlMPCConfig(
            horizon=10,
            max_iterations=1,
            control_dt=0.01,
            max_linear_velocity=0.01,
            convergence_tolerance=1e-6,
        )
    )
    feedback = backend.execute_step(
        _build_step(
            "place_object",
            {"target_object": "the bottle", "target_location": "the tray"},
        ),
        WorldState(
            scene_id="scene-mpc-not-converged",
            objects=[
                _make_object(
                    "the bottle",
                    [0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
                    relations=["held_by:gripper"],
                ),
                _make_object("the tray", [0.9, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ],
            robot_mode="holding",
        ),
        step_index=1,
    )

    assert not feedback.success
    assert feedback.failure_code == "place_mpc_not_converged"
    assert feedback.metrics["controller"] == "mjctrl_mpc"


def test_run_task_writes_trace() -> None:
    temp_dir = ROOT / ".tmp-tests"
    temp_dir.mkdir(exist_ok=True)
    trace_path = temp_dir / "trace.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps" / "run_task.py"),
            "place the bottle on the tray",
            "--trace-out",
            str(trace_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["events"][0]["stage"] == "task_parser"
    assert any(event["message"] == "step_1_feedback_observed" for event in data["events"])
    assert data["events"][-1]["message"] == "execution_completed"
    assert "final_world_state" in data["events"][-1]["payload"]
    assert '"trace_id"' in result.stdout
    shutil.rmtree(temp_dir)


def test_run_benchmark_outputs_report() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps" / "run_benchmark.py"),
            "--cases",
            str(ROOT / "eval" / "benchmarks" / "tabletop_v0.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert len(report) == 2
    assert all(entry["success"] for entry in report)


def _build_step(skill_name: str, parameters: dict[str, str]) -> PlanStep:
    return PlanStep(
        skill=SkillSpec(name=skill_name),
        parameters=parameters,
    )


def _make_object(
    label: str,
    pose: list[float],
    relations: list[str] | None = None,
) -> ObjectState:
    return ObjectState(
        object_id="-".join(label.lower().split()),
        label=label,
        pose=pose,
        relations=[] if relations is None else list(relations),
    )


def _require_object(world_state: WorldState, label: str) -> ObjectState:
    for obj in world_state.objects:
        if obj.label == label:
            return obj
    raise AssertionError(f"Object not found: {label}")
