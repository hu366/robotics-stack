from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_model(mujoco: Any, scene_path: Path) -> Any:
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except ValueError:
        # Fallback for non-ASCII paths and include/asset resolution.
        model_xml = scene_path.read_text(encoding="utf-8")
        assets: dict[str, bytes] = {}
        for file_path in scene_path.parent.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(scene_path.parent).as_posix()
                assets[rel] = file_path.read_bytes()
        return mujoco.MjModel.from_xml_string(model_xml, assets=assets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open MuJoCo viewer for a scene XML.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("sim/scenes/mujoco_minimal.xml"),
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Sleep time (seconds) between viewer sync steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional max simulation steps; 0 means run until window is closed.",
    )
    parser.add_argument(
        "--camera",
        default="",
        help="Optional camera name to use (e.g. wrist_rgbd).",
    )
    args = parser.parse_args()

    import mujoco  # type: ignore[import-untyped]
    import mujoco.viewer  # type: ignore[import-untyped]

    scene_path = args.scene if args.scene.is_absolute() else ROOT / args.scene
    model = _load_model(mujoco, scene_path)
    data = mujoco.MjData(model)

    print(f"scene={scene_path}")
    print("MuJoCo viewer started. Close the window to exit.")

    step_count = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.camera:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
            if cam_id < 0:
                raise ValueError(f"Camera not found: {args.camera}")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1
            if args.max_steps > 0 and step_count >= args.max_steps:
                break
            time.sleep(args.dt)

    print(f"steps={step_count}")
    print(f"time={data.time:.6f}")


if __name__ == "__main__":
    main()
