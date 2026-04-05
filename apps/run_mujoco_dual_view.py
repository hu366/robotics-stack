from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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


def _depth_to_meters(model: Any, depth_buffer: Any) -> Any:
    near = model.vis.map.znear * model.stat.extent
    far = model.vis.map.zfar * model.stat.extent
    return near / (1.0 - depth_buffer * (1.0 - near / far))


def _save_depth_vis(path: Path, depth_m: np.ndarray, cv2: Any) -> None:
    valid = np.isfinite(depth_m)
    vis = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        dmin = float(np.min(depth_m[valid]))
        dmax = float(np.max(depth_m[valid]))
        if dmax - dmin > 1e-9:
            norm = (depth_m - dmin) / (dmax - dmin)
            vis = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", vis)
    if not ok:
        raise RuntimeError(f"Failed to encode depth visualization: {path}")
    path.write_bytes(encoded.tobytes())


def _save_png(path: Path, image_bgr: np.ndarray, cv2: Any) -> None:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open two views: MuJoCo full-arm viewer + wrist camera RGB window."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("sim/assets/franka_emika_panda/scene.xml"),
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--wrist-camera",
        default="wrist_rgbd",
        help="Camera name used for the RGB stream window.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Wrist RGB window width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Wrist RGB window height.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Sleep time (seconds) between frames.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional max simulation steps; 0 means run until closed.",
    )
    parser.add_argument(
        "--gripper-step",
        type=float,
        default=8.0,
        help="How much to change gripper control per key press.",
    )
    parser.add_argument(
        "--joint-step",
        type=float,
        default=0.04,
        help="How much to change one joint target per key press.",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("artifacts/teleop_captures"),
        help="Directory to save keyboard snapshots.",
    )
    args = parser.parse_args()

    import cv2
    import mujoco  # type: ignore[import-untyped]
    import mujoco.viewer  # type: ignore[import-untyped]

    scene_path = args.scene if args.scene.is_absolute() else ROOT / args.scene
    model = _load_model(mujoco, scene_path)
    data = mujoco.MjData(model)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.wrist_camera)
    if cam_id < 0:
        raise ValueError(f"Camera not found: {args.wrist_camera}")

    arm_act_ids: list[int] = []
    for idx in range(1, 8):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{idx}")
        if act_id >= 0:
            arm_act_ids.append(act_id)
    if len(arm_act_ids) != 7:
        arm_act_ids = list(range(min(7, model.nu)))

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if gripper_act_id < 0:
        gripper_act_id = model.nu - 1

    gripper_min = float(model.actuator_ctrlrange[gripper_act_id, 0])
    gripper_max = float(model.actuator_ctrlrange[gripper_act_id, 1])
    if model.nkey > 0 and model.nu > 0:
        data.ctrl[:] = model.key_ctrl[0]

    ctrl_targets = data.ctrl.copy()
    gripper_ctrl = float(ctrl_targets[gripper_act_id]) if model.nu > 0 else gripper_max
    joint_step = args.joint_step
    capture_dir = args.capture_dir if args.capture_dir.is_absolute() else ROOT / args.capture_dir
    capture_dir.mkdir(parents=True, exist_ok=True)

    print(f"scene={scene_path}")
    print("MuJoCo viewer + wrist camera window started.")
    print("Focus RGB window, then use:")
    print("Q/A W/S E/D R/F T/G Y/H U/J -> joint1..7 +/-")
    print("O/P -> gripper open/close, SPACE -> snapshot, [ ] -> joint step, M -> home, ESC -> quit")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    step_count = 0

    cv2.namedWindow("Wrist RGB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Wrist RGB", args.width, args.height)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                data.ctrl[:] = ctrl_targets
                mujoco.mj_step(model, data)
                viewer.sync()

                renderer.update_scene(data, camera=args.wrist_camera)
                rgb = renderer.render()
                bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
                display_bgr = bgr.copy()
                cv2.putText(
                    display_bgr,
                    f"joint_step={joint_step:.3f} | gripper={gripper_ctrl:.1f}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_bgr,
                    "Q/A W/S E/D R/F T/G Y/H U/J move joints | O/P gripper | SPACE snap | ESC quit",
                    (10, args.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Wrist RGB", display_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("["):
                    joint_step = max(0.001, joint_step * 0.8)
                    print(f"joint_step={joint_step:.4f}")
                if key == ord("]"):
                    joint_step = min(0.5, joint_step * 1.25)
                    print(f"joint_step={joint_step:.4f}")
                if key == ord("m") and model.nkey > 0 and model.nu > 0:
                    ctrl_targets[:] = model.key_ctrl[0]
                    gripper_ctrl = float(ctrl_targets[gripper_act_id])
                    print("Reset to home keyframe controls.")
                if key == ord("o"):
                    gripper_ctrl = min(gripper_max, gripper_ctrl + args.gripper_step)
                    ctrl_targets[gripper_act_id] = gripper_ctrl
                    print(f"gripper_ctrl={gripper_ctrl:.2f}")
                if key == ord("p"):
                    gripper_ctrl = max(gripper_min, gripper_ctrl - args.gripper_step)
                    ctrl_targets[gripper_act_id] = gripper_ctrl
                    print(f"gripper_ctrl={gripper_ctrl:.2f}")

                # Easy two-row layout for joint teleop.
                joint_key_pairs = [
                    (ord("q"), ord("a")),
                    (ord("w"), ord("s")),
                    (ord("e"), ord("d")),
                    (ord("r"), ord("f")),
                    (ord("t"), ord("g")),
                    (ord("y"), ord("h")),
                    (ord("u"), ord("j")),
                ]
                for j, (key_pos, key_neg) in enumerate(joint_key_pairs):
                    if j >= len(arm_act_ids):
                        continue
                    act_id = arm_act_ids[j]
                    cmin = float(model.actuator_ctrlrange[act_id, 0])
                    cmax = float(model.actuator_ctrlrange[act_id, 1])
                    if key == key_pos:
                        ctrl_targets[act_id] = min(cmax, float(ctrl_targets[act_id]) + joint_step)
                        print(f"joint{j + 1}={ctrl_targets[act_id]:.4f}")
                    if key == key_neg:
                        ctrl_targets[act_id] = max(cmin, float(ctrl_targets[act_id]) - joint_step)
                        print(f"joint{j + 1}={ctrl_targets[act_id]:.4f}")

                if key == 32:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    shot_dir = capture_dir / stamp
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    rgb_path = shot_dir / "rgb.png"
                    depth_path = shot_dir / "depth.npy"
                    depth_vis_path = shot_dir / "depth_vis.png"

                    _save_png(rgb_path, bgr, cv2)
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera=args.wrist_camera)
                    depth_buf = renderer.render()
                    renderer.disable_depth_rendering()
                    depth_m = _depth_to_meters(model, depth_buf)
                    np.save(depth_path, depth_m)
                    _save_depth_vis(depth_vis_path, depth_m, cv2)
                    print(f"snapshot saved: {shot_dir}")

                step_count += 1
                if args.max_steps > 0 and step_count >= args.max_steps:
                    break
                time.sleep(args.dt)
    finally:
        renderer.close()
        cv2.destroyAllWindows()

    print(f"steps={step_count}")
    print(f"time={data.time:.6f}")


if __name__ == "__main__":
    main()
