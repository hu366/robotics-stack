from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_model_with_ascii_fallback(scene_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except ValueError as primary_err:
        source_dir = scene_path.parent
        temp_root = Path(tempfile.gettempdir())
        fallback_root = temp_root / "mjctrl_ascii_fallback"
        fallback_scene_dir = fallback_root / "ur5e_scene"
        fallback_scene_path = fallback_scene_dir / scene_path.name
        fallback_root.mkdir(parents=True, exist_ok=True)
        if fallback_scene_dir.exists():
            shutil.rmtree(fallback_scene_dir)
        shutil.copytree(source_dir, fallback_scene_dir)
        print(
            "Direct MuJoCo XML load failed; using ASCII fallback path: "
            f"{fallback_scene_path}"
        )
        try:
            return mujoco.MjModel.from_xml_path(str(fallback_scene_path))
        except ValueError as fallback_err:
            raise RuntimeError(
                "MuJoCo model load failed on both primary and fallback paths.\n"
                f"Primary: {scene_path}\n"
                f"Primary error: {primary_err}\n"
                f"Fallback: {fallback_scene_path}\n"
                f"Fallback error: {fallback_err}"
            ) from None


def _input_worker(command_queue: Queue[str], stop_event: threading.Event) -> None:
    print("Input target: `x y z` or `--x X --y Y --z Z`; type `quit` to exit.")
    while not stop_event.is_set():
        try:
            line = input("> ")
        except EOFError:
            stop_event.set()
            return
        command_queue.put(line)


def main() -> None:
    from modules.control.mujoco_target_driver import parse_target_input, plan_linear_waypoints

    parser = argparse.ArgumentParser(
        description="Move MuJoCo end-effector to xyz targets with repeated commands."
    )
    parser.add_argument("--x", type=float, required=True, help="Initial target x.")
    parser.add_argument("--y", type=float, required=True, help="Initial target y.")
    parser.add_argument("--z", type=float, required=True, help="Initial target z.")
    parser.add_argument(
        "--waypoint-steps",
        type=int,
        default=30,
        help="Number of linear waypoints per command.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Simulation timestep.",
    )
    parser.add_argument(
        "--integration-dt",
        type=float,
        default=1.0,
        help="Integration horizon for differential IK.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=1e-4,
        help="Damping regularizer for differential IK.",
    )
    parser.add_argument(
        "--max-angvel",
        type=float,
        default=0.0,
        help="Max joint angular velocity. Set 0 to disable clipping.",
    )
    args = parser.parse_args()

    scene_path = ROOT / "third_party" / "mjctrl" / "universal_robots_ur5e" / "scene.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"scene xml not found: {scene_path}")

    model = _load_model_with_ascii_fallback(scene_path)
    data = mujoco.MjData(model)
    model.opt.timestep = args.dt

    site_id = model.site("attachment_site").id
    dof_ids = np.array(
        [
            model.joint("shoulder_pan").id,
            model.joint("shoulder_lift").id,
            model.joint("elbow").id,
            model.joint("wrist_1").id,
            model.joint("wrist_2").id,
            model.joint("wrist_3").id,
        ]
    )
    actuator_ids = np.array(
        [
            model.actuator("shoulder_pan").id,
            model.actuator("shoulder_lift").id,
            model.actuator("elbow").id,
            model.actuator("wrist_1").id,
            model.actuator("wrist_2").id,
            model.actuator("wrist_3").id,
        ]
    )
    key_id = model.key("home").id
    mocap_id = model.body("target").mocapid[0]

    jac = np.zeros((6, model.nv))
    diag = args.damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    command_queue: Queue[str] = Queue()
    stop_event = threading.Event()
    waypoints: deque[np.ndarray] = deque()

    input_thread = threading.Thread(
        target=_input_worker,
        args=(command_queue, stop_event),
        daemon=True,
    )
    input_thread.start()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        initial_target = np.array([args.x, args.y, args.z], dtype=np.float64)
        start_position = data.mocap_pos[mocap_id].copy().tolist()
        for waypoint in plan_linear_waypoints(
            start_position,
            initial_target.tolist(),
            args.waypoint_steps,
        ):
            waypoints.append(waypoint)
        print(f"Initial target set to ({args.x:.3f}, {args.y:.3f}, {args.z:.3f})")

        while viewer.is_running() and not stop_event.is_set():
            step_start = time.time()

            while True:
                try:
                    raw = command_queue.get_nowait()
                except Empty:
                    break
                parsed = parse_target_input(raw)
                if parsed.kind == "quit":
                    stop_event.set()
                    break
                if parsed.kind == "help":
                    print("Use: x y z  OR  --x X --y Y --z Z; type quit to exit.")
                    continue
                if parsed.kind == "invalid":
                    print(f"Invalid input: {parsed.message}")
                    continue
                assert parsed.target is not None
                target = np.array(parsed.target, dtype=np.float64)
                waypoints.clear()
                for waypoint in plan_linear_waypoints(
                    data.mocap_pos[mocap_id].copy().tolist(),
                    target.tolist(),
                    args.waypoint_steps,
                ):
                    waypoints.append(waypoint)
                print(
                    "New target accepted: "
                    f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}), "
                    f"planned {len(waypoints)} waypoints."
                )

            if stop_event.is_set():
                break

            if waypoints:
                next_waypoint = waypoints[0]
                data.mocap_pos[mocap_id] = next_waypoint
                if np.linalg.norm(data.site(site_id).xpos - next_waypoint) < 0.01:
                    waypoints.popleft()

            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            if args.max_angvel > 0:
                dq_abs_max = float(np.abs(dq).max())
                if dq_abs_max > args.max_angvel:
                    dq *= args.max_angvel / dq_abs_max

            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, args.integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = args.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    stop_event.set()
    print("Controller stopped.")


if __name__ == "__main__":
    main()
