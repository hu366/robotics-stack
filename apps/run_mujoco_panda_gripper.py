from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_model_with_ascii_fallback(model_path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(model_path))
    except ValueError as primary_err:
        source_dir = model_path.parent
        temp_root = Path(tempfile.gettempdir())
        fallback_root = temp_root / "mjctrl_ascii_fallback"
        fallback_scene_dir = fallback_root / "panda_scene"
        fallback_scene_path = fallback_scene_dir / model_path.name
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
                f"Primary: {model_path}\n"
                f"Primary error: {primary_err}\n"
                f"Fallback: {fallback_scene_path}\n"
                f"Fallback error: {fallback_err}"
            ) from None


def _input_worker(command_queue: Queue[str], stop_event: threading.Event) -> None:
    print("Commands: open | close | set <0-255> | width <0-255> | quit")
    while not stop_event.is_set():
        try:
            line = input("> ")
        except EOFError:
            stop_event.set()
            return
        command_queue.put(line)


def main() -> None:
    from modules.control.mujoco_gripper_driver import clamp_gripper_ctrl, parse_gripper_command

    parser = argparse.ArgumentParser(
        description="Interactive Panda gripper driver (mjctrl model)."
    )
    parser.add_argument(
        "--gripper",
        type=float,
        default=255.0,
        help="Initial gripper control value in [0, 255]. 255=open, 0=close.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Simulation timestep.",
    )
    args = parser.parse_args()

    model_path = ROOT / "third_party" / "mjctrl" / "franka_emika_panda" / "panda.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"panda xml not found: {model_path}")

    model = _load_model_with_ascii_fallback(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = args.dt

    key_id = model.key("home").id
    arm_actuator_names = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
    ]
    arm_actuator_ids = [model.actuator(name).id for name in arm_actuator_names]
    gripper_actuator_id = model.actuator("actuator8").id

    command_queue: Queue[str] = Queue()
    stop_event = threading.Event()
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

        # Hold the arm at home and only update the gripper command interactively.
        home_ctrl = data.ctrl.copy()
        for act_id in arm_actuator_ids:
            home_ctrl[act_id] = data.ctrl[act_id]

        gripper_ctrl = clamp_gripper_ctrl(args.gripper)
        print(f"Initial gripper ctrl = {gripper_ctrl:.1f}")

        while viewer.is_running() and not stop_event.is_set():
            step_start = time.time()

            while True:
                try:
                    raw = command_queue.get_nowait()
                except Empty:
                    break
                parsed = parse_gripper_command(raw)
                if parsed.kind == "quit":
                    stop_event.set()
                    break
                if parsed.kind == "help":
                    print("Commands: open | close | set <0-255> | width <0-255> | quit")
                    continue
                if parsed.kind == "invalid":
                    print(f"Invalid input: {parsed.message}")
                    continue
                if parsed.target_ctrl is None:
                    print("Invalid command: missing target")
                    continue
                gripper_ctrl = clamp_gripper_ctrl(parsed.target_ctrl)
                print(f"Gripper target ctrl set to {gripper_ctrl:.1f}")

            if stop_event.is_set():
                break

            data.ctrl[:] = home_ctrl
            data.ctrl[gripper_actuator_id] = gripper_ctrl
            mujoco.mj_step(model, data)
            viewer.sync()

            left = float(data.joint("finger_joint1").qpos[0])
            right = float(data.joint("finger_joint2").qpos[0])
            if int(data.time * 10) != int((data.time - model.opt.timestep) * 10):
                print(
                    "Finger width state: "
                    f"left={left:.4f}, right={right:.4f}, total={left + right:.4f}"
                )

            time_until_next_step = args.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    stop_event.set()
    print("Gripper driver stopped.")


if __name__ == "__main__":
    main()
