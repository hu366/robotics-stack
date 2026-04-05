# Apps

This directory contains user-facing entry points.

## Prerequisites

Install dependencies first:

```bash
uv sync --group dev --group sim
```

## Scripts

- `run_task.py`
  - Purpose: run one natural-language instruction through parser -> grounding -> planner -> executor.
  - Example:
    ```bash
    uv run python apps/run_task.py "place the bottle on the tray" --trace-out artifacts/trace.json
    ```

- `run_benchmark.py`
  - Purpose: execute a benchmark JSON suite and print report JSON.
  - Example:
    ```bash
    uv run python apps/run_benchmark.py --cases eval/benchmarks/tabletop_v0.json
    ```

- `replay_trace.py`
  - Purpose: replay a saved execution trace JSON in terminal output.
  - Example:
    ```bash
    uv run python apps/replay_trace.py artifacts/trace.json
    ```

- `run_mujoco_smoke.py`
  - Purpose: headless MuJoCo smoke test (no GUI), useful for fast environment validation.
  - Default scene: `sim/scenes/mujoco_minimal.xml`
  - Panda scene example:
    ```bash
    uv run python apps/run_mujoco_smoke.py --scene sim/assets/franka_emika_panda/scene.xml --steps 200
    ```

- `run_mujoco_viewer.py`
  - Purpose: open MuJoCo GUI viewer and run simulation in real time.
  - Common options: `--scene`, `--dt`, `--max-steps`, `--camera`
  - Panda scene example:
    ```bash
    uv run python apps/run_mujoco_viewer.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd
    ```

- `run_mujoco_dual_view.py`
  - Purpose: open two views at once:
    1) MuJoCo full-arm viewer window, 2) wrist camera RGB window.
  - Teleop keys (focus RGB window first):
    - Joint control:
      - `q/a`: joint1 +/-
      - `w/s`: joint2 +/-
      - `e/d`: joint3 +/-
      - `r/f`: joint4 +/-
      - `t/g`: joint5 +/-
      - `y/h`: joint6 +/-
      - `u/j`: joint7 +/-
    - Gripper:
      - `o`: open
      - `p`: close
    - Snapshot:
      - `space`: create a timestamp folder under `artifacts/teleop_captures/` and save:
        - `rgb.png`
        - `depth.npy`
        - `depth_vis.png`
    - Utility:
      - `[` / `]`: decrease/increase joint step
      - `m`: reset to home keyframe controls
      - `esc`: quit
  - Note: wrist camera marker is visualized in scene as:
    - `wrist_rgbd_origin` (red sphere)
  - Example:
    ```bash
    uv run python apps/run_mujoco_dual_view.py --scene sim/assets/franka_emika_panda/scene.xml --wrist-camera wrist_rgbd
    ```

- `capture_mujoco_rgbd.py`
  - Purpose: capture one RGB frame and one depth frame from a MuJoCo camera.
  - Default camera: `wrist_rgbd` (mounted on Panda hand link).
  - Outputs: `rgb.png`, `depth.npy` (meters), `depth_vis.png`
  - Panda scene example:
    ```bash
    uv run python apps/capture_mujoco_rgbd.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd --out-dir artifacts/rgbd
    ```
