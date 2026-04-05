# Simulation

This directory contains simulated scenes, object assets, task definitions, and sensor configs.

## MuJoCo smoke scene

This repository includes a tiny MuJoCo scene for local environment validation:

- scene file: `sim/scenes/mujoco_minimal.xml`
- runner: `apps/run_mujoco_smoke.py`

Run with:

```bash
uv sync --group dev --group sim
uv run python apps/run_mujoco_smoke.py --steps 500
```

## Franka Emika Panda (MuJoCo)

Franka Panda model is available under:

- `sim/assets/franka_emika_panda/scene.xml`

Run smoke test:

```bash
uv run python apps/run_mujoco_smoke.py --scene sim/assets/franka_emika_panda/scene.xml --steps 200
```

Open viewer:

```bash
uv run python apps/run_mujoco_viewer.py --scene sim/assets/franka_emika_panda/scene.xml
```

Open dual view (full-arm + wrist camera RGB):

```bash
uv run python apps/run_mujoco_dual_view.py --scene sim/assets/franka_emika_panda/scene.xml --wrist-camera wrist_rgbd
```

In dual view RGB window, use keyboard teleop:
`q/a w/s e/d r/f t/g y/h u/j` for joints, `o/p` for gripper, `space` for snapshot.

Capture RGB-D:

```bash
uv run python apps/capture_mujoco_rgbd.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd --out-dir artifacts/rgbd
```
