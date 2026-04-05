# Simulation（中文）

本目录包含仿真场景、对象资产、任务定义和传感器配置。

## MuJoCo 冒烟场景

```bash
uv sync --group dev --group sim
uv run python apps/run_mujoco_smoke.py --steps 500
```

## Franka Emika Panda（MuJoCo）

场景文件：`sim/assets/franka_emika_panda/scene.xml`

### 无界面测试

```bash
uv run python apps/run_mujoco_smoke.py --scene sim/assets/franka_emika_panda/scene.xml --steps 200
```

### 图形界面

```bash
uv run python apps/run_mujoco_viewer.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd
```

### 双视窗 + 键盘遥操作

```bash
uv run python apps/run_mujoco_dual_view.py --scene sim/assets/franka_emika_panda/scene.xml --wrist-camera wrist_rgbd
```

### RGB-D 采集

```bash
uv run python apps/capture_mujoco_rgbd.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd --out-dir artifacts/rgbd
```
