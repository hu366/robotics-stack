# Simulation（中文）

本目录包含仿真场景、对象资产、任务定义和传感器配置。

## MuJoCo 冒烟场景

仓库内置了一个用于本地环境验证的小型 MuJoCo 场景：

- 场景文件：`sim/scenes/mujoco_minimal.xml`
- 运行脚本：`apps/run_mujoco_smoke.py`

运行方式：

```bash
uv sync --group dev --group sim
uv run python apps/run_mujoco_smoke.py --steps 500
```

## Franka Emika Panda（MuJoCo）

Franka Panda 模型位于：

- `sim/assets/franka_emika_panda/scene.xml`

运行无界面冒烟测试：

```bash
uv run python apps/run_mujoco_smoke.py --scene sim/assets/franka_emika_panda/scene.xml --steps 200
```

打开图形界面：

```bash
uv run python apps/run_mujoco_viewer.py --scene sim/assets/franka_emika_panda/scene.xml
```

打开双视图（整机 + 腕部相机 RGB）：

```bash
uv run python apps/run_mujoco_dual_view.py --scene sim/assets/franka_emika_panda/scene.xml --wrist-camera wrist_rgbd
```

在双视图 RGB 窗口中可使用键盘遥操作：
`q/a w/s e/d r/f t/g y/h u/j` 控关节，`o/p` 控夹爪，`space` 触发快照。

采集 RGB-D：

```bash
uv run python apps/capture_mujoco_rgbd.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd --out-dir artifacts/rgbd
```
