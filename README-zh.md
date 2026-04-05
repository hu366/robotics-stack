# Robotics Stack（中文）

模块化、可解释的机器人系统基线项目。

## 项目定位

本仓库不是黑盒端到端 VLA 演示，而是一个可检查、可调试、可测试的工程化基线。
系统强调显式任务分解、几何与语义结合、可追踪执行过程。

## 系统流程

`instruction -> task parser -> grounding -> world model -> planner -> skills -> control -> evaluation`

## 仓库结构

- `apps/`：可运行入口脚本
- `modules/`：核心能力模块
- `interfaces/`：模块间契约与数据结构
- `sim/`：仿真资产与场景
- `eval/`：评测与报告
- `docs/`：设计与工具链文档
- `tests/`：回归测试

## 本地开发

```bash
uv sync --group dev --group sim
uv run pytest -q
uv run ruff check .
uv run mypy .
```

## 仿真相关常用命令

```bash
uv run python apps/run_mujoco_viewer.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd
uv run python apps/run_mujoco_dual_view.py --scene sim/assets/franka_emika_panda/scene.xml --wrist-camera wrist_rgbd
uv run python apps/capture_mujoco_rgbd.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd --out-dir artifacts/rgbd
```

## 说明

英文详细说明请查看 [README.md](./README.md)。
