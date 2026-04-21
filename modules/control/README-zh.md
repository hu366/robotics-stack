# Control（中文）

本目录包含控制相关能力，例如：

- 轨迹跟踪
- 伺服回路
- 力控或阻抗控制

## 控制后端

- `SymbolicControlBackend`：用于回归测试的确定性基线控制后端。
- `MjctrlMPCBackend`：控制层闭环 MPC 后端，控制律参考 `third_party/mjctrl` 中的差分 IK 与操作空间控制实现。

## 命令行使用

- `uv run python apps/run_task.py "place the bottle on the tray" --control-backend mjctrl_mpc`
- `uv run python apps/run_benchmark.py --control-backend mjctrl_mpc`
- `uv run python apps/run_mujoco_move_to_point.py --x 0.5 --y 0.0 --z 0.3`
  - 启动后可持续输入新的目标点（`x y z` 或 `--x --y --z`），控制器会重新规划轨迹并驱动机械臂移动。
