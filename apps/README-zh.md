# Apps（中文）

本目录提供面向使用者的入口脚本。

## 前置依赖

```bash
uv sync --group dev --group sim
```

## 脚本说明

- `run_task.py`：执行单条自然语言任务，走完整流水线。
- `run_benchmark.py`：执行基准测试集并输出报告 JSON。
- `replay_trace.py`：在终端回放已有执行轨迹。
- `run_mujoco_smoke.py`：无界面 MuJoCo 冒烟测试。
- `run_mujoco_viewer.py`：打开 MuJoCo 图形界面。
- `run_mujoco_dual_view.py`：双视窗（整机视角 + 腕部相机视角）并支持键盘遥操作。
- `capture_mujoco_rgbd.py`：采集单帧 RGB 与深度数据。

## 双视窗遥操作按键

请先点击 RGB 小窗口以获取键盘焦点。

- 关节控制：
  - `q/a`：关节1 正/负
  - `w/s`：关节2 正/负
  - `e/d`：关节3 正/负
  - `r/f`：关节4 正/负
  - `t/g`：关节5 正/负
  - `y/h`：关节6 正/负
  - `u/j`：关节7 正/负
- 夹爪：
  - `o`：张开
  - `p`：闭合
- 拍照：
  - `space`：在 `artifacts/teleop_captures/` 下新建时间戳目录，并保存
    - `rgb.png`
    - `depth.npy`
    - `depth_vis.png`
- 其他：
  - `[` / `]`：减小/增大关节步长
  - `m`：回到 home 控制值
  - `esc`：退出

## 示例命令

```bash
uv run python apps/run_mujoco_dual_view.py --scene sim/assets/franka_emika_panda/scene.xml --wrist-camera wrist_rgbd
```
