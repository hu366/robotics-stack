# Apps（中文）

本目录包含面向用户的入口脚本。

## 前置条件

先安装依赖：

```bash
uv sync --group dev --group sim
```

## 脚本

- `run_task.py`
  - 用途：将一条自然语言指令按 `parser -> grounding -> planner -> executor` 流程完整执行。
  - 示例：
    ```bash
    uv run python apps/run_task.py "place the bottle on the tray" --trace-out artifacts/trace.json
    ```

- `run_benchmark.py`
  - 用途：执行基准测试 JSON 套件并输出报告 JSON。
  - 示例：
    ```bash
    uv run python apps/run_benchmark.py --cases eval/benchmarks/tabletop_v0.json
    ```

- `replay_trace.py`
  - 用途：在终端中回放已保存的执行轨迹 JSON。
  - 示例：
    ```bash
    uv run python apps/replay_trace.py artifacts/trace.json
    ```

- `run_mujoco_smoke.py`
  - 用途：无界面 MuJoCo 冒烟测试（无 GUI），用于快速验证环境是否可用。
  - 默认场景：`sim/scenes/mujoco_minimal.xml`
  - Panda 场景示例：
    ```bash
    uv run python apps/run_mujoco_smoke.py --scene sim/assets/franka_emika_panda/scene.xml --steps 200
    ```

- `run_mujoco_viewer.py`
  - 用途：打开 MuJoCo 图形界面并实时运行仿真。
  - 常用参数：`--scene`、`--dt`、`--max-steps`、`--camera`
  - Panda 场景示例：
    ```bash
    uv run python apps/run_mujoco_viewer.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd
    ```

- `run_mujoco_dual_view.py`
  - 用途：同时打开两个视图：
    1) MuJoCo 整机视窗；2) 腕部相机 RGB 视窗。
  - 遥操作按键（先让 RGB 窗口获得焦点）：
    - 关节控制：
      - `q/a`：关节1 +/-
      - `w/s`：关节2 +/-
      - `e/d`：关节3 +/-
      - `r/f`：关节4 +/-
      - `t/g`：关节5 +/-
      - `y/h`：关节6 +/-
      - `u/j`：关节7 +/-
    - 夹爪：
      - `o`：张开
      - `p`：闭合
    - 快照：
      - `space`：在 `artifacts/teleop_captures/` 下创建时间戳目录，并保存：
        - `rgb.png`
        - `depth.npy`
        - `depth_vis.png`
    - 其他：
      - `[` / `]`：减小/增大关节步长
      - `m`：重置到 home 关键帧控制值
      - `esc`：退出
  - 说明：场景中使用 `wrist_rgbd_origin`（红色小球）标记腕部相机位置。
  - 示例：
    ```bash
    uv run python apps/run_mujoco_dual_view.py --scene sim/assets/franka_emika_panda/scene.xml --wrist-camera wrist_rgbd
    ```

- `capture_mujoco_rgbd.py`
  - 用途：从 MuJoCo 相机采集一帧 RGB 与深度图。
  - 默认相机：`wrist_rgbd`（挂载在 Panda 手部链路）。
  - 输出：`rgb.png`、`depth.npy`（米）、`depth_vis.png`
  - Panda 场景示例：
    ```bash
    uv run python apps/capture_mujoco_rgbd.py --scene sim/assets/franka_emika_panda/scene.xml --camera wrist_rgbd --out-dir artifacts/rgbd
    ```
