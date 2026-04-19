# Robotics Stack / 机器人栈

[English Version](./README.md)

模块化、可解释的机器人系统基线。

## 项目定位

这个仓库不是一个黑盒的端到端 VLA demo。
它是一个模块化基线，用来构建在任务复杂度上升时依然可检查、可调试、可测试的机器人系统。

我们不把语言当成几何、规划、控制或评测的替代品。
我们把语言视为更大系统中的一个输入；这个系统仍然必须显式建模任务、世界状态、约束和执行过程。

## 项目动机

这个仓库的动机部分来自下面这篇文章：

- [如何看待目前VLA的具身智能技术？ - 弗雷尔卓德的回答 - 知乎](https://www.zhihu.com/question/1920708362489828723/answer/1920722548087292522)

这个项目并不试图逐字复述那篇文章中的所有观点，但认同它提出的工程性担忧：
机器人系统不应把任务失败隐藏在黑盒流水线、选择性 demo 或模糊的“泛化”表述后面。

## 设计态度

我们更倾向于：

- 用显式任务分解，而不是不透明的端到端捷径
- 用语义解析加几何 grounding，而不是把指令直接压缩成动作
- 用可复用的 skill 计划，而不是被 prompt 塑形的策略行为
- 用结构化 trace，而不是一句话“成功率”
- 用可复现 benchmark，而不是挑选过的 demo

## 非目标

这个仓库当前并不声称自己是：

- 通用机器人 foundation model
- 真实世界零样本操作系统
- 面向生产的 ROS 2、MoveIt、Gazebo 或硬件运行时
- 围绕展示性指标优化的 benchmark 榜单项目

相反，它是一个用于验证接口、任务分解、trace 和评测纪律的基线工程。

## 系统流水线

预期执行流程为：

`instruction -> task parser -> grounding -> world model -> planner -> skills -> control -> evaluation`

为什么采用这种结构：

- 它能让失败可归因。
- 它允许检查中间状态。
- 它让测试能直接命中真正拥有该行为的那一层。
- 它避免未来接入仿真或硬件时，整套系统塌缩成一个无法解释的 policy 黑盒。

## 仓库结构

- `apps/`：可运行入口
- `modules/`：核心系统能力
- `interfaces/`：共享 schema 与契约
- `sim/`：仿真资产与任务
- `eval/`：benchmark、指标与报告
- `docs/`：设计文档
- `tests/`：流水线回归测试

## 当前状态

当前已具备：

- 任务解析基线
- 场景 grounding 基线
- 规划器基线
- CLI 任务执行
- execution trace 输出
- benchmark 运行器
- 基础端到端测试

当前仍然有意缺失：

- 更丰富的 3D 感知与场景图构建
- 低层闭环控制
- 物理一致的仿真集成
- 面向硬件的适配层
- 强真实世界操作能力宣称

## 示例

输入：

```text
把瓶子放到托盘上
```

系统应产出可检查的中间结果，例如：

- 解析后的 `TaskSpec`
- 物体与目标位置的 grounding 结果
- 基于 skill 的计划
- 执行器写出的步骤 trace

重点不只是“任务做完了”。
重点是要知道系统究竟在哪一层成功、失败，或做了没有根据的假设。

## 评测态度

我们不希望评测退化成剪辑过的 demo 和模糊的成功率数字。

benchmark case 应明确记录：

- 场景设置
- 物体数量与杂乱程度
- 目标指令
- 允许重试次数
- 成功标准
- 失败分类
- 随机种子与重复次数

失败案例和 trace 与成功运行同样重要。

## 本地开发

本仓库使用 `uv` 管理 Python 环境。

常用本地命令：

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run mypy .
uv run python apps/run_task.py "把瓶子放到托盘上"
uv run python apps/run_benchmark.py --cases eval/benchmarks/tabletop_v0.json
```

## 容器环境

仓库提供了一个轻量级 Docker 环境，用于可复现的纯 Python 开发任务。

它适用于：

- 运行单任务入口
- 运行 benchmark 脚本
- 运行测试套件

它目前并不是 `docs/toolchain.md` 中未来可能接入的 ROS 2、MoveIt、Gazebo 或硬件运行时环境。

构建镜像：

```bash
docker compose build
```

运行默认单任务示例，并将 trace 写入 `./artifacts/trace.json`：

```bash
docker compose run --rm app
```

覆盖任务指令：

```bash
docker compose run --rm app uv run python apps/run_task.py "把杯子放到架子上"
```

运行 benchmark：

```bash
docker compose run --rm benchmark
```

运行测试：

```bash
docker compose run --rm test
```

## 贡献约定

如果你要扩展这个仓库：

- 把行为加在真正拥有它的模块里
- 更新接口，而不是传递临时拼凑的原始结构
- 对非平凡变更补充或更新测试
- 保持 trace 与 benchmark 预期和行为一致
- 架构变化要写进文档，而不是藏在实现里

## 路线图

近期方向：

- 更丰富的任务 schema 与失败语义
- 更显式的世界状态与场景建模
- 更强的 planner 可检查性
- 更严格的 benchmark 协议与报告
- 具有清晰几何和物理边界的仿真适配层
