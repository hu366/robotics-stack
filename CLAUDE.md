# AGENTS.md

本文件定义了编码代理在这个仓库中的工作方式。

## 仓库目标

`robotics-stack` 是一个模块化、可解释的机器人系统基线。预期执行流程为：

`instruction -> task parser -> grounding -> world model -> planner -> skills -> control -> evaluation`

代理在工作时应保持这套分解结构。不要把本应属于某个明确模块边界的改动，压缩成一条不透明的端到端路径。

## 核心原则

1. 优先使用显式接口，而不是隐式耦合。
2. 保持任务语义、几何 grounding、规划、执行和评估彼此分离。
3. 通过 trace、结构化输出和测试，让系统行为可检查。
4. 在引入抽象、启发式或学习式复杂度之前，优先采用简单、确定性的基线方案。
5. 当行为发生变化时，应修改真正拥有该行为的最近一层，而不是在下游打补丁。

## 仓库结构说明

- `apps/`：可运行入口，例如单任务执行、trace 回放和 benchmark 运行
- `interfaces/`：共享 schema 和契约
- `modules/task_parser/`：将指令解析为 `TaskSpec`
- `modules/grounding/`：语义到场景的 grounding
- `modules/world_model/`：状态存储和世界表示
- `modules/planner/`：基于 skill 的计划构建
- `modules/skills/`：可复用 skill 定义与查找
- `modules/control/`：执行层
- `eval/`：benchmark 输入和报告
- `sim/`：面向仿真的资产与任务脚手架
- `docs/`：架构与协议文档
- `tests/`：流水线与 CLI 的回归测试

## 工作规则

### 1. 尊重模块归属

- 解析相关改动应放在 `modules/task_parser/` 和 `interfaces/task_spec.py`。
- grounding 相关改动应放在 `modules/grounding/` 以及相关 world-state 接口。
- 规划逻辑应放在 `modules/planner/`，并消费结构化输入，而不是重新解析原始文本。
- 执行行为应放在 `modules/control/`。
- 跨模块的数据结构应定义在 `interfaces/` 中，不要在实现里临时拼装。

### 2. 保持流水线可检查

- 修改流水线行为时，应保留或增强 execution trace。
- 当信息跨模块传递时，优先使用结构化字段，而不是自由文本字符串。
- 如果 CLI 会写出 artifact，在任务没有明确要求 breaking change 的情况下，应保持输出格式稳定。

### 3. 测试是改动的一部分

- 任何非平凡的行为变更，都要新增或更新测试。
- 优先为改动所在模块补充聚焦的单元测试。
- 除非用户明确要求故意改变行为，否则应保持 `tests/test_pipeline.py` 通过。

### 4. 文档必须跟上设计变更

- 用户可见的工作流变更，应更新 `README.md`。
- 架构预期、任务 schema 或 benchmark protocol 变化时，应更新 `docs/` 下文档。
- 不要让 `AGENTS.md`、文档和代码对流水线的描述互相矛盾。

## 开发工作流

本仓库使用 `uv` 执行本地命令。

常用命令：

```powershell
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run mypy .
uv run python apps/run_task.py "place the bottle on the tray"
uv run python apps/run_benchmark.py --cases eval/benchmarks/tabletop_v0.json
```

如果修改了 CLI，在条件允许时应本地运行最相关的 CLI 命令做验证。

## 实现指导

### 添加新能力时

1. 先扩展相关接口类型。
2. 再实现拥有该能力的模块。
3. 以最小耦合方式把新数据贯通到相邻层。
4. 添加或更新测试。
5. 如果该能力改变了公开行为或仓库约定，则更新文档。

### 修改规划行为时

- 优先使用可复用的 skill step，而不是为整条指令写特殊分支。
- 保持计划可读、可调试。
- 不要把执行层假设直接编码进 parser 输出。

### 修改评估或 benchmark 时

- 保持 benchmark case 可复现。
- 尽量保留明确的成功标准、重试次数、随机种子和失败分类。
- 不要在没有更新协议或测试依据的情况下夸大成功效果。

## 风格要求

- 目标 Python 版本为 3.11。
- 代码应兼容仓库中严格的 `mypy` 配置。
- 遵循 `pyproject.toml` 中配置的 `ruff` 规则。
- 在仓库已有模式下，优先使用小而类型明确的函数，以及基于 dataclass 或 schema 的接口。
- 除非至少有两个调用点能立即受益，否则不要引入预防式抽象。

## 需要避免的事情

- 不要在模块间引入隐藏的全局状态。
- 当已有类型化结构存在时，不要绕过接口直接传 raw dict。
- 不要把仅供 benchmark 使用的捷径混入生产路径，除非显式标明。
- 不要在没有明确必要的情况下增加重量级依赖。
- 不要因为更短，就用单体式捷径替代模块化流水线逻辑。

## 验证清单

在完成任务前，代理通常应根据情况执行以下检查：

1. 运行目标测试或 `uv run pytest -q`。
2. 如果改了 Python 文件，运行 `uv run ruff check .`。
3. 如果改了接口或类型密集代码，运行 `uv run mypy .`。
4. 对受影响的 CLI 入口做一次基本验证。
5. 确认文档与实现行为保持一致。

## 决策标准

当存在多个可选方案时，优先选择满足以下条件的方案：

- 保持流水线模块化，
- 提升可检查性，
- 最容易测试，
- 并且符合当前仓库的基线复杂度。
