# Robotics Stack

[中文说明 / Chinese Version](./README-zh.md)

Modular, interpretable robotics stack baseline.

## Positioning

This repository is not a black-box end-to-end VLA demo.
It is a modular baseline for building robotics systems that remain inspectable, debuggable, and testable as task complexity grows.

We do not treat language as a substitute for geometry, planning, control, or evaluation.
We treat language as one input to a larger system that must still model tasks, world state, constraints, and execution explicitly.

## Motivation

This repository was motivated in part by the following essay:

- [如何看待目前VLA的具身智能技术？ - 弗雷尔卓德的回答 - 知乎](https://www.zhihu.com/question/1920708362489828723/answer/1920722548087292522)

The project does not attempt to mirror every claim in that article verbatim, but it shares the same engineering concern:
robotics should not hide task failure behind black-box pipelines, selective demos, or vague claims of generalization.

## Design Stance

We prefer:

- explicit task decomposition over opaque end-to-end shortcuts
- semantic parsing plus geometric grounding over direct instruction-to-action collapse
- reusable skill plans over prompt-shaped policy behavior
- structured traces over one-line success claims
- reproducible benchmarks over hand-picked demos

## Non-Goals

This repository does not currently claim to be:

- a general-purpose robot foundation model
- a zero-shot real-world manipulation system
- a production ROS 2, MoveIt, Gazebo, or hardware runtime
- a benchmark leaderboard project optimized around presentation metrics

Instead, it is a baseline for validating interfaces, decomposition, traces, and evaluation discipline.

## System Flow

The intended execution flow is:

`instruction -> task parser -> grounding -> world model -> planner -> skills -> control -> evaluation`

Why this structure:

- It makes failure attributable.
- It allows intermediate state inspection.
- It lets tests target the layer that actually owns the behavior.
- It keeps future simulation and hardware integration from collapsing into one opaque policy blob.

## Repository Layout

- `apps/`: runnable entry points
- `modules/`: core system capabilities
- `interfaces/`: shared schemas and contracts
- `sim/`: simulation assets and tasks
- `eval/`: benchmarks, metrics, and reports
- `docs/`: design documents
- `tests/`: regression coverage for the pipeline

## Current Status

Implemented today:

- task parsing baseline
- scene grounding baseline
- planner baseline
- CLI task execution
- execution trace output
- benchmark runner
- basic end-to-end tests

Still intentionally missing:

- rich 3D perception and scene graph construction
- closed-loop low-level control
- physics-consistent simulation integration
- hardware-facing adapters
- strong real-world manipulation claims

## Example

Given:

```text
把瓶子放到托盘上
```

The pipeline is expected to produce inspectable intermediate outputs such as:

- a parsed `TaskSpec`
- grounded object and location references
- a skill-level plan
- a step trace written by the executor

The point is not just to finish the task.
The point is to know where the system succeeded, failed, or made an unjustified assumption.

## Evaluation Attitude

We do not want evaluation to degrade into edited demos and vague success rates.

Benchmark cases should be explicit about:

- scene setup
- object count and clutter level
- target instruction
- allowed retries
- success criteria
- failure taxonomy
- seeds and repeat count

Failure cases and traces matter as much as successful runs.

## Local Development

This repository uses `uv` for Python environment management.

Typical local commands:

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run mypy .
uv run python apps/run_task.py "把瓶子放到托盘上"
uv run python apps/run_benchmark.py --cases eval/benchmarks/tabletop_v0.json
uv sync --group dev --group sim
uv run python apps/run_mujoco_smoke.py --steps 500
```

## Docker

The repository includes a lightweight Docker setup for reproducible Python-only development tasks.

It is intended for:

- running a single task entry point
- running the benchmark script
- running the test suite

It is not intended yet for the future ROS 2, MoveIt, Gazebo, or hardware-facing runtime described in `docs/toolchain.md`.

Build the image:

```bash
docker compose build
```

Run the default single-task example and write a trace to `./artifacts/trace.json`:

```bash
docker compose run --rm app
```

Override the task instruction:

```bash
docker compose run --rm app uv run python apps/run_task.py "把杯子放到架子上"
```

Run the benchmark:

```bash
docker compose run --rm benchmark
```

Run tests:

```bash
docker compose run --rm test
```

## Contribution Expectations

If you extend this repository:

- add behavior in the module that owns it
- update interfaces instead of passing raw ad hoc structures
- add or update tests for non-trivial changes
- keep traces and benchmark expectations aligned with behavior
- document architectural changes instead of hiding them in implementation

## Roadmap

Near-term directions:

- richer task schema and failure semantics
- more explicit world-state and scene modeling
- stronger planner introspection
- tighter benchmark protocol and reporting
- simulation adapters with clearer geometry and physics boundaries
