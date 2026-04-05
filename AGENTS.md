# AGENTS.md

This file defines how coding agents should work in this repository.

## Repository Intent

`robotics-stack` is a modular, interpretable robotics baseline. The intended execution flow is:

`instruction -> task parser -> grounding -> world model -> planner -> skills -> control -> evaluation`

Agents should preserve that decomposition. Do not collapse new features into a single opaque end-to-end path when the change belongs in a specific module boundary.

## Core Principles

1. Prefer explicit interfaces over hidden coupling.
2. Keep task semantics, geometric grounding, planning, execution, and evaluation separable.
3. Make behavior inspectable through traces, structured outputs, and tests.
4. Favor simple deterministic baselines before adding abstractions, heuristics, or learning-oriented complexity.
5. When a behavior changes, update the closest layer that owns that behavior instead of patching around it downstream.

## Repository Map

- `apps/`: runnable entry points such as single-task execution, trace replay, and benchmark runs
- `interfaces/`: shared schemas and contracts
- `modules/task_parser/`: instruction parsing into `TaskSpec`
- `modules/grounding/`: semantic-to-scene grounding
- `modules/world_model/`: state storage and world representation
- `modules/planner/`: skill-level plan construction
- `modules/skills/`: reusable skill definitions and lookup
- `modules/control/`: execution layer
- `eval/`: benchmark inputs and reports
- `sim/`: simulation-facing assets and task scaffolding
- `docs/`: architecture and protocol documents
- `tests/`: regression coverage for the pipeline and CLIs

## Working Rules

### 1. Respect module ownership

- Parsing changes belong in `modules/task_parser/` and `interfaces/task_spec.py`.
- Grounding changes belong in `modules/grounding/` and relevant world-state interfaces.
- Planning logic belongs in `modules/planner/` and should consume structured inputs instead of reparsing raw text.
- Execution behavior belongs in `modules/control/`.
- Cross-module shapes should be defined in `interfaces/`, not ad hoc inside implementations.

### 2. Keep the pipeline inspectable

- Preserve or improve execution traces when changing pipeline behavior.
- Prefer structured fields over free-form strings when information crosses module boundaries.
- If a CLI writes artifacts, keep output formats stable unless the task explicitly requires a breaking change.

### 3. Tests are part of the change

- Add or update tests for any non-trivial behavior change.
- Prefer focused unit coverage close to the changed module.
- Keep `tests/test_pipeline.py` passing unless the user explicitly asks for a deliberate behavior change.

### 4. Documentation should track design changes

- Update `README.md` for user-facing workflow changes.
- Update docs under `docs/` when architectural expectations, task schema, or benchmark protocol change.
- Do not leave `AGENTS.md`, docs, and code disagreeing on the pipeline.

## Development Workflow

Use `uv` for local commands.

Typical commands:

```powershell
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run mypy .
uv run python apps/run_task.py "place the bottle on the tray"
uv run python apps/run_benchmark.py --cases eval/benchmarks/tabletop_v0.json
```

If you change a CLI, run the most relevant CLI command locally when feasible.

## Implementation Guidance

### When adding a new capability

1. Extend the relevant interface type first.
2. Implement the owning module.
3. Thread the new data through adjacent layers with minimal coupling.
4. Add or update tests.
5. Update docs if the capability changes public behavior or repository conventions.

### When changing planning behavior

- Prefer reusable skill steps over special-casing full instructions.
- Keep plans readable and debuggable.
- Avoid embedding execution-side assumptions directly into parser output.

### When changing evaluation or benchmarks

- Keep benchmark cases reproducible.
- Preserve explicit success criteria, retries, seeds, and failure taxonomy where applicable.
- Avoid overstating success without updating the protocol or tests that justify it.

## Style Expectations

- Target Python 3.11.
- Keep code compatible with the repository's strict `mypy` configuration.
- Follow `ruff` defaults configured in `pyproject.toml`.
- Prefer small, typed functions and dataclass-based or schema-based interfaces where the repository already uses them.
- Avoid speculative abstractions unless at least two call sites benefit immediately.

## What To Avoid

- Do not introduce hidden global state across modules.
- Do not bypass interfaces by passing raw dicts when typed structures already exist.
- Do not mix benchmark-only shortcuts into production code paths without making them explicit.
- Do not add heavyweight dependencies without clear need.
- Do not replace modular pipeline logic with a monolithic shortcut just because it is shorter.

## Validation Checklist

Before finishing a task, agents should usually do the following when relevant:

1. Run targeted tests or `uv run pytest -q`.
2. Run `uv run ruff check .` if Python files changed.
3. Run `uv run mypy .` when interface or type-heavy code changed.
4. Sanity-check any affected CLI entry point.
5. Confirm docs stay aligned with the implemented behavior.

## Decision Standard

When multiple solutions are possible, prefer the one that:

- keeps the pipeline modular,
- improves inspectability,
- is easiest to test,
- and matches the repository's current baseline complexity.
