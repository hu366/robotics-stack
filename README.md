# Robotics Stack

This repository is organized around a modular robotics stack:

- `apps/`: runnable entry points
- `modules/`: core system capabilities
- `interfaces/`: shared schemas and contracts
- `sim/`: simulation assets and tasks
- `eval/`: benchmarks, metrics, and reports
- `docs/`: design documents

The intended execution flow is:

`instruction -> task parser -> grounding -> world model -> planner -> skills -> control -> evaluation`

## Local Development

This repository uses `uv` for Python environment management.

Typical local commands:

```bash
uv sync --group dev
uv run pytest -q
uv run python apps/run_task.py "place the bottle on the tray"
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
docker compose run --rm app uv run python apps/run_task.py "put the cup on the shelf"
```

Run the benchmark:

```bash
docker compose run --rm benchmark
```

Run tests:

```bash
docker compose run --rm test
```
