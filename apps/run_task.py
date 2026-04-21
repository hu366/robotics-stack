from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from interfaces.execution_trace import ExecutionTrace
    from modules.control import MjctrlMPCBackend, PlanExecutor, SymbolicControlBackend
    from modules.grounding import SceneGrounder
    from modules.planner import TaskPlanner
    from modules.task_parser import TaskParser
    from modules.world_model import WorldModelStore

    parser = argparse.ArgumentParser(description="Run a single task instruction.")
    parser.add_argument("instruction", help="Natural language instruction to execute.")
    parser.add_argument(
        "--trace-out",
        type=Path,
        help="Optional path to write the execution trace as JSON.",
    )
    parser.add_argument(
        "--control-backend",
        choices=["symbolic", "mjctrl_mpc"],
        default="mjctrl_mpc",
        help="Control backend implementation used by the closed-loop executor.",
    )
    args = parser.parse_args()

    task_parser = TaskParser()
    grounder = SceneGrounder()
    world_model = WorldModelStore()
    planner = TaskPlanner()
    backend = SymbolicControlBackend() if args.control_backend == "symbolic" else MjctrlMPCBackend()
    executor = PlanExecutor(world_model=world_model, backend=backend)

    task = task_parser.parse(args.instruction)
    trace = ExecutionTrace(trace_id=f"trace-{uuid4().hex[:8]}", task_id=task.task_id)
    trace.add_event("task_parser", "task_parsed", "success", payload=task.to_dict())

    world_state = world_model.update(grounder.ground(task))
    trace.add_event("grounding", "world_state_grounded", "success", payload=world_state.to_dict())

    plan = planner.build_plan(task, world_state)
    trace.add_event("planner", "plan_built", "success", payload=plan.to_dict())

    executor.execute(plan, trace)

    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    print(json.dumps(trace.to_dict(), indent=2))


if __name__ == "__main__":
    main()
