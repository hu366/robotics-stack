from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from interfaces.execution_trace import ExecutionTrace
    from modules.control import PlanExecutor
    from modules.grounding import SceneGrounder
    from modules.planner import TaskPlanner
    from modules.task_parser import TaskParser
    from modules.world_model import WorldModelStore

    parser = argparse.ArgumentParser(description="Run benchmark scenarios.")
    parser.add_argument(
        "--suite",
        default="tabletop_v0",
        help="Benchmark suite name.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/benchmarks/tabletop_v0.json"),
        help="Path to benchmark case JSON.",
    )
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    parser_module = TaskParser()
    grounder = SceneGrounder()
    world_model = WorldModelStore()
    planner = TaskPlanner()
    executor = PlanExecutor(world_model=world_model)

    report: list[dict[str, object]] = []
    for case in cases:
        task = parser_module.parse(case["instruction"])
        trace = ExecutionTrace(trace_id=f"trace-{case['case_id']}", task_id=task.task_id)
        trace.add_event("benchmark", "case_loaded", "success", payload=case)
        world_state = world_model.update(grounder.ground(task))
        trace.add_event(
            "grounding",
            "world_state_grounded",
            "success",
            payload=world_state.to_dict(),
        )
        plan = planner.build_plan(task, world_state)
        trace.add_event("planner", "plan_built", "success", payload=plan.to_dict())
        result = executor.execute(plan, trace)
        report.append(
            {
                "case_id": case["case_id"],
                "suite": args.suite,
                "instruction": case["instruction"],
                "success": result.success,
                "event_count": len(trace.events),
            }
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
