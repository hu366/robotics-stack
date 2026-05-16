from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_llm_backend(args: argparse.Namespace) -> Any:
    from modules.vlm import LocalVLMBackend, MockVLMBackend, OpenAIVLMBackend

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if args.llm_provider == "mock":
        return MockVLMBackend(model=args.llm_model or "mock-vlm")
    if args.llm_provider == "openai":
        if not api_key:
            raise ValueError("OpenAI backend requires --api-key or OPENAI_API_KEY")
        return OpenAIVLMBackend(
            model=args.llm_model or "gpt-4.1-mini",
            api_key=api_key,
            base_url=args.llm_base_url or "https://api.openai.com/v1",
        )
    return LocalVLMBackend(
        model=args.llm_model or "llava",
        api_key=api_key,
        base_url=args.llm_base_url or "http://localhost:8000/v1",
    )


def _build_task_parser(args: argparse.Namespace) -> Any:
    from modules.task_parser import LLMTaskParserBackend, RuleTaskParserBackend, TaskParser

    if args.parser_backend == "rule":
        return TaskParser()
    return TaskParser(
        backend=LLMTaskParserBackend(
            backend=_build_llm_backend(args),
            fallback_backend=RuleTaskParserBackend(),
        )
    )


def main() -> None:
    from interfaces.execution_trace import ExecutionTrace
    from modules.control import MjctrlMPCBackend, PlanExecutor, SymbolicControlBackend
    from modules.grounding import SceneGrounder
    from modules.planner import TaskPlanner
    from modules.task_parser import LOW_CONFIDENCE_THRESHOLD
    from modules.world_model import WorldModelStore

    parser = argparse.ArgumentParser(description="Run a single task instruction.")
    parser.add_argument("instruction", help="Natural language instruction to execute.")
    parser.add_argument(
        "--parser-backend",
        choices=("rule", "llm"),
        default="rule",
        help="Task parser backend to use.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("openai", "local", "mock"),
        default="openai",
        help="LLM provider used when --parser-backend=llm.",
    )
    parser.add_argument(
        "--llm-model",
        help="Model identifier for the selected LLM provider.",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Base URL for OpenAI-compatible LLM backends.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the selected backend. Defaults to OPENAI_API_KEY.",
    )
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

    task_parser = _build_task_parser(args)
    grounder = SceneGrounder()
    world_model = WorldModelStore()
    planner = TaskPlanner()
    backend = SymbolicControlBackend() if args.control_backend == "symbolic" else MjctrlMPCBackend()
    executor = PlanExecutor(world_model=world_model, backend=backend)

    task = task_parser.parse(args.instruction)
    diagnostics = task_parser.last_diagnostics
    trace = ExecutionTrace(trace_id=f"trace-{uuid4().hex[:8]}", task_id=task.task_id)

    if diagnostics.fallback_used:
        trace.add_event(
            "task_parser",
            "task_parser_fallback",
            "warning",
            payload={
                "parser_backend": diagnostics.backend,
                "parser_diagnostics": diagnostics.to_dict(),
            },
        )

    task_payload = task.to_dict()
    task_payload["parser_backend"] = diagnostics.backend
    task_payload["parser_diagnostics"] = diagnostics.to_dict()
    task_status = (
        "warning"
        if diagnostics.backend == "llm"
        and not diagnostics.fallback_used
        and diagnostics.has_soft_warnings(threshold=LOW_CONFIDENCE_THRESHOLD)
        else "success"
    )
    trace.add_event("task_parser", "task_parsed", task_status, payload=task_payload)

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
