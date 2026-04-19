from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from interfaces.vlm import VLMQueryContext, VLMResponse
from modules.task_parser import (
    LLMTaskParserBackend,
    ParseError,
    ParseFailureMode,
    RuleTaskParserBackend,
    TaskParser,
)

ROOT = Path(__file__).resolve().parents[1]


class StaticBackend:
    def __init__(self, text: str, *, confidence: float | None = None) -> None:
        self.text = text
        self.confidence = confidence
        self.model = "static-llm"

    def query(
        self,
        images: list[bytes],
        prompt: str,
        context: VLMQueryContext | None = None,
    ) -> VLMResponse:
        return VLMResponse(
            text=self.text,
            confidence=self.confidence,
            model_id=self.model,
            latency_ms=5,
            raw={"prompt": prompt, "stage": "" if context is None else context.stage},
        )


class FailingBackend:
    model = "failing-llm"

    def query(
        self,
        images: list[bytes],
        prompt: str,
        context: VLMQueryContext | None = None,
    ) -> VLMResponse:
        raise RuntimeError("backend unavailable")


def test_llm_backend_parses_structured_intent() -> None:
    backend = StaticBackend(
        json.dumps(
            {
                "goal": "place_object",
                "action": "place",
                "arguments": [
                    {"role": "target_object", "text": "瓶子", "entity_type": "object"},
                    {"role": "target_location", "text": "托盘", "entity_type": "object"},
                ],
                "spatial_relation": "on",
                "confidence": 0.93,
                "ambiguities": [],
                "omitted_details": [],
            }
        )
    )
    parser = TaskParser(backend=LLMTaskParserBackend(backend=backend))

    task = parser.parse("把瓶子放到托盘上")

    assert task.goal == "place_object"
    assert task.action == "place"
    assert task.target_object == "瓶子"
    assert task.target_location == "托盘"
    assert task.spatial_relation == "on"
    assert parser.last_diagnostics.backend == "llm"
    assert parser.last_diagnostics.model_id == "static-llm"
    assert parser.last_diagnostics.confidence == 0.93
    assert not parser.last_diagnostics.fallback_used


def test_llm_backend_records_omitted_details_for_compressed_instruction() -> None:
    backend = StaticBackend(
        json.dumps(
            {
                "goal": "place_object",
                "action": "place",
                "arguments": [
                    {"role": "target_object", "text": "杯子", "entity_type": "object"},
                    {"role": "target_location", "text": "托盘", "entity_type": "object"},
                ],
                "spatial_relation": "on",
                "confidence": 0.52,
                "ambiguities": [],
                "omitted_details": ["先拿起杯子"],
            }
        )
    )
    parser = TaskParser(backend=LLMTaskParserBackend(backend=backend))

    task = parser.parse("先拿起杯子再放到托盘上")

    assert task.goal == "place_object"
    assert parser.last_diagnostics.omitted_details == ["先拿起杯子"]
    assert parser.last_diagnostics.has_soft_warnings()


def test_llm_call_failure_falls_back_to_rule_backend() -> None:
    parser = TaskParser(
        backend=LLMTaskParserBackend(
            backend=FailingBackend(),
            fallback_backend=RuleTaskParserBackend(),
        )
    )

    task = parser.parse("把瓶子放到托盘上")

    assert task.goal == "place_object"
    assert parser.last_diagnostics.backend == "llm"
    assert parser.last_diagnostics.fallback_used
    assert parser.last_diagnostics.failure_mode == ParseFailureMode.LLM_CALL_FAILED.value


@pytest.mark.parametrize(
    ("payload", "failure_mode"),
    [
        ("not-json", ParseFailureMode.INVALID_SCHEMA),
        (
            json.dumps(
                {
                    "goal": "place_object",
                    "action": "teleport",
                    "arguments": [
                        {"role": "target_object", "text": "瓶子", "entity_type": "object"}
                    ],
                    "spatial_relation": None,
                    "confidence": 0.7,
                    "ambiguities": [],
                    "omitted_details": [],
                }
            ),
            ParseFailureMode.ACTION_NOT_IN_VOCAB,
        ),
        (
            json.dumps(
                {
                    "goal": "inspect_scene",
                    "action": "inspect",
                    "arguments": [],
                    "spatial_relation": None,
                    "confidence": 0.7,
                    "ambiguities": [],
                    "omitted_details": [],
                }
            ),
            ParseFailureMode.NO_ARGUMENTS,
        ),
    ],
)
def test_llm_backend_raises_on_hard_errors(
    payload: str,
    failure_mode: ParseFailureMode,
) -> None:
    parser = TaskParser(backend=LLMTaskParserBackend(backend=StaticBackend(payload)))

    with pytest.raises(ParseError) as exc_info:
        parser.parse("把瓶子放到托盘上")

    assert exc_info.value.failure_mode == failure_mode


def test_run_task_supports_llm_parser_backend() -> None:
    temp_dir = ROOT / ".tmp-tests-llm"
    temp_dir.mkdir(exist_ok=True)
    trace_path = temp_dir / "trace.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps" / "run_task.py"),
            "把瓶子放到托盘上",
            "--parser-backend",
            "llm",
            "--llm-provider",
            "mock",
            "--trace-out",
            str(trace_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    payload = data["events"][0]["payload"]
    assert payload["parser_backend"] == "llm"
    assert payload["parser_diagnostics"]["backend"] == "llm"
    assert payload["parser_diagnostics"]["fallback_used"] is False
    assert '"trace_id"' in result.stdout


def test_run_task_records_llm_fallback_event() -> None:
    temp_dir = ROOT / ".tmp-tests-llm-fallback"
    temp_dir.mkdir(exist_ok=True)
    trace_path = temp_dir / "trace.json"
    env = dict(os.environ)
    env["ROBOTICS_STACK_MOCK_LLM_FAIL_TASK_PARSE"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps" / "run_task.py"),
            "把瓶子放到托盘上",
            "--parser-backend",
            "llm",
            "--llm-provider",
            "mock",
            "--trace-out",
            str(trace_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["events"][0]["message"] == "task_parser_fallback"
    assert data["events"][0]["status"] == "warning"
    assert data["events"][1]["message"] == "task_parsed"
    assert data["events"][1]["payload"]["parser_diagnostics"]["fallback_used"] is True
