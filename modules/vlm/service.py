from __future__ import annotations

import json
from typing import Any

from interfaces.task_spec import TaskSpec
from interfaces.vlm import (
    ExecutionVerification,
    PlanFeasibility,
    SceneDescription,
    VLMBackend,
    VLMQueryContext,
)
from modules.planner import ExecutionPlan
from modules.vlm.prompts import (
    build_execution_verification_prompt,
    build_plan_review_prompt,
    build_scene_description_prompt,
)


class VLMService:
    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def describe_scene(
        self,
        image: bytes,
        task: TaskSpec | str | None = None,
    ) -> SceneDescription:
        task_instruction = _task_instruction(task)
        response = self.backend.query(
            [image],
            build_scene_description_prompt(task_instruction),
            context=VLMQueryContext(
                stage="scene_description",
                task_instruction=task_instruction,
            ),
        )
        payload = _parse_json_payload(response.text)
        return SceneDescription(
            objects_described=_string_list(payload.get("objects_described")),
            spatial_summary=_string_value(
                payload.get("spatial_summary"),
                default=response.text.strip(),
            ),
            raw_response=response,
        )

    def review_plan(
        self,
        image: bytes,
        plan: ExecutionPlan | dict[str, Any],
        task: TaskSpec | str,
    ) -> PlanFeasibility:
        task_instruction = _task_instruction(task) or ""
        plan_payload = _plan_payload(plan)
        response = self.backend.query(
            [image],
            build_plan_review_prompt(task_instruction, plan_payload),
            context=VLMQueryContext(
                stage="plan_review",
                task_instruction=task_instruction,
                metadata={"plan": plan_payload},
            ),
        )
        payload = _parse_json_payload(response.text)
        concerns = _string_list(payload.get("concerns"))
        suggestions = _string_list(payload.get("suggestions"))
        if not payload:
            concerns = ["Unable to parse structured VLM plan review."]
        return PlanFeasibility(
            feasible=_bool_value(payload.get("feasible"), default=True),
            concerns=concerns,
            suggestions=suggestions,
            raw_response=response,
        )

    def verify_execution(
        self,
        before: bytes,
        after: bytes,
        task: TaskSpec | str,
    ) -> ExecutionVerification:
        task_instruction = _task_instruction(task) or ""
        response = self.backend.query(
            [before, after],
            build_execution_verification_prompt(task_instruction),
            context=VLMQueryContext(
                stage="execution_verification",
                task_instruction=task_instruction,
            ),
        )
        payload = _parse_json_payload(response.text)
        discrepancies = _string_list(payload.get("discrepancies"))
        if not payload:
            discrepancies = ["Unable to parse structured VLM execution verification."]
        return ExecutionVerification(
            task_completed=_bool_value(payload.get("task_completed"), default=False),
            discrepancies=discrepancies,
            confidence=_float_value(
                payload.get("confidence"),
                default=response.confidence,
            ),
            raw_response=response,
        )


def _task_instruction(task: TaskSpec | str | None) -> str | None:
    if task is None:
        return None
    if isinstance(task, TaskSpec):
        return task.instruction
    return task


def _plan_payload(plan: ExecutionPlan | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan, ExecutionPlan):
        return plan.to_dict()
    return dict(plan)


def _parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    candidates = [stripped]
    if "```" in stripped:
        for segment in stripped.split("```"):
            candidate = segment.strip()
            if not candidate or candidate == "json":
                continue
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            candidates.append(candidate)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
    return result


def _string_value(value: Any, default: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return default


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _float_value(value: Any, default: float | None) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return default
