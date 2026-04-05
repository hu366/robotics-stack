from __future__ import annotations

import re
from uuid import uuid4

from interfaces.task_spec import TaskSpec

_ACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(place|put|move)\b", re.IGNORECASE), "place_object"),
    (re.compile(r"\b(open)\b", re.IGNORECASE), "open_object"),
    (re.compile(r"\b(close)\b", re.IGNORECASE), "close_object"),
    (re.compile(r"\b(insert)\b", re.IGNORECASE), "insert_object"),
)

_PREPOSITIONS = (" on ", " onto ", " in ", " into ", " inside ", " at ", " to ")


class TaskParser:
    """Rule-based parser used as the baseline interpretable task frontend."""

    def parse(self, instruction: str) -> TaskSpec:
        normalized = " ".join(instruction.strip().split())
        if not normalized:
            raise ValueError("instruction must not be empty")

        goal = self._infer_goal(normalized)
        target_object, target_location = self._extract_entities(normalized)

        constraints = ["maintain_collision_safety", "keep_traceability"]
        if target_location:
            constraints.append("respect_target_relation")

        return TaskSpec(
            task_id=f"task-{uuid4().hex[:8]}",
            instruction=normalized,
            goal=goal,
            target_object=target_object,
            target_location=target_location,
            constraints=constraints,
            recovery_policy="replan",
        )

    def _infer_goal(self, instruction: str) -> str:
        for pattern, goal in _ACTION_PATTERNS:
            if pattern.search(instruction):
                return goal
        return "inspect_scene"

    def _extract_entities(self, instruction: str) -> tuple[str | None, str | None]:
        lowered = instruction.lower()
        for prep in _PREPOSITIONS:
            if prep in lowered:
                idx = lowered.index(prep)
                left = instruction[:idx].strip()
                right = instruction[idx + len(prep) :].strip(" .")
                return self._strip_verb_prefix(left), right or None
        return self._strip_verb_prefix(instruction), None

    def _strip_verb_prefix(self, text: str) -> str | None:
        cleaned = re.sub(
            r"^(please\s+)?(place|put|move|open|close|insert)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip(" .")
        return cleaned or None
