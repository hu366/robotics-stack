from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from interfaces.task_spec import TaskArgument, TaskSpec, TaskStepSpec
from interfaces.vlm import VLMBackend, VLMQueryContext, VLMResponse
from modules.task_parser.prompts import build_task_parse_prompt

GOAL_VOCAB = {
    "place_object",
    "open_object",
    "close_object",
    "insert_object",
    "inspect_scene",
}
ACTION_VOCAB = {"place", "move", "open", "close", "insert", "inspect"}
ARGUMENT_ROLE_VOCAB = {"target_object", "target_location"}
SPATIAL_RELATION_VOCAB = {"on", "in", "to", "inside"}
LOW_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True, slots=True)
class _IntentTemplate:
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    substeps: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...]


_INTENT_TEMPLATES: dict[str, _IntentTemplate] = {
    "place_object": _IntentTemplate(
        preconditions=("target_object_identified", "target_location_identified"),
        postconditions=("object_transferred", "target_relation_satisfied"),
        substeps=(
            ("step_1", "locate", ("target_object",), ("object_pose_resolved",)),
            ("step_2", "grasp", ("target_object",), ("stable_grasp",)),
            (
                "step_3",
                "place",
                ("target_object", "target_location"),
                ("target_relation_satisfied",),
            ),
        ),
    ),
    "open_object": _IntentTemplate(
        preconditions=("target_object_identified",),
        postconditions=("object_opened",),
        substeps=(
            ("step_1", "locate", ("target_object",), ("object_pose_resolved",)),
            ("step_2", "open", ("target_object",), ("object_opened",)),
        ),
    ),
    "close_object": _IntentTemplate(
        preconditions=("target_object_identified",),
        postconditions=("object_closed",),
        substeps=(
            ("step_1", "locate", ("target_object",), ("object_pose_resolved",)),
            ("step_2", "close", ("target_object",), ("object_closed",)),
        ),
    ),
    "insert_object": _IntentTemplate(
        preconditions=("target_object_identified", "target_location_identified"),
        postconditions=("object_inserted",),
        substeps=(
            ("step_1", "locate", ("target_object",), ("object_pose_resolved",)),
            ("step_2", "grasp", ("target_object",), ("stable_grasp",)),
            ("step_3", "insert", ("target_object", "target_location"), ("object_inserted",)),
        ),
    ),
    "inspect_scene": _IntentTemplate(
        preconditions=("scene_accessible",),
        postconditions=("scene_observed",),
        substeps=(("step_1", "inspect", tuple(), ("scene_observed",)),),
    ),
}

_RELATION_SUFFIXES: tuple[tuple[str, str | None], ...] = (
    ("里面", "inside"),
    ("内", "in"),
    ("里", "in"),
    ("上", "on"),
)

_RULE_ACTIONS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("放进", "放入", "放在", "放到", "放"), "place", "place_object"),
    (("移动", "移到"), "move", "place_object"),
    (("打开",), "open", "open_object"),
    (("关闭",), "close", "close_object"),
    (("插入", "塞进"), "insert", "insert_object"),
    (("查看", "检查", "观察"), "inspect", "inspect_scene"),
)

_PLACE_CONNECTORS: tuple[tuple[str, str | None], ...] = (
    ("放到", "to"),
    ("放在", "on"),
    ("放进", "in"),
    ("放入", "in"),
    ("移到", "to"),
    ("到", "to"),
)

_INSERT_CONNECTORS: tuple[tuple[str, str | None], ...] = (
    ("插入", "in"),
    ("塞进", "in"),
)


class ParseFailureMode(str, Enum):
    LLM_CALL_FAILED = "llm_call_failed"
    INVALID_SCHEMA = "invalid_schema"
    ACTION_NOT_IN_VOCAB = "action_not_in_vocab"
    NO_ARGUMENTS = "no_arguments"


class ParseError(RuntimeError):
    def __init__(self, failure_mode: ParseFailureMode, instruction: str, message: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode
        self.instruction = instruction


@dataclass(slots=True)
class ParsedIntentArgument:
    role: str
    text: str
    entity_type: str = "object"

    def to_task_argument(self) -> TaskArgument:
        return TaskArgument(role=self.role, text=self.text, entity_type=self.entity_type)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParsedIntent:
    goal: str
    action: str
    arguments: list[ParsedIntentArgument] = field(default_factory=list)
    spatial_relation: str | None = None
    confidence: float | None = None
    ambiguities: list[str] = field(default_factory=list)
    omitted_details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "action": self.action,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "spatial_relation": self.spatial_relation,
            "confidence": self.confidence,
            "ambiguities": list(self.ambiguities),
            "omitted_details": list(self.omitted_details),
        }


@dataclass(slots=True)
class TaskParseDiagnostics:
    backend: str
    model_id: str | None = None
    confidence: float | None = None
    ambiguities: list[str] = field(default_factory=list)
    omitted_details: list[str] = field(default_factory=list)
    fallback_used: bool = False
    failure_mode: str | None = None
    raw_response: dict[str, Any] | None = None

    def has_soft_warnings(self, *, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> bool:
        if self.ambiguities or self.omitted_details:
            return True
        return self.confidence is not None and self.confidence < threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "confidence": self.confidence,
            "ambiguities": list(self.ambiguities),
            "omitted_details": list(self.omitted_details),
            "fallback_used": self.fallback_used,
            "failure_mode": self.failure_mode,
            "raw_response": None if self.raw_response is None else dict(self.raw_response),
        }


@dataclass(slots=True)
class TaskParseOutput:
    task: TaskSpec
    diagnostics: TaskParseDiagnostics


class TaskParserBackend(Protocol):
    def parse(self, instruction: str) -> TaskParseOutput: ...


class RuleTaskParserBackend:
    """Deterministic rule-based parser for the baseline single-task schema."""

    def parse(self, instruction: str) -> TaskParseOutput:
        normalized = normalize_instruction(instruction)
        if not normalized:
            raise ValueError("instruction must not be empty")

        goal, action = self._detect_intent(normalized)
        target_object, target_location, spatial_relation = self._extract_arguments(
            normalized,
            goal,
            action,
        )
        arguments = build_arguments(target_object, target_location)
        task = build_task_spec(
            instruction=normalized,
            goal=goal,
            action=action,
            arguments=arguments,
            spatial_relation=spatial_relation,
        )
        return TaskParseOutput(
            task=task,
            diagnostics=TaskParseDiagnostics(backend="rule"),
        )

    def _detect_intent(self, instruction: str) -> tuple[str, str]:
        for keywords, action, goal in _RULE_ACTIONS:
            for keyword in keywords:
                if keyword in instruction:
                    return goal, action
        return "inspect_scene", "inspect"

    def _extract_arguments(
        self,
        instruction: str,
        goal: str,
        action: str,
    ) -> tuple[str | None, str | None, str | None]:
        if goal in {"place_object", "insert_object"}:
            return self._extract_transfer_arguments(instruction, action)
        if goal in {"open_object", "close_object", "inspect_scene"}:
            target_object = clean_object_text(instruction, action)
            return target_object, None, None
        return None, None, None

    def _extract_transfer_arguments(
        self,
        instruction: str,
        action: str,
    ) -> tuple[str | None, str | None, str | None]:
        connectors = _INSERT_CONNECTORS if action == "insert" else _PLACE_CONNECTORS
        left, right, connector_relation = split_once(instruction, connectors)
        if left is None or right is None:
            target_object = clean_object_text(instruction, action)
            return target_object, None, None

        target_object = clean_object_text(left, action)
        target_location, suffix_relation = clean_location_text(right)
        return target_object, target_location, suffix_relation or connector_relation


class LLMTaskParserBackend:
    """Structured parser that fills a constrained intent schema via an LLM backend."""

    def __init__(
        self,
        backend: VLMBackend,
        *,
        fallback_backend: TaskParserBackend | None = None,
    ) -> None:
        self.backend = backend
        self.fallback_backend = (
            RuleTaskParserBackend() if fallback_backend is None else fallback_backend
        )

    def parse(self, instruction: str) -> TaskParseOutput:
        normalized = normalize_instruction(instruction)
        if not normalized:
            raise ValueError("instruction must not be empty")

        try:
            response = self.backend.query(
                [],
                build_task_parse_prompt(normalized),
                context=VLMQueryContext(stage="task_parse", task_instruction=normalized),
            )
        except Exception:
            fallback_output = self.fallback_backend.parse(normalized)
            return TaskParseOutput(
                task=fallback_output.task,
                diagnostics=TaskParseDiagnostics(
                    backend="llm",
                    model_id=backend_model_id(self.backend),
                    fallback_used=True,
                    failure_mode=ParseFailureMode.LLM_CALL_FAILED.value,
                ),
            )

        intent = parse_structured_intent(response, normalized)
        task = build_task_spec(
            instruction=normalized,
            goal=intent.goal,
            action=intent.action,
            arguments=[argument.to_task_argument() for argument in intent.arguments],
            spatial_relation=intent.spatial_relation,
        )
        return TaskParseOutput(
            task=task,
            diagnostics=TaskParseDiagnostics(
                backend="llm",
                model_id=response.model_id,
                confidence=intent.confidence,
                ambiguities=list(intent.ambiguities),
                omitted_details=list(intent.omitted_details),
                raw_response=response.to_dict(),
            ),
        )


class TaskParser:
    """Facade that keeps the stable parse interface while allowing backend selection."""

    def __init__(self, backend: TaskParserBackend | None = None) -> None:
        self.backend = RuleTaskParserBackend() if backend is None else backend
        self.last_diagnostics = TaskParseDiagnostics(backend="rule")

    def parse(self, instruction: str) -> TaskSpec:
        output = self.backend.parse(instruction)
        self.last_diagnostics = output.diagnostics
        return output.task


def normalize_instruction(instruction: str) -> str:
    stripped = " ".join(instruction.strip().split())
    return stripped.rstrip("。！？!?")


def build_task_spec(
    *,
    instruction: str,
    goal: str,
    action: str,
    arguments: list[TaskArgument],
    spatial_relation: str | None,
) -> TaskSpec:
    template = _template_for_goal(goal)
    constraints = ["maintain_collision_safety", "keep_traceability"]
    if any(argument.role == "target_location" for argument in arguments):
        constraints.append("respect_target_relation")
    return TaskSpec(
        task_id=f"task-{uuid4().hex[:8]}",
        instruction=instruction,
        goal=goal,
        action=action,
        arguments=list(arguments),
        spatial_relation=spatial_relation,
        preconditions=list(template.preconditions),
        postconditions=list(template.postconditions),
        constraints=constraints,
        recovery_policy="replan",
        substeps=build_substeps(template),
    )


def build_arguments(
    target_object: str | None,
    target_location: str | None,
) -> list[TaskArgument]:
    arguments: list[TaskArgument] = []
    if target_object:
        arguments.append(TaskArgument(role="target_object", text=target_object))
    if target_location:
        arguments.append(TaskArgument(role="target_location", text=target_location))
    return arguments


def build_substeps(template: _IntentTemplate) -> list[TaskStepSpec]:
    return [
        TaskStepSpec(
            step_id=step_id,
            action=action,
            description=substep_description(action),
            required_arguments=list(required_arguments),
            success_criteria=list(success_criteria),
        )
        for step_id, action, required_arguments, success_criteria in template.substeps
    ]


def parse_structured_intent(response: VLMResponse, instruction: str) -> ParsedIntent:
    payload = parse_json_payload(response.text)
    validate_top_level_payload(payload, instruction)
    action = payload["action"]
    if action not in ACTION_VOCAB:
        raise ParseError(
            ParseFailureMode.ACTION_NOT_IN_VOCAB,
            instruction,
            f"Unsupported action from LLM parser: {action}",
        )
    arguments_payload = payload["arguments"]
    if not arguments_payload:
        raise ParseError(
            ParseFailureMode.NO_ARGUMENTS,
            instruction,
            "LLM parser returned no arguments.",
        )
    arguments = [parse_intent_argument(item, instruction) for item in arguments_payload]
    return ParsedIntent(
        goal=parse_goal(payload["goal"], instruction),
        action=action,
        arguments=arguments,
        spatial_relation=parse_spatial_relation(payload["spatial_relation"], instruction),
        confidence=parse_confidence(payload["confidence"], instruction, response.confidence),
        ambiguities=parse_string_list(payload["ambiguities"], instruction, "ambiguities"),
        omitted_details=parse_string_list(
            payload["omitted_details"],
            instruction,
            "omitted_details",
        ),
    )


def parse_json_payload(text: str) -> dict[str, Any]:
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
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def validate_top_level_payload(payload: dict[str, Any], instruction: str) -> None:
    expected_keys = {
        "goal",
        "action",
        "arguments",
        "spatial_relation",
        "confidence",
        "ambiguities",
        "omitted_details",
    }
    if set(payload) != expected_keys:
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            "LLM parser returned an unexpected top-level schema.",
        )
    if not isinstance(payload["arguments"], list):
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            "LLM parser arguments must be a list.",
        )


def parse_intent_argument(payload: Any, instruction: str) -> ParsedIntentArgument:
    if not isinstance(payload, dict):
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            "LLM parser argument entries must be objects.",
        )
    if set(payload) != {"role", "text", "entity_type"}:
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            "LLM parser argument schema is invalid.",
        )
    role = payload["role"]
    text = payload["text"]
    entity_type = payload["entity_type"]
    if role not in ARGUMENT_ROLE_VOCAB:
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            f"Unsupported argument role from LLM parser: {role}",
        )
    if not isinstance(text, str) or not text.strip():
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            "LLM parser argument text must be a non-empty string.",
        )
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            "LLM parser argument entity_type must be a non-empty string.",
        )
    return ParsedIntentArgument(role=role, text=text.strip(), entity_type=entity_type.strip())


def parse_goal(value: Any, instruction: str) -> str:
    if not isinstance(value, str) or value not in GOAL_VOCAB:
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            f"Unsupported goal from LLM parser: {value}",
        )
    return value


def parse_spatial_relation(value: Any, instruction: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in SPATIAL_RELATION_VOCAB:
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            f"Unsupported spatial relation from LLM parser: {value}",
        )
    return value


def parse_confidence(
    value: Any,
    instruction: str,
    default: float | None,
) -> float | None:
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            "LLM parser confidence must be numeric or null.",
        )
    return float(value)


def parse_string_list(value: Any, instruction: str, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ParseError(
            ParseFailureMode.INVALID_SCHEMA,
            instruction,
            f"LLM parser {field_name} must be a list.",
        )
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ParseError(
                ParseFailureMode.INVALID_SCHEMA,
                instruction,
                f"LLM parser {field_name} entries must be strings.",
            )
        cleaned = item.strip()
        if cleaned:
            parsed.append(cleaned)
    return parsed


def split_once(
    instruction: str,
    connectors: tuple[tuple[str, str | None], ...],
) -> tuple[str | None, str | None, str | None]:
    for connector, relation in connectors:
        if connector not in instruction:
            continue
        left, right = instruction.split(connector, 1)
        return left.strip(), right.strip(), relation
    return None, None, None


def clean_object_text(text: str, action: str) -> str | None:
    cleaned = text.strip()
    if cleaned.startswith("把"):
        cleaned = cleaned[1:].strip()
    for prefix in action_prefixes(action):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned or None


def clean_location_text(text: str) -> tuple[str | None, str | None]:
    cleaned = text.strip()
    relation: str | None = None
    for suffix, mapped_relation in _RELATION_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            relation = mapped_relation
            break
    return cleaned or None, relation


def action_prefixes(action: str) -> tuple[str, ...]:
    if action == "place":
        return ("放进", "放入", "放在", "放到", "放")
    if action == "move":
        return ("移到", "移动")
    if action == "insert":
        return ("插入", "塞进")
    if action == "open":
        return ("打开",)
    if action == "close":
        return ("关闭",)
    if action == "inspect":
        return ("查看", "检查", "观察")
    return tuple()


def substep_description(action: str) -> str:
    descriptions = {
        "locate": "Resolve task entities in the scene.",
        "grasp": "Establish a stable grasp on the target object.",
        "place": "Place the target object at the requested location.",
        "open": "Open the requested object.",
        "close": "Close the requested object.",
        "insert": "Insert the target object into the requested location.",
        "inspect": "Inspect the current scene state.",
    }
    return descriptions[action]


def backend_model_id(backend: VLMBackend) -> str | None:
    model = getattr(backend, "model", None)
    if isinstance(model, str):
        return model
    return None


def _template_for_goal(goal: str) -> _IntentTemplate:
    return _INTENT_TEMPLATES[goal]
