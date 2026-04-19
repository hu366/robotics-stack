from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from interfaces.task_spec import TaskArgument, TaskSpec, TaskStepSpec


@dataclass(frozen=True, slots=True)
class _IntentSpec:
    action: str
    goal: str
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    substeps: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...]


_INTENT_LIBRARY: tuple[tuple[tuple[str, ...], _IntentSpec], ...] = (
    (
        ("放进", "放入", "放在", "放到", "移动", "移到", "放"),
        _IntentSpec(
            action="place",
            goal="place_object",
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
    ),
    (
        ("打开",),
        _IntentSpec(
            action="open",
            goal="open_object",
            preconditions=("target_object_identified",),
            postconditions=("object_opened",),
            substeps=(
                ("step_1", "locate", ("target_object",), ("object_pose_resolved",)),
                ("step_2", "open", ("target_object",), ("object_opened",)),
            ),
        ),
    ),
    (
        ("关闭",),
        _IntentSpec(
            action="close",
            goal="close_object",
            preconditions=("target_object_identified",),
            postconditions=("object_closed",),
            substeps=(
                ("step_1", "locate", ("target_object",), ("object_pose_resolved",)),
                ("step_2", "close", ("target_object",), ("object_closed",)),
            ),
        ),
    ),
    (
        ("插入", "塞进"),
        _IntentSpec(
            action="insert",
            goal="insert_object",
            preconditions=("target_object_identified", "target_location_identified"),
            postconditions=("object_inserted",),
            substeps=(
                ("step_1", "locate", ("target_object",), ("object_pose_resolved",)),
                ("step_2", "grasp", ("target_object",), ("stable_grasp",)),
                ("step_3", "insert", ("target_object", "target_location"), ("object_inserted",)),
            ),
        ),
    ),
    (
        ("查看", "检查", "观察"),
        _IntentSpec(
            action="inspect",
            goal="inspect_scene",
            preconditions=("scene_accessible",),
            postconditions=("scene_observed",),
            substeps=(
                ("step_1", "inspect", tuple(), ("scene_observed",)),
            ),
        ),
    ),
)

_RELATION_SUFFIXES: tuple[tuple[str, str | None], ...] = (
    ("里面", "inside"),
    ("内", "in"),
    ("里", "in"),
    ("上", "on"),
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


class TaskParser:
    """Rule-based parser used as the baseline interpretable task frontend."""

    def parse(self, instruction: str) -> TaskSpec:
        normalized = self._normalize_instruction(instruction)
        if not normalized:
            raise ValueError("instruction must not be empty")

        intent = self._detect_intent(normalized)
        target_object, target_location, spatial_relation = self._extract_arguments(
            normalized,
            intent,
        )
        arguments = self._build_arguments(target_object, target_location)
        constraints = ["maintain_collision_safety", "keep_traceability"]
        if target_location:
            constraints.append("respect_target_relation")

        return TaskSpec(
            task_id=f"task-{uuid4().hex[:8]}",
            instruction=normalized,
            goal=intent.goal,
            action=intent.action,
            arguments=arguments,
            spatial_relation=spatial_relation,
            preconditions=list(intent.preconditions),
            postconditions=list(intent.postconditions),
            constraints=constraints,
            recovery_policy="replan",
            substeps=self._build_substeps(intent),
        )

    def _normalize_instruction(self, instruction: str) -> str:
        stripped = " ".join(instruction.strip().split())
        return stripped.rstrip("。！？!?")

    def _detect_intent(self, instruction: str) -> _IntentSpec:
        for keywords, intent in _INTENT_LIBRARY:
            for keyword in keywords:
                if keyword in instruction:
                    if keyword == "移动":
                        return _IntentSpec(
                            action="move",
                            goal=intent.goal,
                            preconditions=intent.preconditions,
                            postconditions=intent.postconditions,
                            substeps=intent.substeps,
                        )
                    if keyword == "移到":
                        return _IntentSpec(
                            action="move",
                            goal=intent.goal,
                            preconditions=intent.preconditions,
                            postconditions=intent.postconditions,
                            substeps=intent.substeps,
                        )
                    return intent
        return _IntentSpec(
            action="inspect",
            goal="inspect_scene",
            preconditions=("scene_accessible",),
            postconditions=("scene_observed",),
            substeps=(("step_1", "inspect", tuple(), ("scene_observed",)),),
        )

    def _extract_arguments(
        self,
        instruction: str,
        intent: _IntentSpec,
    ) -> tuple[str | None, str | None, str | None]:
        if intent.goal in {"place_object", "insert_object"}:
            return self._extract_transfer_arguments(instruction, intent.action)
        if intent.goal in {"open_object", "close_object", "inspect_scene"}:
            target_object = self._clean_object_text(instruction, intent.action)
            return target_object, None, None
        return None, None, None

    def _extract_transfer_arguments(
        self,
        instruction: str,
        action: str,
    ) -> tuple[str | None, str | None, str | None]:
        connectors = _INSERT_CONNECTORS if action == "insert" else _PLACE_CONNECTORS
        left, right, connector_relation = self._split_once(instruction, connectors)
        if left is None or right is None:
            target_object = self._clean_object_text(instruction, action)
            return target_object, None, None

        target_object = self._clean_object_text(left, action)
        target_location, suffix_relation = self._clean_location_text(right)
        spatial_relation = suffix_relation or connector_relation
        return target_object, target_location, spatial_relation

    def _split_once(
        self,
        instruction: str,
        connectors: tuple[tuple[str, str | None], ...],
    ) -> tuple[str | None, str | None, str | None]:
        for connector, relation in connectors:
            if connector not in instruction:
                continue
            left, right = instruction.split(connector, 1)
            return left.strip(), right.strip(), relation
        return None, None, None

    def _clean_object_text(self, text: str, action: str) -> str | None:
        cleaned = text.strip()
        if cleaned.startswith("把"):
            cleaned = cleaned[1:].strip()
        for prefix in self._action_prefixes(action):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break
        return cleaned or None

    def _clean_location_text(self, text: str) -> tuple[str | None, str | None]:
        cleaned = text.strip()
        relation: str | None = None
        for suffix, mapped_relation in _RELATION_SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                relation = mapped_relation
                break
        return cleaned or None, relation

    def _action_prefixes(self, action: str) -> tuple[str, ...]:
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

    def _build_arguments(
        self,
        target_object: str | None,
        target_location: str | None,
    ) -> list[TaskArgument]:
        arguments: list[TaskArgument] = []
        if target_object:
            arguments.append(TaskArgument(role="target_object", text=target_object))
        if target_location:
            arguments.append(TaskArgument(role="target_location", text=target_location))
        return arguments

    def _build_substeps(self, intent: _IntentSpec) -> list[TaskStepSpec]:
        return [
            TaskStepSpec(
                step_id=step_id,
                action=action,
                description=self._substep_description(action),
                required_arguments=list(required_arguments),
                success_criteria=list(success_criteria),
            )
            for step_id, action, required_arguments, success_criteria in intent.substeps
        ]

    def _substep_description(self, action: str) -> str:
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
