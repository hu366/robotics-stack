from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskArgument:
    role: str
    text: str
    entity_type: str = "object"
    is_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskStepSpec:
    step_id: str
    action: str
    description: str
    required_arguments: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    instruction: str
    goal: str
    action: str
    arguments: list[TaskArgument] = field(default_factory=list)
    spatial_relation: str | None = None
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    recovery_policy: str = "replan"
    substeps: list[TaskStepSpec] = field(default_factory=list)

    def get_argument(self, role: str) -> TaskArgument | None:
        for argument in self.arguments:
            if argument.role == role:
                return argument
        return None

    def argument_text(self, role: str) -> str | None:
        argument = self.get_argument(role)
        if argument is None:
            return None
        return argument.text

    @property
    def target_object(self) -> str | None:
        return self.argument_text("target_object")

    @property
    def target_location(self) -> str | None:
        return self.argument_text("target_location")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "goal": self.goal,
            "action": self.action,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "spatial_relation": self.spatial_relation,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "constraints": list(self.constraints),
            "recovery_policy": self.recovery_policy,
            "substeps": [step.to_dict() for step in self.substeps],
            "target_object": self.target_object,
            "target_location": self.target_location,
        }
