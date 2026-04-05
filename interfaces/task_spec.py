from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    instruction: str
    goal: str
    target_object: str | None = None
    target_location: str | None = None
    constraints: list[str] = field(default_factory=list)
    recovery_policy: str = "replan"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
