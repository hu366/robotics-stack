from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillSpec:
    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    success_conditions: list[str] = field(default_factory=list)
    failure_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
