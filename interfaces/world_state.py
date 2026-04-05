from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ObjectState:
    object_id: str
    label: str
    pose: list[float]
    relations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorldState:
    scene_id: str
    objects: list[ObjectState] = field(default_factory=list)
    robot_mode: str = "idle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "robot_mode": self.robot_mode,
            "objects": [obj.to_dict() for obj in self.objects],
        }
