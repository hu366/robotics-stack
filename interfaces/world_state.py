from dataclasses import asdict, dataclass, field
from typing import Any

from interfaces.perception import GeometryMapSummary, MapArtifactSummary
from interfaces.scene_graph import ConstraintState, SceneGraph


@dataclass(slots=True)
class ObjectState:
    object_id: str
    label: str
    pose: list[float]
    relations: list[str] = field(default_factory=list)
    source_node_id: str | None = None
    pose_source: str = "grounding"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorldState:
    scene_id: str
    objects: list[ObjectState] = field(default_factory=list)
    robot_mode: str = "idle"
    geometry_map: GeometryMapSummary | None = None
    scene_graph: SceneGraph | None = None
    constraint_states: list[ConstraintState] = field(default_factory=list)
    map_artifacts: MapArtifactSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "robot_mode": self.robot_mode,
            "objects": [obj.to_dict() for obj in self.objects],
            "geometry_map": None
            if self.geometry_map is None
            else self.geometry_map.to_dict(),
            "scene_graph": None if self.scene_graph is None else self.scene_graph.to_dict(),
            "constraint_states": [item.to_dict() for item in self.constraint_states],
            "map_artifacts": None
            if self.map_artifacts is None
            else self.map_artifacts.to_dict(),
        }
