from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from interfaces.common import Pose7D, SourceAttribution, Vector3


@dataclass(slots=True)
class SceneNode:
    node_id: str
    label: str
    node_type: str
    center_pose: Pose7D | None = None
    bbox_extent_m: Vector3 | None = None
    geometry_anchor_id: str | None = None
    semantic_tags: list[str] = field(default_factory=list)
    state_tags: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=dict)
    provenance: list[SourceAttribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "center_pose": None if self.center_pose is None else list(self.center_pose),
            "bbox_extent_m": None
            if self.bbox_extent_m is None
            else list(self.bbox_extent_m),
            "geometry_anchor_id": self.geometry_anchor_id,
            "semantic_tags": list(self.semantic_tags),
            "state_tags": list(self.state_tags),
            "properties": dict(self.properties),
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class SceneEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    relation_family: str = "spatial"
    directed: bool = True
    numeric: dict[str, float] = field(default_factory=dict)
    properties: dict[str, str] = field(default_factory=dict)
    provenance: list[SourceAttribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation_type": self.relation_type,
            "relation_family": self.relation_family,
            "directed": self.directed,
            "numeric": dict(self.numeric),
            "properties": dict(self.properties),
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class ConstraintState:
    constraint_id: str
    subject_id: str
    constraint_type: str
    status: str
    object_id: str | None = None
    metric_values: dict[str, float] = field(default_factory=dict)
    symbolic_values: dict[str, str] = field(default_factory=dict)
    provenance: list[SourceAttribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "subject_id": self.subject_id,
            "constraint_type": self.constraint_type,
            "status": self.status,
            "object_id": self.object_id,
            "metric_values": dict(self.metric_values),
            "symbolic_values": dict(self.symbolic_values),
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class SceneGraph:
    scene_id: str
    root_frame_id: str
    revision: int = 0
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "root_frame_id": self.root_frame_id,
            "revision": self.revision,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "summary": dict(self.summary),
        }
