from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from interfaces.common import Pose7D, ProvenanceSourceType, Vector3
from interfaces.scene_graph import ConstraintState


@dataclass(slots=True)
class SceneGraphQuery:
    query_id: str
    text: str
    target_label: str | None = None
    node_types: list[str] = field(default_factory=list)
    relation_filters: list[str] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)
    preferred_sources: list[ProvenanceSourceType] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "target_label": self.target_label,
            "node_types": list(self.node_types),
            "relation_filters": list(self.relation_filters),
            "required_tags": list(self.required_tags),
            "preferred_sources": list(self.preferred_sources),
        }


@dataclass(slots=True)
class GroundingRequest:
    request_id: str
    task_id: str
    scene_id: str
    object_query: SceneGraphQuery | None = None
    surface_query: SceneGraphQuery | None = None
    required_constraint_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "scene_id": self.scene_id,
            "object_query": None
            if self.object_query is None
            else self.object_query.to_dict(),
            "surface_query": None
            if self.surface_query is None
            else self.surface_query.to_dict(),
            "required_constraint_types": list(self.required_constraint_types),
        }


@dataclass(slots=True)
class ObjectCandidate:
    candidate_id: str
    object_id: str
    node_id: str
    label: str
    score: float
    pose: Pose7D | None = None
    bbox_extent_m: Vector3 | None = None
    supporting_surface_id: str | None = None
    source_types: list[ProvenanceSourceType] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "object_id": self.object_id,
            "node_id": self.node_id,
            "label": self.label,
            "score": self.score,
            "pose": None if self.pose is None else list(self.pose),
            "bbox_extent_m": None
            if self.bbox_extent_m is None
            else list(self.bbox_extent_m),
            "supporting_surface_id": self.supporting_surface_id,
            "source_types": list(self.source_types),
            "evidence": list(self.evidence),
        }


@dataclass(slots=True)
class SurfaceCandidate:
    candidate_id: str
    surface_id: str
    node_id: str
    label: str
    score: float
    pose: Pose7D | None = None
    normal_xyz: Vector3 | None = None
    parent_object_id: str | None = None
    source_types: list[ProvenanceSourceType] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "surface_id": self.surface_id,
            "node_id": self.node_id,
            "label": self.label,
            "score": self.score,
            "pose": None if self.pose is None else list(self.pose),
            "normal_xyz": None if self.normal_xyz is None else list(self.normal_xyz),
            "parent_object_id": self.parent_object_id,
            "source_types": list(self.source_types),
            "evidence": list(self.evidence),
        }


@dataclass(slots=True)
class PoseCandidate:
    candidate_id: str
    frame_id: str
    pose: Pose7D
    score: float
    reason: str
    reference_node_id: str | None = None
    source_types: list[ProvenanceSourceType] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "frame_id": self.frame_id,
            "pose": list(self.pose),
            "score": self.score,
            "reason": self.reason,
            "reference_node_id": self.reference_node_id,
            "source_types": list(self.source_types),
        }


@dataclass(slots=True)
class GroundingResult:
    request_id: str
    scene_id: str
    object_candidates: list[ObjectCandidate] = field(default_factory=list)
    surface_candidates: list[SurfaceCandidate] = field(default_factory=list)
    pose_candidates: list[PoseCandidate] = field(default_factory=list)
    constraint_states: list[ConstraintState] = field(default_factory=list)
    unresolved_slots: list[str] = field(default_factory=list)
    trace_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scene_id": self.scene_id,
            "object_candidates": [item.to_dict() for item in self.object_candidates],
            "surface_candidates": [item.to_dict() for item in self.surface_candidates],
            "pose_candidates": [item.to_dict() for item in self.pose_candidates],
            "constraint_states": [item.to_dict() for item in self.constraint_states],
            "unresolved_slots": list(self.unresolved_slots),
            "trace_notes": list(self.trace_notes),
        }
