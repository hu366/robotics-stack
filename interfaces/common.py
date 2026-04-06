from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Vector3 = list[float]
Pose7D = list[float]
ProvenanceSourceType = Literal[
    "sim_ground_truth",
    "rgbd_fused_geometry",
    "inferred_rule",
    "future_learned_model",
]


@dataclass(slots=True)
class ArtifactReference:
    artifact_id: str
    artifact_type: str
    path: str
    format: str
    role: str
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceAttribution:
    source_type: ProvenanceSourceType
    source_id: str
    confidence: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
