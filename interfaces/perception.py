from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from interfaces.common import ArtifactReference, SourceAttribution, Vector3


@dataclass(slots=True)
class CameraIntrinsics:
    camera_name: str
    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_m: float | None = None
    far_m: float | None = None
    distortion_model: str = "none"
    depth_unit: str = "meter"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CameraPose:
    frame_id: str
    parent_frame_id: str
    position_xyz: Vector3
    orientation_xyzw: list[float]
    timestamp_s: float | None = None
    pose_source: str = "sim_ground_truth"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RGBDFrame:
    frame_id: str
    sequence_index: int
    timestamp_s: float
    camera_name: str
    intrinsics: CameraIntrinsics
    pose: CameraPose
    rgb_artifact: ArtifactReference
    depth_artifact: ArtifactReference
    valid_depth_ratio: float | None = None
    provenance: list[SourceAttribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "sequence_index": self.sequence_index,
            "timestamp_s": self.timestamp_s,
            "camera_name": self.camera_name,
            "intrinsics": self.intrinsics.to_dict(),
            "pose": self.pose.to_dict(),
            "rgb_artifact": self.rgb_artifact.to_dict(),
            "depth_artifact": self.depth_artifact.to_dict(),
            "valid_depth_ratio": self.valid_depth_ratio,
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class PointCloudFrame:
    frame_id: str
    source_rgbd_frame_id: str
    point_count: int
    bounds_min_m: Vector3
    bounds_max_m: Vector3
    pointcloud_artifact: ArtifactReference
    voxel_size_m: float | None = None
    provenance: list[SourceAttribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "source_rgbd_frame_id": self.source_rgbd_frame_id,
            "point_count": self.point_count,
            "bounds_min_m": list(self.bounds_min_m),
            "bounds_max_m": list(self.bounds_max_m),
            "pointcloud_artifact": self.pointcloud_artifact.to_dict(),
            "voxel_size_m": self.voxel_size_m,
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class GeometryMapSummary:
    map_id: str
    scene_id: str
    frame_count: int
    fused_point_count: int
    voxel_size_m: float | None
    bounds_min_m: Vector3
    bounds_max_m: Vector3
    occupied_voxel_count: int | None = None
    support_surface_ids: list[str] = field(default_factory=list)
    provenance: list[SourceAttribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "scene_id": self.scene_id,
            "frame_count": self.frame_count,
            "fused_point_count": self.fused_point_count,
            "voxel_size_m": self.voxel_size_m,
            "bounds_min_m": list(self.bounds_min_m),
            "bounds_max_m": list(self.bounds_max_m),
            "occupied_voxel_count": self.occupied_voxel_count,
            "support_surface_ids": list(self.support_surface_ids),
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(slots=True)
class MapArtifactSummary:
    mapping_run_id: str
    root_dir: str
    manifest_artifact: ArtifactReference | None = None
    frame_manifest_artifact: ArtifactReference | None = None
    fused_pointcloud_artifact: ArtifactReference | None = None
    geometry_map_artifact: ArtifactReference | None = None
    scene_graph_artifact: ArtifactReference | None = None
    constraint_artifact: ArtifactReference | None = None
    trace_artifact: ArtifactReference | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping_run_id": self.mapping_run_id,
            "root_dir": self.root_dir,
            "manifest_artifact": None
            if self.manifest_artifact is None
            else self.manifest_artifact.to_dict(),
            "frame_manifest_artifact": None
            if self.frame_manifest_artifact is None
            else self.frame_manifest_artifact.to_dict(),
            "fused_pointcloud_artifact": None
            if self.fused_pointcloud_artifact is None
            else self.fused_pointcloud_artifact.to_dict(),
            "geometry_map_artifact": None
            if self.geometry_map_artifact is None
            else self.geometry_map_artifact.to_dict(),
            "scene_graph_artifact": None
            if self.scene_graph_artifact is None
            else self.scene_graph_artifact.to_dict(),
            "constraint_artifact": None
            if self.constraint_artifact is None
            else self.constraint_artifact.to_dict(),
            "trace_artifact": None
            if self.trace_artifact is None
            else self.trace_artifact.to_dict(),
        }


@dataclass(slots=True)
class MappingTracePayload:
    mapping_run_id: str
    scene_id: str
    camera_name: str
    pose_source: str
    frame_ids: list[str]
    fused_frame_count: int
    dropped_frame_ids: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    map_summary: GeometryMapSummary | None = None
    artifact_summary: MapArtifactSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping_run_id": self.mapping_run_id,
            "scene_id": self.scene_id,
            "camera_name": self.camera_name,
            "pose_source": self.pose_source,
            "frame_ids": list(self.frame_ids),
            "fused_frame_count": self.fused_frame_count,
            "dropped_frame_ids": list(self.dropped_frame_ids),
            "timings_ms": dict(self.timings_ms),
            "notes": list(self.notes),
            "map_summary": None if self.map_summary is None else self.map_summary.to_dict(),
            "artifact_summary": None
            if self.artifact_summary is None
            else self.artifact_summary.to_dict(),
        }
