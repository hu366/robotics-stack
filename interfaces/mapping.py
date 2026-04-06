from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CameraPose:
    position: list[float]
    rotation: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameMetadata:
    frame_id: str
    camera_name: str
    timestamp: float
    intrinsics: CameraIntrinsics
    pose: CameraPose
    point_count: int
    rgb_path: str
    depth_npy_path: str
    depth_vis_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera_name": self.camera_name,
            "timestamp": self.timestamp,
            "intrinsics": self.intrinsics.to_dict(),
            "pose": self.pose.to_dict(),
            "point_count": self.point_count,
            "rgb_path": self.rgb_path,
            "depth_npy_path": self.depth_npy_path,
            "depth_vis_path": self.depth_vis_path,
        }


@dataclass(slots=True)
class MappingSummary:
    scene_id: str
    camera_name: str
    frame_count: int
    input_point_count: int
    total_points: int
    voxel_size: float
    bbox_min: list[float]
    bbox_max: list[float]
    object_count: int
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "camera_name": self.camera_name,
            "frame_count": self.frame_count,
            "input_point_count": self.input_point_count,
            "total_points": self.total_points,
            "voxel_size": self.voxel_size,
            "bbox": {
                "min": self.bbox_min,
                "max": self.bbox_max,
            },
            "object_count": self.object_count,
            "artifacts": self.artifacts,
        }
