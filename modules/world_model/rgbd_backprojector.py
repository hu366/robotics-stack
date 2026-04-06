from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from interfaces.mapping import CameraPose

FloatArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]


@dataclass(slots=True)
class BackprojectedPointCloud:
    points: FloatArray
    colors: UInt8Array | None = None


class RGBDBackprojector:
    """Back-project metric depth into the MuJoCo camera frame."""

    def backproject(
        self,
        depth_m: FloatArray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        rgb: UInt8Array | None = None,
    ) -> BackprojectedPointCloud:
        if depth_m.ndim != 2:
            raise ValueError("depth image must be HxW")
        if rgb is not None and rgb.shape[:2] != depth_m.shape:
            raise ValueError("rgb image must match depth resolution")

        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
        v_coords, u_coords = np.nonzero(valid_mask)
        depth = depth_m[valid_mask].astype(np.float64, copy=False)

        x = (u_coords.astype(np.float64) - cx) * depth / fx
        y = (cy - v_coords.astype(np.float64)) * depth / fy
        points = np.column_stack((x, y, -depth)).astype(np.float64, copy=False)

        colors: UInt8Array | None = None
        if rgb is not None:
            colors = rgb[valid_mask].astype(np.uint8, copy=False)

        return BackprojectedPointCloud(points=points, colors=colors)

    def transform_to_world(
        self,
        cloud: BackprojectedPointCloud,
        pose: CameraPose,
    ) -> BackprojectedPointCloud:
        rotation = np.asarray(pose.rotation, dtype=np.float64)
        position = np.asarray(pose.position, dtype=np.float64)
        world_points = cloud.points @ rotation.T + position
        return BackprojectedPointCloud(points=world_points, colors=cloud.colors)
