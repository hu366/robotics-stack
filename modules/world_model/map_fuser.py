from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
UInt8Array = npt.NDArray[np.uint8]


@dataclass(slots=True)
class FusedPointCloud:
    points: FloatArray
    colors: UInt8Array | None = None

    def bbox(self) -> tuple[list[float], list[float]]:
        if self.points.size == 0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        bbox_min = np.min(self.points, axis=0)
        bbox_max = np.max(self.points, axis=0)
        return bbox_min.tolist(), bbox_max.tolist()


class PointCloudMapFuser:
    def __init__(self, voxel_size: float) -> None:
        if voxel_size < 0.0:
            raise ValueError("voxel_size must be non-negative")
        self._voxel_size = voxel_size
        self._point_batches: list[FloatArray] = []
        self._color_batches: list[UInt8Array] = []
        self._has_colors = True
        self._total_input_points = 0

    @property
    def total_input_points(self) -> int:
        return self._total_input_points

    def add_points(self, points: FloatArray, colors: UInt8Array | None = None) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape Nx3")
        if colors is not None and (colors.ndim != 2 or colors.shape != (points.shape[0], 3)):
            raise ValueError("colors must have shape Nx3")

        if points.size == 0:
            return

        self._point_batches.append(points.astype(np.float64, copy=False))
        self._total_input_points += int(points.shape[0])

        if colors is None:
            self._has_colors = False
        else:
            self._color_batches.append(colors.astype(np.uint8, copy=False))

    def build(self) -> FusedPointCloud:
        if not self._point_batches:
            empty_points = np.zeros((0, 3), dtype=np.float64)
            empty_colors = np.zeros((0, 3), dtype=np.uint8) if self._has_colors else None
            return FusedPointCloud(points=empty_points, colors=empty_colors)

        points = np.concatenate(self._point_batches, axis=0)
        colors = None
        if self._has_colors and len(self._color_batches) == len(self._point_batches):
            colors = np.concatenate(self._color_batches, axis=0)

        if self._voxel_size <= 0.0:
            return FusedPointCloud(points=points, colors=colors)

        return self._voxel_downsample(points, colors)

    def _voxel_downsample(self, points: FloatArray, colors: UInt8Array | None) -> FusedPointCloud:
        voxel_indices = np.floor(points / self._voxel_size).astype(np.int64)
        unique_keys, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)

        reduced_points = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        reduced_colors = (
            np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
            if colors is not None
            else None
        )
        counts = np.zeros(unique_keys.shape[0], dtype=np.int64)

        for index, voxel_id in enumerate(inverse):
            reduced_points[voxel_id] += points[index]
            if reduced_colors is not None and colors is not None:
                reduced_colors[voxel_id] += colors[index].astype(np.float64)
            counts[voxel_id] += 1

        reduced_points /= counts[:, None]
        colors_out: UInt8Array | None = None
        if reduced_colors is not None:
            reduced_colors /= counts[:, None]
            colors_out = np.clip(np.rint(reduced_colors), 0, 255).astype(np.uint8)

        order = self._lexsort_keys(unique_keys)
        return FusedPointCloud(
            points=reduced_points[order],
            colors=None if colors_out is None else colors_out[order],
        )

    def _lexsort_keys(self, keys: IntArray) -> IntArray:
        return np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
