from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from interfaces.execution_trace import ExecutionTrace
from interfaces.mapping import FrameMetadata, MappingSummary
from interfaces.scene_graph import SceneGraph

FloatArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]


class MappingArtifactStore:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._frames_dir = out_dir / "frames"
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    def save_frame_artifacts(
        self,
        frame_id: str,
        rgb: UInt8Array,
        depth_m: FloatArray,
    ) -> tuple[str, str, str]:
        from PIL import Image

        rgb_path = (self._frames_dir / f"{frame_id}_rgb.png").resolve()
        depth_npy_path = (self._frames_dir / f"{frame_id}_depth.npy").resolve()
        depth_vis_path = (self._frames_dir / f"{frame_id}_depth_vis.png").resolve()

        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_npy_path, depth_m)
        self._save_depth_vis(depth_vis_path, depth_m)
        return str(rgb_path), str(depth_npy_path), str(depth_vis_path)

    def write_poses(self, frame_metadata: list[FrameMetadata]) -> str:
        payload = [
            {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "camera_name": frame.camera_name,
                "pose": frame.pose.to_dict(),
            }
            for frame in frame_metadata
        ]
        return self._write_json("poses.json", payload)

    def write_frame_metadata(self, frame_metadata: list[FrameMetadata]) -> str:
        return self._write_json(
            "frame_metadata.json",
            [frame.to_dict() for frame in frame_metadata],
        )

    def write_summary(self, summary: MappingSummary) -> str:
        return self._write_json("summary.json", summary.to_dict())

    def write_scene_graph(self, scene_graph: SceneGraph) -> str:
        return self._write_json("scene_graph.json", scene_graph.to_dict())

    def write_trace(self, trace: ExecutionTrace) -> str:
        return self._write_json("trace.json", trace.to_dict())

    def write_global_cloud(self, points: FloatArray, colors: UInt8Array | None) -> str:
        cloud_path = (self._out_dir / "global_cloud.ply").resolve()
        colors_to_write = colors
        if colors_to_write is None:
            colors_to_write = np.full((points.shape[0], 3), 200, dtype=np.uint8)
        with cloud_path.open("w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {points.shape[0]}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
            handle.write("end_header\n")
            for point, color in zip(points, colors_to_write, strict=True):
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        return str(cloud_path)

    def _write_json(self, name: str, payload: Any) -> str:
        path = (self._out_dir / name).resolve()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    def _save_depth_vis(self, path: Path, depth_m: FloatArray) -> None:
        from PIL import Image

        valid = np.isfinite(depth_m)
        if not np.any(valid):
            vis = np.zeros_like(depth_m, dtype=np.uint8)
        else:
            dmin = float(np.min(depth_m[valid]))
            dmax = float(np.max(depth_m[valid]))
            if dmax - dmin < 1e-9:
                vis = np.zeros_like(depth_m, dtype=np.uint8)
            else:
                norm = (depth_m - dmin) / (dmax - dmin)
                vis = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(vis).save(path)
