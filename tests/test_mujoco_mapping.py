from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import apps.run_mujoco_mapping as mapping_cli
from interfaces.mapping import CameraIntrinsics, CameraPose
from modules.world_model import (
    PointCloudMapFuser,
    RGBDBackprojector,
    SceneGraphBuilder,
    SceneObjectObservation,
)


def test_rgbd_backprojector_outputs_expected_shape_and_values() -> None:
    depth_m = np.asarray([[1.0, 2.0], [np.nan, 0.0]], dtype=np.float64)
    rgb = np.asarray(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )

    cloud = RGBDBackprojector().backproject(depth_m, fx=1.0, fy=1.0, cx=0.5, cy=0.5, rgb=rgb)

    assert cloud.points.shape == (2, 3)
    np.testing.assert_allclose(
        cloud.points,
        np.asarray([[-0.5, 0.5, -1.0], [1.0, 1.0, -2.0]], dtype=np.float64),
    )
    assert cloud.colors is not None
    assert cloud.colors.tolist() == [[10, 20, 30], [40, 50, 60]]


def test_point_cloud_map_fuser_voxel_downsamples_points() -> None:
    fuser = PointCloudMapFuser(voxel_size=0.1)
    points = np.asarray(
        [
            [0.01, 0.01, 0.0],
            [0.02, 0.02, 0.0],
            [0.19, 0.01, 0.0],
        ],
        dtype=np.float64,
    )
    colors = np.asarray(
        [
            [10, 10, 10],
            [20, 20, 20],
            [40, 50, 60],
        ],
        dtype=np.uint8,
    )

    fuser.add_points(points, colors)
    fused = fuser.build()

    assert fused.points.shape == (2, 3)
    assert fused.colors is not None
    np.testing.assert_allclose(fused.points[0], np.asarray([0.015, 0.015, 0.0], dtype=np.float64))
    assert fused.colors[0].tolist() == [15, 15, 15]


def test_scene_graph_builder_infers_support_and_reachability() -> None:
    builder = SceneGraphBuilder(reachable_radius=1.0)
    objects = [
        SceneObjectObservation(
            object_id="floor",
            label="floor",
            category="support_surface",
            position=[0.0, 0.0, 0.0],
            bbox_min=[-1.0, -1.0, -0.01],
            bbox_max=[1.0, 1.0, 0.0],
            static=True,
        ),
        SceneObjectObservation(
            object_id="box",
            label="box",
            category="sim_object",
            position=[0.4, 0.0, 0.03],
            bbox_min=[0.35, -0.05, 0.0],
            bbox_max=[0.45, 0.05, 0.06],
            static=True,
        ),
        SceneObjectObservation(
            object_id="far_box",
            label="far_box",
            category="sim_object",
            position=[1.5, 0.0, 0.03],
            bbox_min=[1.45, -0.05, 0.0],
            bbox_max=[1.55, 0.05, 0.06],
            static=True,
        ),
    ]

    graph = builder.build(scene_id="scene", objects=objects, robot_position=[0.0, 0.0, 0.0])
    relations = {
        (edge.source_node_id, edge.target_node_id, edge.relation_type)
        for edge in graph.edges
    }

    assert ("floor", "box", "supports") in relations
    assert ("box", "floor", "supported_by") in relations
    assert ("robot_base", "box", "can_reach") in relations
    assert ("robot_base", "far_box", "can_reach") not in relations


def test_run_mujoco_mapping_cli_writes_expected_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intrinsics = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=0.5, cy=0.5)
    pose_identity = CameraPose(
        position=[0.0, 0.0, 0.0],
        rotation=np.eye(3, dtype=np.float64).tolist(),
    )
    pose_shifted = CameraPose(
        position=[0.1, 0.0, 0.0],
        rotation=np.eye(3, dtype=np.float64).tolist(),
    )
    rgb = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    depth = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    def fake_capture_mapping_session(
        scene_path: Path,
        camera_name: str,
        width: int,
        height: int,
        frame_count: int,
    ) -> tuple[str, list[mapping_cli.CapturedFrame], list[SceneObjectObservation], list[float]]:
        assert camera_name == "wrist_rgbd"
        assert width == 2
        assert height == 2
        assert frame_count == 2
        frames = [
            mapping_cli.CapturedFrame(
                frame_id="frame_000",
                timestamp=0.0,
                rgb=rgb,
                depth_m=depth,
                intrinsics=intrinsics,
                pose=pose_identity,
            ),
            mapping_cli.CapturedFrame(
                frame_id="frame_001",
                timestamp=1.0,
                rgb=rgb,
                depth_m=depth,
                intrinsics=intrinsics,
                pose=pose_shifted,
            ),
        ]
        objects = [
            SceneObjectObservation(
                object_id="floor",
                label="floor",
                category="support_surface",
                position=[0.0, 0.0, 0.0],
                bbox_min=[-1.0, -1.0, -0.01],
                bbox_max=[1.0, 1.0, 0.0],
                static=True,
            ),
            SceneObjectObservation(
                object_id="mapping_box",
                label="mapping_box",
                category="sim_object",
                position=[0.5, 0.0, 0.03],
                bbox_min=[0.45, -0.05, 0.0],
                bbox_max=[0.55, 0.05, 0.06],
                static=True,
            ),
        ]
        return "synthetic_scene", frames, objects, [0.0, 0.0, 0.0]

    monkeypatch.setattr(mapping_cli, "capture_mapping_session", fake_capture_mapping_session)

    temp_dir = mapping_cli.ROOT / ".tmp-tests-mapping"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = temp_dir / "mapping_artifacts"
    mapping_cli.main(
        [
            "--width",
            "2",
            "--height",
            "2",
            "--frame-count",
            "2",
            "--voxel-size",
            "0.05",
            "--out-dir",
            str(out_dir),
        ]
    )

    expected_files = [
        out_dir / "poses.json",
        out_dir / "frame_metadata.json",
        out_dir / "global_cloud.ply",
        out_dir / "summary.json",
        out_dir / "scene_graph.json",
        out_dir / "trace.json",
    ]
    assert all(path.exists() for path in expected_files)

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["frame_count"] == 2
    assert summary["total_points"] > 0
    assert summary["object_count"] == 3
    assert "trace" in summary["artifacts"]

    trace = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
    assert trace["events"][-1]["payload"]["frame_count"] == 2
    shutil.rmtree(temp_dir)
