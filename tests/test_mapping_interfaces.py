from __future__ import annotations

from interfaces.common import ArtifactReference, SourceAttribution
from interfaces.grounding import (
    GroundingRequest,
    GroundingResult,
    ObjectCandidate,
    PoseCandidate,
    SceneGraphQuery,
    SurfaceCandidate,
)
from interfaces.perception import (
    CameraIntrinsics,
    CameraPose,
    GeometryMapSummary,
    MapArtifactSummary,
    MappingTracePayload,
    PointCloudFrame,
    RGBDFrame,
)
from interfaces.scene_graph import ConstraintState, SceneEdge, SceneGraph, SceneNode
from interfaces.world_state import ObjectState, WorldState


def test_mapping_interfaces_serialize_nested_payloads() -> None:
    provenance = [
        SourceAttribution(
            source_type="sim_ground_truth",
            source_id="mujoco/camera/wrist_rgbd",
            confidence=1.0,
        )
    ]
    intrinsics = CameraIntrinsics(
        camera_name="wrist_rgbd",
        width_px=640,
        height_px=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        near_m=0.01,
        far_m=5.0,
    )
    pose = CameraPose(
        frame_id="wrist_rgbd",
        parent_frame_id="world",
        position_xyz=[0.5, 0.0, 0.6],
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        timestamp_s=0.25,
    )
    rgb = ArtifactReference(
        artifact_id="rgb-0",
        artifact_type="image",
        path="frames/frame_000000/rgb.png",
        format="png",
        role="rgb",
    )
    depth = ArtifactReference(
        artifact_id="depth-0",
        artifact_type="depth",
        path="frames/frame_000000/depth.npy",
        format="npy",
        role="depth",
    )
    frame = RGBDFrame(
        frame_id="frame_000000",
        sequence_index=0,
        timestamp_s=0.25,
        camera_name="wrist_rgbd",
        intrinsics=intrinsics,
        pose=pose,
        rgb_artifact=rgb,
        depth_artifact=depth,
        valid_depth_ratio=0.98,
        provenance=provenance,
    )
    cloud = PointCloudFrame(
        frame_id="pc_000000",
        source_rgbd_frame_id=frame.frame_id,
        point_count=1024,
        bounds_min_m=[-0.1, -0.2, 0.0],
        bounds_max_m=[0.6, 0.2, 0.8],
        pointcloud_artifact=ArtifactReference(
            artifact_id="cloud-0",
            artifact_type="pointcloud",
            path="frames/frame_000000/pointcloud.npz",
            format="npz",
            role="pointcloud",
        ),
        voxel_size_m=0.01,
        provenance=provenance,
    )
    summary = GeometryMapSummary(
        map_id="map-0001",
        scene_id="scene-1",
        frame_count=1,
        fused_point_count=cloud.point_count,
        voxel_size_m=0.01,
        bounds_min_m=cloud.bounds_min_m,
        bounds_max_m=cloud.bounds_max_m,
        support_surface_ids=["table_top"],
        provenance=[
            SourceAttribution(
                source_type="rgbd_fused_geometry",
                source_id="fusion_v0",
            )
        ],
    )
    artifacts = MapArtifactSummary(
        mapping_run_id="map-0001",
        root_dir="artifacts/maps/map-0001",
        fused_pointcloud_artifact=ArtifactReference(
            artifact_id="fused-cloud",
            artifact_type="pointcloud",
            path="fused/fused_pointcloud.npz",
            format="npz",
            role="fused_pointcloud",
        ),
    )
    payload = MappingTracePayload(
        mapping_run_id="map-0001",
        scene_id="scene-1",
        camera_name="wrist_rgbd",
        pose_source="sim_ground_truth",
        frame_ids=[frame.frame_id],
        fused_frame_count=1,
        timings_ms={"capture": 8.0, "fusion": 3.0},
        map_summary=summary,
        artifact_summary=artifacts,
    ).to_dict()

    assert payload["frame_ids"] == ["frame_000000"]
    assert payload["map_summary"]["support_surface_ids"] == ["table_top"]
    assert payload["artifact_summary"]["fused_pointcloud_artifact"]["path"].endswith(
        "fused_pointcloud.npz"
    )
    assert frame.to_dict()["pose"]["pose_source"] == "sim_ground_truth"


def test_world_state_and_grounding_result_keep_summary_level_contracts() -> None:
    scene_graph = SceneGraph(
        scene_id="scene-1",
        root_frame_id="world",
        revision=2,
        nodes=[
            SceneNode(
                node_id="bottle_1",
                label="bottle",
                node_type="object",
                center_pose=[0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                bbox_extent_m=[0.05, 0.05, 0.25],
                geometry_anchor_id="cluster_1",
            ),
            SceneNode(
                node_id="tray_surface",
                label="tray",
                node_type="surface",
                center_pose=[0.7, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0],
                bbox_extent_m=[0.2, 0.2, 0.02],
                geometry_anchor_id="surface_1",
            ),
        ],
        edges=[
            SceneEdge(
                edge_id="bottle_on_tray",
                source_node_id="bottle_1",
                target_node_id="tray_surface",
                relation_type="on",
                relation_family="support",
            )
        ],
    )
    constraints = [
        ConstraintState(
            constraint_id="reachability_bottle",
            subject_id="bottle_1",
            constraint_type="reachability",
            status="satisfied",
            metric_values={"max_arm_extension_m": 0.82},
            symbolic_values={"method": "rule_based"},
        )
    ]
    summary = GeometryMapSummary(
        map_id="map-0001",
        scene_id="scene-1",
        frame_count=12,
        fused_point_count=2048,
        voxel_size_m=0.01,
        bounds_min_m=[-0.3, -0.4, 0.0],
        bounds_max_m=[0.9, 0.4, 0.8],
        support_surface_ids=["tray_surface"],
    )
    world_state = WorldState(
        scene_id="scene-1",
        objects=[
            ObjectState(
                object_id="bottle_1",
                label="bottle",
                pose=[0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                source_node_id="bottle_1",
                pose_source="sim_ground_truth",
            )
        ],
        robot_mode="ready",
        geometry_map=summary,
        scene_graph=scene_graph,
        constraint_states=constraints,
    ).to_dict()

    request = GroundingRequest(
        request_id="ground-1",
        task_id="task-1",
        scene_id="scene-1",
        object_query=SceneGraphQuery(
            query_id="target_object",
            text="bottle",
            target_label="bottle",
            node_types=["object"],
        ),
        surface_query=SceneGraphQuery(
            query_id="target_surface",
            text="tray",
            target_label="tray",
            node_types=["surface"],
        ),
        required_constraint_types=["reachability"],
    )
    result = GroundingResult(
        request_id=request.request_id,
        scene_id="scene-1",
        object_candidates=[
            ObjectCandidate(
                candidate_id="obj-1",
                object_id="bottle_1",
                node_id="bottle_1",
                label="bottle",
                score=0.98,
                pose=[0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                supporting_surface_id="tray_surface",
                source_types=["sim_ground_truth", "rgbd_fused_geometry"],
            )
        ],
        surface_candidates=[
            SurfaceCandidate(
                candidate_id="surf-1",
                surface_id="tray_surface",
                node_id="tray_surface",
                label="tray",
                score=0.91,
                pose=[0.7, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0],
                normal_xyz=[0.0, 0.0, 1.0],
                source_types=["rgbd_fused_geometry"],
            )
        ],
        pose_candidates=[
            PoseCandidate(
                candidate_id="pose-1",
                frame_id="world",
                pose=[0.7, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                score=0.88,
                reason="place_pose_above_surface",
                reference_node_id="tray_surface",
                source_types=["inferred_rule"],
            )
        ],
        constraint_states=constraints,
    ).to_dict()

    assert world_state["geometry_map"]["fused_point_count"] == 2048
    assert world_state["scene_graph"]["nodes"][0]["geometry_anchor_id"] == "cluster_1"
    assert result["object_candidates"][0]["supporting_surface_id"] == "tray_surface"
    assert result["pose_candidates"][0]["reason"] == "place_pose_above_surface"
