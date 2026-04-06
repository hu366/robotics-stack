# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import numpy.typing as npt

from interfaces.execution_trace import ExecutionTrace
from interfaces.mapping import CameraIntrinsics, CameraPose, FrameMetadata, MappingSummary
from modules.world_model import (
    MappingArtifactStore,
    PointCloudMapFuser,
    RGBDBackprojector,
    SceneGraphBuilder,
    SceneObjectObservation,
)

FloatArray = npt.NDArray[np.float64]
UInt8Array = npt.NDArray[np.uint8]

DEFAULT_JOINT_DELTAS = np.asarray(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.10, 0.0, -0.10, 0.10, 0.15, -0.10],
        [-0.25, 0.10, 0.0, -0.10, -0.10, 0.15, 0.10],
        [0.0, 0.25, -0.15, -0.20, 0.0, 0.25, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class CapturedFrame:
    frame_id: str
    timestamp: float
    rgb: UInt8Array
    depth_m: FloatArray
    intrinsics: CameraIntrinsics
    pose: CameraPose


def _load_model(mujoco: Any, scene_path: Path) -> Any:
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except ValueError:
        model_xml = scene_path.read_text(encoding="utf-8")
        assets: dict[str, bytes] = {}
        for file_path in scene_path.parent.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(scene_path.parent).as_posix()
                assets[rel] = file_path.read_bytes()
        return mujoco.MjModel.from_xml_string(model_xml, assets=assets)


def _depth_to_meters(model: Any, depth_buffer: npt.NDArray[np.floating[Any]]) -> FloatArray:
    near = float(model.vis.map.znear * model.stat.extent)
    far = float(model.vis.map.zfar * model.stat.extent)
    return (near / (1.0 - depth_buffer * (1.0 - near / far))).astype(np.float64, copy=False)


def _camera_intrinsics(model: Any, cam_id: int, width: int, height: int) -> CameraIntrinsics:
    fovy = np.deg2rad(float(model.cam_fovy[cam_id]))
    focal = 0.5 * height / np.tan(fovy / 2.0)
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(focal),
        fy=float(focal),
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )


def _camera_pose(data: Any, cam_id: int) -> CameraPose:
    position = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    rotation = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    return CameraPose(position=position.tolist(), rotation=rotation.tolist())


def _initialize_to_home(mujoco: Any, model: Any, data: Any) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def _robot_waypoints(model: Any, data: Any, frame_count: int) -> list[FloatArray]:
    home_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    if home_qpos.shape[0] < 9:
        raise ValueError("expected Panda qpos to include 7 arm joints and 2 fingers")

    joint_limits = np.asarray(model.jnt_range[:7], dtype=np.float64)
    waypoints: list[FloatArray] = []
    for index in range(frame_count):
        waypoint = home_qpos.copy()
        delta = DEFAULT_JOINT_DELTAS[index % len(DEFAULT_JOINT_DELTAS)]
        waypoint[:7] = np.clip(home_qpos[:7] + delta, joint_limits[:, 0], joint_limits[:, 1])
        waypoint[7:9] = home_qpos[7:9]
        waypoints.append(waypoint)
    return waypoints


def capture_mapping_session(
    scene_path: Path,
    camera_name: str,
    width: int,
    height: int,
    frame_count: int,
) -> tuple[str, list[CapturedFrame], list[SceneObjectObservation], list[float]]:
    import mujoco

    model = _load_model(mujoco, scene_path)
    data = mujoco.MjData(model)
    _initialize_to_home(mujoco, model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"camera not found: {camera_name}")

    intrinsics = _camera_intrinsics(model, cam_id, width, height)
    waypoints = _robot_waypoints(model, data, frame_count)
    frames: list[CapturedFrame] = []

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for index, waypoint in enumerate(waypoints):
            data.qpos[:] = waypoint
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera_name)
            rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=camera_name)
            depth_buffer = np.asarray(renderer.render(), dtype=np.float64).copy()
            renderer.disable_depth_rendering()

            frames.append(
                CapturedFrame(
                    frame_id=f"frame_{index:03d}",
                    timestamp=float(index),
                    rgb=rgb,
                    depth_m=_depth_to_meters(model, depth_buffer),
                    intrinsics=intrinsics,
                    pose=_camera_pose(data, cam_id),
                )
            )
    finally:
        renderer.close()

    objects = _collect_scene_objects(mujoco, model, data)
    robot_position = _robot_position(mujoco, model, data)
    return scene_path.stem, frames, objects, robot_position


def _collect_scene_objects(mujoco: Any, model: Any, data: Any) -> list[SceneObjectObservation]:
    objects: list[SceneObjectObservation] = []
    extent = float(model.stat.extent)

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name != "floor":
            continue
        objects.append(
            SceneObjectObservation(
                object_id="floor",
                label="floor",
                category="support_surface",
                position=[0.0, 0.0, 0.0],
                bbox_min=[-extent, -extent, -0.001],
                bbox_max=[extent, extent, 0.0],
                static=True,
                attributes={"known_from_sim": "true"},
            )
        )

    robot_prefixes = ("link", "hand", "left_finger", "right_finger", "mocap")
    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None or body_name.startswith(robot_prefixes):
            continue

        bbox = _body_aabb(mujoco, model, data, body_id)
        if bbox is None:
            continue
        bbox_min, bbox_max = bbox
        center = ((bbox_min + bbox_max) / 2.0).tolist()

        objects.append(
            SceneObjectObservation(
                object_id=body_name,
                label=body_name,
                category="sim_object",
                position=center,
                bbox_min=bbox_min.tolist(),
                bbox_max=bbox_max.tolist(),
                static=bool(model.body_jntnum[body_id] == 0),
                attributes={"known_from_sim": "true"},
            )
        )

    return objects


def _body_aabb(
    mujoco: Any,
    model: Any,
    data: Any,
    body_id: int,
) -> tuple[FloatArray, FloatArray] | None:
    geom_start = int(model.body_geomadr[body_id])
    geom_count = int(model.body_geomnum[body_id])
    if geom_count <= 0:
        return None

    mins: list[FloatArray] = []
    maxs: list[FloatArray] = []
    for geom_id in range(geom_start, geom_start + geom_count):
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        geom_bbox = _geom_aabb(mujoco, model, data, geom_id)
        if geom_bbox is None:
            continue
        geom_min, geom_max = geom_bbox
        mins.append(geom_min)
        maxs.append(geom_max)

    if not mins:
        return None
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def _geom_aabb(
    mujoco: Any,
    model: Any,
    data: Any,
    geom_id: int,
) -> tuple[FloatArray, FloatArray] | None:
    geom_type = int(model.geom_type[geom_id])
    position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half_extents = size[:3]
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half_extents = np.full(3, size[0], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        half_extents = size[:3]
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half_extents = np.asarray([size[0], size[0], size[1]], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_extents = np.asarray([size[0], size[0], size[1] + size[0]], dtype=np.float64)
    else:
        radius = float(model.geom_rbound[geom_id])
        if radius <= 0.0:
            return None
        half_extents = np.full(3, radius, dtype=np.float64)

    world_half = np.abs(rotation) @ half_extents
    return position - world_half, position + world_half


def _robot_position(mujoco: Any, model: Any, data: Any) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    if body_id < 0:
        return [0.0, 0.0, 0.0]
    position = np.asarray(data.xpos[body_id], dtype=np.float64)
    return [float(value) for value in position]


def run_mapping_pipeline(
    scene_id: str,
    camera_name: str,
    frames: list[CapturedFrame],
    objects: list[SceneObjectObservation],
    robot_position: list[float],
    out_dir: Path,
    voxel_size: float,
) -> MappingSummary:
    trace = ExecutionTrace(trace_id=f"trace-{uuid4().hex[:8]}", task_id=f"mapping-{scene_id}")
    trace.add_event(
        "capture",
        "frames_captured",
        "success",
        payload={"scene_id": scene_id, "camera_name": camera_name, "frame_count": len(frames)},
    )

    store = MappingArtifactStore(out_dir)
    backprojector = RGBDBackprojector()
    fuser = PointCloudMapFuser(voxel_size=voxel_size)
    frame_metadata: list[FrameMetadata] = []

    for frame in frames:
        cloud_camera = backprojector.backproject(
            frame.depth_m,
            fx=frame.intrinsics.fx,
            fy=frame.intrinsics.fy,
            cx=frame.intrinsics.cx,
            cy=frame.intrinsics.cy,
            rgb=frame.rgb,
        )
        cloud_world = backprojector.transform_to_world(cloud_camera, frame.pose)
        fuser.add_points(cloud_world.points, cloud_world.colors)

        rgb_path, depth_npy_path, depth_vis_path = store.save_frame_artifacts(
            frame.frame_id,
            frame.rgb,
            frame.depth_m,
        )
        metadata = FrameMetadata(
            frame_id=frame.frame_id,
            camera_name=camera_name,
            timestamp=frame.timestamp,
            intrinsics=frame.intrinsics,
            pose=frame.pose,
            point_count=int(cloud_world.points.shape[0]),
            rgb_path=rgb_path,
            depth_npy_path=depth_npy_path,
            depth_vis_path=depth_vis_path,
        )
        frame_metadata.append(metadata)
        trace.add_event(
            "fusion",
            "frame_fused",
            "success",
            payload={"frame_id": frame.frame_id, "point_count": metadata.point_count},
        )

    fused = fuser.build()
    scene_graph = SceneGraphBuilder().build(
        scene_id=scene_id,
        objects=objects,
        robot_position=robot_position,
    )

    artifacts = {
        "poses": store.write_poses(frame_metadata),
        "frame_metadata": store.write_frame_metadata(frame_metadata),
        "global_cloud": store.write_global_cloud(fused.points, fused.colors),
        "scene_graph": store.write_scene_graph(scene_graph),
    }

    bbox_min, bbox_max = fused.bbox()
    summary = MappingSummary(
        scene_id=scene_id,
        camera_name=camera_name,
        frame_count=len(frame_metadata),
        input_point_count=fuser.total_input_points,
        total_points=int(fused.points.shape[0]),
        voxel_size=voxel_size,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        object_count=len(scene_graph.nodes),
        artifacts=artifacts,
    )
    artifacts["summary"] = store.write_summary(summary)

    trace.add_event(
        "scene_graph",
        "scene_graph_built",
        "success",
        payload={"object_count": len(scene_graph.nodes), "edge_count": len(scene_graph.edges)},
    )
    trace.add_event(
        "mapping",
        "mapping_completed",
        "success",
        payload=summary.to_dict(),
    )
    artifacts["trace"] = store.write_trace(trace)
    summary.artifacts = artifacts
    artifacts["summary"] = store.write_summary(summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture multi-frame MuJoCo wrist RGBD and build a fused map baseline."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("sim/scenes/mujoco_mapping_panda.xml"),
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--camera",
        default="wrist_rgbd",
        help="Camera name from the scene XML.",
    )
    parser.add_argument("--width", type=int, default=320, help="RGBD image width.")
    parser.add_argument("--height", type=int, default=240, help="RGBD image height.")
    parser.add_argument(
        "--frame-count",
        type=int,
        default=4,
        help="Number of wrist RGBD frames to capture.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size in meters for global cloud downsampling. Use 0 to disable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/mujoco_mapping"),
        help="Directory for exported artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scene_path = args.scene if args.scene.is_absolute() else ROOT / args.scene
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    scene_id, frames, objects, robot_position = capture_mapping_session(
        scene_path=scene_path,
        camera_name=args.camera,
        width=args.width,
        height=args.height,
        frame_count=args.frame_count,
    )
    summary = run_mapping_pipeline(
        scene_id=scene_id,
        camera_name=args.camera,
        frames=frames,
        objects=objects,
        robot_position=robot_position,
        out_dir=out_dir,
        voxel_size=args.voxel_size,
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
