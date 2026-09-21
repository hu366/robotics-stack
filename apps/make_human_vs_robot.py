"""Split-screen: human head-cam demo | dual-Piper execution from the same ego view.

Left  = original head-mounted first-person video.
Right = MuJoCo ego_head camera looking down at both grippers (training view).
Frames are aligned by clip frame_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.replay_piper_ee import (  # noqa: E402
    CONTACT_BOX_SIZE_M,
    CONTACT_PAD_SIZE_M,
    DEFAULT_CLIP,
    PIPER_XML,
    _joint_qpos_adr,
    arm_joint_names,
    cam_traj_to_base,
    directional_support_radius,
    load_ee_item,
    load_model,
    map_gripper,
    solve_ik,
)

DEFAULT_VIDEO = Path(r"C:\Users\Administrator\Downloads\video_20260914_210702..mp4")
DEFAULT_OUT = Path("artifacts/human_vs_robot")

PANEL_W, PANEL_H = 960, 540
HEADER_H = 56
LABEL_HUMAN = "Left: Human demonstration"
LABEL_ROBOT = "Right: Robot execution"
SUB_HUMAN = "head-mounted first-person camera"
SUB_ROBOT = "ego-head camera (same view as training)"

# The ER preview returns sparse camera-frame poses. These values keep the
# visualization inside the Piper workspace and make the two parallel grippers
# hold opposite sides of the box instead of following two unrelated points.
OBJECT_X_TARGET_M = 0.48
BOX_GRASP_WIDTH_M = 0.18
GRASP_CLEARANCE_M = 0.04
GRASP_SEPARATION_M = BOX_GRASP_WIDTH_M + GRASP_CLEARANCE_M
# The pad body target is placed slightly outside the object's supporting face;
# the pad's projected half-thickness then provides physical contact overlap.
CONTACT_PAD_PRELOAD_M = 0.0003
CONTACT_MAX_SIDE_FORCE_N = 50.0
# Kinematic replay is a visual deliverable: any negative MuJoCo contact
# distance is a visible mesh intersection, so its acceptance limit is zero.
# The contact-dynamics diagnostic keeps the looser physical penetration limit.
KINEMATIC_MAX_PENETRATION_M = 0.0
CONTACT_MAX_PENETRATION_M = 0.0005
CONTACT_KINEMATIC_CLEARANCE_M = 0.002
MAX_JOINT_SPEED_RAD_S = 3.0
MAX_GRIPPER_SPEED_M_S = 0.20
MAX_IK_ERROR_M = 0.03
# The contact object is placed directly at the first recentered trajectory pose.
# Keeping this in one transform avoids applying an object offset a second time
# while solving the explicit fingertip-pad body targets.
CONTACT_OBJECT_OFFSET_M = np.zeros(3, dtype=np.float64)
# Keep the rotated box inside the overlap of both Piper workspaces.  The raw
# camera calibration places the +X grasp side too close to the left arm's
# reach limit once the box turns; this recentering leaves room for the vendor
# gripper-base mesh, not just the synthetic fingertip pad.
CONTACT_BOX_POSITION_M = np.array([0.35, 0.0, 0.135], dtype=np.float64)
CONTACT_STABLE_RATIO = 0.80
CONTACT_DROP_Z_M = 0.04
OBJECT_POSITION_SUCCESS_MEAN_M = 0.05
OBJECT_POSITION_SUCCESS_MAX_M = 0.10
OBJECT_ORIENTATION_SUCCESS_MEAN_RAD = 0.35
OBJECT_ORIENTATION_SUCCESS_MAX_RAD = 0.70


def _grasp_axis_world(world: dict[str, np.ndarray]) -> np.ndarray:
    """Infer the horizontal line joining the two initial fingertip targets."""
    raw_axis = (
        np.asarray(world["left"][0, :3, 3], dtype=np.float64)
        - np.asarray(world["right"][0, :3, 3], dtype=np.float64)
    )
    raw_axis[2] = 0.0
    norm = float(np.linalg.norm(raw_axis))
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return raw_axis / norm


def contact_grasp_geometry(
    object_rotation: np.ndarray,
    axis_world: np.ndarray,
    pad_rotations: dict[str, np.ndarray],
    *,
    pad_center_offset: float = CONTACT_PAD_PRELOAD_M,
) -> dict[str, Any]:
    """Compute a designed center spacing for the explicit contact pads.

    ``pad_center_offset`` is a small, explicit pad preload.  The target
    spacing is computed from the actual directional support of each pad, so
    the resulting geometry cannot rely on several millimetres of overlap.
    """
    if not pad_rotations:
        raise ValueError("pad_rotations must contain at least one pad")
    object_support = directional_support_radius(
        CONTACT_BOX_SIZE_M, object_rotation, axis_world
    )
    pad_supports = {
        side: directional_support_radius(CONTACT_PAD_SIZE_M, rotation, axis_world)
        for side, rotation in pad_rotations.items()
    }
    center_offset = max(float(pad_center_offset), 0.0)
    separation = (
        2.0 * object_support
        + sum(pad_supports.values())
        - 2.0 * center_offset
    )
    pad_interference = {
        side: center_offset for side in pad_supports
    }
    normalized_axis = np.asarray(axis_world, dtype=np.float64) / np.linalg.norm(axis_world)
    return {
        "axis_world": normalized_axis.tolist(),
        "object_support_m": object_support,
        "pad_support_m": max(pad_supports.values()),
        "pad_supports_m": pad_supports,
        "pad_interference_m": pad_interference,
        "pad_center_offset_m": center_offset,
        "separation_m": separation,
    }


def _is_stable_grasp(
    *,
    physics_steps: int,
    both_contact_ratio: float,
    min_object_z_m: float,
    max_side_force_n: float,
    floor_contact_physics_steps: int,
    max_penetration_m: float = 0.0,
) -> bool:
    return bool(
        physics_steps > 0
        and both_contact_ratio >= CONTACT_STABLE_RATIO
        and min_object_z_m >= CONTACT_DROP_Z_M
        and max_side_force_n <= CONTACT_MAX_SIDE_FORCE_N
        and floor_contact_physics_steps == 0
        and max_penetration_m <= CONTACT_MAX_PENETRATION_M
    )


def _hold_to_video_length(
    frame_ids: np.ndarray,
    traj: np.ndarray,
    n_video: int,
) -> np.ndarray:
    """Map sparse retarget frames onto every source frame, holding last pose."""
    by_id = {int(fid): row for fid, row in zip(frame_ids, traj, strict=True)}
    ordered = sorted(by_id)
    out = np.zeros((n_video, traj.shape[-1]), dtype=np.float64)
    if not ordered:
        return out
    last = by_id[ordered[0]]
    j = 0
    for i in range(n_video):
        while j < len(ordered) and ordered[j] <= i:
            last = by_id[ordered[j]]
            j += 1
        out[i] = last
    return out


def _interpolate_ee_trajectory(
    frame_ids: np.ndarray,
    traj: np.ndarray,
    n_video: int,
) -> np.ndarray:
    """Densify [xyz, quat_wxyz, gripper] ER waypoints to source frames."""
    ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(traj, dtype=np.float64)
    if n_video <= 0:
        raise ValueError("n_video must be positive")
    if values.ndim != 2 or values.shape != (len(ids), 8):
        raise ValueError(f"EE trajectory must have shape ({len(ids)}, 8), got {values.shape}")
    if not len(ids):
        raise ValueError("EE trajectory is empty")
    by_id = {int(fid): row for fid, row in zip(ids, values, strict=True)}
    ordered = sorted(by_id.items())
    sparse_ids = np.asarray([item[0] for item in ordered], dtype=np.int64)
    sparse_traj = np.asarray([item[1] for item in ordered], dtype=np.float64)
    sparse_traj[:, 3:7] = np.asarray(
        [_normalize_quat_wxyz(quat) for quat in sparse_traj[:, 3:7]],
        dtype=np.float64,
    )
    dense = np.empty((n_video, 8), dtype=np.float64)
    for frame in range(n_video):
        right = int(np.searchsorted(sparse_ids, frame, side="right"))
        if right == 0:
            dense[frame] = sparse_traj[0]
            continue
        if right >= len(sparse_ids):
            dense[frame] = sparse_traj[-1]
            continue
        left = right - 1
        span = int(sparse_ids[right] - sparse_ids[left])
        amount = 0.0 if span <= 0 else (frame - sparse_ids[left]) / span
        dense[frame, :3] = (
            (1.0 - amount) * sparse_traj[left, :3]
            + amount * sparse_traj[right, :3]
        )
        dense[frame, 3:7] = _slerp_quat_wxyz(
            sparse_traj[left, 3:7], sparse_traj[right, 3:7], float(amount)
        )
        dense[frame, 7] = (
            (1.0 - amount) * sparse_traj[left, 7]
            + amount * sparse_traj[right, 7]
        )
    return dense


def _load_dense_ee_trajectory(clip: Path, side: str, n_video: int) -> np.ndarray:
    """Interpolate each hand from its own sparse timestamps.

    Using per-hand timestamps avoids manufacturing a zero/hold waypoint when
    the detector emits asynchronous left and right keyframes.
    """
    item = load_ee_item(clip, side)
    return _interpolate_ee_trajectory(
        np.asarray(item["frame_ids"], dtype=np.int64),
        np.asarray(item["traj"], dtype=np.float64),
        n_video,
    )


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return value / norm


def _slerp_quat_wxyz(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    """Interpolate two wxyz quaternions while keeping the shortest rotation."""
    q0 = _normalize_quat_wxyz(first)
    q1 = _normalize_quat_wxyz(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat_wxyz(q0 + amount * (q1 - q0))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < 1e-12:
        return q0
    weight0 = float(np.sin((1.0 - amount) * theta) / sin_theta)
    weight1 = float(np.sin(amount * theta) / sin_theta)
    return _normalize_quat_wxyz(weight0 * q0 + weight1 * q1)


def _interpolate_object_trajectory(
    frame_ids: np.ndarray,
    poses: np.ndarray,
    n_video: int,
) -> np.ndarray:
    """Densify camera-frame [xyz, quat_wxyz] object poses to source frames."""
    ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(poses, dtype=np.float64)
    if n_video <= 0:
        raise ValueError("n_video must be positive")
    if values.ndim != 2 or values.shape != (len(ids), 7):
        raise ValueError(f"object poses must have shape ({len(ids)}, 7), got {values.shape}")
    if not len(ids):
        raise ValueError("object trajectory is empty")
    by_id = {int(fid): row for fid, row in zip(ids, values, strict=True)}
    ordered = sorted(by_id.items())
    sparse_ids = np.asarray([item[0] for item in ordered], dtype=np.int64)
    sparse_poses = np.asarray([item[1] for item in ordered], dtype=np.float64)
    sparse_poses[:, 3:] = np.asarray(
        [_normalize_quat_wxyz(quat) for quat in sparse_poses[:, 3:]], dtype=np.float64
    )
    dense = np.empty((n_video, 7), dtype=np.float64)
    for frame in range(n_video):
        right = int(np.searchsorted(sparse_ids, frame, side="right"))
        if right == 0:
            dense[frame] = sparse_poses[0]
        elif right >= len(sparse_ids):
            dense[frame] = sparse_poses[-1]
        else:
            left = right - 1
            span = int(sparse_ids[right] - sparse_ids[left])
            amount = 0.0 if span <= 0 else (frame - sparse_ids[left]) / span
            dense[frame, :3] = (
                (1.0 - amount) * sparse_poses[left, :3]
                + amount * sparse_poses[right, :3]
            )
            dense[frame, 3:] = _slerp_quat_wxyz(
                sparse_poses[left, 3:], sparse_poses[right, 3:], float(amount)
            )
    return dense


def _mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to MuJoCo's wxyz quaternion order."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * float(np.sqrt(trace + 1.0))
        quat = np.array(
            [
                0.25 * scale,
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(m)
        major = int(np.argmax(diagonal))
        if major == 0:
            scale = 2.0 * float(np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)))
            quat = np.array(
                [
                    (m[2, 1] - m[1, 2]) / scale,
                    0.25 * scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif major == 1:
            scale = 2.0 * float(np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)))
            quat = np.array(
                [
                    (m[0, 2] - m[2, 0]) / scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    0.25 * scale,
                    (m[1, 2] + m[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * float(np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)))
            quat = np.array(
                [
                    (m[1, 0] - m[0, 1]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return _normalize_quat_wxyz(quat)


def _load_object_world_trajectory(clip: Path, n_video: int) -> np.ndarray | None:
    payload = json.loads(clip.read_text(encoding="utf-8"))
    object_data = payload.get("object_trajectory")
    if not isinstance(object_data, dict):
        stem = clip.parent.name
        candidates = [
            clip.parent / "clip_retargeted.json",
            clip.parent.parent / f"{stem}_retarget" / "clip_retargeted.json",
            clip.parent.parent
            / f"{stem.removesuffix('_ego2dex')}_ego2dex_retarget"
            / "clip_retargeted.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    cand_payload = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(cand_payload.get("object_trajectory"), dict):
                        payload = cand_payload
                        object_data = payload.get("object_trajectory")
                        break
                except Exception:
                    pass
    if not isinstance(object_data, dict):
        return None
    frame_ids = np.asarray(object_data.get("frame_ids", []), dtype=np.int64)
    poses = np.asarray(object_data.get("poses", []), dtype=np.float64)
    dense_camera = _interpolate_object_trajectory(frame_ids, poses, n_video)
    # cam_traj_to_base expects the final gripper column; it is unused when side=None.
    with_gripper = np.column_stack([dense_camera, np.zeros(n_video, dtype=np.float64)])
    world, _ = cam_traj_to_base(with_gripper, apply_calib=True, side=None)
    return world


def _recenter_object_workspace(
    object_world: np.ndarray,
    target_x: float = OBJECT_X_TARGET_M,
    target_position: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the approximate camera calibration into a reachable replay area."""
    shifted = np.asarray(object_world, dtype=np.float64).copy()
    if target_position is None:
        target = np.array(
            [target_x, float(shifted[0, 1, 3]), float(shifted[0, 2, 3])],
            dtype=np.float64,
        )
    else:
        target = np.asarray(target_position, dtype=np.float64).reshape(3)
    translation = target - np.asarray(shifted[0, :3, 3], dtype=np.float64)
    shifted[:, :3, 3] += translation
    return shifted, translation


def _constrain_bimanual_grasp(
    world: dict[str, np.ndarray],
    object_world: np.ndarray,
    separation: float = GRASP_SEPARATION_M,
) -> dict[str, np.ndarray]:
    """Place both grippers on opposite sides of one rigid object trajectory."""
    if set(world) != {"left", "right"}:
        raise ValueError("bimanual grasp requires left and right trajectories")
    if object_world.ndim != 3 or object_world.shape[1:] != (4, 4):
        raise ValueError(f"object_world must have shape (T, 4, 4), got {object_world.shape}")
    if any(len(world[side]) != len(object_world) for side in ("left", "right")):
        raise ValueError("arm and object trajectories must have equal length")
    axis_world_0 = _grasp_axis_world(world)
    object_rot_0 = object_world[0, :3, :3]
    axis_object = object_rot_0.T @ axis_world_0
    half_separation = max(float(separation), 0.02) * 0.5
    constrained = {side: values.copy() for side, values in world.items()}
    relative_rot = {
        side: object_rot_0.T @ world[side][0, :3, :3]
        for side in ("left", "right")
    }
    for i, object_mat in enumerate(object_world):
        axis_world = object_mat[:3, :3] @ axis_object
        axis_world[2] = 0.0
        norm = float(np.linalg.norm(axis_world))
        if norm < 1e-8:
            axis_world = axis_world_0
        else:
            axis_world /= norm
        center = object_mat[:3, 3]
        constrained["left"][i, :3, 3] = center + half_separation * axis_world
        constrained["right"][i, :3, 3] = center - half_separation * axis_world
        for side in ("left", "right"):
            constrained[side][i, :3, :3] = object_mat[:3, :3] @ relative_rot[side]
    return constrained


def _banner(width: int, height: int) -> np.ndarray:
    import cv2

    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (28, 28, 28)
    cv2.putText(
        bar, LABEL_HUMAN, (24, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (240, 240, 240), 1, cv2.LINE_AA,
    )
    cv2.putText(
        bar, SUB_HUMAN, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (170, 170, 170), 1, cv2.LINE_AA,
    )
    cv2.putText(
        bar, LABEL_ROBOT, (width // 2 + 24, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (240, 240, 240), 1, cv2.LINE_AA,
    )
    cv2.putText(
        bar, SUB_ROBOT, (width // 2 + 24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (170, 170, 170), 1, cv2.LINE_AA,
    )
    cv2.line(bar, (width // 2, 8), (width // 2, height - 8), (70, 70, 70), 1)
    return bar


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    out = np.zeros((h, w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


HEAD_POS = np.array([0.10, 0.0, 0.80], dtype=np.float64)


def _ego_mjv_camera(mujoco: Any, lookat: np.ndarray, head_pos: np.ndarray = HEAD_POS) -> Any:
    """Head-mounted view: stand behind the bases, look down at both grippers."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    delta = np.asarray(head_pos, dtype=np.float64) - np.asarray(lookat, dtype=np.float64)
    cam.distance = max(float(np.linalg.norm(delta)), 1e-3)
    cam.azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])))
    cam.elevation = float(np.degrees(np.arctan2(delta[2], np.hypot(delta[0], delta[1]))))
    return cam


def _actuator_ids(mujoco: Any, model: Any, prefix: str) -> list[int]:
    ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}joint{i}"))
        for i in range(1, 9)
    ]
    if any(item < 0 for item in ids):
        raise SystemExit(f"actuators missing for arm prefix={prefix!r}")
    return ids


def _set_arm_controls(data: Any, actuator_ids: list[int], q: np.ndarray, gripper: float) -> None:
    j7, j8 = map_gripper(float(gripper))
    for actuator_id, value in zip(actuator_ids[:6], q, strict=True):
        data.ctrl[actuator_id] = float(value)
    data.ctrl[actuator_ids[6]] = j7
    data.ctrl[actuator_ids[7]] = j8


def _set_arm_qpos(
    mujoco: Any,
    model: Any,
    data: Any,
    prefix: str,
    q: np.ndarray,
    gripper: float,
) -> None:
    """Apply an arm pose directly for collision-checked kinematic replay."""
    names = arm_joint_names(prefix)
    for name, value in zip(names[:6], q, strict=True):
        data.qpos[_joint_qpos_adr(mujoco, model, name)] = float(value)
    j7, j8 = map_gripper(float(gripper))
    data.qpos[_joint_qpos_adr(mujoco, model, names[6])] = j7
    data.qpos[_joint_qpos_adr(mujoco, model, names[7])] = j8


def _move_vector_towards(
    current: np.ndarray,
    target: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """Limit a joint command by distance per physics step, independent of stride."""
    current_f = np.asarray(current, dtype=np.float64)
    target_f = np.asarray(target, dtype=np.float64)
    delta = target_f - current_f
    distance = float(np.linalg.norm(delta))
    if distance <= max_distance or max_distance <= 0.0:
        return target_f.copy() if max_distance > 0.0 else current_f.copy()
    return current_f + delta * (max_distance / distance)


def _move_scalar_towards(current: float, target: float, max_distance: float) -> float:
    delta = float(target) - float(current)
    if max_distance <= 0.0:
        return float(current)
    return float(current) + float(np.clip(delta, -max_distance, max_distance))


def _set_free_body_pose(
    mujoco: Any,
    model: Any,
    data: Any,
    body_name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    joint_name = f"{body_name}_free"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise SystemExit(f"free joint {joint_name!r} missing")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(position, dtype=np.float64)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = _normalize_quat_wxyz(quaternion)


def _arm_geom_sides(mujoco: Any, model: Any) -> dict[int, str]:
    """Return the arm side for every vendor mesh geom in a bimanual scene."""
    sides: dict[int, str] = {}
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or ""
        if body_name.startswith("left_") and not body_name.startswith("left_grasp_pad"):
            sides[geom_id] = "left"
        elif body_name.startswith("right_") and not body_name.startswith("right_grasp_pad"):
            sides[geom_id] = "right"
    return sides


def _contact_metrics(
    mujoco: Any,
    model: Any,
    data: Any,
    box_geom_id: int,
    floor_geom_id: int,
    pad_geom_ids: dict[str, int],
    arm_geom_sides: dict[int, str] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "total_contacts": 0,
        "left_contacts": 0,
        "right_contacts": 0,
        "left_force_n": 0.0,
        "right_force_n": 0.0,
        "max_force_n": 0.0,
        "left_penetration_m": 0.0,
        "right_penetration_m": 0.0,
        "max_penetration_m": 0.0,
        "floor_contacts": 0,
        "floor_force_n": 0.0,
        "arm_contacts": 0,
        "left_arm_contacts": 0,
        "right_arm_contacts": 0,
        "arm_max_penetration_m": 0.0,
        "left_arm_penetration_m": 0.0,
        "right_arm_penetration_m": 0.0,
        "arm_contact_names": [],
        "arm_contact_details": [],
    }
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom_a, geom_b = int(contact.geom[0]), int(contact.geom[1])
        if box_geom_id not in (geom_a, geom_b):
            continue
        metrics["total_contacts"] = int(metrics["total_contacts"]) + 1
        other_geom = geom_b if geom_a == box_geom_id else geom_a
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        force_norm = float(np.linalg.norm(force[:3]))
        metrics["max_force_n"] = max(float(metrics["max_force_n"]), force_norm)
        if other_geom == floor_geom_id:
            metrics["floor_contacts"] = int(metrics["floor_contacts"]) + 1
            metrics["floor_force_n"] = float(metrics["floor_force_n"]) + force_norm
            continue
        penetration = max(0.0, -float(contact.dist))
        for side, pad_geom_id in pad_geom_ids.items():
            if other_geom != pad_geom_id:
                continue
            count_key = f"{side}_contacts"
            force_key = f"{side}_force_n"
            penetration_key = f"{side}_penetration_m"
            metrics[count_key] = int(metrics[count_key]) + 1
            metrics[force_key] = float(metrics[force_key]) + force_norm
            metrics[penetration_key] = max(
                float(metrics[penetration_key]), penetration
            )
            metrics["max_penetration_m"] = max(
                float(metrics["max_penetration_m"]), penetration
            )
        if arm_geom_sides is not None:
            arm_side = arm_geom_sides.get(other_geom)
            if arm_side in ("left", "right"):
                metrics["arm_contacts"] = int(metrics["arm_contacts"]) + 1
                metrics[f"{arm_side}_arm_contacts"] = (
                    int(metrics[f"{arm_side}_arm_contacts"]) + 1
                )
                metrics["arm_max_penetration_m"] = max(
                    float(metrics["arm_max_penetration_m"]), penetration
                )
                metrics[f"{arm_side}_arm_penetration_m"] = max(
                    float(metrics[f"{arm_side}_arm_penetration_m"]), penetration
                )
                geom_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                ) or f"geom_{other_geom}"
                if geom_name not in metrics["arm_contact_names"]:
                    metrics["arm_contact_names"].append(geom_name)
                body_id = int(model.geom_bodyid[other_geom])
                body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id
                ) or f"body_{body_id}"
                metrics["arm_contact_details"].append(
                    {
                        "geom": geom_name,
                        "body": body_name,
                        "penetration_m": penetration,
                        "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
                    }
                )
                metrics["max_penetration_m"] = max(
                    float(metrics["max_penetration_m"]), penetration
                )
    return metrics


def _activate_contact_welds(
    mujoco: Any,
    model: Any,
    data: Any,
    box_body_id: int,
    pad_body_ids: dict[str, int],
    metrics: dict[str, float | int],
) -> list[str]:
    """Activate welds only after both explicit fingertip pads touch the box."""
    if not (
        int(metrics["left_contacts"]) > 0
        and int(metrics["right_contacts"]) > 0
    ):
        return []
    box_rotation = np.asarray(data.xmat[box_body_id], dtype=np.float64).reshape(3, 3)
    activated: list[str] = []
    for side, pad_body_id in pad_body_ids.items():
        equality_name = f"{side}_grasp_weld"
        equality_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
        )
        if equality_id < 0:
            continue
        pad_rotation = np.asarray(data.xmat[pad_body_id], dtype=np.float64).reshape(3, 3)
        relative_position = box_rotation.T @ (
            np.asarray(data.xpos[pad_body_id]) - np.asarray(data.xpos[box_body_id])
        )
        relative_quaternion = _mat_to_quat_wxyz(box_rotation.T @ pad_rotation)
        model.eq_data[equality_id, 0:3] = 0.0
        model.eq_data[equality_id, 3:10] = np.concatenate(
            [relative_position, relative_quaternion]
        )
        data.eq_active[equality_id] = 1
        activated.append(equality_name)
    return activated


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).reshape(3, 3) @ np.asarray(
        second, dtype=np.float64
    ).reshape(3, 3).T
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human ego demo vs robot ego execution, side by side."
    )
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--xml", type=Path, default=PIPER_XML)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--pos-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Track only the explicit fingertip-pad position. Contact replays default "
            "to position priority; pass --no-pos-only for pose diagnostics."
        ),
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=4,
        help="MuJoCo dynamics steps per rendered source frame.",
    )
    parser.add_argument(
        "--grasp-constraint",
        choices=("none", "contact_weld"),
        default="none",
        help="Use pure contacts or activate a weld after both fingertip pads touch.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=("kinematic", "contact"),
        default="kinematic",
        help=(
            "Kinematic collision-checked replay is the default; use contact for "
            "free-body dynamics diagnostics."
        ),
    )
    parser.add_argument(
        "--require-stable-grasp",
        action="store_true",
        help="Exit nonzero if both-sided contact is not maintained for 80%% of frames.",
    )
    parser.add_argument(
        "--require-task-success",
        action="store_true",
        help="Exit nonzero unless grasp stability and object pose tracking both pass.",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    kinematic_replay = args.replay_mode == "kinematic"
    if kinematic_replay and args.grasp_constraint != "none":
        raise SystemExit("--grasp-constraint requires --replay-mode contact")

    import cv2
    import mujoco

    if args.stride <= 0:
        raise SystemExit("--stride must be a positive integer")
    if not args.video.exists():
        raise SystemExit(f"demo video missing: {args.video}")

    out_dir = args.output if args.output.is_absolute() else ROOT / args.output
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    if n_video <= 0:
        raise SystemExit("could not read demo video frames")

    dense = {
        side: _load_dense_ee_trajectory(args.clip, side, n_video)
        for side in ("left", "right")
    }
    stride = max(1, args.stride)
    indices = list(range(0, n_video, stride))
    render_indices = {frame_id: index for index, frame_id in enumerate(indices)}

    world: dict[str, np.ndarray] = {}
    grips: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        world[side], grips[side] = cam_traj_to_base(dense[side], apply_calib=True, side=None)
    object_world = _load_object_world_trajectory(args.clip, n_video)
    if object_world is not None:
        contact_object_world, object_shift = _recenter_object_workspace(
            object_world,
            target_position=CONTACT_BOX_POSITION_M,
        )
    else:
        object_shift = np.zeros(3, dtype=np.float64)
        contact_object_world = None
    pos_only = (
        bool(args.pos_only)
        if args.pos_only is not None
        else contact_object_world is not None
    )

    sim_substeps = max(1, int(args.sim_substeps))
    model = load_model(
        mujoco,
        args.xml,
        bimanual=True,
        ego_camera=True,
        contact_grasp=True,
        contact_weld=args.grasp_constraint == "contact_weld",
        exclude_arm_box_contacts=not kinematic_replay,
    )
    model.opt.timestep = float(1.0 / max(fps * sim_substeps, 1.0))
    physics_steps = sim_substeps
    physics_dt = float(model.opt.timestep)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_head")
    if cam_id < 0:
        raise SystemExit("ego_head camera missing")
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "demo_box")
    box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "demo_box_collision")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if box_id < 0 or box_geom_id < 0 or floor_geom_id < 0:
        raise SystemExit("contact scene is missing demo_box or demo_box_collision")
    pad_body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_grasp_pad")
        for side in ("left", "right")
    }
    pad_geom_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_grasp_pad_geom")
        for side in ("left", "right")
    }
    if any(value < 0 for value in (*pad_body_ids.values(), *pad_geom_ids.values())):
        raise SystemExit("contact scene is missing explicit fingertip pads")
    arm_geom_sides = _arm_geom_sides(mujoco, model)
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)

    q = {
        "left": np.array([0.0, 0.8, -0.9, 0.0, 0.4, 0.0], dtype=np.float64),
        "right": np.array([0.0, 0.8, -0.9, 0.0, 0.4, 0.0], dtype=np.float64),
    }
    previous_q = {side: values.copy() for side, values in q.items()}
    previous_grippers = {side: float(grips[side][0]) for side in ("left", "right")}
    ik_errors: dict[str, list[float]] = {side: [] for side in ("left", "right")}
    joint_steps: dict[str, list[float]] = {side: [] for side in ("left", "right")}
    actuator_ids = {side: _actuator_ids(mujoco, model, f"{side}_") for side in ("left", "right")}
    if contact_object_world is not None:
        initial_object_position = contact_object_world[0, :3, 3]
        initial_object_rotation = contact_object_world[0, :3, :3]
    else:
        initial_object_position = CONTACT_BOX_POSITION_M
        initial_object_rotation = np.eye(3, dtype=np.float64)

    # Establish the nominal reachable pad orientation first.  This is the
    # position-priority contact frame, rather than the often-unreachable raw
    # camera orientation emitted by the retargeter.
    for side in ("left", "right"):
        names = arm_joint_names(f"{side}_")
        for name, value in zip(names[:6], q[side], strict=True):
            data.qpos[_joint_qpos_adr(mujoco, model, name)] = value
        j7, j8 = map_gripper(previous_grippers[side])
        data.qpos[_joint_qpos_adr(mujoco, model, names[6])] = j7
        data.qpos[_joint_qpos_adr(mujoco, model, names[7])] = j8
    mujoco.mj_forward(model, data)

    contact_geometry: dict[str, Any] | None = None
    grasp_separation_m: float | None = None
    axis_world: np.ndarray | None = None
    if contact_object_world is not None:
        axis_world = _grasp_axis_world(world)
        object_support = directional_support_radius(
            CONTACT_BOX_SIZE_M, initial_object_rotation, axis_world
        )
        probe_pad_supports = {
            side: directional_support_radius(
                CONTACT_PAD_SIZE_M,
                np.asarray(data.xmat[pad_body_ids[side]], dtype=np.float64).reshape(3, 3),
                axis_world,
            )
            for side in ("left", "right")
        }
        # The first IK pass is only a reachability probe. Leave a visible gap
        # here, then recompute spacing from the pad orientation IK actually reaches.
        grasp_separation_m = (
            2.0 * object_support
            + sum(probe_pad_supports.values())
            + 2.0 * CONTACT_KINEMATIC_CLEARANCE_M
        )
        world = _constrain_bimanual_grasp(
            world,
            contact_object_world,
            separation=grasp_separation_m,
        )

    _set_free_body_pose(
        mujoco,
        model,
        data,
        "demo_box",
        initial_object_position,
        _mat_to_quat_wxyz(initial_object_rotation),
    )
    mujoco.mj_forward(model, data)
    def solve_initial_pad_targets() -> None:
        """Solve explicit pad targets before starting contact dynamics."""
        ik_data.qpos[:] = data.qpos
        mujoco.mj_forward(model, ik_data)
        for side in ("left", "right"):
            names = arm_joint_names(f"{side}_")
            candidate_q, ik_error = solve_ik(
                mujoco,
                model,
                ik_data,
                pad_body_ids[side],
                world[side][0, :3, 3],
                world[side][0, :3, :3],
                q[side],
                joint_names=names,
                pos_only=True,
            )
            q[side] = candidate_q
            previous_q[side] = candidate_q.copy()
            ik_errors[side].append(float(ik_error))
            for name, value in zip(names[:6], candidate_q, strict=True):
                data.qpos[_joint_qpos_adr(mujoco, model, name)] = value
            j7, j8 = map_gripper(float(grips[side][0]))
            data.qpos[_joint_qpos_adr(mujoco, model, names[6])] = j7
            data.qpos[_joint_qpos_adr(mujoco, model, names[7])] = j8
            _set_arm_controls(data, actuator_ids[side], candidate_q, float(grips[side][0]))
        mujoco.mj_forward(model, data)

    if contact_object_world is not None:
        assert axis_world is not None
        # IK changes the reachable pad orientation. Recompute spacing from the
        # orientation it actually reaches before any physics step is taken.
        for _ in range(3):
            solve_initial_pad_targets()
            contact_geometry = contact_grasp_geometry(
                initial_object_rotation,
                axis_world,
                {
                    side: np.asarray(data.xmat[pad_body_ids[side]], dtype=np.float64).reshape(3, 3)
                    for side in ("left", "right")
                },
            )
            new_separation = float(contact_geometry["separation_m"])
            if grasp_separation_m is not None and abs(new_separation - grasp_separation_m) < 1e-5:
                grasp_separation_m = new_separation
                break
            grasp_separation_m = new_separation
            world = _constrain_bimanual_grasp(
                world,
                contact_object_world,
                separation=grasp_separation_m,
            )
    else:
        solve_initial_pad_targets()
    if kinematic_replay and contact_object_world is not None:
        # ``contact_grasp_geometry`` describes a small physical preload for
        # contact mode. Kinematic replay needs the opposite: a visible,
        # collision-free clearance after the final IK calibration pass.
        grasp_separation_m = float(grasp_separation_m or 0.0) + 2.0 * (
            CONTACT_PAD_PRELOAD_M + CONTACT_KINEMATIC_CLEARANCE_M
        )
        world = _constrain_bimanual_grasp(
            world,
            contact_object_world,
            separation=grasp_separation_m,
        )
        solve_initial_pad_targets()
        if contact_geometry is not None:
            contact_geometry["kinematic_clearance_m"] = CONTACT_KINEMATIC_CLEARANCE_M
            contact_geometry["separation_m"] = grasp_separation_m
            contact_geometry["pad_center_offset_m"] = -CONTACT_KINEMATIC_CLEARANCE_M
            contact_geometry["pad_interference_m"] = {
                side: -CONTACT_KINEMATIC_CLEARANCE_M
                for side in ("left", "right")
            }
    ik_data.qpos[:] = data.qpos
    mujoco.mj_forward(model, ik_data)
    pad_relative_rotations = {
        side: np.asarray(
            contact_object_world[0, :3, :3].T
            @ np.asarray(
                data.xmat[
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"{side}_grasp_pad",
                    )
                ],
                dtype=np.float64,
            ).reshape(3, 3),
            dtype=np.float64,
        )
        for side in ("left", "right")
    } if contact_object_world is not None else {
        side: world[side][0, :3, :3].copy() for side in ("left", "right")
    }
    # A collision-free parked pose is kept as a deterministic retreat when a
    # rotated object makes the retargeted arm branch unreachable.  This keeps
    # the visualization honest instead of leaving a mesh inside the box.
    parked_q = {side: q[side].copy() for side in ("left", "right")}
    last_collision_free_q = {
        side: values.copy() for side, values in parked_q.items()
    }
    active_welds: list[str] = []
    contact_trace: list[dict[str, Any]] = []
    banner = _banner(PANEL_W * 2, HEADER_H)
    video_path = out_dir / "human_vs_robot.mp4"
    out_h, out_w = HEADER_H + PANEL_H, PANEL_W * 2
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (out_w, out_h)
    )
    n_out = 0
    snapshots = {0, len(indices) // 2, len(indices) - 1}
    src_i = -1
    ok, human_raw = True, None

    active_weld_frame: int | None = None
    total_physics_steps = 0
    contact_physics_steps = 0
    both_contact_physics_steps = 0
    max_contact_force_n = 0.0
    max_left_force_n = 0.0
    max_right_force_n = 0.0
    max_penetration_m = 0.0
    floor_contact_physics_steps = 0
    max_floor_force_n = 0.0
    min_object_z = float(np.asarray(data.xpos[box_id])[2])
    for i in range(n_video):
        for side in ("left", "right"):
            previous_grippers[side] = _move_scalar_towards(
                previous_grippers[side],
                float(grips[side][i]),
                MAX_GRIPPER_SPEED_M_S / max(fps, 1.0),
            )

        # IK must see the same gripper state as the command sent to MuJoCo;
        # otherwise the pad body target is solved against stale finger slides.
        ik_data.qpos[:] = data.qpos
        for side in ("left", "right"):
            names = arm_joint_names(f"{side}_")
            for name, value in zip(names[:6], q[side], strict=True):
                ik_data.qpos[_joint_qpos_adr(mujoco, model, name)] = value
            j7, j8 = map_gripper(previous_grippers[side])
            ik_data.qpos[_joint_qpos_adr(mujoco, model, names[6])] = j7
            ik_data.qpos[_joint_qpos_adr(mujoco, model, names[7])] = j8
        mujoco.mj_forward(model, ik_data)

        # Position-only IK can change the pad orientation substantially as the
        # box rotates. In kinematic replay, recompute each pad's support radius
        # after IK and move its target outward until the actual collision shape
        # has the requested clearance from the current box pose.
        frame_targets = {
            side: np.asarray(world[side][i, :3, 3], dtype=np.float64).copy()
            for side in ("left", "right")
        }
        frame_q = {side: q[side].copy() for side in ("left", "right")}
        frame_errors = {side: 0.0 for side in ("left", "right")}
        collision_offsets = {"left": 0.0, "right": 0.0}
        n_target_passes = 10 if kinematic_replay and contact_object_world is not None else 1
        for _ in range(n_target_passes):
            ik_data.qpos[:] = data.qpos
            if kinematic_replay and contact_object_world is not None:
                _set_free_body_pose(
                    mujoco,
                    model,
                    ik_data,
                    "demo_box",
                    contact_object_world[i, :3, 3],
                    _mat_to_quat_wxyz(contact_object_world[i, :3, :3]),
                )
            for side in ("left", "right"):
                _set_arm_qpos(
                    mujoco,
                    model,
                    ik_data,
                    f"{side}_",
                    frame_q[side],
                    previous_grippers[side],
                )
            mujoco.mj_forward(model, ik_data)
            for side in ("left", "right"):
                prefix = f"{side}_"
                names = arm_joint_names(prefix)
                body_id = pad_body_ids[side]
                mat = world[side][i]
                target_pad_rotation = mat[:3, :3]
                if not pos_only and contact_object_world is not None:
                    target_pad_rotation = (
                        contact_object_world[i, :3, :3] @ pad_relative_rotations[side]
                    )
                candidate_q, ik_error = solve_ik(
                    mujoco,
                    model,
                    ik_data,
                    body_id,
                    frame_targets[side],
                    target_pad_rotation,
                    frame_q[side],
                    joint_names=names,
                    pos_only=pos_only,
                )
                frame_errors[side] = float(ik_error)
                if ik_error > MAX_IK_ERROR_M:
                    candidate_q = previous_q[side].copy()
                if not kinematic_replay:
                    candidate_q = _move_vector_towards(
                        previous_q[side],
                        candidate_q,
                        MAX_JOINT_SPEED_RAD_S * physics_dt * sim_substeps,
                    )
                frame_q[side] = candidate_q

            if not (kinematic_replay and contact_object_world is not None):
                break
            mujoco.mj_forward(model, ik_data)
            raw_axis = frame_targets["left"] - frame_targets["right"]
            raw_axis[2] = 0.0
            axis_norm = float(np.linalg.norm(raw_axis))
            if axis_norm < 1e-8:
                break
            axis = raw_axis / axis_norm
            object_rotation = np.asarray(
                contact_object_world[i, :3, :3], dtype=np.float64
            )
            object_center = np.asarray(
                contact_object_world[i, :3, 3], dtype=np.float64
            )
            object_support = directional_support_radius(
                CONTACT_BOX_SIZE_M, object_rotation, axis
            )
            adjusted_targets: dict[str, np.ndarray] = {}
            for side, sign in (("left", 1.0), ("right", -1.0)):
                pad_rotation = np.asarray(
                    ik_data.xmat[pad_body_ids[side]], dtype=np.float64
                ).reshape(3, 3)
                pad_support = directional_support_radius(
                    CONTACT_PAD_SIZE_M, pad_rotation, axis
                )
                adjusted_targets[side] = object_center + sign * (
                    object_support + pad_support + CONTACT_KINEMATIC_CLEARANCE_M
                ) * axis + sign * collision_offsets[side] * axis
            arm_penetrations = {side: 0.0 for side in ("left", "right")}
            if not arm_geom_sides:
                raise SystemExit("contact scene has no arm collision geometries")
            probe = _contact_metrics(
                mujoco,
                model,
                ik_data,
                box_geom_id,
                floor_geom_id,
                pad_geom_ids,
                arm_geom_sides,
            )
            for side in ("left", "right"):
                arm_penetration = float(probe[f"{side}_arm_penetration_m"])
                arm_penetrations[side] = arm_penetration
                if arm_penetration > KINEMATIC_MAX_PENETRATION_M:
                    collision_offsets[side] += (
                        arm_penetration
                        + CONTACT_KINEMATIC_CLEARANCE_M
                    )
                    adjusted_targets[side] += (
                        (1.0 if side == "left" else -1.0)
                        * (arm_penetration + CONTACT_KINEMATIC_CLEARANCE_M)
                        * axis
                    )
            target_delta = max(
                float(np.linalg.norm(adjusted_targets[side] - frame_targets[side]))
                for side in ("left", "right")
            )
            frame_targets = adjusted_targets
            if (
                target_delta < 1e-5
                and max(arm_penetrations.values()) <= KINEMATIC_MAX_PENETRATION_M
            ):
                break
            for side in ("left", "right"):
                frame_q[side] = frame_q[side].copy()

        for side in ("left", "right"):
            q[side] = frame_q[side]
            ik_errors[side].append(frame_errors[side])
            joint_steps[side].append(float(np.linalg.norm(q[side] - previous_q[side])))
            _set_arm_controls(data, actuator_ids[side], q[side], previous_grippers[side])
            previous_q[side] = q[side].copy()
        executed_pad_targets = frame_targets
        frame_contact_samples: list[dict[str, Any]] = []
        if kinematic_replay and contact_object_world is not None:
            # In the documented replay mode the object follows its supplied
            # trajectory and the arms are applied as poses. This is a
            # collision-checked visualization, not a claim of physical grasp.
            _set_free_body_pose(
                mujoco,
                model,
                data,
                "demo_box",
                contact_object_world[i, :3, 3],
                _mat_to_quat_wxyz(contact_object_world[i, :3, :3]),
            )
            for side in ("left", "right"):
                _set_arm_qpos(
                    mujoco,
                    model,
                    data,
                    f"{side}_",
                    q[side],
                    previous_grippers[side],
                )
            mujoco.mj_forward(model, data)
            sample = _contact_metrics(
                mujoco,
                model,
                data,
                box_geom_id,
                floor_geom_id,
                pad_geom_ids,
                arm_geom_sides,
            )
            if int(sample["arm_contacts"]) > 0 and float(
                sample["arm_max_penetration_m"]
            ) > KINEMATIC_MAX_PENETRATION_M:
                # Prefer a nearby retreat, then the known parked pose.  The
                # interpolation is checked against the same mesh contacts as
                # the final frame, so this cannot merely hide a bad metric.
                for side in ("left", "right"):
                    if float(sample[f"{side}_arm_penetration_m"]) <= KINEMATIC_MAX_PENETRATION_M:
                        continue
                    original_q = q[side].copy()
                    best_q = original_q.copy()
                    best_sample = sample
                    best_penetration = float(sample[f"{side}_arm_penetration_m"])
                    references = (last_collision_free_q[side], parked_q[side])
                    for reference in references:
                        for alpha in np.linspace(0.1, 1.0, 10):
                            candidate_q = (
                                (1.0 - alpha) * original_q + alpha * reference
                            )
                            _set_arm_qpos(
                                mujoco,
                                model,
                                data,
                                f"{side}_",
                                candidate_q,
                                previous_grippers[side],
                            )
                            mujoco.mj_forward(model, data)
                            candidate_sample = _contact_metrics(
                                mujoco,
                                model,
                                data,
                                box_geom_id,
                                floor_geom_id,
                                pad_geom_ids,
                                arm_geom_sides,
                            )
                            candidate_penetration = float(
                                candidate_sample[f"{side}_arm_penetration_m"]
                            )
                            if candidate_penetration < best_penetration:
                                best_penetration = candidate_penetration
                                best_q = candidate_q.copy()
                                best_sample = candidate_sample
                            if candidate_penetration <= KINEMATIC_MAX_PENETRATION_M:
                                break
                        if best_penetration <= KINEMATIC_MAX_PENETRATION_M:
                            break
                    q[side] = best_q
                    _set_arm_qpos(
                        mujoco,
                        model,
                        data,
                        f"{side}_",
                        q[side],
                        previous_grippers[side],
                    )
                    sample = best_sample
                    if float(sample["arm_max_penetration_m"]) <= KINEMATIC_MAX_PENETRATION_M:
                        break
                mujoco.mj_forward(model, data)
                sample = _contact_metrics(
                    mujoco,
                    model,
                    data,
                    box_geom_id,
                    floor_geom_id,
                    pad_geom_ids,
                    arm_geom_sides,
                )
            if float(sample["arm_max_penetration_m"]) <= KINEMATIC_MAX_PENETRATION_M:
                last_collision_free_q = {
                    side: q[side].copy() for side in ("left", "right")
                }
            frame_contact_samples = [sample]
            total_physics_steps += 1
            if int(sample["total_contacts"]) > 0:
                contact_physics_steps += 1
            if int(sample["left_contacts"]) > 0 and int(sample["right_contacts"]) > 0:
                both_contact_physics_steps += 1
            max_contact_force_n = max(max_contact_force_n, float(sample["max_force_n"]))
            max_left_force_n = max(max_left_force_n, float(sample["left_force_n"]))
            max_right_force_n = max(max_right_force_n, float(sample["right_force_n"]))
            max_penetration_m = max(max_penetration_m, float(sample["max_penetration_m"]))
            if int(sample["floor_contacts"]) > 0:
                floor_contact_physics_steps += 1
            max_floor_force_n = max(max_floor_force_n, float(sample["floor_force_n"]))
            min_object_z = min(min_object_z, float(np.asarray(data.xpos[box_id])[2]))
        else:
            for _ in range(physics_steps):
                if args.grasp_constraint == "contact_weld" and not active_welds:
                    newly_active = _activate_contact_welds(
                        mujoco,
                        model,
                        data,
                        box_id,
                        pad_body_ids,
                        _contact_metrics(
                            mujoco,
                            model,
                            data,
                            box_geom_id,
                            floor_geom_id,
                            pad_geom_ids,
                            arm_geom_sides,
                        ),
                    )
                    if newly_active:
                        active_welds = newly_active
                        active_weld_frame = int(i)
                mujoco.mj_step(model, data)
                sample = _contact_metrics(
                    mujoco,
                    model,
                    data,
                    box_geom_id,
                    floor_geom_id,
                    pad_geom_ids,
                    arm_geom_sides,
                )
                frame_contact_samples.append(sample)
                total_physics_steps += 1
                if int(sample["total_contacts"]) > 0:
                    contact_physics_steps += 1
                if (
                    int(sample["left_contacts"]) > 0
                    and int(sample["right_contacts"]) > 0
                ):
                    both_contact_physics_steps += 1
                max_contact_force_n = max(max_contact_force_n, float(sample["max_force_n"]))
                max_left_force_n = max(max_left_force_n, float(sample["left_force_n"]))
                max_right_force_n = max(max_right_force_n, float(sample["right_force_n"]))
                max_penetration_m = max(
                    max_penetration_m, float(sample["max_penetration_m"])
                )
                if int(sample["floor_contacts"]) > 0:
                    floor_contact_physics_steps += 1
                max_floor_force_n = max(max_floor_force_n, float(sample["floor_force_n"]))
                min_object_z = min(min_object_z, float(np.asarray(data.xpos[box_id])[2]))
        contact = frame_contact_samples[-1] if frame_contact_samples else _contact_metrics(
            mujoco,
            model,
            data,
            box_geom_id,
            floor_geom_id,
            pad_geom_ids,
            arm_geom_sides,
        )
        contact = dict(contact)
        contact["physics_steps"] = len(frame_contact_samples)
        contact["contact_physics_steps"] = sum(
            int(int(sample["total_contacts"]) > 0) for sample in frame_contact_samples
        )
        contact["both_contact_physics_steps"] = sum(
            int(
                int(sample["left_contacts"]) > 0
                and int(sample["right_contacts"]) > 0
            )
            for sample in frame_contact_samples
        )
        contact["max_force_n"] = max(
            (float(sample["max_force_n"]) for sample in frame_contact_samples),
            default=0.0,
        )
        contact["left_force_n"] = max(
            (float(sample["left_force_n"]) for sample in frame_contact_samples),
            default=0.0,
        )
        contact["right_force_n"] = max(
            (float(sample["right_force_n"]) for sample in frame_contact_samples),
            default=0.0,
        )
        contact["left_penetration_m"] = max(
            (float(sample["left_penetration_m"]) for sample in frame_contact_samples),
            default=0.0,
        )
        contact["right_penetration_m"] = max(
            (float(sample["right_penetration_m"]) for sample in frame_contact_samples),
            default=0.0,
        )
        contact["max_penetration_m"] = max(
            (float(sample["max_penetration_m"]) for sample in frame_contact_samples),
            default=0.0,
        )
        contact["floor_contact_physics_steps"] = sum(
            int(int(sample["floor_contacts"]) > 0) for sample in frame_contact_samples
        )
        contact["floor_force_n"] = max(
            (float(sample["floor_force_n"]) for sample in frame_contact_samples),
            default=0.0,
        )
        if i not in render_indices:
            continue
        while src_i < i and ok:
            ok, human_raw = cap.read()
            src_i += 1
        if not ok or human_raw is None:
            break
        k = render_indices[i]
        object_position = np.asarray(data.xpos[box_id], dtype=np.float64).copy()
        object_rotation = np.asarray(data.xmat[box_id], dtype=np.float64).reshape(3, 3).copy()
        target_position = (
            contact_object_world[i, :3, 3] if contact_object_world is not None else None
        )
        target_rotation = (
            contact_object_world[i, :3, :3]
            if contact_object_world is not None
            else None
        )
        contact_trace.append(
            {
                "frame_id": int(i),
                **contact,
                "pad_positions_m": {
                    side: np.asarray(data.xpos[pad_body_ids[side]], dtype=np.float64).tolist()
                    for side in ("left", "right")
                },
                "pad_target_positions_m": {
                    side: np.asarray(executed_pad_targets[side], dtype=np.float64).tolist()
                    for side in ("left", "right")
                },
                "arm_q": {side: q[side].tolist() for side in ("left", "right")},
                "object_position_m": object_position.tolist(),
                "object_target_error_m": (
                    float(np.linalg.norm(object_position - target_position))
                    if target_position is not None
                    else None
                ),
                "object_orientation_error_rad": (
                    _rotation_angle(object_rotation, target_rotation)
                    if target_rotation is not None
                    else None
                ),
            }
        )
        look = object_position
        renderer.update_scene(data, camera=_ego_mjv_camera(mujoco, look))
        robot = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(
            robot,
            f"contacts L={int(contact['left_contacts'])} R={int(contact['right_contacts'])} "
            f"F={float(contact['max_force_n']):.1f}N "
            f"pen={float(contact['max_penetration_m']) * 1000.0:.1f}mm "
            f"z={object_position[2]:.3f}m",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        human = _fit(human_raw, PANEL_W, PANEL_H)
        frame = np.concatenate([banner, np.concatenate([human, robot], axis=1)], axis=0)
        writer.write(frame)
        n_out += 1
        if k in snapshots or k % 60 == 0:
            cv2.imwrite(str(frames_dir / f"compare_{i:04d}.png"), frame)

    cap.release()
    writer.release()

    both_contact_frames = sum(
        int(
            int(row["left_contacts"]) > 0
            and int(row["right_contacts"]) > 0
        )
        for row in contact_trace
    )
    contact_frames = sum(int(int(row["total_contacts"]) > 0) for row in contact_trace)
    object_errors = [
        float(row["object_target_error_m"])
        for row in contact_trace
        if row["object_target_error_m"] is not None
    ]
    orientation_errors = [
        float(row["object_orientation_error_rad"])
        for row in contact_trace
        if row["object_orientation_error_rad"] is not None
    ]
    both_contact_ratio = both_contact_physics_steps / max(total_physics_steps, 1)
    force_limit_ok = bool(
        max_left_force_n <= CONTACT_MAX_SIDE_FORCE_N
        and max_right_force_n <= CONTACT_MAX_SIDE_FORCE_N
    )
    stable_grasp = _is_stable_grasp(
        physics_steps=total_physics_steps,
        both_contact_ratio=both_contact_ratio,
        min_object_z_m=min_object_z,
        max_side_force_n=max(max_left_force_n, max_right_force_n),
        floor_contact_physics_steps=floor_contact_physics_steps,
        max_penetration_m=max_penetration_m,
    )
    trajectory_tracking_success = bool(
        contact_object_world is not None
        and object_errors
        and orientation_errors
        and float(np.mean(object_errors)) <= OBJECT_POSITION_SUCCESS_MEAN_M
        and float(np.max(object_errors)) <= OBJECT_POSITION_SUCCESS_MAX_M
        and float(np.mean(orientation_errors)) <= OBJECT_ORIENTATION_SUCCESS_MEAN_RAD
        and float(np.max(orientation_errors)) <= OBJECT_ORIENTATION_SUCCESS_MAX_RAD
    )
    collision_check_success = bool(
        kinematic_replay and max_penetration_m <= KINEMATIC_MAX_PENETRATION_M
    )
    task_success = bool(
        trajectory_tracking_success
        and (collision_check_success if kinematic_replay else stable_grasp)
    )
    failure_reasons: list[str] = []
    if not stable_grasp and not kinematic_replay:
        failure_reasons.append(
            "contact grasp was not stable: "
            f"both_contact_ratio={both_contact_ratio:.3f}, min_object_z={min_object_z:.3f}, "
            f"left_force_n={max_left_force_n:.2f}, right_force_n={max_right_force_n:.2f}, "
            f"force_limit_n={CONTACT_MAX_SIDE_FORCE_N:.2f}, "
            f"floor_contact_steps={floor_contact_physics_steps}, "
            f"max_penetration_m={max_penetration_m:.6f}, "
            f"penetration_limit_m={CONTACT_MAX_PENETRATION_M:.6f}"
        )
    if kinematic_replay and not collision_check_success:
        failure_reasons.append(
            "kinematic collision check failed: "
            f"max_penetration_m={max_penetration_m:.6f}, "
            f"penetration_limit_m={KINEMATIC_MAX_PENETRATION_M:.6f}"
        )
    if contact_object_world is None:
        failure_reasons.append(
            "object trajectory unavailable; task pose tracking was not evaluated"
        )
    elif not trajectory_tracking_success:
        failure_reasons.append(
            "task pose tracking failed: "
            f"position_mean={float(np.mean(object_errors)) if object_errors else None}, "
            f"orientation_mean={float(np.mean(orientation_errors)) if orientation_errors else None}"
        )
    required_failures: list[str] = []
    if args.require_stable_grasp and not stable_grasp:
        required_failures.append(
            failure_reasons[0]
            if failure_reasons
            else "stable_grasp is only evaluated in --replay-mode contact"
        )
    if args.require_task_success and not task_success:
        required_failures.extend(failure_reasons)

    summary = {
        "human_video": str(args.video),
        "clip": str(args.clip),
        "n_source_frames": n_video,
        "n_out_frames": n_out,
        "fps": fps / stride,
        "camera": "ego_head",
        "object_trajectory": object_world is not None,
        "object_shift_m": object_shift.tolist(),
        "contact_object_offset_m": CONTACT_OBJECT_OFFSET_M.tolist(),
        "grasp_separation_m": grasp_separation_m,
        "grasp_geometry": contact_geometry,
        "contact_ik": "position" if pos_only else "pose",
        "arm_trajectory": (
            "interpolated_and_object_constrained"
            if object_world is not None
            else "interpolated_camera_trajectory"
        ),
        "simulation_mode": (
            "kinematic_collision_checked_replay"
            if kinematic_replay
            else "contact_dynamics_replay"
        ),
        "object_dynamics": (
            "trajectory_driven_collision_check"
            if kinematic_replay
            else "free_body_with_gravity_and_collision"
        ),
        "replay_mode": args.replay_mode,
        "grasp_constraint": args.grasp_constraint,
        "active_grasp_constraints": active_welds,
        "active_weld_frame": active_weld_frame,
        "constraint_assisted": bool(active_welds),
        "sim_substeps": sim_substeps,
        "physics_steps_per_render": 1 if kinematic_replay else physics_steps,
        "source_frames_per_render": stride,
        "contact_metrics": {
            "contact_frames": contact_frames,
            "both_contact_frames": both_contact_frames,
            "physics_steps": total_physics_steps,
            "contact_physics_steps": contact_physics_steps,
            "both_contact_physics_steps": both_contact_physics_steps,
            "both_contact_ratio": both_contact_ratio,
            "max_force_n": max_contact_force_n,
            "left_force_n_max": max_left_force_n,
            "right_force_n_max": max_right_force_n,
            "side_force_limit_n": CONTACT_MAX_SIDE_FORCE_N,
            "force_limit_ok": force_limit_ok,
        "max_penetration_m": max_penetration_m,
        "penetration_limit_m": (
            KINEMATIC_MAX_PENETRATION_M
            if kinematic_replay
            else CONTACT_MAX_PENETRATION_M
        ),
        "penetration_ok": max_penetration_m <= (
            KINEMATIC_MAX_PENETRATION_M
            if kinematic_replay
            else CONTACT_MAX_PENETRATION_M
        ),
            "floor_contact_physics_steps": floor_contact_physics_steps,
            "max_floor_force_n": max_floor_force_n,
        },
        "object_actual_min_z_m": min_object_z,
        "object_target_error_m": {
            "mean": float(np.mean(object_errors)) if object_errors else None,
            "max": float(np.max(object_errors)) if object_errors else None,
        },
        "object_orientation_error_rad": {
            "mean": float(np.mean(orientation_errors)) if orientation_errors else None,
            "max": float(np.max(orientation_errors)) if orientation_errors else None,
        },
        "stable_grasp": stable_grasp,
        "collision_check_success": collision_check_success,
        "trajectory_tracking_success": trajectory_tracking_success,
        "task_success": task_success,
        "failure_reasons": failure_reasons,
        "max_ik_error_m": {
            side: max(ik_errors[side], default=0.0) for side in ("left", "right")
        },
        "max_joint_step_rad": {
            side: max(joint_steps[side], default=0.0) for side in ("left", "right")
        },
        "video": str(video_path),
        "contact_trace": str(out_dir / "contact_trace.json"),
        "note": (
            "right panel is a collision-checked kinematic replay; object pose follows "
            "the supplied trajectory and stable_grasp is not claimed"
            if kinematic_replay
            else (
                "right panel is a MuJoCo contact-dynamics replay, not a trained policy rollout; "
                "contact_weld is activated only after both named fingertip pads touch"
                if args.grasp_constraint == "contact_weld"
                else "right panel is pure MuJoCo contact-dynamics replay without a grasp weld"
            )
        ),
    }
    (out_dir / "contact_trace.json").write_text(
        json.dumps(contact_trace, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if required_failures:
        raise SystemExit("; ".join(dict.fromkeys(required_failures)))


if __name__ == "__main__":
    main()
