"""Kinematic playback of ego2dex virtual-gripper T_ee+g on Piper MuJoCo.

Camera-frame EE poses are left-multiplied by the hacky T_base←cam, then a
damped Jacobian IK (arm joints only) tracks link6. Gripper slides are
joint7 = g/2, joint8 = -g/2. No Pinocchio, no ROS.
"""

# ruff: noqa: E402
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

from modules.umi_pose.hacky_calib import T_base_from_cam

PIPER_XML = Path(
    r"C:\Users\Administrator\Desktop\robotic-data\models\Piper"
    r"\src\piper_description\mujoco_model\piper_description.xml"
)
DEFAULT_CLIP = Path(
    r"C:\Users\Administrator\Desktop\robotic-data\video_pipeline\outputs"
    r"\video_20260914_210702._ego2dex_retarget\clip_retargeted.json"
)


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotvec_from_mats(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    r_err = r_tgt @ r_cur.T
    angle = np.arccos(np.clip((np.trace(r_err) - 1.0) * 0.5, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.zeros(3)
    return axis / n * angle


ARM_Y = {"left": 0.22, "right": -0.22}

# The vendor Piper model contains visual mesh collision geometry, but it has no
# named fingertip contact surface suitable for a small tabletop object.  These
# pads are deliberately small boxes attached to the parallel gripper fingers;
# they are only used by the contact-replay scene below, not by the normal
# kinematic playback path.
CONTACT_BOX_SIZE_M = (0.090, 0.060, 0.100)
CONTACT_BOX_MASS_KG = 0.05
CONTACT_PAD_SIZE_M = (0.025, 0.006, 0.035)
CONTACT_PAD_LOCAL_POSE = {
    "left": (
        "link7",
        "0.000318 -0.046055 -0.000398",
        "0.300638 -0.379425 -0.560266 0.672127",
    ),
    "right": (
        "link8",
        "-0.000318 -0.046055 -0.000397",
        "0.672123 0.560270 0.379425 0.300641",
    ),
}


def directional_support_radius(
    half_sizes: tuple[float, float, float] | np.ndarray,
    rotation: np.ndarray,
    axis_world: np.ndarray,
) -> float:
    """Return an oriented box's support radius along a world-space axis.

    ``half_sizes`` follows MuJoCo's box ``size`` convention.  The projection is
    evaluated in the box frame, so this remains valid when the object rotates
    during a contact replay.
    """
    sizes = np.asarray(half_sizes, dtype=np.float64).reshape(3)
    rotation_matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    axis = np.asarray(axis_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("axis_world must be non-zero")
    local_axis = rotation_matrix.T @ (axis / norm)
    return float(np.dot(np.abs(local_axis), sizes))


def load_ee_item(clip: Path, hand_side: str) -> dict[str, Any]:
    payload = json.loads(clip.read_text(encoding="utf-8"))
    retargeting = payload.get("retargeting") or []
    if not retargeting:
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
                    if cand_payload.get("retargeting"):
                        payload = cand_payload
                        retargeting = cand_payload.get("retargeting") or []
                        break
                except Exception:
                    pass
    hits = [
        item
        for item in retargeting
        if item.get("hand_side") == hand_side
    ]
    if not hits:
        sides = [item.get("hand_side") for item in retargeting]
        raise SystemExit(f"no retargeting for side={hand_side!r}; have {sides}")
    item = hits[0]
    traj = np.asarray(item["joint_trajectory"], dtype=np.float64)
    if traj.ndim != 2 or traj.shape[-1] != 8:
        raise SystemExit(f"expected (T,8) T_ee+g, got {traj.shape}")
    frame_ids = np.asarray(item.get("frame_ids", np.arange(len(traj))), dtype=np.int64)
    return {"traj": traj, "frame_ids": frame_ids, "hand_side": hand_side}


def load_ee_traj(clip: Path, hand_side: str) -> np.ndarray:
    return load_ee_item(clip, hand_side)["traj"]


def align_bimanual(clip: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Union of frame_ids; hold last pose when a hand is missing that frame."""
    items = {side: load_ee_item(clip, side) for side in ("left", "right")}
    by_id = {
        side: {int(fid): row for fid, row in zip(item["frame_ids"], item["traj"], strict=True)}
        for side, item in items.items()
    }
    all_ids = np.array(sorted(set(by_id["left"]) | set(by_id["right"])), dtype=np.int64)
    mats: dict[str, np.ndarray] = {}
    grips: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        last = None
        rows = np.zeros((len(all_ids), 8), dtype=np.float64)
        seen = np.zeros(len(all_ids), dtype=bool)
        for i, fid in enumerate(all_ids):
            if fid in by_id[side]:
                last = by_id[side][fid]
                seen[i] = True
            if last is None:
                continue
            rows[i] = last
            seen[i] = seen[i] or True
        # frames before this hand appears stay zero + unseen
        first = int(np.argmax(seen)) if seen.any() else 0
        if seen.any():
            rows[:first] = rows[first]
        mats[side] = rows
        grips[side] = rows[:, 7].copy()
        present[side] = seen
    return all_ids, mats, grips


def cam_traj_to_base(
    traj: np.ndarray,
    apply_calib: bool,
    *,
    side: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tx = T_base_from_cam() if apply_calib else np.eye(4)
    mats = np.zeros((len(traj), 4, 4), dtype=np.float64)
    offset = np.array([0.0, ARM_Y.get(side, 0.0), 0.0]) if side else np.zeros(3)
    for i, row in enumerate(traj):
        m = np.eye(4)
        m[:3, :3] = _quat_wxyz_to_mat(row[3:7])
        m[:3, 3] = row[:3]
        world = tx @ m
        world[:3, 3] = world[:3, 3] + offset
        mats[i] = world
    return mats, traj[:, 7].copy()


def _prefix_arm_chunk(chunk: str, prefix: str) -> str:
    for i in range(8, 0, -1):
        chunk = chunk.replace(f'name="joint{i}"', f'name="{prefix}joint{i}"')
        chunk = chunk.replace(f'joint="joint{i}"', f'joint="{prefix}joint{i}"')
        chunk = chunk.replace(f'name="link{i}"', f'name="{prefix}link{i}"')
    return chunk


def _contact_box_xml() -> str:
    sx, sy, sz = CONTACT_BOX_SIZE_M
    return f"""
        <body name="demo_box" pos="0.55 0 0.135">
            <freejoint name="demo_box_free"/>
            <geom name="demo_box_collision" type="box" size="{sx} {sy} {sz}"
                  mass="{CONTACT_BOX_MASS_KG}" rgba="0.82 0.72 0.58 1"
                  friction="5 1 0.2" contype="2" conaffinity="5"
                  solref="0.002 1" solimp="0.97 0.995 0.001"/>
        </body>
    """


def _contact_pad_xml(side: str) -> str:
    parent, position, quaternion = CONTACT_PAD_LOCAL_POSE[side]
    sx, sy, sz = CONTACT_PAD_SIZE_M
    return f"""
        <body name="{side}_grasp_pad" pos="{position}" quat="{quaternion}">
            <geom name="{side}_grasp_pad_geom" type="box" size="{sx} {sy} {sz}"
                  mass="0.002" rgba="0.18 0.78 0.30 1" friction="5 1 0.2"
                  contype="4" conaffinity="2" solref="0.002 1" solimp="0.97 0.995 0.001"/>
        </body>
    """


def _insert_child_body(chunk: str, parent_name: str, child: str) -> str:
    start = chunk.find(f'<body name="{parent_name}"')
    if start < 0:
        raise SystemExit(f"piper xml missing body {parent_name!r}")
    end = chunk.find("</body>", start)
    if end < 0:
        raise SystemExit(f"piper xml body {parent_name!r} is not closed")
    return chunk[:end] + child + chunk[end:]


def _contact_exclusions() -> str:
    excluded_links = (
        "base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link8",
    )
    entries = "\n".join(
        f'            <exclude body1="demo_box" body2="{side}_{link}"/>'
        for side in ("left", "right")
        for link in excluded_links
    )
    return f"\n    <contact>\n{entries}\n    </contact>"


def _contact_welds() -> str:
    return """
    <equality>
        <weld name="left_grasp_weld" body1="demo_box" body2="left_grasp_pad"
              active="false" solref="0.002 1" solimp="0.99 0.999 0.001"/>
        <weld name="right_grasp_weld" body1="demo_box" body2="right_grasp_pad"
              active="false" solref="0.002 1" solimp="0.99 0.999 0.001"/>
    </equality>
    """


def scene_xml(
    src: Path,
    *,
    bimanual: bool,
    ego_camera: bool = False,
    contact_grasp: bool = False,
    contact_weld: bool = False,
    exclude_arm_box_contacts: bool = True,
) -> str:
    if contact_grasp and not bimanual:
        raise ValueError("contact_grasp requires a bimanual scene")
    if contact_weld and not contact_grasp:
        raise ValueError("contact_weld requires contact_grasp=True")
    text = src.read_text(encoding="utf-8")
    start = text.find("<worldbody>")
    end = text.find("</worldbody>")
    act_start = text.find("<actuator>")
    act_end = text.find("</actuator>")
    if min(start, end, act_start, act_end) < 0:
        raise SystemExit("piper xml missing worldbody/actuator")
    inner = text[start + len("<worldbody>") : end]
    actuators = text[act_start + len("<actuator>") : act_end]
    lights = """
        <light pos="0.4 0.4 1.4" dir="-0.15 -0.15 -1" diffuse="0.9 0.9 0.9"/>
        <light pos="0.1 0 0.9" dir="0.25 0 -1" diffuse="0.55 0.55 0.55"/>
        <geom name="floor" type="plane" size="1.2 1.2 0.05"
              rgba="0.55 0.54 0.52 1" contype="1" conaffinity="2"/>
"""
    if ego_camera:
        # Head-mounted: between the bases, look +X and down at both grippers.
        # xyaxes x=(0,-1,0)=world -Y as camera-right so left arm stays on image-left.
        lights = """
        <light pos="0.25 0.2 1.1" dir="0 -0.1 -1" diffuse="0.85 0.85 0.85"/>
        <light pos="0.35 -0.25 0.9" dir="-0.2 0.15 -1" diffuse="0.45 0.45 0.45"/>
        <geom name="floor" type="plane" size="1.6 1.6 0.05" material="tile"
              contype="1" conaffinity="2"/>
        <camera name="ego_head" pos="-0.04 0 0.62" xyaxes="0 -1 0 0.74 0 0.67" fovy="72"/>
        {box}
"""
        box = _contact_box_xml() if contact_grasp else """
        <body name="demo_box" mocap="true" pos="0.28 0 0.22">
            <geom type="box" size="0.032 0.090 0.028" rgba="0.82 0.72 0.58 1"
                  contype="0" conaffinity="0"/>
        </body>
"""
        # Substitute after selecting the scene object to keep the non-contact
        # replay XML byte-for-byte equivalent apart from formatting.
        lights = lights.replace("{box}", box)
        ego = ""
    else:
        if contact_grasp:
            lights += _contact_box_xml()
        ego = ""
    if not bimanual:
        world = lights + ego + inner + """
        <body name="ee_target" mocap="true" pos="0.25 0 0.25">
            <geom type="sphere" size="0.012" rgba="1 0.25 0.2 0.85" contype="0" conaffinity="0"/>
        </body>
"""
        act = actuators
        center = "0.2 0 0.2"
        extent = "0.7"
    else:
        world = lights + ego
        act_parts = []
        for side, rgba in (("left", "0.35 0.55 0.95 0.9"), ("right", "0.95 0.40 0.28 0.9")):
            y = ARM_Y[side]
            chunk = _prefix_arm_chunk(inner, f"{side}_")
            if contact_grasp and bimanual:
                parent, _, _ = CONTACT_PAD_LOCAL_POSE[side]
                chunk = _insert_child_body(chunk, f"{side}_{parent}", _contact_pad_xml(side))
            act_parts.append(_prefix_arm_chunk(actuators, f"{side}_"))
            target = "" if ego_camera else f"""
        <body name="{side}_ee_target" mocap="true" pos="0.25 {y} 0.25">
            <geom type="sphere" size="0.012" rgba="{rgba}" contype="0" conaffinity="0"/>
        </body>"""
            world += f"""
        <body name="{side}_base" pos="0 {y} 0">
            {chunk}
        </body>
        {target}
"""
        act = "\n".join(act_parts)
        center = "0.22 0 0.2"
        extent = "0.95"
    out = (
        text[:start]
        + "<worldbody>\n"
        + world
        + "\n    </worldbody>"
        + text[end + len("</worldbody>") :]
    )
    act_start = out.find("<actuator>")
    act_end = out.find("</actuator>")
    out = (
        out[:act_start]
        + "<actuator>\n"
        + act
        + "\n    </actuator>"
        + out[act_end + len("</actuator>") :]
    )
    if contact_grasp and exclude_arm_box_contacts:
        out = out.replace("</actuator>", "</actuator>" + _contact_exclusions(), 1)
    if contact_weld:
        out = out.replace("</actuator>", "</actuator>" + _contact_welds(), 1)
    extra = ""
    if "<statistic" not in out:
        extra += f'\n    <statistic center="{center}" extent="{extent}"/>'
    if "<visual" not in out:
        extra += """
    <visual>
      <global offwidth="1920" offheight="1080"/>
      <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    </visual>"""
    if extra:
        out = out.replace("<mujoco model=\"piper\">", f'<mujoco model="piper">{extra}', 1)
    if ego_camera and 'name="tile"' not in out:
        tile = """
        <texture name="tile" type="2d" builtin="checker" rgb1="0.62 0.61 0.59" rgb2="0.50 0.49 0.47"
                 width="256" height="256"/>
        <material name="tile" texture="tile" texrepeat="8 8" reflectance="0.04"/>
"""
        out = out.replace("<asset>", "<asset>" + tile, 1)
    return out


def load_model(
    mujoco: Any,
    xml_path: Path,
    *,
    bimanual: bool,
    ego_camera: bool = False,
    contact_grasp: bool = False,
    contact_weld: bool = False,
    exclude_arm_box_contacts: bool = True,
) -> Any:
    xml = scene_xml(
        xml_path,
        bimanual=bimanual,
        ego_camera=ego_camera,
        contact_grasp=contact_grasp,
        contact_weld=contact_weld,
        exclude_arm_box_contacts=exclude_arm_box_contacts,
    )
    assets: dict[str, bytes] = {}
    mesh_dir = (xml_path.parent / ".." / "meshes").resolve()
    if mesh_dir.exists():
        by_lower: dict[str, Path] = {}
        for file_path in mesh_dir.iterdir():
            if file_path.is_file():
                by_lower[file_path.name.lower()] = file_path
        for file_path in by_lower.values():
            data = file_path.read_bytes()
            assets[f"../meshes/{file_path.name}"] = data
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def _joint_qpos_adr(mujoco: Any, model: Any, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise SystemExit(f"joint {name!r} missing")
    return int(model.jnt_qposadr[jid])


def _joint_dof_adr(mujoco: Any, model: Any, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_dofadr[jid])


def arm_joint_names(prefix: str) -> list[str]:
    return [f"{prefix}joint{i}" for i in range(1, 9)]


def clip_named_q(mujoco: Any, model: Any, names: list[str], q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i, name in enumerate(names[: len(q)]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def solve_ik(
    mujoco: Any,
    model: Any,
    data: Any,
    body_id: int,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_seed: np.ndarray,
    *,
    joint_names: list[str],
    max_iter: int = 60,
    damp: float = 1e-3,
    pos_only: bool = False,
) -> tuple[np.ndarray, float]:
    arm_names = joint_names[:6]
    q = clip_named_q(mujoco, model, arm_names, q_seed)
    qpos_adr = [_joint_qpos_adr(mujoco, model, n) for n in arm_names]
    dof_adr = [_joint_dof_adr(mujoco, model, n) for n in arm_names]
    for adr, val in zip(qpos_adr, q, strict=True):
        data.qpos[adr] = val
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err_norm = 0.0
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        err_pos = target_pos - np.asarray(data.xpos[body_id])
        if pos_only:
            err = err_pos
        else:
            err_rot = _rotvec_from_mats(data.xmat[body_id].reshape(3, 3), target_rot)
            err = np.concatenate([err_pos, 0.4 * err_rot])
        err_norm = float(np.linalg.norm(err_pos))
        if err_norm < 2e-3 and (pos_only or float(np.linalg.norm(err[3:])) < 0.05):
            break
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        jac_full = jacp if pos_only else np.vstack([jacp, jacr])
        jac = jac_full[:, dof_adr]
        jjt = jac @ jac.T + damp * np.eye(jac.shape[0])
        dq = jac.T @ np.linalg.solve(jjt, err)
        q = clip_named_q(mujoco, model, arm_names, q + dq)
        for adr, val in zip(qpos_adr, q, strict=True):
            data.qpos[adr] = val
    mujoco.mj_forward(model, data)
    err_norm = float(np.linalg.norm(target_pos - np.asarray(data.xpos[body_id])))
    return q.copy(), err_norm


def map_gripper(g: float) -> tuple[float, float]:
    half = float(np.clip(g, 0.0, 0.07) * 0.5)
    half = float(np.clip(half, 0.0, 0.035))
    return half, -half


def _write_video(path: Path, images: list[np.ndarray], width: int, height: int) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    for img in images:
        writer.write(img)
    writer.release()


def replay_one_arm(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    prefix: str,
    mats: np.ndarray,
    gripper: np.ndarray,
    pos_only: bool,
    cam: Any,
    renderer: Any,
    label_side: str,
    frames_dir: Path,
) -> tuple[list[np.ndarray], list[float]]:
    import cv2

    names = arm_joint_names(prefix)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}link6")
    target_name = "ee_target" if prefix == "" else f"{label_side}_ee_target"
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
    mocap_bid = int(model.body_mocapid[mocap_id]) if mocap_id >= 0 else -1
    q = np.array([0.0, 0.8, -0.9, 0.0, 0.4, 0.0], dtype=np.float64)
    images: list[np.ndarray] = []
    err_log: list[float] = []
    for i, (mat, g) in enumerate(zip(mats, gripper, strict=True)):
        if mocap_bid >= 0:
            data.mocap_pos[mocap_bid] = mat[:3, 3]
        q, err = solve_ik(
            mujoco, model, data, body_id, mat[:3, 3], mat[:3, :3], q,
            joint_names=names, pos_only=pos_only,
        )
        j7, j8 = map_gripper(float(g))
        for name, val in zip(names[:6], q, strict=True):
            data.qpos[_joint_qpos_adr(mujoco, model, name)] = val
        data.qpos[_joint_qpos_adr(mujoco, model, names[6])] = j7
        data.qpos[_joint_qpos_adr(mujoco, model, names[7])] = j8
        mujoco.mj_forward(model, data)
        err_log.append(err)
        renderer.update_scene(data, camera=cam)
        bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            f"{label_side} i={i:03d} g={g*1000:.0f}mm pos_err={err*1000:.1f}mm",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        images.append(bgr)
        if i in (0, len(mats) // 2, len(mats) - 1) or i % 40 == 0:
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), bgr)
    return images, err_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Play virtual-gripper T_ee+g on Piper MuJoCo.")
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--xml", type=Path, default=PIPER_XML)
    parser.add_argument("--hand-side", default="both", choices=("right", "left", "both"))
    parser.add_argument("--no-calib", action="store_true")
    parser.add_argument(
        "--pos-only", action="store_true", help="IK position only (ignore EE orientation)"
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("-o", "--output", type=Path, default=Path("artifacts/piper_playback"))
    args = parser.parse_args()

    import cv2
    import mujoco

    out_dir = args.output if args.output.is_absolute() else ROOT / args.output
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    bimanual = args.hand_side == "both"

    model = load_model(mujoco, args.xml, bimanual=bimanual)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = np.array([0.22, 0.0, 0.18])
    cam.distance = 1.35 if bimanual else 1.15
    cam.azimuth = 135
    cam.elevation = -18

    if not bimanual:
        traj = load_ee_traj(args.clip, args.hand_side)
        mats, gripper = cam_traj_to_base(traj, apply_calib=not args.no_calib, side=args.hand_side)
        mats = mats[:: args.stride]
        gripper = gripper[:: args.stride]
        images, err_log = replay_one_arm(
            mujoco, model, data,
            prefix="", mats=mats, gripper=gripper, pos_only=args.pos_only,
            cam=cam, renderer=renderer, label_side=args.hand_side, frames_dir=frames_dir,
        )
        video_path = out_dir / "piper_ee_playback.mp4"
        _write_video(video_path, images, args.width, args.height)
        summary = {
            "clip": str(args.clip),
            "hand_side": args.hand_side,
            "n_frames": len(mats),
            "hacky_calib": not args.no_calib,
            "pos_only": args.pos_only,
            "mean_pos_err_m": float(np.mean(err_log)),
            "max_pos_err_m": float(np.max(err_log)),
            "g_min": float(np.min(gripper)),
            "g_max": float(np.max(gripper)),
            "video": str(video_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    frame_ids, cam_trajs, _ = align_bimanual(args.clip)
    frame_ids = frame_ids[:: args.stride]
    world: dict[str, np.ndarray] = {}
    grips: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        world[side], grips[side] = cam_traj_to_base(
            cam_trajs[side][:: args.stride], apply_calib=not args.no_calib, side=side
        )

    q = {
        "left": np.array([0.0, 0.8, -0.9, 0.0, 0.4, 0.0], dtype=np.float64),
        "right": np.array([0.0, 0.8, -0.9, 0.0, 0.4, 0.0], dtype=np.float64),
    }
    err_log = {"left": [], "right": []}
    images = []
    for i in range(len(frame_ids)):
        for side in ("left", "right"):
            prefix = f"{side}_"
            names = arm_joint_names(prefix)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}link6")
            mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ee_target")
            mocap_bid = int(model.body_mocapid[mocap_id])
            mat = world[side][i]
            g = float(grips[side][i])
            data.mocap_pos[mocap_bid] = mat[:3, 3]
            q[side], err = solve_ik(
                mujoco, model, data, body_id, mat[:3, 3], mat[:3, :3], q[side],
                joint_names=names, pos_only=args.pos_only,
            )
            err_log[side].append(err)
            j7, j8 = map_gripper(g)
            for name, val in zip(names[:6], q[side], strict=True):
                data.qpos[_joint_qpos_adr(mujoco, model, name)] = val
            data.qpos[_joint_qpos_adr(mujoco, model, names[6])] = j7
            data.qpos[_joint_qpos_adr(mujoco, model, names[7])] = j8
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        lg, rg = grips["left"][i] * 1000, grips["right"][i] * 1000
        le, re = err_log["left"][-1] * 1000, err_log["right"][-1] * 1000
        cv2.putText(
            bgr,
            f"dual f={int(frame_ids[i])}  L g={lg:.0f}mm e={le:.1f}  R g={rg:.0f}mm e={re:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        images.append(bgr)
        if i in (0, len(frame_ids) // 2, len(frame_ids) - 1) or i % 40 == 0:
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), bgr)

    video_path = out_dir / "piper_ee_playback.mp4"
    _write_video(video_path, images, args.width, args.height)
    summary = {
        "clip": str(args.clip),
        "hand_side": "both",
        "n_frames": len(frame_ids),
        "hacky_calib": not args.no_calib,
        "pos_only": args.pos_only,
        "arm_y": ARM_Y,
        "left_mean_pos_err_m": float(np.mean(err_log["left"])),
        "right_mean_pos_err_m": float(np.mean(err_log["right"])),
        "left_g_range": [float(np.min(grips["left"])), float(np.max(grips["left"]))],
        "right_g_range": [float(np.min(grips["right"])), float(np.max(grips["right"]))],
        "video": str(video_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
