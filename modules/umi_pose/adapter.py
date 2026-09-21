"""Thin adapter: virtual-gripper ``T_ee + g`` → UMI pose10d action.

Does not reimplement relative/delta math. The only operations here are:

- wxyz quaternion (ego2dex) → xyzw (scipy / UMI ``RotationTransformer``)
- optional left-multiply of a camera→base SE(3)
- calls to vendored ``convert_pose_mat_rep`` / ``mat_to_pose10d``
- concatenating gripper as a **separate last dim** (UMI ``umi_dataset.py``)

``pose_rep`` values are those of Cheng Chi's ``convert_pose_mat_rep``:

- ``relative`` — body-frame ``inv(g0) @ g``  (the UMI "base moved and it still works" path)
- ``delta`` — consecutive body-frame increments
- ``abs`` — pass-through
- ``rel`` — leftover **buggy** world-mix in the vendored file; kept only for compatibility
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from modules.umi_pose.hacky_calib import T_base_from_cam
from modules.umi_pose.pose_repr_util import convert_pose_mat_rep
from modules.umi_pose.pose_util import mat_to_pose10d, pose10d_to_mat
from modules.umi_pose.rotation_transformer import RotationTransformer

# scipy / UMI quaternion is xyzw; ego2dex virtual gripper is wxyz.
_WXYZ_TO_XYZW = (1, 2, 3, 0)
_XYZW_TO_WXYZ = (3, 0, 1, 2)

_QUAT_TO_MAT = RotationTransformer("quaternion", "matrix")
_MAT_TO_QUAT = RotationTransformer("matrix", "quaternion")

POSE_GRIPPER_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
    "gripper",
)
POSE10D_GRIPPER_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "r6d_0",
    "r6d_1",
    "r6d_2",
    "r6d_3",
    "r6d_4",
    "r6d_5",
    "gripper",
)


def wxyz_gripper_to_mats(
    traj: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(T, 8)`` x,y,z,qw,qx,qy,qz,g → ``(T, 4, 4)`` mats and ``(T, 1)`` gripper."""
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 8:
        raise ValueError(f"expected (T, 8) x,y,z,qw,qx,qy,qz,g; got {arr.shape}")
    xyz = arr[:, :3]
    quat_xyzw = arr[:, 3:7][:, list(_WXYZ_TO_XYZW)]
    rot = np.asarray(_QUAT_TO_MAT.forward(quat_xyzw), dtype=np.float64)
    mats = np.zeros((arr.shape[0], 4, 4), dtype=np.float64)
    mats[:, :3, :3] = rot
    mats[:, :3, 3] = xyz
    mats[:, 3, 3] = 1.0
    gripper = arr[:, 7:8]
    return mats, gripper


def mats_to_wxyz_gripper(
    mats: NDArray[np.floating],
    gripper: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Inverse of :func:`wxyz_gripper_to_mats` (scipy xyzw → wxyz)."""
    mats_f = np.asarray(mats, dtype=np.float64)
    g = np.asarray(gripper, dtype=np.float64).reshape(-1, 1)
    quat_xyzw = np.asarray(_MAT_TO_QUAT.forward(mats_f[:, :3, :3]), dtype=np.float64)
    quat_wxyz = quat_xyzw[:, list(_XYZW_TO_WXYZ)]
    return np.concatenate([mats_f[:, :3, 3], quat_wxyz, g], axis=-1)


def apply_cam_to_base(
    mats: NDArray[np.floating],
    T_base_cam: NDArray[np.floating] | None = None,
    *,
    apply: bool = True,
) -> NDArray[np.float64]:
    """Left-multiply ``T_base←cam``. No-op when ``apply`` is false."""
    mats_f = np.asarray(mats, dtype=np.float64)
    if not apply:
        return mats_f
    tx = np.asarray(T_base_from_cam() if T_base_cam is None else T_base_cam, dtype=np.float64)
    if tx.shape != (4, 4):
        raise ValueError(f"T_base_cam must be 4x4, got {tx.shape}")
    return tx @ mats_f


def convert_ee_trajectory(
    traj: NDArray[np.floating],
    *,
    pose_rep: str = "relative",
    apply_hacky_calib: bool = True,
    T_base_cam: NDArray[np.floating] | None = None,
    base_index: int = 0,
) -> dict[str, Any]:
    """Train path: abs ``(T, 8)`` wxyz+g → pose10d+g via ``convert_pose_mat_rep``.

    ``pose_rep`` is forwarded unchanged. Default ``relative`` is body-frame
    ``g0^{-1} g``. Hacky ``T_base←cam`` is applied on the **absolute** mats
    before that call (needed for MuJoCo parking, not for the delta math).
    """
    if pose_rep not in ("abs", "rel", "relative", "delta"):
        raise ValueError(f"unsupported pose_rep {pose_rep!r}")

    mats, gripper = wxyz_gripper_to_mats(traj)
    mats = apply_cam_to_base(mats, T_base_cam, apply=apply_hacky_calib)
    base = mats[base_index]
    converted = convert_pose_mat_rep(mats, base_pose_mat=base, pose_rep=pose_rep, backward=False)
    pose10d = mat_to_pose10d(converted)
    action = np.concatenate([pose10d, gripper], axis=-1)
    return {
        "action": action,
        "pose10d": pose10d,
        "gripper": gripper,
        "pose_mat": converted,
        "abs_mat": mats,
        "base_pose_mat": base,
        "pose_rep": pose_rep,
        "joint_names": list(POSE10D_GRIPPER_NAMES),
    }


def invert_action(
    action: NDArray[np.floating],
    base_pose_mat: NDArray[np.floating],
    *,
    pose_rep: str = "relative",
) -> NDArray[np.float64]:
    """Eval path: pose10d+g → absolute ``(T, 8)`` wxyz+g, using the train base."""
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 10:
        raise ValueError(f"expected (T, 10) pose10d+g; got {arr.shape}")
    rel_mat = pose10d_to_mat(arr[:, :9])
    abs_mat = convert_pose_mat_rep(
        rel_mat,
        base_pose_mat=np.asarray(base_pose_mat, dtype=np.float64),
        pose_rep=pose_rep,
        backward=True,
    )
    return mats_to_wxyz_gripper(abs_mat, arr[:, 9:10])
