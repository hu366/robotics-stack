"""Vendored Cheng Chi / real-stanford pose math + a thin T_ee+g adapter.

Source files (do not rewrite):

- ``pose_util.py`` — UMI ``umi/common/pose_util.py``
- ``pose_repr_util.py`` — UMI ``diffusion_policy/common/pose_repr_util.py``
- ``rotation_transformer.py`` — UMI ``diffusion_policy/model/common/rotation_transformer.py``
"""

from modules.umi_pose.adapter import (
    POSE10D_GRIPPER_NAMES,
    POSE_GRIPPER_NAMES,
    convert_ee_trajectory,
    invert_action,
    wxyz_gripper_to_mats,
)
from modules.umi_pose.hacky_calib import T_base_from_cam
from modules.umi_pose.pose_repr_util import convert_pose_mat_rep
from modules.umi_pose.pose_util import (
    mat_to_pose,
    mat_to_pose10d,
    pose10d_to_mat,
    pose_to_mat,
)
from modules.umi_pose.rotation_transformer import RotationTransformer

__all__ = [
    "POSE10D_GRIPPER_NAMES",
    "POSE_GRIPPER_NAMES",
    "RotationTransformer",
    "T_base_from_cam",
    "convert_ee_trajectory",
    "convert_pose_mat_rep",
    "invert_action",
    "mat_to_pose",
    "mat_to_pose10d",
    "pose10d_to_mat",
    "pose_to_mat",
    "wxyz_gripper_to_mats",
]
