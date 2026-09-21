"""Drive the vendored Cheng Chi convert_pose_mat_rep, not a reimplementation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from modules.umi_pose.adapter import convert_ee_trajectory, invert_action
from modules.umi_pose.hacky_calib import T_base_from_cam
from modules.umi_pose.pose_repr_util import convert_pose_mat_rep
from modules.umi_pose.pose_util import mat_to_pose10d, pose10d_to_mat, pose_to_mat
from modules.umi_pose.rotation_transformer import RotationTransformer


def _rand_mats(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mats = np.zeros((n, 4, 4), dtype=np.float64)
    mats[:, 3, 3] = 1.0
    mats[:, :3, 3] = rng.randn(n, 3) * 0.1 + np.array([0.3, 0.0, 0.2])
    mats[:, :3, :3] = Rotation.random(n, random_state=rng).as_matrix()
    return mats


def test_relative_is_body_frame_left_multiply() -> None:
    mats = _rand_mats(5)
    base = mats[0]
    got = convert_pose_mat_rep(mats, base_pose_mat=base, pose_rep="relative", backward=False)
    expected = np.linalg.inv(base) @ mats
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_rel_is_not_the_body_frame_path() -> None:
    mats = _rand_mats(4, seed=1)
    base = mats[0]
    body = convert_pose_mat_rep(mats, base_pose_mat=base, pose_rep="relative", backward=False)
    legacy = convert_pose_mat_rep(mats, base_pose_mat=base, pose_rep="rel", backward=False)
    assert not np.allclose(body, legacy)


def test_relative_forward_backward_inverse() -> None:
    mats = _rand_mats(8, seed=2)
    base = mats[0]
    rel = convert_pose_mat_rep(mats, base_pose_mat=base, pose_rep="relative", backward=False)
    back = convert_pose_mat_rep(rel, base_pose_mat=base, pose_rep="relative", backward=True)
    np.testing.assert_allclose(back, mats, atol=1e-10)


def test_delta_forward_backward_inverse() -> None:
    mats = _rand_mats(6, seed=3)
    base = mats[0]
    delta = convert_pose_mat_rep(mats, base_pose_mat=base, pose_rep="delta", backward=False)
    back = convert_pose_mat_rep(delta, base_pose_mat=base, pose_rep="delta", backward=True)
    np.testing.assert_allclose(back, mats, atol=1e-10)


def test_pose10d_roundtrip_through_official_helpers() -> None:
    mats = _rand_mats(5, seed=4)
    d10 = mat_to_pose10d(mats)
    assert d10.shape == (5, 9)
    back = pose10d_to_mat(d10)
    np.testing.assert_allclose(back[:, :3, 3], mats[:, :3, 3], atol=1e-12)
    np.testing.assert_allclose(back[:, :3, :3], mats[:, :3, :3], atol=1e-12)


def test_rotation_transformer_quat_to_6d_uses_scipy_not_euler() -> None:
    tf = RotationTransformer("quaternion", "rotation_6d")
    quat_xyzw = Rotation.from_rotvec([0.2, -0.1, 0.4]).as_quat()
    d6 = tf.forward(quat_xyzw)
    assert d6.shape == (6,)
    back = RotationTransformer("rotation_6d", "quaternion").forward(d6)
    # q and -q are the same rotation
    same = np.allclose(back, quat_xyzw, atol=1e-7) or np.allclose(back, -quat_xyzw, atol=1e-7)
    assert same


def test_adapter_keeps_gripper_as_last_dim() -> None:
    t = 5
    traj = np.zeros((t, 8), dtype=np.float64)
    traj[:, 0] = 0.4
    traj[:, 2] = 0.5
    traj[:, 3] = 1.0  # qw
    traj[:, 7] = np.linspace(0.0, 0.07, t)
    out = convert_ee_trajectory(traj, pose_rep="relative", apply_hacky_calib=True)
    action = out["action"]
    assert action.shape == (t, 10)
    np.testing.assert_allclose(action[:, -1], traj[:, 7])
    np.testing.assert_allclose(action[0, :3], 0.0, atol=1e-12)
    restored = invert_action(action, out["base_pose_mat"], pose_rep="relative")
    # restored is in the calibrated (base) frame, not the raw camera frame
    abs_cal = out["abs_mat"]
    np.testing.assert_allclose(restored[:, :3], abs_cal[:, :3, 3], atol=1e-9)
    np.testing.assert_allclose(restored[:, 7], traj[:, 7], atol=1e-12)


def test_hacky_calib_is_left_multiply_not_rewritten_delta() -> None:
    traj = np.zeros((3, 8), dtype=np.float64)
    traj[:, 3] = 1.0
    traj[0, :3] = [0.1, 0.0, 0.4]
    traj[1, :3] = [0.15, 0.02, 0.41]
    traj[2, :3] = [0.2, 0.04, 0.42]
    traj[:, 7] = 0.03
    cam = convert_ee_trajectory(traj, pose_rep="relative", apply_hacky_calib=False)
    base = convert_ee_trajectory(traj, pose_rep="relative", apply_hacky_calib=True)
    tx = T_base_from_cam()
    np.testing.assert_allclose(base["abs_mat"], tx @ cam["abs_mat"], atol=1e-12)
    # relative action itself is body-frame, so first-frame pose10d translation is 0 either way
    np.testing.assert_allclose(cam["action"][0, :3], 0.0, atol=1e-12)
    np.testing.assert_allclose(base["action"][0, :3], 0.0, atol=1e-12)


def test_pose_to_mat_axis_angle_layout() -> None:
    pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    mat = pose_to_mat(pose)
    np.testing.assert_allclose(mat[:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(mat[:3, :3], np.eye(3))
