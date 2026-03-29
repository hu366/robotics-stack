"""Kinematics utilities for Franka Panda 7-DoF arm."""

import numpy as np
import mujoco


def forward_kinematics(model, data, body_name):
    """Compute forward kinematics for a body.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of body to compute pose for

    Returns:
        pos: Position (3,)
        quat: Quaternion (4,)
    """
    body_id = model.body(body_name)
    pos = data.xpos[body_id].copy()
    quat = data.xquat[body_id].copy()
    return pos, quat


def compute_jacobian(model, data, body_name):
    """Compute geometric Jacobian for a body.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of body

    Returns:
        J: 6xDOF Jacobian (position + orientation)
    """
    body_id = model.body(body_name)
    J = np.zeros((6, model.nv))
    mujoco.mj_jacBody(model, data, J[:3, :], J[3:, :], body_id)
    return J
