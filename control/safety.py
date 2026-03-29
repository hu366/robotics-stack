"""Safety monitor for torque/velocity/force limits."""

import numpy as np


def check_torque_limits(tau, limits):
    """Check if torques exceed limits.

    Args:
        tau: Commanded torques
        limits: Torque limit array

    Returns:
        exceeded: Boolean indicating if any limit exceeded
    """
    return np.any(np.abs(tau) > limits)


def check_joint_limits(q, qmin, qmax):
    """Check if joints exceed position limits.

    Args:
        q: Joint positions
        qmin: Minimum joint positions
        qmax: Maximum joint positions

    Returns:
        exceeded: Boolean indicating if any limit exceeded
    """
    return np.any(q < qmin) or np.any(q > qmax)


def check_force_limit(force, limit):
    """Check if contact force exceeds limit.

    Args:
        force: Contact force magnitude
        limit: Force limit

    Returns:
        exceeded: Boolean indicating if limit exceeded
    """
    return abs(force) > limit
