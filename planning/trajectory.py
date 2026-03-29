"""Trajectory generation and time parameterization.

M0 stub: placeholder for M1 implementation.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class TrajectorySegment:
    """A single trajectory segment with time parameterization."""
    waypoints: np.ndarray  # (N, 6) or (N, 7) poses
    durations: np.ndarray  # (N,) segment durations

    def total_time(self) -> float:
        """Return total segment duration."""
        return float(np.sum(self.durations))


def generate_minimum_jerk_trajectory(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    duration: float,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate minimum jerk trajectory between two poses.

    Args:
        start_pose: Starting pose
        end_pose: Ending pose
        duration: Total duration in seconds
        num_points: Number of waypoints

    Returns:
        poses: (num_points, 6) array of poses
        times: (num_points,) array of timestamps
    """
    t = np.linspace(0, duration, num_points)
    t_norm = t / duration

    # Minimum jerk polynomial: s(t) = 10t^3 - 15t^4 + 6t^5
    s = 10 * t_norm**3 - 15 * t_norm**4 + 6 * t_norm**5

    # Interpolate positions
    positions = np.outer(1 - s, start_pose[:3]) + np.outer(s, end_pose[:3])

    # Simple quaternion SLERP
    q0, q1 = start_pose[3:7], end_pose[3:7]
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    # Linear interpolation for M0 (proper SLERP in M1)
    quaternions = np.outer(1 - s, q0) + np.outer(s, q1)
    quaternions = quaternions / (np.linalg.norm(quaternions, axis=1, keepdims=True) + 1e-10)

    poses = np.hstack([positions, quaternions])
    return poses, t


def generate_trapezoidal_velocity_profile(
    distance: float,
    max_velocity: float,
    max_acceleration: float
) -> Tuple[float, float, float]:
    """Compute trapezoidal velocity profile parameters.

    Args:
        distance: Total distance to travel
        max_velocity: Maximum velocity
        max_acceleration: Maximum acceleration

    Returns:
        t_accel: Acceleration phase duration
        t_constant: Constant velocity phase duration
        t_decel: Deceleration phase duration
    """
    # Time to reach max velocity
    t_accel = max_velocity / max_acceleration
    t_decel = t_accel

    # Distance covered during accel/decel
    d_accel = 0.5 * max_acceleration * t_accel**2
    d_decel = d_accel

    # Remaining distance at constant velocity
    d_constant = distance - d_accel - d_decel
    if d_constant < 0:
        # Triangle profile (no constant phase)
        t_constant = 0
    else:
        t_constant = d_constant / max_velocity

    return t_accel, t_constant, t_decel
