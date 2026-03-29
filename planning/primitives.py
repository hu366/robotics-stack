"""Motion primitives library.

M0 stub: placeholder for M1/M3 implementation.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MotionPrimitive:
    """Base motion primitive."""
    name: str
    target_pose: Optional[np.ndarray] = None
    constraints: Optional[dict] = None

    def generate_trajectory(self, current_pose: np.ndarray) -> np.ndarray:
        """Generate trajectory waypoints.

        Args:
            current_pose: Current end-effector pose

        Returns:
            waypoints: Array of pose waypoints
        """
        # M0 stub
        return np.array([current_pose])


def reach_pose(target_pose: np.ndarray, orientation_constraint: bool = False) -> MotionPrimitive:
    """Create a reach pose primitive.

    Args:
        target_pose: 6D or 7D pose (position + quaternion)
        orientation_constraint: Whether to enforce orientation

    Returns:
        MotionPrimitive for reaching target pose
    """
    return MotionPrimitive(
        name="ReachPose",
        target_pose=target_pose,
        constraints={"orientation": orientation_constraint}
    )


def approach_contact(contact_normal: np.ndarray, approach_distance: float = 0.1) -> MotionPrimitive:
    """Create an approach until contact primitive.

    Args:
        contact_normal: Normal vector of contact surface
        approach_distance: Distance to approach before contact

    Returns:
        MotionPrimitive for approaching contact
    """
    return MotionPrimitive(
        name="ApproachUntilContact",
        constraints={
            "contact_normal": contact_normal,
            "approach_distance": approach_distance
        }
    )


def retreat(retreat_distance: float = 0.1) -> MotionPrimitive:
    """Create a retreat primitive.

    Args:
        retreat_distance: Distance to retreat

    Returns:
        MotionPrimitive for retreating
    """
    return MotionPrimitive(
        name="Retreat",
        constraints={"retreat_distance": retreat_distance}
    )
