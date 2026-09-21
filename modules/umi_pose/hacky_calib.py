"""Hacky camera→Piper-base SE(3). Not a real hand-eye calibration.

OpenCV camera (x right, y down, z forward) mapped onto a ROS-style arm base
(x forward, y left, z up), then shoved into a reachable box in front of the
Piper (~0.30 m forward, 0.20 m up). Replace this matrix when a real extrinsics
file exists; relative/delta actions do not depend on it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# cam z (forward) -> base x; cam x (right) -> base -y; cam y (down) -> base -z
R_BASE_FROM_CAM: NDArray[np.float64] = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
T_BASE_FROM_CAM_TRANSLATION: NDArray[np.float64] = np.array(
    [0.30, 0.0, 0.20],
    dtype=np.float64,
)


def T_base_from_cam(
    translation: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Return a 4×4 ``T_base←cam`` (left-multiply onto camera-frame EE poses)."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_BASE_FROM_CAM
    if translation is None:
        out[:3, 3] = T_BASE_FROM_CAM_TRANSLATION
    else:
        out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out
