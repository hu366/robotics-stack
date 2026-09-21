from __future__ import annotations

import numpy as np

from modules.control.mujoco_target_driver import parse_target_input, plan_linear_waypoints


def test_parse_target_input_with_flag_form() -> None:
    parsed = parse_target_input("--x 0.4 --y -0.1 --z 0.3")
    assert parsed.kind == "target"
    assert parsed.target == (0.4, -0.1, 0.3)


def test_parse_target_input_with_triplet_form() -> None:
    parsed = parse_target_input("0.25 0.15 0.5")
    assert parsed.kind == "target"
    assert parsed.target == (0.25, 0.15, 0.5)


def test_parse_target_input_quit() -> None:
    parsed = parse_target_input("quit")
    assert parsed.kind == "quit"


def test_plan_linear_waypoints_reaches_target() -> None:
    waypoints = plan_linear_waypoints([0.0, 0.0, 0.0], [0.3, -0.3, 0.6], steps=3)
    assert len(waypoints) == 3
    assert np.allclose(waypoints[0], np.array([0.1, -0.1, 0.2]))
    assert np.allclose(waypoints[-1], np.array([0.3, -0.3, 0.6]))
