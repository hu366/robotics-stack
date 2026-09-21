from __future__ import annotations

from modules.control.mujoco_gripper_driver import clamp_gripper_ctrl, parse_gripper_command


def test_parse_gripper_open_close() -> None:
    assert parse_gripper_command("open").target_ctrl == 255.0
    assert parse_gripper_command("close").target_ctrl == 0.0


def test_parse_gripper_numeric_set() -> None:
    parsed = parse_gripper_command("set 120")
    assert parsed.kind == "set"
    assert parsed.target_ctrl == 120.0


def test_parse_gripper_prefixed_command() -> None:
    parsed = parse_gripper_command("gripper width 64")
    assert parsed.kind == "set"
    assert parsed.target_ctrl == 64.0


def test_clamp_gripper_ctrl() -> None:
    assert clamp_gripper_ctrl(-3.0) == 0.0
    assert clamp_gripper_ctrl(20.0) == 20.0
    assert clamp_gripper_ctrl(999.0) == 255.0
