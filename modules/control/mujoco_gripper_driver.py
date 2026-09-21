from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedGripperCommand:
    kind: str
    target_ctrl: float | None = None
    message: str | None = None


def parse_gripper_command(raw: str) -> ParsedGripperCommand:
    text = raw.strip()
    if not text:
        return ParsedGripperCommand(kind="invalid", message="empty input")

    lower = text.lower()
    if lower in {"q", "quit", "exit"}:
        return ParsedGripperCommand(kind="quit")
    if lower in {"h", "help", "?"}:
        return ParsedGripperCommand(kind="help")
    if lower in {"open", "gripper open"}:
        return ParsedGripperCommand(kind="set", target_ctrl=255.0)
    if lower in {"close", "gripper close"}:
        return ParsedGripperCommand(kind="set", target_ctrl=0.0)

    tokens = shlex.split(text)
    if not tokens:
        return ParsedGripperCommand(kind="invalid", message="empty input")

    if tokens[0] == "gripper":
        tokens = tokens[1:]

    if len(tokens) == 2 and tokens[0] in {"set", "width", "ctrl"}:
        try:
            value = float(tokens[1])
        except ValueError:
            return ParsedGripperCommand(
                kind="invalid",
                message=f"invalid numeric value: {tokens[1]}",
            )
        return ParsedGripperCommand(kind="set", target_ctrl=value)

    return ParsedGripperCommand(
        kind="invalid",
        message=(
            "supported: open | close | set <0-255> | width <0-255> | "
            "gripper open/close/set <v>"
        ),
    )


def clamp_gripper_ctrl(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return value
