from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class ParsedTargetInput:
    kind: str
    target: tuple[float, float, float] | None = None
    message: str | None = None


def parse_target_input(raw: str) -> ParsedTargetInput:
    text = raw.strip()
    if not text:
        return ParsedTargetInput(kind="invalid", message="empty input")
    if text.lower() in {"q", "quit", "exit"}:
        return ParsedTargetInput(kind="quit")
    if text.lower() in {"h", "help", "?"}:
        return ParsedTargetInput(kind="help")

    tokens = shlex.split(text)
    if not tokens:
        return ParsedTargetInput(kind="invalid", message="empty input")

    if any(token.startswith("--") for token in tokens):
        values: dict[str, float] = {}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token not in {"--x", "--y", "--z"}:
                return ParsedTargetInput(
                    kind="invalid",
                    message=f"unsupported option: {token}",
                )
            if index + 1 >= len(tokens):
                return ParsedTargetInput(
                    kind="invalid",
                    message=f"missing value for {token}",
                )
            try:
                values[token] = float(tokens[index + 1])
            except ValueError:
                return ParsedTargetInput(
                    kind="invalid",
                    message=f"invalid float for {token}: {tokens[index + 1]}",
                )
            index += 2
        if set(values) != {"--x", "--y", "--z"}:
            return ParsedTargetInput(
                kind="invalid",
                message="must provide --x --y --z together",
            )
        return ParsedTargetInput(
            kind="target",
            target=(values["--x"], values["--y"], values["--z"]),
        )

    if len(tokens) != 3:
        return ParsedTargetInput(
            kind="invalid",
            message="use either: x y z or --x X --y Y --z Z",
        )
    try:
        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
    except ValueError:
        return ParsedTargetInput(
            kind="invalid",
            message="x y z must be numeric",
        )
    return ParsedTargetInput(kind="target", target=(x, y, z))


def plan_linear_waypoints(
    start_xyz: Sequence[float],
    target_xyz: Sequence[float],
    steps: int,
) -> list[np.ndarray]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    start = np.asarray(start_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    waypoints: list[np.ndarray] = []
    for i in range(1, steps + 1):
        alpha = float(i) / float(steps)
        point = (1.0 - alpha) * start + alpha * target
        waypoints.append(point)
    return waypoints
