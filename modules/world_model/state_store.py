from __future__ import annotations

from interfaces.world_state import WorldState


class WorldModelStore:
    """Minimal in-memory world model until ROS-backed state ingestion is added."""

    def __init__(self) -> None:
        self._current: WorldState | None = None

    def update(self, world_state: WorldState) -> WorldState:
        self._current = world_state
        return world_state

    def current(self) -> WorldState:
        if self._current is None:
            raise RuntimeError("world state has not been initialized")
        return self._current
