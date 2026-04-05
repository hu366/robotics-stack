from __future__ import annotations

from interfaces.task_spec import TaskSpec
from interfaces.world_state import ObjectState, WorldState


class SceneGrounder:
    """Ground task entities into a simple geometric world state baseline."""

    def ground(self, task: TaskSpec) -> WorldState:
        objects: list[ObjectState] = []
        if task.target_object:
            objects.append(
                ObjectState(
                    object_id=self._slug(task.target_object),
                    label=task.target_object,
                    pose=[0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                    relations=[],
                )
            )
        if task.target_location:
            objects.append(
                ObjectState(
                    object_id=self._slug(task.target_location),
                    label=task.target_location,
                    pose=[0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    relations=["reference_frame:tabletop"],
                )
            )

        return WorldState(
            scene_id=f"scene-{task.task_id}",
            objects=objects,
            robot_mode="ready",
        )

    def _slug(self, label: str) -> str:
        return "-".join(label.lower().split())
