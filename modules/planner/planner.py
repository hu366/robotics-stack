from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from interfaces.skill_spec import SkillSpec
from interfaces.task_spec import TaskSpec
from interfaces.world_state import WorldState


@dataclass(slots=True)
class PlanStep:
    skill: SkillSpec
    parameters: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.to_dict(),
            "parameters": dict(self.parameters),
        }


@dataclass(slots=True)
class ExecutionPlan:
    task_id: str
    steps: list[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [step.to_dict() for step in self.steps],
        }


class TaskPlanner:
    """Build a transparent skill sequence from parsed task intent."""

    def build_plan(self, task: TaskSpec, world_state: WorldState) -> ExecutionPlan:
        steps: list[PlanStep] = []
        target_object = task.target_object or "object"
        target_location = task.target_location or "inspection_zone"

        if task.goal == "place_object":
            steps.extend(
                [
                    PlanStep(
                        skill=SkillSpec(
                            name="locate_object",
                            inputs=["world_state", "target_object"],
                            outputs=["object_pose"],
                            success_conditions=["object_pose_resolved"],
                            failure_codes=["object_not_found"],
                        ),
                        parameters={"target_object": target_object},
                    ),
                    PlanStep(
                        skill=SkillSpec(
                            name="grasp_object",
                            inputs=["object_pose"],
                            outputs=["grasped_object"],
                            success_conditions=["stable_grasp"],
                            failure_codes=["grasp_failed"],
                        ),
                        parameters={"target_object": target_object},
                    ),
                    PlanStep(
                        skill=SkillSpec(
                            name="place_object",
                            inputs=["grasped_object", "target_location"],
                            outputs=["object_placed"],
                            success_conditions=["target_relation_satisfied"],
                            failure_codes=["placement_failed"],
                        ),
                        parameters={
                            "target_object": target_object,
                            "target_location": target_location,
                        },
                    ),
                ]
            )
        else:
            steps.append(
                PlanStep(
                    skill=SkillSpec(
                        name="inspect_scene",
                        inputs=["world_state"],
                        outputs=["scene_summary"],
                        success_conditions=["scene_summarized"],
                        failure_codes=["inspection_failed"],
                    ),
                    parameters={"scene_id": world_state.scene_id},
                )
            )

        return ExecutionPlan(task_id=task.task_id, steps=steps)
