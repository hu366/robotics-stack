from __future__ import annotations

from copy import deepcopy
from typing import Protocol

from interfaces.control_feedback import StepFeedback
from interfaces.world_state import ObjectState, WorldState
from modules.planner.planner import PlanStep


class ControlBackend(Protocol):
    def execute_step(
        self,
        step: PlanStep,
        world_state: WorldState,
        step_index: int,
    ) -> StepFeedback: ...


class SymbolicControlBackend:
    """Minimal symbolic backend for validating the execution feedback loop."""

    def execute_step(
        self,
        step: PlanStep,
        world_state: WorldState,
        step_index: int,
    ) -> StepFeedback:
        skill_name = step.skill.name
        if skill_name == "locate_object":
            return self._locate_object(step, world_state, step_index)
        if skill_name == "grasp_object":
            return self._grasp_object(step, world_state, step_index)
        if skill_name == "place_object":
            return self._place_object(step, world_state, step_index)
        if skill_name == "inspect_scene":
            return self._inspect_scene(world_state, step_index)
        return StepFeedback(
            step_index=step_index,
            skill_name=skill_name,
            success=False,
            failure_code="unsupported_skill",
            observed_world_state=_clone_world_state(world_state),
            metrics={},
            notes=[f"Unsupported skill: {skill_name}"],
        )

    def _locate_object(
        self,
        step: PlanStep,
        world_state: WorldState,
        step_index: int,
    ) -> StepFeedback:
        target_object = step.parameters.get("target_object", "")
        matched_object = _find_object(world_state, target_object)
        if matched_object is None:
            return StepFeedback(
                step_index=step_index,
                skill_name=step.skill.name,
                success=False,
                failure_code="object_not_found",
                observed_world_state=_clone_world_state(world_state),
                metrics={"target_object": target_object},
                notes=[f"Target object not found: {target_object}"],
            )
        return StepFeedback(
            step_index=step_index,
            skill_name=step.skill.name,
            success=True,
            failure_code=None,
            observed_world_state=_clone_world_state(world_state),
            metrics={
                "target_object": target_object,
                "matched_object_id": matched_object.object_id,
            },
            notes=[f"Resolved target object: {matched_object.object_id}"],
        )

    def _grasp_object(
        self,
        step: PlanStep,
        world_state: WorldState,
        step_index: int,
    ) -> StepFeedback:
        updated_world_state = _clone_world_state(world_state)
        target_object = step.parameters.get("target_object", "")
        matched_object = _find_object(updated_world_state, target_object)
        if matched_object is None:
            return StepFeedback(
                step_index=step_index,
                skill_name=step.skill.name,
                success=False,
                failure_code="object_not_found",
                observed_world_state=updated_world_state,
                metrics={"target_object": target_object},
                notes=[f"Cannot grasp missing object: {target_object}"],
            )

        matched_object.relations = _dedupe_relations(
            [relation for relation in matched_object.relations if relation != "held_by:gripper"]
            + ["held_by:gripper"]
        )
        updated_world_state.robot_mode = "holding"
        return StepFeedback(
            step_index=step_index,
            skill_name=step.skill.name,
            success=True,
            failure_code=None,
            observed_world_state=updated_world_state,
            metrics={"target_object": matched_object.object_id},
            notes=[f"Grasped object: {matched_object.object_id}"],
        )

    def _place_object(
        self,
        step: PlanStep,
        world_state: WorldState,
        step_index: int,
    ) -> StepFeedback:
        updated_world_state = _clone_world_state(world_state)
        target_object = step.parameters.get("target_object", "")
        target_location = step.parameters.get("target_location", "")
        matched_object = _find_object(updated_world_state, target_object)
        if matched_object is None:
            return StepFeedback(
                step_index=step_index,
                skill_name=step.skill.name,
                success=False,
                failure_code="object_not_found",
                observed_world_state=updated_world_state,
                metrics={"target_object": target_object, "target_location": target_location},
                notes=[f"Cannot place missing object: {target_object}"],
            )

        matched_location = _find_object(updated_world_state, target_location)
        if matched_location is None:
            return StepFeedback(
                step_index=step_index,
                skill_name=step.skill.name,
                success=False,
                failure_code="target_location_not_found",
                observed_world_state=updated_world_state,
                metrics={
                    "target_object": matched_object.object_id,
                    "target_location": target_location,
                },
                notes=[f"Cannot resolve target location: {target_location}"],
            )

        if "held_by:gripper" not in matched_object.relations:
            return StepFeedback(
                step_index=step_index,
                skill_name=step.skill.name,
                success=False,
                failure_code="object_not_grasped",
                observed_world_state=updated_world_state,
                metrics={
                    "target_object": matched_object.object_id,
                    "target_location": matched_location.object_id,
                },
                notes=[f"Object is not grasped: {matched_object.object_id}"],
            )

        matched_object.pose = list(matched_location.pose)
        matched_object.relations = _dedupe_relations(
            [
                relation
                for relation in matched_object.relations
                if relation != "held_by:gripper" and not relation.startswith("on:")
            ]
            + [f"on:{matched_location.object_id}"]
        )
        updated_world_state.robot_mode = "ready"
        return StepFeedback(
            step_index=step_index,
            skill_name=step.skill.name,
            success=True,
            failure_code=None,
            observed_world_state=updated_world_state,
            metrics={
                "target_object": matched_object.object_id,
                "target_location": matched_location.object_id,
            },
            notes=[
                f"Placed {matched_object.object_id} on {matched_location.object_id}",
            ],
        )

    def _inspect_scene(self, world_state: WorldState, step_index: int) -> StepFeedback:
        return StepFeedback(
            step_index=step_index,
            skill_name="inspect_scene",
            success=True,
            failure_code=None,
            observed_world_state=_clone_world_state(world_state),
            metrics={"object_count": len(world_state.objects)},
            notes=["Scene inspected without mutation."],
        )


def _clone_world_state(world_state: WorldState) -> WorldState:
    return deepcopy(world_state)


def _find_object(world_state: WorldState, query: str) -> ObjectState | None:
    normalized_query = _normalize(query)
    query_slug = _slug(query)
    for obj in world_state.objects:
        candidate_keys = {
            _normalize(obj.label),
            _normalize(obj.object_id),
            _slug(obj.label),
            _slug(obj.object_id),
        }
        if normalized_query in candidate_keys or query_slug in candidate_keys:
            return obj
    return None


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _slug(value: str) -> str:
    return "-".join(_normalize(value).split())


def _dedupe_relations(relations: list[str]) -> list[str]:
    return list(dict.fromkeys(relations))
