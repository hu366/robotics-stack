from dataclasses import dataclass, field
from typing import Any

from interfaces.world_state import WorldState


@dataclass(slots=True)
class StepFeedback:
    step_index: int
    skill_name: str
    success: bool
    failure_code: str | None
    observed_world_state: WorldState
    metrics: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "skill_name": self.skill_name,
            "success": self.success,
            "failure_code": self.failure_code,
            "observed_world_state": self.observed_world_state.to_dict(),
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class ExecutionResult:
    task_id: str
    success: bool
    completed_steps: int
    failed_step_index: int | None
    final_world_state: WorldState
    failure_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "completed_steps": self.completed_steps,
            "failed_step_index": self.failed_step_index,
            "final_world_state": self.final_world_state.to_dict(),
            "failure_code": self.failure_code,
        }
