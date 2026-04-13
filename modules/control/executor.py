from __future__ import annotations

from interfaces.control_feedback import ExecutionResult
from interfaces.execution_trace import ExecutionTrace
from modules.control.backend import ControlBackend, SymbolicControlBackend
from modules.planner.planner import ExecutionPlan
from modules.skills.library import SkillLibrary
from modules.world_model.state_store import WorldModelStore


class PlanExecutor:
    """Closed-loop executor that records actions, feedback, and final outcomes."""

    def __init__(
        self,
        skill_library: SkillLibrary | None = None,
        backend: ControlBackend | None = None,
        world_model: WorldModelStore | None = None,
    ) -> None:
        self.skill_library = skill_library or SkillLibrary()
        self.backend = backend or SymbolicControlBackend()
        self.world_model = world_model

    def execute(self, plan: ExecutionPlan, trace: ExecutionTrace) -> ExecutionResult:
        if self.world_model is None:
            raise RuntimeError("world model must be provided for closed-loop execution")

        completed_steps = 0
        for index, step in enumerate(plan.steps, start=1):
            current_world_state = self.world_model.current()
            action = self.skill_library.describe(step)
            trace.add_event(
                stage="control",
                message=f"step_{index}_started",
                status="running",
                payload={
                    "action": action,
                    "world_state": current_world_state.to_dict(),
                },
            )
            feedback = self.backend.execute_step(step, current_world_state, index)
            self.world_model.update(feedback.observed_world_state)
            trace.add_event(
                stage="control",
                message=f"step_{index}_feedback_observed",
                status="success",
                payload=feedback.to_dict(),
            )
            if feedback.success:
                completed_steps = index
                trace.add_event(
                    stage="control",
                    message=f"step_{index}_succeeded",
                    status="success",
                    payload={
                        "action": action,
                        "feedback": feedback.to_dict(),
                    },
                )
                continue

            result = ExecutionResult(
                task_id=plan.task_id,
                success=False,
                completed_steps=completed_steps,
                failed_step_index=index,
                final_world_state=self.world_model.current(),
                failure_code=feedback.failure_code,
            )
            trace.add_event(
                stage="control",
                message=f"step_{index}_failed",
                status="failed",
                payload={
                    "action": action,
                    "feedback": feedback.to_dict(),
                },
            )
            trace.add_event(
                stage="control",
                message="execution_failed",
                status="failed",
                payload=result.to_dict(),
            )
            return result

        result = ExecutionResult(
            task_id=plan.task_id,
            success=True,
            completed_steps=completed_steps,
            failed_step_index=None,
            final_world_state=self.world_model.current(),
            failure_code=None,
        )
        trace.add_event(
            stage="control",
            message="execution_completed",
            status="success",
            payload=result.to_dict(),
        )
        return result
