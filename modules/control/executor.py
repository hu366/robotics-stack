from __future__ import annotations

from interfaces.execution_trace import ExecutionTrace
from modules.planner.planner import ExecutionPlan
from modules.skills.library import SkillLibrary


class PlanExecutor:
    """Deterministic baseline executor that emits an auditable trace."""

    def __init__(self, skill_library: SkillLibrary | None = None) -> None:
        self.skill_library = skill_library or SkillLibrary()

    def execute(self, plan: ExecutionPlan, trace: ExecutionTrace) -> ExecutionTrace:
        for index, step in enumerate(plan.steps, start=1):
            trace.add_event(
                stage="control",
                message=f"execute_step_{index}",
                status="running",
                payload={"action": self.skill_library.describe(step)},
            )
            trace.add_event(
                stage="control",
                message=f"step_{index}_completed",
                status="success",
                payload={"action": self.skill_library.describe(step)},
            )
        return trace
