from __future__ import annotations

from modules.planner.planner import PlanStep


class SkillLibrary:
    """Resolve plan steps into executable baseline actions."""

    def describe(self, step: PlanStep) -> str:
        parameters = ", ".join(f"{key}={value}" for key, value in step.parameters.items())
        return f"{step.skill.name}({parameters})" if parameters else step.skill.name
