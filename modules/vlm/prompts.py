from __future__ import annotations

import json
from typing import Any


def build_scene_description_prompt(task_instruction: str | None = None) -> str:
    task_line = (
        f"The user instruction is: {task_instruction}"
        if task_instruction
        else "No user instruction was provided."
    )
    return "\n".join(
        [
            "You are inspecting a robotics simulation image.",
            task_line,
            "Describe only what is visually grounded in the image.",
            (
                'Return JSON with keys: "objects_described" '
                '(array of strings), "spatial_summary" (string).'
            ),
            "Do not include markdown fences or extra commentary.",
        ]
    )


def build_plan_review_prompt(task_instruction: str, plan_payload: dict[str, Any]) -> str:
    serialized_plan = json.dumps(plan_payload, indent=2, sort_keys=True)
    return "\n".join(
        [
            "You are reviewing whether a robot plan matches the visible scene.",
            f"Instruction: {task_instruction}",
            "Plan JSON:",
            serialized_plan,
            (
                'Return JSON with keys: "feasible" (boolean), '
                '"concerns" (array of strings), "suggestions" (array of strings).'
            ),
            "Treat this as advisory review only. Do not assume hidden objects or state.",
            "Do not include markdown fences or extra commentary.",
        ]
    )


def build_execution_verification_prompt(task_instruction: str) -> str:
    return "\n".join(
        [
            "You are comparing before and after robotics simulation images.",
            f"Instruction: {task_instruction}",
            "Determine whether the task appears completed in the after image.",
            (
                'Return JSON with keys: "task_completed" (boolean), '
                '"discrepancies" (array of strings), '
                '"confidence" (number from 0.0 to 1.0).'
            ),
            "Use the first image as before and the second image as after.",
            "Do not include markdown fences or extra commentary.",
        ]
    )
