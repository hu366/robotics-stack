from __future__ import annotations

from interfaces.execution_trace import ExecutionTrace
from interfaces.skill_spec import SkillSpec
from interfaces.task_spec import TaskSpec
from interfaces.vlm import VLMQueryContext, VLMResponse
from modules.planner import ExecutionPlan, PlanStep
from modules.vlm.service import VLMService


class RecordingBackend:
    def __init__(self, payloads: dict[str, str]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[list[bytes], str, VLMQueryContext | None]] = []

    def query(
        self,
        images: list[bytes],
        prompt: str,
        context: VLMQueryContext | None = None,
    ) -> VLMResponse:
        self.calls.append((images, prompt, context))
        stage = "" if context is None else context.stage
        return VLMResponse(
            text=self.payloads[stage],
            confidence=0.73,
            model_id="test-vlm",
            latency_ms=12,
            raw={"stage": stage},
        )


def test_describe_scene_parses_structured_output() -> None:
    backend = RecordingBackend(
        {
            "scene_description": (
                '{"objects_described": ["robot arm", "tray"], '
                '"spatial_summary": "The tray is in front of the robot arm."}'
            )
        }
    )

    result = VLMService(backend).describe_scene(b"image-bytes", "place the bottle on the tray")

    assert result.objects_described == ["robot arm", "tray"]
    assert result.spatial_summary == "The tray is in front of the robot arm."
    assert result.raw_response is not None
    assert result.raw_response.model_id == "test-vlm"


def test_review_plan_parses_feasibility() -> None:
    backend = RecordingBackend(
        {
            "plan_review": (
                '{"feasible": false, "concerns": ["Target object is not visible."], '
                '"suggestions": ["Reacquire a wider camera view."]}'
            )
        }
    )
    plan = ExecutionPlan(
        task_id="task-123",
        steps=[
            PlanStep(
                skill=SkillSpec(name="inspect_scene"),
                parameters={"scene_id": "scene-123"},
            )
        ],
    )

    result = VLMService(backend).review_plan(
        b"image-bytes",
        plan,
        "place the bottle on the tray",
    )

    assert not result.feasible
    assert result.concerns == ["Target object is not visible."]
    assert result.suggestions == ["Reacquire a wider camera view."]


def test_verify_execution_parses_completion_and_confidence() -> None:
    backend = RecordingBackend(
        {
            "execution_verification": (
                '{"task_completed": true, "discrepancies": [], "confidence": 0.91}'
            )
        }
    )
    task = TaskSpec(
        task_id="task-verify",
        instruction="place the bottle on the tray",
        goal="place_object",
        target_object="the bottle",
        target_location="the tray",
    )

    result = VLMService(backend).verify_execution(b"before", b"after", task)

    assert result.task_completed
    assert result.discrepancies == []
    assert result.confidence == 0.91
    assert len(backend.calls[0][0]) == 2


def test_vlm_structures_serialize_into_trace_events() -> None:
    backend = RecordingBackend(
        {
            "scene_description": (
                '{"objects_described": ["robot arm"], "spatial_summary": "Robot above table."}'
            ),
            "plan_review": '{"feasible": true, "concerns": [], "suggestions": []}',
            "execution_verification": (
                '{"task_completed": false, '
                '"discrepancies": ["Object did not move."], '
                '"confidence": 0.42}'
            ),
        }
    )
    service = VLMService(backend)
    plan = {"task_id": "task-1", "steps": []}
    trace = ExecutionTrace(trace_id="trace-vlm", task_id="task-1")

    trace.add_event(
        "vlm_scene",
        "scene_described",
        "success",
        payload=service.describe_scene(b"before", "inspect the scene").to_dict(),
    )
    trace.add_event(
        "vlm_plan_review",
        "plan_reviewed",
        "success",
        payload=service.review_plan(b"before", plan, "inspect the scene").to_dict(),
    )
    trace.add_event(
        "vlm_verification",
        "execution_verified",
        "warning",
        payload=service.verify_execution(b"before", b"after", "inspect the scene").to_dict(),
    )

    serialized = trace.to_dict()
    assert [event["stage"] for event in serialized["events"]] == [
        "vlm_scene",
        "vlm_plan_review",
        "vlm_verification",
    ]
    assert serialized["events"][0]["payload"]["raw_response"]["model_id"] == "test-vlm"
    assert serialized["events"][2]["payload"]["discrepancies"] == ["Object did not move."]
