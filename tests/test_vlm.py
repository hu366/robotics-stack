from __future__ import annotations

import json
from typing import Any

from interfaces.execution_trace import ExecutionTrace
from interfaces.skill_spec import SkillSpec
from interfaces.task_spec import TaskArgument, TaskSpec, TaskStepSpec
from interfaces.vlm import VLMQueryContext, VLMResponse
from modules.planner import ExecutionPlan, PlanStep
from modules.vlm.backends import OpenAIVLMBackend
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

    result = VLMService(backend).describe_scene(b"image-bytes", "把瓶子放到托盘上")

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
        "把瓶子放到托盘上",
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
        instruction="把瓶子放到托盘上",
        goal="place_object",
        action="place",
        arguments=[
            TaskArgument(role="target_object", text="瓶子"),
            TaskArgument(role="target_location", text="托盘"),
        ],
        spatial_relation="on",
        preconditions=["target_object_identified", "target_location_identified"],
        postconditions=["object_transferred", "target_relation_satisfied"],
        constraints=[
            "maintain_collision_safety",
            "keep_traceability",
            "respect_target_relation",
        ],
        substeps=[
            TaskStepSpec(
                step_id="step_1",
                action="locate",
                description="Resolve task entities in the scene.",
                required_arguments=["target_object"],
                success_criteria=["object_pose_resolved"],
            )
        ],
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
        payload=service.describe_scene(b"before", "检查场景").to_dict(),
    )
    trace.add_event(
        "vlm_plan_review",
        "plan_reviewed",
        "success",
        payload=service.review_plan(b"before", plan, "检查场景").to_dict(),
    )
    trace.add_event(
        "vlm_verification",
        "execution_verified",
        "warning",
        payload=service.verify_execution(b"before", b"after", "检查场景").to_dict(),
    )

    serialized = trace.to_dict()
    assert [event["stage"] for event in serialized["events"]] == [
        "vlm_scene",
        "vlm_plan_review",
        "vlm_verification",
    ]
    assert serialized["events"][0]["payload"]["raw_response"]["model_id"] == "test-vlm"
    assert serialized["events"][2]["payload"]["discrepancies"] == ["Object did not move."]


def test_openai_backend_sends_compatibility_headers(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        status = 200

        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-5.4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"feasible": true}'},
                    "finish_reason": "stop",
                }
            ],
        }
        return FakeResponse(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OpenAIVLMBackend(
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://codeapi.icu/v1",
    )

    response = backend.query([b"fake-image"], "Return JSON.", context=None)

    assert response.model_id == "gpt-5.4"
    assert captured["url"] == "https://codeapi.icu/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["headers"]["user-agent"] == "curl/8.7.1"
    assert captured["headers"]["accept"] == "application/json"
    assert captured["body"]["model"] == "gpt-5.4"
    assert captured["body"]["messages"][0]["content"][1]["type"] == "image_url"
