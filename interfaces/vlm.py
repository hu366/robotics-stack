from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class VLMQueryContext:
    stage: str
    task_instruction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VLMResponse:
    text: str
    confidence: float | None
    model_id: str
    latency_ms: int
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLMBackend(Protocol):
    def query(
        self,
        images: list[bytes],
        prompt: str,
        context: VLMQueryContext | None = None,
    ) -> VLMResponse: ...


@dataclass(slots=True)
class SceneDescription:
    objects_described: list[str] = field(default_factory=list)
    spatial_summary: str = ""
    raw_response: VLMResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects_described": list(self.objects_described),
            "spatial_summary": self.spatial_summary,
            "raw_response": None if self.raw_response is None else self.raw_response.to_dict(),
        }


@dataclass(slots=True)
class PlanFeasibility:
    feasible: bool
    concerns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw_response: VLMResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "concerns": list(self.concerns),
            "suggestions": list(self.suggestions),
            "raw_response": None if self.raw_response is None else self.raw_response.to_dict(),
        }


@dataclass(slots=True)
class ExecutionVerification:
    task_completed: bool
    discrepancies: list[str] = field(default_factory=list)
    confidence: float | None = None
    raw_response: VLMResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_completed": self.task_completed,
            "discrepancies": list(self.discrepancies),
            "confidence": self.confidence,
            "raw_response": None if self.raw_response is None else self.raw_response.to_dict(),
        }
