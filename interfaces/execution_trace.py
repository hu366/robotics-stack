from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    stage: str
    message: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionTrace:
    trace_id: str
    task_id: str
    events: list[TraceEvent] = field(default_factory=list)

    def add_event(
        self,
        stage: str,
        message: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                stage=stage,
                message=message,
                status=status,
                payload=payload or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
        }
