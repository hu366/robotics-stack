from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any

from interfaces.vlm import VLMQueryContext, VLMResponse


class _ChatCompletionsBackend:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def query(
        self,
        images: list[bytes],
        prompt: str,
        context: VLMQueryContext | None = None,
    ) -> VLMResponse:
        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_content(images, prompt),
                }
            ],
            "temperature": self._context_number(context, "temperature", 0.0),
            "max_tokens": int(self._context_number(context, "max_tokens", 400.0)),
        }
        started = time.perf_counter()
        response_payload = self._post_json("/chat/completions", request_body)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return VLMResponse(
            text=_extract_message_text(response_payload),
            confidence=_extract_confidence(response_payload),
            model_id=str(response_payload.get("model", self.model)),
            latency_ms=latency_ms,
            raw=response_payload,
        )

    def _build_content(self, images: list[bytes], prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_bytes in images:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        return content

    def _context_number(
        self,
        context: VLMQueryContext | None,
        key: str,
        default: float,
    ) -> float:
        if context is None:
            return default
        value = context.metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return default

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                parsed = json.loads(response.read().decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError("VLM response must be a JSON object")
                return parsed
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM request failed: {exc.reason}") from exc


class OpenAIVLMBackend(_ChatCompletionsBackend):
    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
    ) -> None:
        super().__init__(model=model, base_url=base_url, api_key=api_key, timeout_s=timeout_s)


class LocalVLMBackend(_ChatCompletionsBackend):
    def __init__(
        self,
        *,
        model: str = "llava",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        super().__init__(model=model, base_url=base_url, api_key=api_key, timeout_s=timeout_s)


class MockVLMBackend:
    def __init__(self, model: str = "mock-vlm") -> None:
        self.model = model

    def query(
        self,
        images: list[bytes],
        prompt: str,
        context: VLMQueryContext | None = None,
    ) -> VLMResponse:
        stage = "" if context is None else context.stage
        if stage == "scene_description":
            payload: dict[str, Any] = {
                "objects_described": ["robot arm", "tabletop", "workspace"],
                "spatial_summary": "A robot arm is positioned above a tabletop workspace.",
            }
        elif stage == "plan_review":
            payload = {
                "feasible": True,
                "concerns": [],
                "suggestions": ["Verify the target object is visible before grasping."],
            }
        elif stage == "execution_verification":
            payload = {
                "task_completed": len(images) >= 2,
                "discrepancies": [],
                "confidence": 0.88,
            }
        else:
            payload = {"message": "unrecognized stage", "prompt": prompt}

        return VLMResponse(
            text=json.dumps(payload),
            confidence=0.88,
            model_id=self.model,
            latency_ms=1,
            raw={"mock": True, "stage": stage, "payload": payload},
        )


def _extract_message_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return ""


def _extract_confidence(response_payload: dict[str, Any]) -> float | None:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    logprobs = first_choice.get("logprobs")
    if isinstance(logprobs, dict):
        return 1.0
    return None
