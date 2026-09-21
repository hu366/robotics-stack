"""Call Gemini Robotics-ER with a video and optionally replay the result on Piper."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com"
LIVE_WS_ROOT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
STREAMING_MODEL = "gemini-robotics-er-2-streaming-preview"
LIVE_VIDEO_FPS = 1.0
DEFAULT_MODEL = os.getenv("GEMINI_ROBOTICS_ER_MODEL", STREAMING_MODEL)
DEFAULT_INSTRUCTION = (
    "根据视频抓住纸箱，用双臂从纸箱左右两侧夹持；夹爪只与纸箱外表面接触，"
    "不得穿过纸箱表面；将纸箱绕竖直轴旋转约90度，然后稳定保持。"
)


def build_er_prompt(instruction: str) -> str:
    task = instruction.strip()
    if not task:
        raise ValueError("instruction cannot be empty")
    return (
        "You are Gemini Robotics-ER 2. Analyze this robot manipulation video and "
        "return only JSON. The video is a head-mounted RGB camera recording a "
        "dual-arm Piper-H parallel-gripper demonstration.\n\n"
        "The required task is:\n"
        f"{task}\n\n"
        "Do not merely describe the demonstrated motion: propose the bimanual "
        "motion needed to accomplish this task from the video. In particular, "
        "the box pose at the first and last keyframes must show a clear rotation "
        "around the vertical axis, and both grippers should stay closed or in "
        "estimated contact during the rotation phase.\n\n"
        "Return this exact semantic shape:\n"
        "{\n"
        '  "task_summary": "short description of the requested box rotation",\n'
        '  "confidence": 0.0,\n'
        '  "waypoints": [\n'
        "    {\n"
        '      "t_sec": 0.0,\n'
        '      "left": {\n'
        '        "xyz_m": [x, y, z],\n'
        '        "quat_wxyz": [qw, qx, qy, qz],\n'
        '        "gripper_m": 0.035,\n'
        '        "confidence": 0.0\n'
        "      },\n"
        '      "right": {\n'
        '        "xyz_m": [x, y, z],\n'
        '        "quat_wxyz": [qw, qx, qy, qz],\n'
        '        "gripper_m": 0.035,\n'
        '        "confidence": 0.0\n'
        "      },\n"
        '      "object": {\n'
        '        "xyz_m": [x, y, z],\n'
        '        "quat_wxyz": [qw, qx, qy, qz],\n'
        '        "confidence": 0.0\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "notes": ["..."],\n'
        '  "limitations": ["..."]\n'
        "}\n\n"
        "Use camera-frame coordinates in meters, with x right, y down, z forward. "
        "Use 6 to 12 sparse keyframes and preserve both arms when visible. The "
        "Piper-H gripper opening is 0..0.07 m. Use object pose in the same camera "
        "frame and meters, with quaternion order wxyz. If metric depth, object "
        "orientation, or a hand is uncertain, mark confidence low and explain it "
        "in limitations. Do not output joint angles, a 16-DoF hand, or prose "
        "outside the JSON object. This is a kinematic waypoint proposal, not a "
        "claim of contact-safe control."
    )


ER_PROMPT = build_er_prompt(DEFAULT_INSTRUCTION)


class CloudCallError(RuntimeError):
    pass


def _json_request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CloudCallError(f"Google API HTTP {exc.code}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise CloudCallError(f"Google API connection failed: {exc.reason}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CloudCallError(f"Google API returned non-JSON: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise CloudCallError("Google API response was not a JSON object")
    return payload


def upload_video(video: Path, api_key: str, timeout_s: float) -> dict[str, Any]:
    """Upload through Google's resumable Files API protocol."""
    start_url = f"{GEMINI_API_ROOT}/upload/v1beta/files?key={api_key}"
    metadata = json.dumps(
        {"file": {"display_name": video.name}},
        separators=(",", ":"),
    ).encode("utf-8")
    start_request = urllib.request.Request(
        url=start_url,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(video.stat().st_size),
            "X-Goog-Upload-Header-Content-Type": "video/mp4",
        },
        data=metadata,
    )
    try:
        with urllib.request.urlopen(start_request, timeout=timeout_s) as response:
            upload_url = response.headers.get("X-Goog-Upload-URL")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CloudCallError(
            f"Google upload start failed with HTTP {exc.code}: {detail[:2000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CloudCallError(f"Google upload start failed: {exc.reason}") from exc
    if not upload_url:
        raise CloudCallError("Google upload start did not return X-Goog-Upload-URL")

    upload_request = urllib.request.Request(
        url=upload_url,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Length": str(video.stat().st_size),
            "Content-Type": "video/mp4",
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        data=video.read_bytes(),
    )
    try:
        with urllib.request.urlopen(upload_request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CloudCallError(f"Google upload failed with HTTP {exc.code}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise CloudCallError(f"Google upload failed: {exc.reason}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CloudCallError(f"Google upload returned non-JSON: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise CloudCallError("Google upload response was not a JSON object")
    return payload


def wait_for_file(file_payload: dict[str, Any], api_key: str, timeout_s: float) -> dict[str, Any]:
    file_info = file_payload.get("file", file_payload)
    if not isinstance(file_info, dict):
        raise CloudCallError("Upload response did not contain a file object")
    name = file_info.get("name")
    if not isinstance(name, str):
        raise CloudCallError("Upload response did not contain file.name")
    state = file_info.get("state")
    deadline = time.monotonic() + timeout_s
    while state in (None, "PROCESSING") and time.monotonic() < deadline:
        time.sleep(2.0)
        current = _json_request(
            f"{GEMINI_API_ROOT}/v1beta/{name}?key={api_key}",
            headers={"Accept": "application/json"},
            timeout_s=timeout_s,
        )
        file_info = current.get("file", current)
        if not isinstance(file_info, dict):
            raise CloudCallError("File status response did not contain a file object")
        state = file_info.get("state")
    if state != "ACTIVE":
        raise CloudCallError(f"Google file was not ready: state={state!r}")
    return file_info


def generate_content(
    model: str,
    file_info: dict[str, Any],
    api_key: str,
    timeout_s: float,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    file_uri = file_info.get("uri")
    mime_type = file_info.get("mimeType", "video/mp4")
    if not isinstance(file_uri, str):
        raise CloudCallError("Google file response did not contain file.uri")
    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                    {"text": build_er_prompt(instruction)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    return _json_request(
        f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent?key={api_key}",
        method="POST",
        body=json.dumps(request_body).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout_s=timeout_s,
    )


def generate_content_inline(
    model: str,
    video: Path,
    api_key: str,
    timeout_s: float,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    """Call generateContent with inline video bytes for regions blocking Files API."""
    encoded = base64.b64encode(video.read_bytes()).decode("ascii")
    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "video/mp4",
                            "data": encoded,
                        }
                    },
                    {"text": build_er_prompt(instruction)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    return _json_request(
        f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent?key={api_key}",
        method="POST",
        body=json.dumps(request_body).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout_s=timeout_s,
    )


def _sample_live_video(video: Path) -> tuple[float, list[tuple[int, bytes]]]:
    """Encode chronological JPEG frames at the Live API's one-FPS video limit."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise CloudCallError("opencv-python is required for Live API video streaming") from exc

    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frame_count <= 0:
            raise CloudCallError(f"cannot read frames from {video}")

        frames: list[tuple[int, bytes]] = []
        current_id = -1
        sample_number = 0
        while True:
            frame_id = int(round(sample_number * fps))
            if frame_id >= frame_count:
                break
            ok, frame = True, None
            while current_id < frame_id:
                ok, frame = capture.read()
                current_id += 1
            if not ok or frame is None:
                break
            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not encoded_ok:
                raise CloudCallError(f"cannot JPEG-encode frame {frame_id} from {video}")
            frames.append((frame_id, bytes(encoded)))
            sample_number += 1
        if not frames:
            raise CloudCallError(f"no JPEG frames could be encoded from {video}")
        return fps, frames
    finally:
        capture.release()


def _live_message_text(message: dict[str, Any]) -> str:
    """Extract text from one raw Live API server message."""
    server_content = message.get("serverContent")
    if not isinstance(server_content, dict):
        return ""
    model_turn = server_content.get("modelTurn")
    if not isinstance(model_turn, dict):
        return ""
    parts = model_turn.get("parts")
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part["text"])
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


async def _generate_streaming_content_async(
    model: str,
    video: Path,
    api_key: str,
    timeout_s: float,
    instruction: str,
) -> dict[str, Any]:
    """Send sampled video frames to Gemini Robotics-ER over the Live WebSocket."""
    try:
        import websockets
    except ModuleNotFoundError as exc:
        raise CloudCallError(
            "websockets is required for the streaming model; run `uv sync` first"
        ) from exc

    fps, frames = _sample_live_video(video)
    frame_times = [round(frame_id / fps, 3) for frame_id, _ in frames]
    system_prompt = build_er_prompt(instruction)
    trigger_prompt = (
        "All video frames have now been sent in chronological order at these timestamps "
        f"(seconds): {frame_times}. Analyze the complete sequence and return the requested "
        "JSON object now. Output the JSON object only, without Markdown fences, commentary, "
        "or a reasoning trace."
    )
    setup_message = {
        "setup": {
            "model": f"models/{model}",
            "generationConfig": {
                "responseModalities": ["TEXT"],
                "temperature": 0.0,
                "maxOutputTokens": 4096,
            },
            "systemInstruction": {"parts": [{"text": system_prompt}]},
        }
    }
    ws_url = f"{LIVE_WS_ROOT}?key={urllib.parse.quote(api_key, safe='')}"
    received: list[dict[str, Any]] = []
    response_parts: list[str] = []

    try:
        async with websockets.connect(
            ws_url,
            open_timeout=timeout_s,
            close_timeout=10.0,
            max_size=16 * 1024 * 1024,
        ) as websocket:
            await websocket.send(json.dumps(setup_message, ensure_ascii=False))
            setup_raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
            if isinstance(setup_raw, bytes):
                setup_raw = setup_raw.decode("utf-8")
            setup_reply = json.loads(setup_raw)
            if not isinstance(setup_reply, dict):
                raise CloudCallError("Live API setup response was not a JSON object")
            if isinstance(setup_reply.get("error"), dict):
                raise CloudCallError(
                    "Google Live API setup failed: "
                    + json.dumps(setup_reply["error"], ensure_ascii=False)
                )
            if "setupComplete" not in setup_reply:
                raise CloudCallError(
                    "Live API did not return setupComplete: "
                    + json.dumps(setup_reply, ensure_ascii=False)[:2000]
                )
            received.append(setup_reply)

            loop = asyncio.get_running_loop()
            next_frame_time = loop.time()
            for frame_index, (_, jpeg_bytes) in enumerate(frames):
                if frame_index:
                    next_frame_time += 1.0 / LIVE_VIDEO_FPS
                    delay = next_frame_time - loop.time()
                    if delay > 0.0:
                        await asyncio.sleep(delay)
                await websocket.send(
                    json.dumps(
                        {
                            "realtimeInput": {
                                "video": {
                                    "data": base64.b64encode(jpeg_bytes).decode("ascii"),
                                    "mimeType": "image/jpeg",
                                }
                            }
                        }
                    )
                )

            await websocket.send(
                json.dumps({"realtimeInput": {"text": trigger_prompt}}, ensure_ascii=False)
            )

            while True:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")
                message = json.loads(raw_message)
                if not isinstance(message, dict):
                    continue
                received.append(message)
                if isinstance(message.get("error"), dict):
                    raise CloudCallError(
                        "Google Live API failed: "
                        + json.dumps(message["error"], ensure_ascii=False)
                    )
                text = _live_message_text(message)
                if text:
                    response_parts.append(text)
                server_content = message.get("serverContent")
                if isinstance(server_content, dict) and server_content.get("turnComplete"):
                    break
    except CloudCallError:
        raise
    except Exception as exc:
        raise CloudCallError(f"Google Live API WebSocket failed: {exc}") from exc

    response_text_value = "".join(response_parts).strip()
    if not response_text_value:
        raise CloudCallError("Google Live API completed without a text response")
    return {
        "model": model,
        "input_mode": "live_websocket_jpeg_frames",
        "stream_fps": LIVE_VIDEO_FPS,
        "streamed_frame_times_sec": frame_times,
        "text": response_text_value,
        "messages": received,
    }


def generate_content_streaming(
    model: str,
    video: Path,
    api_key: str,
    timeout_s: float,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    """Synchronous wrapper for the asynchronous Live API WebSocket call."""
    return asyncio.run(
        _generate_streaming_content_async(model, video, api_key, timeout_s, instruction)
    )


def response_text(response: dict[str, Any]) -> str:
    direct_text = response.get("text")
    if isinstance(direct_text, str):
        return direct_text.strip()
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return ""
    return "\n".join(
        str(part["text"])
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ).strip()


def parse_json_text(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    if "```" in text:
        candidates.extend(segment.strip() for segment in text.split("```"))
    for candidate in candidates:
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("ER response did not contain a JSON object")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must have length {length}")
    return [_number(item, label) for item in value]


def _first(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _side_row(value: Any, label: str) -> list[float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    position = _first(value, ("xyz_m", "position_m", "position"))
    quaternion = _first(value, ("quat_wxyz", "quaternion_wxyz", "orientation_wxyz"))
    gripper = _first(value, ("gripper_m", "gripper", "opening_m"))
    xyz = _vector(position, 3, f"{label}.xyz_m")
    quat = _vector(quaternion, 4, f"{label}.quat_wxyz")
    norm = math.sqrt(sum(item * item for item in quat))
    if norm < 1e-9:
        raise ValueError(f"{label}.quat_wxyz cannot be zero")
    quat = [item / norm for item in quat]
    opening = min(0.07, max(0.0, _number(gripper, f"{label}.gripper_m")))
    return xyz + quat + [opening]


def _object_row(value: Any, label: str) -> tuple[list[float], float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    position = _first(value, ("xyz_m", "position_m", "position"))
    quaternion = _first(value, ("quat_wxyz", "quaternion_wxyz", "orientation_wxyz"))
    xyz = _vector(position, 3, f"{label}.xyz_m")
    quat = _vector(quaternion, 4, f"{label}.quat_wxyz")
    norm = math.sqrt(sum(item * item for item in quat))
    if norm < 1e-9:
        raise ValueError(f"{label}.quat_wxyz cannot be zero")
    confidence_value = value.get("confidence", 0.0)
    confidence = min(1.0, max(0.0, _number(confidence_value, f"{label}.confidence")))
    return xyz + [item / norm for item in quat], confidence


def waypoints_to_clip(
    payload: dict[str, Any],
    *,
    video: Path,
    fps: float,
    frame_count: int,
    model: str,
) -> dict[str, Any]:
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("ER response has no waypoints")
    by_side: dict[str, list[tuple[int, list[float]]]] = {"left": [], "right": []}
    object_entries: list[tuple[int, list[float], float]] = []
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, dict):
            raise ValueError(f"waypoints[{index}] must be an object")
        timestamp = _first(waypoint, ("t_sec", "time_sec", "timestamp_sec"))
        if timestamp is None and isinstance(waypoint.get("frame_id"), (int, float)):
            frame_id = int(waypoint["frame_id"])
        else:
            frame_id = int(round(_number(timestamp, f"waypoints[{index}].t_sec") * fps))
        frame_id = max(0, min(max(frame_count - 1, 0), frame_id))
        for side in by_side:
            value = waypoint.get(side)
            if value is not None:
                by_side[side].append((frame_id, _side_row(value, f"waypoints[{index}].{side}")))
        object_value = waypoint.get("object")
        if object_value is not None:
            pose, confidence = _object_row(object_value, f"waypoints[{index}].object")
            object_entries.append((frame_id, pose, confidence))
    if not any(by_side.values()):
        raise ValueError("ER response has no arm waypoints")
    retargeting = []
    for side, entries in by_side.items():
        if not entries:
            continue
        entries.sort(key=lambda item: item[0])
        deduped: dict[int, list[float]] = {frame_id: row for frame_id, row in entries}
        ordered = sorted(deduped.items())
        retargeting.append(
            {
                "hand_side": side,
                "frame_ids": [frame_id for frame_id, _ in ordered],
                "joint_trajectory": [row for _, row in ordered],
                "source": "gemini_robotics_er",
                "coordinate_frame": "camera",
            }
        )
    clip: dict[str, Any] = {
        "schema_version": "0.1.0-er2",
        "video_meta": {
            "path": str(video),
            "fps": fps,
            "num_frames": frame_count,
            "source": "gemini_robotics_er",
        },
        "retargeting": retargeting,
        "er": {
            "model": model,
            "task_summary": payload.get("task_summary"),
            "confidence": payload.get("confidence"),
            "notes": payload.get("notes", []),
            "limitations": payload.get("limitations", []),
        },
    }
    if object_entries:
        deduped_object: dict[int, tuple[list[float], float]] = {
            frame_id: (pose, confidence)
            for frame_id, pose, confidence in object_entries
        }
        ordered_object = sorted(deduped_object.items())
        clip["object_trajectory"] = {
            "frame_ids": [frame_id for frame_id, _ in ordered_object],
            "poses": [pose for _, (pose, _) in ordered_object],
            "confidence": [confidence for _, (_, confidence) in ordered_object],
            "source": "gemini_robotics_er",
            "coordinate_frame": "camera",
        }
    return clip


def _video_meta(video: Path) -> tuple[float, int]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required to inspect the input video") from exc
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if frame_count <= 0:
        raise ValueError(f"cannot read frames from {video}")
    return fps, frame_count


def _run_simulation(
    video: Path,
    clip: Path,
    out_dir: Path,
    *,
    grasp_constraint: str,
    require_task_success: bool,
) -> dict[str, Any]:
    simulation_dir = out_dir / "simulation"
    command = [
        sys.executable,
        str(ROOT / "apps" / "make_human_vs_robot.py"),
        "--video",
        str(video),
        "--clip",
        str(clip),
        "--output",
        str(simulation_dir),
        "--grasp-constraint",
        grasp_constraint,
    ]
    if require_task_success:
        command.append("--require-task-success")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "MuJoCo replay failed: " + (result.stderr.strip() or result.stdout.strip())
        )
    summary_path = simulation_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise RuntimeError("simulation summary was not a JSON object")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--inline-video",
        type=Path,
        help="Use this small MP4 as inline model input instead of the Files API.",
    )
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/robotics_er_video"))
    parser.add_argument("--fallback-clip", type=Path)
    parser.add_argument(
        "--grasp-constraint",
        choices=("none", "contact_weld"),
        default="none",
        help="Pure contact dynamics or a weld activated after dual-pad contact.",
    )
    parser.add_argument(
        "--require-task-success",
        action="store_true",
        help="Return failure when the simulation does not meet grasp and pose thresholds.",
    )
    parser.add_argument("--no-simulate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fps, frame_count = _video_meta(video)
    model_video = args.inline_video.expanduser().resolve() if args.inline_video else video
    is_streaming_model = args.model == STREAMING_MODEL
    request_manifest = {
        "video": str(video),
        "model_video": str(model_video),
        "model": args.model,
        "instruction": args.instruction,
        "fps": fps,
        "frame_count": frame_count,
        "prompt": build_er_prompt(args.instruction),
        "api": (
            "Gemini Live API WebSocket with 1 FPS JPEG frames"
            if is_streaming_model
            else "Gemini generateContent with inline video"
            if args.inline_video
            else "Gemini Files API + generateContent"
        ),
    }
    (out_dir / "request.json").write_text(
        json.dumps(request_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {"status": "not_started", **request_manifest}
    clip_path: Path | None = None
    if args.dry_run:
        manifest["status"] = "dry_run"
    elif not args.api_key:
        manifest["status"] = "missing_gemini_api_key"
        manifest["message"] = "Set GEMINI_API_KEY or pass --api-key to call Google."
    else:
        try:
            if not model_video.exists():
                raise SystemExit(f"model input video not found: {model_video}")
            if is_streaming_model:
                response = generate_content_streaming(
                    args.model,
                    model_video,
                    args.api_key,
                    args.timeout,
                    instruction=args.instruction,
                )
                file_info = {}
            elif args.inline_video:
                response = generate_content_inline(
                    args.model,
                    model_video,
                    args.api_key,
                    args.timeout,
                    instruction=args.instruction,
                )
                file_info = {}
            else:
                uploaded = upload_video(video, args.api_key, args.timeout)
                file_info = wait_for_file(uploaded, args.api_key, args.timeout)
                response = generate_content(
                    args.model,
                    file_info,
                    args.api_key,
                    args.timeout,
                    instruction=args.instruction,
                )
            (out_dir / "er_response.json").write_text(
                json.dumps(response, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
            )
            payload = parse_json_text(response_text(response))
            clip = waypoints_to_clip(
                payload,
                video=video,
                fps=fps,
                frame_count=frame_count,
                model=args.model,
            )
            clip_path = out_dir / "er_clip.json"
            clip_path.write_text(
                json.dumps(clip, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
            )
            manifest["status"] = "cloud_success"
            if file_info:
                manifest["file_name"] = file_info.get("name")
                manifest["cloud_uri"] = file_info.get("uri")
            manifest["model_input_mode"] = response.get(
                "input_mode",
                "inline_video" if args.inline_video else "files_api",
            )
            manifest["clip"] = str(clip_path)
        except (CloudCallError, ValueError, OSError) as exc:
            manifest["status"] = "cloud_failed"
            manifest["error"] = str(exc)

    if clip_path is None and args.fallback_clip:
        fallback = args.fallback_clip.expanduser().resolve()
        if not fallback.exists():
            raise SystemExit(f"fallback clip not found: {fallback}")
        clip_path = out_dir / "fallback_clip.json"
        shutil.copy2(fallback, clip_path)
        manifest["fallback_clip"] = str(fallback)
        manifest["simulation_source"] = "existing clip; no ER trajectory"

    if clip_path is not None and not args.no_simulate:
        try:
            manifest["simulation"] = _run_simulation(
                video,
                clip_path,
                out_dir,
                grasp_constraint=args.grasp_constraint,
                require_task_success=args.require_task_success,
            )
            if not bool(manifest["simulation"].get("task_success", False)):
                manifest["simulation_status"] = "failed"
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            manifest["simulation_error"] = str(exc)
            manifest["simulation_status"] = "error"
    (out_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    simulation_ok = manifest.get("simulation_status", "ok") == "ok"
    return 0 if manifest["status"] in ("cloud_success", "dry_run") and simulation_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
