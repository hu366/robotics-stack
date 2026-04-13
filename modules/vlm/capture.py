from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image


def capture_mujoco_rgb(
    model: Any,
    data: Any,
    camera: str | int,
    width: int,
    height: int,
) -> bytes:
    import mujoco

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
    finally:
        renderer.close()

    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()
