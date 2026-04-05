from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_model(mujoco: Any, scene_path: Path) -> Any:
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except ValueError:
        model_xml = scene_path.read_text(encoding="utf-8")
        assets: dict[str, bytes] = {}
        for file_path in scene_path.parent.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(scene_path.parent).as_posix()
                assets[rel] = file_path.read_bytes()
        return mujoco.MjModel.from_xml_string(model_xml, assets=assets)


def _depth_to_meters(model: Any, depth_buffer: Any) -> Any:
    # MuJoCo depth buffer is non-linear in [0,1]; convert to metric depth.
    near = model.vis.map.znear * model.stat.extent
    far = model.vis.map.zfar * model.stat.extent
    return near / (1.0 - depth_buffer * (1.0 - near / far))


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(path)


def _save_depth_vis_png(path: Path, depth_m: np.ndarray) -> None:
    from PIL import Image

    valid = np.isfinite(depth_m)
    if not np.any(valid):
        vis = np.zeros_like(depth_m, dtype=np.uint8)
    else:
        dmin = float(np.min(depth_m[valid]))
        dmax = float(np.max(depth_m[valid]))
        if dmax - dmin < 1e-9:
            vis = np.zeros_like(depth_m, dtype=np.uint8)
        else:
            norm = (depth_m - dmin) / (dmax - dmin)
            vis = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture RGB + depth from a MuJoCo scene camera.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("sim/assets/franka_emika_panda/scene.xml"),
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--camera",
        default="wrist_rgbd",
        help="Camera name from scene XML.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="RGBD image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="RGBD image height.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=200,
        help="Simulation steps before capture.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/rgbd"),
        help="Directory to save outputs.",
    )
    args = parser.parse_args()

    import mujoco  # type: ignore[import-untyped]

    scene_path = args.scene if args.scene.is_absolute() else ROOT / args.scene
    model = _load_model(mujoco, scene_path)
    data = mujoco.MjData(model)

    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data)

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        renderer.update_scene(data, camera=args.camera)
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=args.camera)
        depth_buf = renderer.render()
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    depth_m = _depth_to_meters(model, depth_buf)

    rgb_path = out_dir / "rgb.png"
    depth_npy_path = out_dir / "depth.npy"
    depth_vis_path = out_dir / "depth_vis.png"

    _save_rgb_png(rgb_path, rgb)
    np.save(depth_npy_path, depth_m)
    _save_depth_vis_png(depth_vis_path, depth_m)

    print(f"scene={scene_path}")
    print(f"camera={args.camera}")
    print(f"rgb={rgb_path}")
    print(f"depth_npy={depth_npy_path}")
    print(f"depth_vis={depth_vis_path}")


if __name__ == "__main__":
    main()
