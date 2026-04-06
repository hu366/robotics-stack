from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_model(mujoco: Any, scene_path: Path) -> Any:
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except ValueError:
        # Fallback for non-ASCII paths and include/asset resolution.
        model_xml = scene_path.read_text(encoding="utf-8")
        assets: dict[str, bytes] = {}
        for file_path in scene_path.parent.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(scene_path.parent).as_posix()
                assets[rel] = file_path.read_bytes()
        return mujoco.MjModel.from_xml_string(model_xml, assets=assets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny MuJoCo smoke simulation.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("sim/scenes/mujoco_minimal.xml"),
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps.",
    )
    args = parser.parse_args()

    import mujoco

    scene_path = args.scene if args.scene.is_absolute() else ROOT / args.scene
    model = _load_model(mujoco, scene_path)
    data = mujoco.MjData(model)

    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    print(f"scene={scene_path}")
    print(f"steps={args.steps}")
    print(f"time={data.time:.6f}")
    print(f"qpos={data.qpos.tolist()}")


if __name__ == "__main__":
    main()
