"""Convert ego2dex virtual-gripper ``T_ee + g`` through UMI pose math.

Reads ``clip.json`` / ``clip_full.json`` ``retargeting[*].joint_trajectory``
(``x,y,z,qw,qx,qy,qz,gripper``) and writes pose10d+gripper actions. Does not
reimplement ``convert_pose_mat_rep``.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.umi_pose.adapter import POSE10D_GRIPPER_NAMES, convert_ee_trajectory
from modules.umi_pose.hacky_calib import T_base_from_cam


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("retargeting") or []
    if not results:
        raise SystemExit(f"no clip.retargeting in {path}")
    out: list[dict[str, Any]] = []
    for item in results:
        names = list(item.get("joint_names") or [])
        traj = np.asarray(item.get("joint_trajectory"), dtype=np.float64)
        if traj.ndim != 2 or traj.shape[-1] != 8:
            raise SystemExit(
                f"expected (T, 8) virtual-gripper trajectory, got {traj.shape} "
                f"joint_names={names}"
            )
        out.append(
            {
                "robot": item.get("robot"),
                "hand_side": item.get("hand_side"),
                "optimizer": item.get("optimizer"),
                "joint_names": names,
                "traj": traj,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ego2dex T_ee+g → UMI pose10d+gripper (vendored convert_pose_mat_rep)."
    )
    parser.add_argument("--clip", type=Path, required=True, help="clip.json / clip_full.json")
    parser.add_argument(
        "--pose-rep",
        default="relative",
        choices=("abs", "rel", "relative", "delta"),
        help="UMI convert_pose_mat_rep mode. 'relative' is body-frame inv(g0)@g.",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Skip the hacky T_base←cam left-multiply (keep camera-frame abs).",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    converted = []
    for item in _load_trajectories(args.clip):
        result = convert_ee_trajectory(
            item["traj"],
            pose_rep=args.pose_rep,
            apply_hacky_calib=not args.no_calib,
            T_base_cam=T_base_from_cam(),
        )
        converted.append(
            {
                "robot": item["robot"],
                "hand_side": item["hand_side"],
                "optimizer": item["optimizer"],
                "source_joint_names": item["joint_names"],
                "pose_rep": args.pose_rep,
                "hacky_calib": not args.no_calib,
                "joint_names": result["joint_names"],
                "action": result["action"].tolist(),
                "base_pose_mat": result["base_pose_mat"].tolist(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"actions": converted}, indent=2), encoding="utf-8")
    first = np.asarray(converted[0]["action"])
    print(
        f"wrote {args.output} n_hands={len(converted)} "
        f"shape={tuple(first.shape)} names={list(POSE10D_GRIPPER_NAMES)}"
    )


if __name__ == "__main__":
    main()
