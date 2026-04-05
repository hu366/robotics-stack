from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from modules.grounding import SceneGrounder
from modules.planner import TaskPlanner
from modules.task_parser import TaskParser

ROOT = Path(__file__).resolve().parents[1]


def test_task_parser_extracts_goal_and_entities() -> None:
    task = TaskParser().parse("place the bottle on the tray")
    assert task.goal == "place_object"
    assert task.target_object == "the bottle"
    assert task.target_location == "the tray"


def test_planner_builds_three_step_place_plan() -> None:
    task = TaskParser().parse("put the cup on the shelf")
    world_state = SceneGrounder().ground(task)
    plan = TaskPlanner().build_plan(task, world_state)
    assert [step.skill.name for step in plan.steps] == [
        "locate_object",
        "grasp_object",
        "place_object",
    ]


def test_run_task_writes_trace() -> None:
    temp_dir = ROOT / ".tmp-tests"
    temp_dir.mkdir(exist_ok=True)
    trace_path = temp_dir / "trace.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps" / "run_task.py"),
            "place the bottle on the tray",
            "--trace-out",
            str(trace_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["events"][0]["stage"] == "task_parser"
    assert any(event["message"] == "step_3_completed" for event in data["events"])
    assert '"trace_id"' in result.stdout
    shutil.rmtree(temp_dir)


def test_run_benchmark_outputs_report() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps" / "run_benchmark.py"),
            "--cases",
            str(ROOT / "eval" / "benchmarks" / "tabletop_v0.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert len(report) == 2
    assert all(entry["success"] for entry in report)
