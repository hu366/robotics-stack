from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from interfaces.execution_trace import ExecutionTrace
from interfaces.skill_spec import SkillSpec
from interfaces.world_state import ObjectState, WorldState
from modules.control import PlanExecutor, SymbolicControlBackend
from modules.grounding import SceneGrounder
from modules.planner import PlanStep, TaskPlanner
from modules.task_parser import TaskParser
from modules.world_model import WorldModelStore

ROOT = Path(__file__).resolve().parents[1]


def test_task_parser_extracts_structured_place_ir() -> None:
    task = TaskParser().parse("把瓶子放到托盘上")

    assert task.goal == "place_object"
    assert task.action == "place"
    assert task.target_object == "瓶子"
    assert task.target_location == "托盘"
    assert task.spatial_relation == "on"
    assert [argument.role for argument in task.arguments] == [
        "target_object",
        "target_location",
    ]
    assert task.preconditions == [
        "target_object_identified",
        "target_location_identified",
    ]
    assert task.postconditions == [
        "object_transferred",
        "target_relation_satisfied",
    ]
    assert [step.action for step in task.substeps] == ["locate", "grasp", "place"]


def test_task_parser_supports_in_relation() -> None:
    task = TaskParser().parse("把杯子放进盒子里")

    assert task.goal == "place_object"
    assert task.action == "place"
    assert task.target_object == "杯子"
    assert task.target_location == "盒子"
    assert task.spatial_relation == "in"


def test_task_parser_supports_move_to_relation() -> None:
    task = TaskParser().parse("移动瓶子到托盘")

    assert task.goal == "place_object"
    assert task.action == "move"
    assert task.target_object == "瓶子"
    assert task.target_location == "托盘"
    assert task.spatial_relation == "to"


def test_task_parser_extracts_open_object() -> None:
    task = TaskParser().parse("打开柜门")

    assert task.goal == "open_object"
    assert task.action == "open"
    assert task.target_object == "柜门"
    assert task.target_location is None
    assert task.preconditions == ["target_object_identified"]
    assert task.postconditions == ["object_opened"]
    assert [step.action for step in task.substeps] == ["locate", "open"]


def test_task_parser_falls_back_to_inspect_scene() -> None:
    task = TaskParser().parse("检查场景")

    assert task.goal == "inspect_scene"
    assert task.action == "inspect"
    assert task.target_object == "场景"
    assert task.target_location is None
    assert task.preconditions == ["scene_accessible"]
    assert task.postconditions == ["scene_observed"]
    assert [step.action for step in task.substeps] == ["inspect"]


def test_planner_builds_three_step_place_plan() -> None:
    task = TaskParser().parse("把瓶子放到托盘上")
    world_state = SceneGrounder().ground(task)
    plan = TaskPlanner().build_plan(task, world_state)

    assert [step.skill.name for step in plan.steps] == [
        "locate_object",
        "grasp_object",
        "place_object",
    ]
    assert plan.steps[-1].parameters["spatial_relation"] == "on"


def test_planner_degrades_open_task_to_inspect_scene() -> None:
    task = TaskParser().parse("打开柜门")
    world_state = SceneGrounder().ground(task)
    plan = TaskPlanner().build_plan(task, world_state)

    assert [step.skill.name for step in plan.steps] == ["inspect_scene"]
    assert plan.steps[0].parameters["requested_goal"] == "open_object"
    assert plan.steps[0].parameters["requested_action"] == "open"


def test_executor_closes_loop_for_place_plan() -> None:
    task = TaskParser().parse("把瓶子放到托盘上")
    world_state = SceneGrounder().ground(task)
    plan = TaskPlanner().build_plan(task, world_state)
    world_model = WorldModelStore()
    world_model.update(world_state)
    trace = ExecutionTrace(trace_id="trace-place", task_id=task.task_id)

    result = PlanExecutor(world_model=world_model).execute(plan, trace)

    assert result.success
    assert result.completed_steps == 3
    bottle = _require_object(result.final_world_state, "瓶子")
    tray = _require_object(result.final_world_state, "托盘")
    assert bottle.pose == tray.pose
    assert "on:托盘" in bottle.relations
    assert result.final_world_state.robot_mode == "ready"
    assert any(event.message == "step_3_feedback_observed" for event in trace.events)
    assert trace.events[-1].message == "execution_completed"


def test_executor_stops_on_missing_target_object() -> None:
    task = TaskParser().parse("把瓶子放到托盘上")
    plan = TaskPlanner().build_plan(task, WorldState(scene_id="scene-plan"))
    world_model = WorldModelStore()
    world_model.update(
        WorldState(
            scene_id="scene-missing-object",
            objects=[_make_object("托盘", [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
            robot_mode="ready",
        )
    )
    trace = ExecutionTrace(trace_id="trace-missing-object", task_id=task.task_id)

    result = PlanExecutor(world_model=world_model).execute(plan, trace)

    assert not result.success
    assert result.completed_steps == 0
    assert result.failed_step_index == 1
    assert result.failure_code == "object_not_found"
    assert any(event.message == "step_1_failed" for event in trace.events)
    assert trace.events[-1].message == "execution_failed"


def test_symbolic_backend_place_reports_missing_target_location() -> None:
    backend = SymbolicControlBackend()
    feedback = backend.execute_step(
        _build_step(
            "place_object",
            {"target_object": "瓶子", "target_location": "托盘"},
        ),
        WorldState(
            scene_id="scene-missing-location",
            objects=[
                _make_object(
                    "瓶子",
                    [0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                    relations=["held_by:gripper"],
                )
            ],
            robot_mode="holding",
        ),
        step_index=1,
    )

    assert not feedback.success
    assert feedback.failure_code == "target_location_not_found"


def test_symbolic_backend_place_requires_grasp() -> None:
    backend = SymbolicControlBackend()
    feedback = backend.execute_step(
        _build_step(
            "place_object",
            {"target_object": "瓶子", "target_location": "托盘"},
        ),
        WorldState(
            scene_id="scene-not-grasped",
            objects=[
                _make_object("瓶子", [0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
                _make_object("托盘", [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ],
            robot_mode="ready",
        ),
        step_index=1,
    )

    assert not feedback.success
    assert feedback.failure_code == "object_not_grasped"


def test_symbolic_backend_inspect_scene_is_non_mutating() -> None:
    backend = SymbolicControlBackend()
    world_state = WorldState(
        scene_id="scene-inspect",
        objects=[_make_object("瓶子", [0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])],
        robot_mode="ready",
    )

    feedback = backend.execute_step(
        _build_step("inspect_scene", {"scene_id": world_state.scene_id}),
        world_state,
        step_index=1,
    )

    assert feedback.success
    assert feedback.failure_code is None
    assert feedback.observed_world_state.to_dict() == world_state.to_dict()


def test_run_task_writes_trace() -> None:
    temp_dir = ROOT / ".tmp-tests"
    temp_dir.mkdir(exist_ok=True)
    trace_path = temp_dir / "trace.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps" / "run_task.py"),
            "把瓶子放到托盘上",
            "--trace-out",
            str(trace_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    task_payload = data["events"][0]["payload"]
    assert data["events"][0]["stage"] == "task_parser"
    assert task_payload["action"] == "place"
    assert task_payload["target_object"] == "瓶子"
    assert task_payload["target_location"] == "托盘"
    assert task_payload["spatial_relation"] == "on"
    assert any(event["message"] == "step_1_feedback_observed" for event in data["events"])
    assert data["events"][-1]["message"] == "execution_completed"
    assert "final_world_state" in data["events"][-1]["payload"]
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


def _build_step(skill_name: str, parameters: dict[str, str]) -> PlanStep:
    return PlanStep(
        skill=SkillSpec(name=skill_name),
        parameters=parameters,
    )


def _make_object(
    label: str,
    pose: list[float],
    relations: list[str] | None = None,
) -> ObjectState:
    return ObjectState(
        object_id="-".join(label.lower().split()),
        label=label,
        pose=pose,
        relations=[] if relations is None else list(relations),
    )


def _require_object(world_state: WorldState, label: str) -> ObjectState:
    for obj in world_state.objects:
        if obj.label == label:
            return obj
    raise AssertionError(f"Object not found: {label}")
