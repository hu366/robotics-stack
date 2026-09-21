from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from apps.make_human_vs_robot import (
    _constrain_bimanual_grasp,
    _contact_metrics,
    _interpolate_ee_trajectory,
    _interpolate_object_trajectory,
    _is_stable_grasp,
    contact_grasp_geometry,
)
from apps.replay_piper_ee import PIPER_XML, directional_support_radius, scene_xml
from apps.run_robotics_er_video import (
    STREAMING_MODEL,
    build_er_prompt,
    parse_json_text,
    response_text,
    waypoints_to_clip,
)


def test_parse_json_text_accepts_markdown_fence() -> None:
    assert parse_json_text('```json\n{"waypoints": []}\n```') == {"waypoints": []}


def test_build_er_prompt_requires_box_rotation_instruction() -> None:
    prompt = build_er_prompt("抓住纸箱并绕竖直轴旋转90度")
    assert "抓住纸箱并绕竖直轴旋转90度" in prompt
    assert '"object"' in prompt
    assert "first and last keyframes" in prompt


def test_streaming_model_response_uses_direct_live_text() -> None:
    assert STREAMING_MODEL == "gemini-robotics-er-2-streaming-preview"
    response = {"text": '{"waypoints": [{"t_sec": 0.0}]}' }
    assert response_text(response) == response["text"]


def test_interpolate_object_trajectory_slerps_quaternion() -> None:
    poses = np.array(
        [
            [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.4, 2**-0.5, 0.0, 2**-0.5, 0.0],
        ]
    )
    dense = _interpolate_object_trajectory(np.array([0, 10]), poses, 11)
    assert dense.shape == (11, 7)
    assert np.allclose(dense[5, :3], [0.1, 0.0, 0.4])
    assert np.isclose(np.linalg.norm(dense[5, 3:]), 1.0)
    assert np.isclose(dense[5, 3], np.cos(np.pi / 8), atol=1e-6)
    assert np.isclose(dense[5, 5], np.sin(np.pi / 8), atol=1e-6)


def test_interpolate_ee_trajectory_removes_waypoint_steps() -> None:
    traj = np.array(
        [
            [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.02],
            [0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.04],
        ]
    )
    dense = _interpolate_ee_trajectory(np.array([0, 10]), traj, 11)
    assert np.allclose(dense[5, :3], [0.05, 0.0, 0.4])
    assert np.isclose(dense[5, 7], 0.03)


def test_constrain_bimanual_grasp_uses_one_object_center() -> None:
    left = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    right = left.copy()
    left[:, :3, 3] = [0.7, 0.3, 0.2]
    right[:, :3, 3] = [0.7, -0.3, 0.2]
    object_world = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    object_world[:, :3, 3] = [0.5, 0.0, 0.2]
    constrained = _constrain_bimanual_grasp(
        {"left": left, "right": right}, object_world, separation=0.22
    )
    separation = np.linalg.norm(
        constrained["left"][:, :3, 3] - constrained["right"][:, :3, 3], axis=1
    )
    midpoint = 0.5 * (
        constrained["left"][:, :3, 3] + constrained["right"][:, :3, 3]
    )
    assert np.allclose(separation, 0.22)
    assert np.allclose(midpoint, object_world[:, :3, 3])


def test_directional_support_radius_follows_object_rotation() -> None:
    object_rotation = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    assert np.isclose(
        directional_support_radius((0.09, 0.06, 0.10), object_rotation, [0.0, 1.0, 0.0]),
        0.09,
    )


def test_contact_grasp_geometry_uses_reachable_pad_support() -> None:
    geometry = contact_grasp_geometry(
        np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.array([0.0, 1.0, 0.0]),
        {"left": np.eye(3), "right": np.eye(3)},
    )
    assert np.isclose(geometry["object_support_m"], 0.09)
    assert np.isclose(geometry["pad_support_m"], 0.006)
    assert np.isclose(geometry["separation_m"], 0.1914)
    assert np.isclose(geometry["pad_interference_m"]["left"], 0.0003)
    assert np.isclose(geometry["pad_interference_m"]["right"], 0.0003)


def test_stable_grasp_rejects_floor_contact_and_excessive_force() -> None:
    good = {
        "physics_steps": 100,
        "both_contact_ratio": 0.95,
        "min_object_z_m": 0.10,
        "max_side_force_n": 30.0,
        "floor_contact_physics_steps": 0,
    }
    assert _is_stable_grasp(**good)
    assert not _is_stable_grasp(**{**good, "floor_contact_physics_steps": 1})
    assert not _is_stable_grasp(**{**good, "max_side_force_n": 51.0})
    assert not _is_stable_grasp(**{**good, "max_penetration_m": 0.0006})


def test_kinematic_scene_keeps_arm_box_collisions_enabled() -> None:
    xml = scene_xml(
        PIPER_XML,
        bimanual=True,
        ego_camera=True,
        contact_grasp=True,
        exclude_arm_box_contacts=False,
    )
    assert "<exclude" not in xml


def test_contact_scene_can_keep_dynamics_exclusions() -> None:
    xml = scene_xml(
        PIPER_XML,
        bimanual=True,
        ego_camera=True,
        contact_grasp=True,
        exclude_arm_box_contacts=True,
    )
    assert xml.count("<exclude") == 18


def test_contact_metrics_include_vendor_arm_mesh_penetration() -> None:
    contacts = [
        SimpleNamespace(geom=np.array([2, 3]), dist=-0.001, pos=np.zeros(3)),
        SimpleNamespace(geom=np.array([2, 4]), dist=-0.004, pos=np.ones(3)),
    ]
    model = SimpleNamespace(geom_bodyid=np.array([0, 0, 0, 30, 40]))
    data = SimpleNamespace(ncon=len(contacts), contact=contacts)
    geom_names = {2: "demo_box_collision", 3: "left_grasp_pad_geom", 4: "left_link5_mesh"}
    body_names = {30: "left_grasp_pad", 40: "left_link5"}
    mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_GEOM="geom", mjOBJ_BODY="body"),
        mj_contactForce=lambda _model, _data, _index, force: force.fill(0.0),
        mj_id2name=lambda _model, kind, object_id: (
            geom_names.get(object_id) if kind == "geom" else body_names.get(object_id)
        ),
    )

    metrics = _contact_metrics(
        mujoco,
        model,
        data,
        box_geom_id=2,
        floor_geom_id=1,
        pad_geom_ids={"left": 3, "right": 5},
        arm_geom_sides={4: "left"},
    )

    assert metrics["left_contacts"] == 1
    assert metrics["left_arm_contacts"] == 1
    assert np.isclose(metrics["left_penetration_m"], 0.001)
    assert np.isclose(metrics["left_arm_penetration_m"], 0.004)
    assert np.isclose(metrics["max_penetration_m"], 0.004)
    assert metrics["arm_contact_names"] == ["left_link5_mesh"]


def test_waypoints_to_clip_normalizes_piper_parallel_gripper_rows() -> None:
    payload = {
        "task_summary": "move an object",
        "confidence": 0.7,
        "waypoints": [
            {
                "t_sec": 0.0,
                "left": {
                    "xyz_m": [0.1, 0.2, 0.3],
                    "quat_wxyz": [2.0, 0.0, 0.0, 0.0],
                    "gripper_m": 0.09,
                },
                "right": {
                    "xyz_m": [0.2, 0.2, 0.3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "gripper_m": -0.01,
                },
                "object": {
                    "xyz_m": [0.15, 0.2, 0.3],
                    "quat_wxyz": [2.0, 0.0, 0.0, 0.0],
                    "confidence": 0.8,
                },
            },
            {
                "t_sec": 1.0,
                "left": {
                    "xyz_m": [0.2, 0.2, 0.3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "gripper_m": 0.035,
                },
                "right": {
                    "xyz_m": [0.3, 0.2, 0.3],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "gripper_m": 0.035,
                },
                "object": {
                    "xyz_m": [0.15, 0.2, 0.3],
                    "quat_wxyz": [0.0, 0.0, 0.0, 2.0],
                    "confidence": 0.9,
                },
            },
        ],
    }

    clip = waypoints_to_clip(
        payload,
        video="demo.mp4",
        fps=10.0,
        frame_count=11,
        model="gemini-robotics-er-2",
    )

    assert [item["hand_side"] for item in clip["retargeting"]] == ["left", "right"]
    left = clip["retargeting"][0]
    right = clip["retargeting"][1]
    assert left["frame_ids"] == [0, 10]
    assert right["frame_ids"] == [0, 10]
    assert left["joint_trajectory"][0] == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.07]
    assert right["joint_trajectory"][0][-1] == 0.0
    assert clip["object_trajectory"]["frame_ids"] == [0, 10]
    assert clip["object_trajectory"]["poses"][0][3:] == [1.0, 0.0, 0.0, 0.0]
    assert clip["object_trajectory"]["poses"][1][3:] == [0.0, 0.0, 0.0, 1.0]
