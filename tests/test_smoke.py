"""
Smoke tests for robotics-stack.

Verifies:
- Dependencies import correctly
- MuJoCo simulation can step
- Basic module structure is in place
"""


def test_mujoco_import():
    """Verify MuJoCo imports correctly."""
    import mujoco
    assert mujoco is not None


def test_mujoco_version():
    """Verify MuJoCo version is available."""
    import mujoco
    assert hasattr(mujoco, '__version__')


def test_numpy_import():
    """Verify numpy imports correctly."""
    import numpy as np
    assert np is not None
    assert hasattr(np, '__version__')


def test_pytest_import():
    """Verify pydantic imports correctly."""
    from pydantic import BaseModel
    assert BaseModel is not None


def test_one_step_sim():
    """
    Verify one-step MuJoCo simulation runs.
    Creates a minimal model and steps once.
    """
    import mujoco
    import numpy as np

    # Minimal MJCF: just a plane and a box
    xml_content = """
    <mujoco model="test_scene">
      <worldbody>
        <light pos="0 0 2"/>
        <geom name="plane" type="plane" size="1 1 0.1" pos="0 0 0"/>
        <body name="box" pos="0 0 0.5">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint type="free"/>
        </body>
      </worldbody>
    </mujoco>
    """

    # Load model and data
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    # Verify model loaded
    assert model is not None
    assert data is not None

    # Step once
    mujoco.mj_step(model, data)

    # Verify step produced valid state
    assert data.qpos is not None
    assert len(data.qpos) > 0
    assert not np.any(np.isnan(data.qpos))


def test_control_module_exists():
    """Verify control module structure."""
    from control import impedance, safety, kinematics
    # Modules exist (may be stubs for M0)
    assert impedance is not None
    assert safety is not None
    assert kinematics is not None


def test_planning_module_exists():
    """Verify planning module structure."""
    from planning import task_graph, primitives, trajectory
    # Modules exist (may be stubs for M0)
    assert task_graph is not None
    assert primitives is not None
    assert trajectory is not None


def test_envs_module_exists():
    """Verify envs module structure."""
    from envs import mujoco_env
    assert mujoco_env is not None


def test_eval_module_exists():
    """Verify eval module structure."""
    from eval import run_trials, metrics, report, plots
    # Modules exist (may be stubs for M0)
    assert run_trials is not None
    assert metrics is not None
    assert report is not None
    assert plots is not None
