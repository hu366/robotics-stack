"""MuJoCo environment base class."""

import mujoco
import numpy as np
from typing import Optional, Dict


class MuJoCoEnv:
    """Base MuJoCo simulation environment."""

    def __init__(self, xml_path: Optional[str] = None, xml_content: Optional[str] = None):
        """Initialize MuJoCo environment.

        Args:
            xml_path: Path to MJCF XML file
            xml_content: Raw MJCF XML content (alternative to xml_path)
        """
        if xml_content is not None:
            self.model = mujoco.MjModel.from_xml_string(xml_content)
        elif xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            raise ValueError("Either xml_path or xml_content must be provided")

        self.data = mujoco.MjData(self.model)
        self._step_count = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the environment.

        Args:
            seed: Optional random seed

        Returns:
            observation: Initial observation dict
        """
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple:
        """Step the environment.

        Args:
            action: Control action (torques or joint commands)

        Returns:
            observation, reward, done, truncated, info
        """
        # Apply action
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Get observation
        obs = self.get_observation()
        reward = 0.0
        done = False
        truncated = False
        info = {"step": self._step_count}

        return obs, reward, done, truncated, info

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation.

        Returns:
            observation dict with qpos, qvel, etc.
        """
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "time": self.data.time,
        }

    def render(self, camera_name: str = "track", width: int = 640, height: int = 480) -> np.ndarray:
        """Render the environment.

        Args:
            camera_name: Camera to use for rendering
            width: Render width
            height: Render height

        Returns:
            RGB image array
        """
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera_name)
        image = renderer.render()
        renderer.close()
        return image
