"""Base interface for robot control."""

from abc import ABC, abstractmethod

import numpy as np

from holosoma_inference.config.config_types import RobotConfig


class BaseInterface(ABC):
    """
    Abstract base class for robot control interfaces.
    """

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        self.robot_config = robot_config
        self.domain_id = domain_id
        self.interface_str = interface_str
        self.use_joystick = use_joystick

    @abstractmethod
    def get_low_state(self) -> np.ndarray:
        """
        Get robot state as numpy array.

        Returns:
            np.ndarray with shape (1, 3+4+N+3+3+N) containing:
            [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]
        """
        raise NotImplementedError

    @abstractmethod
    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """
        Send low-level command to robot.

        Args:
            cmd_q: target joint positions (N,)
            cmd_dq: target joint velocities (N,)
            cmd_tau: feedforward torques (N,)
            dof_pos_latest: latest joint positions (N,)
            kp_override: optional KP gains override (N,)
            kd_override: optional KD gains override (N,)
        """
        raise NotImplementedError

    def update_config(self, robot_config: RobotConfig):
        """
        Update the robot configuration and propagate to internal components.

        Override in subclasses that need to update internal SDK components
        when the config changes (e.g., after loading KP/KD from ONNX metadata).

        Args:
            robot_config: The new robot configuration.
        """
        self.robot_config = robot_config

    @abstractmethod
    def get_joystick_msg(self):
        raise NotImplementedError

    @abstractmethod
    def get_joystick_key(self, wc_msg=None):
        raise NotImplementedError

    @property
    @abstractmethod
    def kp_level(self):
        raise NotImplementedError

    @kp_level.setter
    @abstractmethod
    def kp_level(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def kd_level(self):
        raise NotImplementedError

    @kd_level.setter
    @abstractmethod
    def kd_level(self, value):
        raise NotImplementedError
