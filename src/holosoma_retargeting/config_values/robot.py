"""Configuration values for robot retargeting."""

from __future__ import annotations

from typing import Literal

import tyro

from holosoma_retargeting.config_types.robot import RobotConfig


def get_default_robot_config(robot_type: Literal["g1", "t1"] = "g1") -> RobotConfig:
    """Get default robot configuration.

    Args:
        robot_type: Robot type identifier.

    Returns:
        RobotConfig: Default configuration instance.
    """
    return RobotConfig(robot_type=robot_type)


def get_robot_config_from_cli() -> RobotConfig:
    """Get robot configuration from tyro CLI.

    Returns:
        RobotConfig: Configuration instance from CLI arguments.
    """
    return tyro.cli(RobotConfig)


__all__ = ["get_default_robot_config", "get_robot_config_from_cli"]
