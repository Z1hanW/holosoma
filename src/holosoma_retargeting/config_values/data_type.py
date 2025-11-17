"""Configuration values for motion data format."""

from __future__ import annotations

from typing import Literal

import tyro

from holosoma_retargeting.config_types.data_type import MotionDataConfig


def get_default_motion_data_config(
    data_format: Literal["lafan", "smplh", "mocap"] = "smplh",
    robot_type: Literal["g1", "t1"] = "g1",
) -> MotionDataConfig:
    """Get default motion data configuration.

    Args:
        data_format: Motion data format type.
        robot_type: Robot type for joint mapping.

    Returns:
        MotionDataConfig: Default configuration instance.
    """
    return MotionDataConfig(data_format=data_format, robot_type=robot_type)


def get_motion_data_config_from_cli() -> MotionDataConfig:
    """Get motion data configuration from tyro CLI."""
    return tyro.cli(MotionDataConfig)


__all__ = [
    "get_default_motion_data_config",
    "get_motion_data_config_from_cli",
]
