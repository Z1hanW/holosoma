"""
Configuration types for holosoma run_sim.py script.

This module provides a minimal configuration structure for direct simulation,
following the same pattern as ExperimentConfig. Direct simulations are used for
development and running sim2sim inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tyro
from typing_extensions import Annotated

import holosoma.config_values.robot
import holosoma.config_values.run_sim
import holosoma.config_values.terrain
import holosoma.config_values.image_server
from holosoma.config_types.image_server import ImageServerConfig
from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig, LoggerConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.config_types.camera import CameraManagerCfg
from holosoma.config_types.video import VideoConfig


def default_training_config() -> TrainingConfig:
    """Create minimal training config for direct simulation."""
    return TrainingConfig(num_envs=1, headless=False, seed=42, torch_deterministic=False)


def default_logger_config() -> LoggerConfig:
    """Create minimal logger config for direct simulation."""
    return DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs")


@dataclass(frozen=True)
class MotionInitConfig:
    """Optional clip-driven initialization for direct MuJoCo sim2sim runs."""

    enabled: bool = False
    """Initialize robot/object states from a motion clip before the loop starts."""

    motion_file: str | None = None
    """Motion clip (.npz) used to initialize simulator state."""

    frame_idx: int = 0
    """Motion frame index used for initialization."""

    mode: str = "raw_motion"
    """Initialization mode: 'raw_motion', 'raw_motion_grounded', or 'training_default_pose'."""

    object_name: str = "object"
    """Simulator actor name for the initialized object."""


# Use sim2sim-optimized configs from config_values.run_sim
SIMULATOR_DEFAULTS = holosoma.config_values.run_sim.DEFAULTS


@dataclass(frozen=True)
class RunSimConfig:
    """
    Minimal configuration for direct simulation via run_sim.py.

    Usage Examples:
        python -m holosoma.run_sim simulator:mujoco robot:t1 terrain:terrain-locomotion-plane
        python -m holosoma.run_sim simulator:isaacgym robot:g1 terrain:terrain-locomotion-mix
    """

    # Core components for simulation - using Annotated subcommands like ExperimentConfig
    simulator: Annotated[
        SimulatorConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(SIMULATOR_DEFAULTS)),
    ] = holosoma.config_values.run_sim.mujoco

    robot: Annotated[
        RobotConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.robot.DEFAULTS)),
    ] = holosoma.config_values.robot.g1_29dof

    terrain: Annotated[
        TerrainManagerCfg,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.terrain.DEFAULTS)),
    ] = holosoma.config_values.terrain.terrain_locomotion_plane

    camera: Annotated[
        CameraManagerCfg,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.camera.DEFAULTS)),
    ] = holosoma.config_values.camera.dual_depth_cameras

    image_server: Annotated[
        ImageServerConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.image_server.DEFAULTS)),
    ] = holosoma.config_values.image_server.mujoco

    # Minimal configs needed for FullSimConfig
    training: TrainingConfig = field(default_factory=default_training_config)
    logger: LoggerConfig = field(default_factory=default_logger_config)

    # Optional environment wrapper (only if needed for compatibility)
    env_class: str | None = None

    # Direct simulation timing control
    viewer_dt: float = 1 / 60.0
    """Viewer refresh rate in seconds (60 FPS default).

    Only used by run_sim.py for real-time display synchronization.
    """

    motion_init: MotionInitConfig = field(default_factory=MotionInitConfig)
    """Optional clip-driven robot/object initialization."""

    device: str | None = "cpu"
    """Device to use for simulation. None auto-detects based on the simulator type.
    """
