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
import holosoma.config_values.perception
from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig, LoggerConfig
from holosoma.config_types.perception import PerceptionConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.config_types.video import VideoConfig


def default_training_config() -> TrainingConfig:
    """Create minimal training config for direct simulation."""
    return TrainingConfig(num_envs=1, headless=False, seed=42, torch_deterministic=False)


def default_logger_config() -> LoggerConfig:
    """Create minimal logger config for direct simulation."""
    return DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="/data/logs_new")


@dataclass(frozen=True)
class MotionInitConfig:
    """Optional clip-driven initialization for direct MuJoCo sim2sim runs."""

    enabled: bool = False
    """Initialize robot/object states from a motion clip before the loop starts."""

    motion_file: str | None = None
    """Motion clip (.npz/.h5) used to initialize simulator state."""

    frame_idx: int = 0
    """Motion frame index used for initialization."""

    mode: str = "raw_motion"
    """Initialization mode: 'raw_motion' or 'training_default_pose'."""

    object_name: str = "object"
    """Simulator actor name for the initialized object."""


@dataclass(frozen=True)
class DirectPerceptionRandomizationConfig:
    """Camera-only reset randomization reproduced by direct split simulation.

    Direct simulation deliberately does not run the checkpoint's robot/object
    domain randomizers.  These fields carry only the effective camera producer
    distribution required to reproduce a perception policy's input contract.
    Because training and direct simulation consume a process-global RNG with
    different batch sizes/call interleavings, this does not promise the same
    per-episode sample path as the vectorized training process.
    """

    enabled: bool = False
    # Lists are intentional: authenticated contracts are serialized as strict
    # JSON arrays, and Tyro's literal parser does not coerce JSON lists into
    # nested tuple annotations.
    translation_range: dict[str, list[float]] | None = None
    rotation_range_deg: dict[str, list[float]] | None = None
    noise_std_mult_range: list[float] | None = None
    noise_drop_prob_range: list[float] | None = None


# Tyro already normalizes underscores and hyphens in subcommand names.  Keep
# one canonical registration per preset; duplicate aliases are normalized to
# the same name and otherwise trigger an ambiguous last-registration-wins
# warning while constructing every direct-simulation parser.
SIMULATOR_DEFAULTS = dict(holosoma.config_values.run_sim.DEFAULTS)
ROBOT_DEFAULTS = dict(holosoma.config_values.robot.DEFAULTS)
TERRAIN_DEFAULTS = dict(holosoma.config_values.terrain.DEFAULTS)
PERCEPTION_DEFAULTS = dict(holosoma.config_values.perception.DEFAULTS)


@dataclass(frozen=True)
class RunSimConfig:
    """
    Minimal configuration for direct simulation via run_sim.py.

    Usage Examples:
        python -m holosoma.run_sim simulator:mujoco robot:t1-29dof-waist-wrist terrain:terrain-locomotion-plane
        python -m holosoma.run_sim simulator:isaacgym robot:g1-29dof terrain:terrain-locomotion-mix
    """

    # Core components for simulation - using Annotated subcommands like ExperimentConfig
    simulator: Annotated[
        SimulatorConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(SIMULATOR_DEFAULTS)),
    ] = holosoma.config_values.run_sim.mujoco

    robot: Annotated[
        RobotConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(ROBOT_DEFAULTS)),
    ] = holosoma.config_values.robot.g1_29dof

    terrain: Annotated[
        TerrainManagerCfg,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(TERRAIN_DEFAULTS)),
    ] = holosoma.config_values.terrain.terrain_locomotion_plane

    perception: Annotated[
        PerceptionConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(PERCEPTION_DEFAULTS)),
    ] = holosoma.config_values.perception.none

    perception_randomization: DirectPerceptionRandomizationConfig = field(
        default_factory=DirectPerceptionRandomizationConfig
    )
    """Authenticated camera-only reset distribution for the perception producer."""

    perception_producer_tick_dt: float | None = None
    """Training control-step period used to advance the direct perception producer."""

    perception_allow_mujoco_noise: bool = False
    """Re-enable the checkpoint's effective camera noise in direct MuJoCo production."""

    perception_contract_envelope_b64: str | None = None
    """Canonical base64 JSON envelope carrying the ONNX perception contract and SHA-256."""

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
