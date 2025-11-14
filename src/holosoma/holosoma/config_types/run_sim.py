"""
Configuration types for holosoma run_sim.py script.

This module provides a minimal configuration structure for direct simulation,
following the same pattern as ExperimentConfig. Direct simulations are used for
development and running sim2sim inference.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import tyro
from typing_extensions import Annotated

import holosoma.config_values.robot
import holosoma.config_values.simulator
import holosoma.config_values.terrain
from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig, LoggerConfig
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import BridgeConfig, SimulatorConfig, VirtualGantryCfg
from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.config_types.video import VideoConfig


def default_training_config() -> TrainingConfig:
    """Create minimal training config for direct simulation."""
    return TrainingConfig(num_envs=1, headless=False, seed=42, torch_deterministic=False)


def default_logger_config() -> LoggerConfig:
    """Create minimal logger config for direct simulation."""
    return DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs")


def sim2sim_defaults(config: SimulatorConfig) -> SimulatorConfig:
    """Enable bridge and virtual gantry for run_sim.py usage."""
    return dataclasses.replace(
        config,
        config=dataclasses.replace(
            config.config,
            bridge=BridgeConfig(enabled=True),
            virtual_gantry=VirtualGantryCfg(enabled=True),
            sim=dataclasses.replace(
                config.config.sim,
                fps=1000,  # High FPS for sim2sim
            ),
        ),
    )


# Create local bridge-enabled configs for subcommand system
SIMULATOR_DEFAULTS = {
    name: sim2sim_defaults(config) for name, config in holosoma.config_values.simulator.DEFAULTS.items()
}


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
    ] = sim2sim_defaults(holosoma.config_values.simulator.mujoco)  # noqa: RUF009

    robot: Annotated[
        RobotConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.robot.DEFAULTS)),
    ] = holosoma.config_values.robot.g1_29dof

    terrain: Annotated[
        TerrainManagerCfg,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(holosoma.config_values.terrain.DEFAULTS)),
    ] = holosoma.config_values.terrain.terrain_locomotion_plane

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

    device: str | None = "cpu"
    """Device to use for simulation. If None, auto-detects CUDA availability.

    - None: Auto-detect (uses cuda:0 if available, otherwise cpu)
    - "cpu": Force CPU usage
    - "cuda:0", "cuda:1", etc.: Use specific GPU
    """
