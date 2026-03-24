from __future__ import annotations

import dataclasses

from pydantic.dataclasses import dataclass

import holosoma.config_values.perception
from holosoma.config_types.action import ActionManagerCfg
from holosoma.config_types.command import CommandManagerCfg
from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.logger import LoggerConfig
from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_types.perception import PerceptionConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import SimulatorConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import TerrainManagerCfg


@dataclass(frozen=True)
class EnvConfig:
    """Collection of configs needed for constructing env classes."""

    env_class: str

    simulator: SimulatorConfig
    terrain: TerrainManagerCfg
    perception: PerceptionConfig | None
    teacher_perception: PerceptionConfig | None
    observation: ObservationManagerCfg | None
    action: ActionManagerCfg | None
    reward: RewardManagerCfg | None
    termination: TerminationManagerCfg | None
    randomization: RandomizationManagerCfg | None
    command: CommandManagerCfg | None
    curriculum: CurriculumManagerCfg | None
    robot: RobotConfig
    training: TrainingConfig
    logger: LoggerConfig


def _resolve_teacher_perception_config(tyro_config: ExperimentConfig) -> PerceptionConfig | None:
    algo_wrapper = getattr(tyro_config, "algo", None)
    algo_config = getattr(algo_wrapper, "config", None)
    distill_cfg = getattr(algo_config, "distill", None) if algo_config is not None else None
    preset_name = getattr(distill_cfg, "teacher_perception_preset", None) if distill_cfg is not None else None
    if preset_name is None:
        return None
    preset_name = str(preset_name).strip()
    if not preset_name or preset_name.lower() == "none":
        return None
    defaults = holosoma.config_values.perception.DEFAULTS
    if preset_name not in defaults:
        raise ValueError(f"Unknown distill.teacher_perception_preset: {preset_name}")
    return dataclasses.replace(defaults[preset_name])


def get_tyro_env_config(tyro_config: ExperimentConfig) -> EnvConfig:
    """Convert ExperimentConfig to EnvConfig for environment construction.

    Parameters
    ----------
    tyro_config : ExperimentConfig
        The experiment configuration containing all settings.

    Returns
    -------
    EnvConfig
        Environment configuration with extracted fields.
    """
    return EnvConfig(
        env_class=tyro_config.env_class,
        training=tyro_config.training,
        simulator=tyro_config.simulator,
        terrain=tyro_config.terrain,
        perception=tyro_config.perception,
        teacher_perception=_resolve_teacher_perception_config(tyro_config),
        observation=tyro_config.observation,
        action=tyro_config.action,
        reward=tyro_config.reward,
        termination=tyro_config.termination,
        randomization=tyro_config.randomization,
        command=tyro_config.command,
        curriculum=tyro_config.curriculum,
        robot=tyro_config.robot,
        logger=tyro_config.logger,
    )
