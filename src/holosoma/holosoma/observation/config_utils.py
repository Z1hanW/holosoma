"""Configuration helpers for observation overrides."""

from __future__ import annotations

import dataclasses

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObservationManagerCfg


def apply_observation_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """Apply runtime observation overrides (history length, target groups)."""
    overrides = config.observation_overrides
    if overrides is None:
        return config

    observation = config.observation
    if observation is None:
        return config

    groups = dict(observation.groups)

    if overrides.disable_actor_history and "actor_obs" in groups:
        groups["actor_obs"] = dataclasses.replace(groups["actor_obs"], history_length=1)

    if overrides.disable_critic_history and "critic_obs" in groups:
        groups["critic_obs"] = dataclasses.replace(groups["critic_obs"], history_length=1)

    if overrides.disable_actor_target and "actor_obs_target" in groups:
        groups.pop("actor_obs_target")

    observation = dataclasses.replace(observation, groups=groups)

    if overrides.disable_actor_target:
        algo_cfg = config.algo
        algo_config = getattr(algo_cfg, "config", None)
        module_dict = getattr(algo_config, "module_dict", None) if algo_config is not None else None
        if module_dict is not None:
            actor_cfg = module_dict.actor
            input_dim = [name for name in actor_cfg.input_dim if name != "actor_obs_target"]
            if input_dim != list(actor_cfg.input_dim):
                layer_cfg = actor_cfg.layer_config
                module_inputs = layer_cfg.module_input_name
                if module_inputs:
                    module_inputs = tuple(name for name in module_inputs if name != "actor_obs_target")
                    layer_cfg = dataclasses.replace(layer_cfg, module_input_name=module_inputs)
                actor_cfg = dataclasses.replace(actor_cfg, input_dim=input_dim, layer_config=layer_cfg)
                module_dict = dataclasses.replace(module_dict, actor=actor_cfg)
                algo_config = dataclasses.replace(algo_config, module_dict=module_dict)
                algo_cfg = dataclasses.replace(algo_cfg, config=algo_config)
                return dataclasses.replace(config, observation=observation, algo=algo_cfg)

    return dataclasses.replace(config, observation=observation)
