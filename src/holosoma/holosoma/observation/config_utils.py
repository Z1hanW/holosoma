"""Configuration helpers for observation overrides."""

from __future__ import annotations

import dataclasses

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.algo import ModuleConfig
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

    if overrides.disable_actor_history:
        for group_name, group_cfg in list(groups.items()):
            if group_name.startswith("actor_obs"):
                groups[group_name] = dataclasses.replace(group_cfg, history_length=1)

    if overrides.disable_critic_history:
        for group_name, group_cfg in list(groups.items()):
            if group_name.startswith("critic_obs"):
                groups[group_name] = dataclasses.replace(group_cfg, history_length=1)

    if overrides.disable_actor_target and "actor_obs_target" in groups:
        groups.pop("actor_obs_target")

    if overrides.disable_critic_target and "critic_obs_target" in groups:
        groups.pop("critic_obs_target")

    observation = dataclasses.replace(observation, groups=groups)

    drop_actor_target_inputs = overrides.disable_actor_target or overrides.disable_actor_target_inputs
    drop_critic_target_inputs = overrides.disable_critic_target or overrides.disable_critic_target_inputs

    if drop_actor_target_inputs or drop_critic_target_inputs:
        algo_cfg = config.algo
        algo_config = getattr(algo_cfg, "config", None)
        module_dict = getattr(algo_config, "module_dict", None) if algo_config is not None else None
        if module_dict is not None:
            actor_cfg = module_dict.actor
            critic_cfg = module_dict.critic

            def drop_inputs(module_cfg: ModuleConfig, drop_keys: set[str]) -> ModuleConfig:
                input_dim = [name for name in module_cfg.input_dim if name not in drop_keys]
                if input_dim == list(module_cfg.input_dim):
                    return module_cfg
                layer_cfg = module_cfg.layer_config
                module_inputs = layer_cfg.module_input_name
                if module_inputs:
                    module_inputs = tuple(name for name in module_inputs if name in input_dim)
                    layer_cfg = dataclasses.replace(layer_cfg, module_input_name=module_inputs)
                if layer_cfg.encoder_input_name and layer_cfg.encoder_input_name not in input_dim:
                    layer_cfg = dataclasses.replace(layer_cfg, encoder_input_name="")
                if layer_cfg.encoder_obs_token_name and layer_cfg.encoder_obs_token_name not in input_dim:
                    layer_cfg = dataclasses.replace(layer_cfg, encoder_obs_token_name=None)
                if layer_cfg.perception_input_name and layer_cfg.perception_input_name not in input_dim:
                    layer_cfg = dataclasses.replace(layer_cfg, perception_input_name="")
                return dataclasses.replace(module_cfg, input_dim=input_dim, layer_config=layer_cfg)

            if drop_actor_target_inputs:
                actor_cfg = drop_inputs(actor_cfg, {"actor_obs_target"})
            if drop_critic_target_inputs:
                critic_cfg = drop_inputs(critic_cfg, {"critic_obs_target"})

            module_dict = dataclasses.replace(module_dict, actor=actor_cfg, critic=critic_cfg)
            algo_config = dataclasses.replace(algo_config, module_dict=module_dict)
            algo_cfg = dataclasses.replace(algo_cfg, config=algo_config)
            return dataclasses.replace(config, observation=observation, algo=algo_cfg)

    return dataclasses.replace(config, observation=observation)
