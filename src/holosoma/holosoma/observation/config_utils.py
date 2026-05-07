"""Configuration helpers for observation overrides."""

from __future__ import annotations

import dataclasses

from holosoma.config_types.algo import ModuleConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObsGroupCfg, ObsTermCfg, ObservationManagerCfg


_ACTIONS_TERM_FUNC = "holosoma.managers.observation.terms.wbt:actions"


def _default_actions_term() -> ObsTermCfg:
    return ObsTermCfg(func=_ACTIONS_TERM_FUNC, scale=1.0, noise=0.0)


def _insert_input_after(module_cfg: ModuleConfig, after_keys: tuple[str, ...], new_key: str) -> ModuleConfig:
    input_dim = list(module_cfg.input_dim)
    if new_key in input_dim:
        return module_cfg
    insert_idx = len(input_dim)
    for after_key in after_keys:
        if after_key in input_dim:
            insert_idx = input_dim.index(after_key) + 1
            break
    input_dim.insert(insert_idx, new_key)
    layer_cfg = module_cfg.layer_config
    module_inputs = layer_cfg.module_input_name
    if module_inputs:
        module_inputs_list = list(module_inputs)
        if new_key not in module_inputs_list:
            module_inputs_list.insert(insert_idx, new_key)
            layer_cfg = dataclasses.replace(layer_cfg, module_input_name=tuple(module_inputs_list))
    return dataclasses.replace(module_cfg, input_dim=input_dim, layer_config=layer_cfg)


def _apply_distill_proprio_history_only(
    config: ExperimentConfig,
    observation: ObservationManagerCfg,
    *,
    proprio_history_length: int,
) -> ExperimentConfig:
    groups = dict(observation.groups)
    proprio_history_length = max(1, int(proprio_history_length))

    actor_action_term = None
    critic_action_term = None
    actor_actions_needed = False
    critic_actions_needed = False

    for group_name in ("actor_obs_proprio", "actor_obs_proprio_no_linvel", "actor_obs", "actor_obs_legacy"):
        group_cfg = groups.get(group_name)
        if group_cfg is None:
            continue
        action_term = group_cfg.terms.get("actions")
        if action_term is not None:
            actor_action_term = action_term
            break
    for group_name in ("critic_proprio_history", "critic_obs"):
        group_cfg = groups.get(group_name)
        if group_cfg is None:
            continue
        action_term = group_cfg.terms.get("actions")
        if action_term is not None:
            critic_action_term = action_term
            break
    if actor_action_term is None:
        actor_action_term = _default_actions_term()
    if critic_action_term is None:
        critic_action_term = actor_action_term

    for group_name in ("actor_obs_proprio", "actor_obs_proprio_no_linvel"):
        group_cfg = groups.get(group_name)
        if group_cfg is None:
            continue
        terms = dict(group_cfg.terms)
        if terms.pop("actions", None) is not None:
            actor_actions_needed = True
        groups[group_name] = dataclasses.replace(group_cfg, history_length=proprio_history_length, terms=terms)

    for group_name in (
        "actor_obs_root",
        "actor_obs_torso",
        "actor_obs_track",
        "actor_obs_box",
        "actor_obs_drop",
    ):
        group_cfg = groups.get(group_name)
        if group_cfg is not None:
            groups[group_name] = dataclasses.replace(group_cfg, history_length=1)

    critic_proprio_group = groups.get("critic_proprio_history")
    if critic_proprio_group is not None:
        terms = dict(critic_proprio_group.terms)
        if terms.pop("actions", None) is not None:
            critic_actions_needed = True
        groups["critic_proprio_history"] = dataclasses.replace(
            critic_proprio_group,
            history_length=proprio_history_length,
            terms=terms,
        )

    if actor_actions_needed:
        groups["actor_obs_actions"] = ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={"actions": actor_action_term},
        )
    if critic_actions_needed:
        groups["critic_actions"] = ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={"actions": critic_action_term},
        )

    observation = dataclasses.replace(observation, groups=groups)

    algo_cfg = config.algo
    algo_config = getattr(algo_cfg, "config", None)
    module_dict = getattr(algo_config, "module_dict", None) if algo_config is not None else None
    if module_dict is None:
        return dataclasses.replace(config, observation=observation)

    actor_cfg = module_dict.actor
    critic_cfg = module_dict.critic
    if actor_actions_needed:
        actor_cfg = _insert_input_after(
            actor_cfg,
            after_keys=("actor_obs_proprio", "actor_obs_proprio_no_linvel"),
            new_key="actor_obs_actions",
        )
    if critic_actions_needed and "critic_actions" not in critic_cfg.input_dim:
        critic_cfg = _insert_input_after(
            critic_cfg,
            after_keys=("critic_proprio_history",),
            new_key="critic_actions",
        )

    module_dict = dataclasses.replace(module_dict, actor=actor_cfg, critic=critic_cfg)
    algo_config = dataclasses.replace(algo_config, module_dict=module_dict)
    algo_cfg = dataclasses.replace(algo_cfg, config=algo_config)
    return dataclasses.replace(config, observation=observation, algo=algo_cfg)


def apply_observation_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """Apply runtime observation overrides (history length, target groups)."""
    overrides = config.observation_overrides
    if overrides is None:
        return config

    observation = config.observation
    if observation is None:
        return config

    if overrides.distill_proprio_history_only:
        config = _apply_distill_proprio_history_only(
            config,
            observation,
            proprio_history_length=overrides.distill_proprio_history_length,
        )
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
