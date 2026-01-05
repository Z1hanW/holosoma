"""Configuration helpers for perception-driven overrides."""

from __future__ import annotations

import dataclasses

from holosoma.config_types.algo import ModuleConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg


def apply_perception_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """Inject perception observations and encoder settings when enabled."""
    if config.perception is None or not config.perception.enabled:
        return config

    observation = _add_perception_group(config.observation)
    algo = _add_perception_modules(config)
    return dataclasses.replace(config, observation=observation, algo=algo)


def _add_perception_group(observation: ObservationManagerCfg | None) -> ObservationManagerCfg:
    if observation is None:
        raise ValueError("Perception requires an observation manager configuration.")

    if "perception_obs" in observation.groups:
        return observation

    perception_group = ObsGroupCfg(
        concatenate=True,
        enable_noise=False,
        history_length=1,
        terms={
            "perception": ObsTermCfg(
                func="holosoma.managers.observation.terms.perception:perception_obs",
                scale=1.0,
                noise=0.0,
            )
        },
    )

    groups = dict(observation.groups)
    groups["perception_obs"] = perception_group
    return dataclasses.replace(observation, groups=groups)


def _add_perception_modules(config: ExperimentConfig) -> object:
    algo_cfg = config.algo
    algo_config = getattr(algo_cfg, "config", None)
    module_dict = getattr(algo_config, "module_dict", None) if algo_config is not None else None
    if module_dict is None:
        return algo_cfg

    actor_cfg = _update_module_config(module_dict.actor, config)
    critic_cfg = _update_module_config(module_dict.critic, config, is_critic=True)
    module_dict = dataclasses.replace(module_dict, actor=actor_cfg, critic=critic_cfg)
    algo_config = dataclasses.replace(algo_config, module_dict=module_dict)
    return dataclasses.replace(algo_cfg, config=algo_config)


def _update_module_config(
    module_cfg: ModuleConfig, config: ExperimentConfig, *, is_critic: bool = False
) -> ModuleConfig:
    input_dim = list(module_cfg.input_dim)
    if "perception_obs" not in input_dim:
        input_dim.append("perception_obs")

    layer_cfg = module_cfg.layer_config
    layer_cfg = dataclasses.replace(
        layer_cfg,
        perception_input_name="perception_obs",
        perception_output_dim=config.perception.encoder_output_dim,
    )

    module_type = module_cfg.type
    if module_type == "MLP":
        module_type = "MLPPerceptionEncoder"
        module_inputs = tuple(name for name in input_dim if name != "perception_obs")
        layer_cfg = dataclasses.replace(
            layer_cfg,
            module_input_name=module_inputs,
            encoder_input_name="",
            encoder_obs_token_name=None,
        )

    return dataclasses.replace(module_cfg, type=module_type, input_dim=input_dim, layer_config=layer_cfg)
