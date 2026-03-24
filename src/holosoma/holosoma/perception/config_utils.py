"""Configuration helpers for perception-driven overrides."""

from __future__ import annotations

import dataclasses
import os

import holosoma.config_values.perception
from holosoma.config_types.algo import ModuleConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg


def apply_perception_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """Inject perception observations and encoder settings when enabled."""
    student_perception_enabled = config.perception is not None and config.perception.enabled
    teacher_perception_preset, teacher_perception_obs_key = _get_teacher_perception_settings(config)
    if not student_perception_enabled and teacher_perception_obs_key is None:
        return config

    if student_perception_enabled:
        inject_env = os.getenv("HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES")
        if inject_env is not None:
            inject_str = inject_env.strip().lower()
            if inject_str in {"1", "true", "yes", "y", "on"}:
                inject_override = True
            elif inject_str in {"0", "false", "no", "n", "off"}:
                inject_override = False
            else:
                raise ValueError(
                    "Invalid HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES value: "
                    f"{inject_env!r}. Expected one of true/false/1/0."
                )
            config = dataclasses.replace(
                config,
                perception=dataclasses.replace(config.perception, inject_into_policy_modules=inject_override),
            )

    observation = config.observation
    if student_perception_enabled:
        observation = _add_perception_group(observation)
    if teacher_perception_obs_key is not None:
        observation = _add_named_perception_group(
            observation,
            group_name=teacher_perception_obs_key,
            func="holosoma.managers.observation.terms.perception:teacher_perception_obs",
        )
    if not student_perception_enabled or not config.perception.inject_into_policy_modules:
        return dataclasses.replace(config, observation=observation)
    algo = _add_perception_modules(config)
    return dataclasses.replace(config, observation=observation, algo=algo)


def _add_perception_group(observation: ObservationManagerCfg | None) -> ObservationManagerCfg:
    return _add_named_perception_group(
        observation,
        group_name="perception_obs",
        func="holosoma.managers.observation.terms.perception:perception_obs",
    )


def _add_named_perception_group(
    observation: ObservationManagerCfg | None, *, group_name: str, func: str
) -> ObservationManagerCfg:
    if observation is None:
        raise ValueError("Perception requires an observation manager configuration.")

    if group_name in observation.groups:
        return observation

    perception_group = ObsGroupCfg(
        concatenate=True,
        enable_noise=False,
        history_length=1,
        terms={
            "perception": ObsTermCfg(
                func=func,
                scale=1.0,
                noise=0.0,
            )
        },
    )

    groups = dict(observation.groups)
    groups[group_name] = perception_group
    return dataclasses.replace(observation, groups=groups)


def _get_teacher_perception_settings(config: ExperimentConfig) -> tuple[str | None, str | None]:
    algo_cfg = getattr(config.algo, "config", None)
    distill_cfg = getattr(algo_cfg, "distill", None) if algo_cfg is not None else None
    preset_name = getattr(distill_cfg, "teacher_perception_preset", None) if distill_cfg is not None else None
    if preset_name is None:
        return None, None
    preset_name = str(preset_name).strip()
    if not preset_name or preset_name.lower() == "none":
        return None, None
    if preset_name not in holosoma.config_values.perception.DEFAULTS:
        raise ValueError(f"Unknown distill.teacher_perception_preset: {preset_name}")
    obs_key = getattr(distill_cfg, "teacher_perception_obs_key", None)
    obs_key = str(obs_key).strip() if obs_key is not None else "teacher_perception_obs"
    if not obs_key:
        obs_key = "teacher_perception_obs"
    return preset_name, obs_key


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
    input_dim = [name for name in module_cfg.input_dim if name != "perception_obs"]

    layer_cfg = module_cfg.layer_config
    use_extra = config.perception.encoder_type != "time_gru"
    layer_cfg = dataclasses.replace(
        layer_cfg,
        extra_input_to_hidden=use_extra,
        perception_input_name="perception_obs",
        perception_output_dim=config.perception.encoder_output_dim,
        perception_encoder_type=config.perception.encoder_type,
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
