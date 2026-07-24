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
    critic_perception_cfg, critic_perception_obs_key = _get_critic_perception_settings(config)
    if not student_perception_enabled and teacher_perception_obs_key is None and critic_perception_cfg is None:
        return config

    if student_perception_enabled:
        _validate_perception_encoder_input_contract(config.perception, context="student perception")
    if teacher_perception_preset is not None:
        _validate_perception_encoder_input_contract(
            holosoma.config_values.perception.DEFAULTS[teacher_perception_preset],
            context=f"teacher perception preset {teacher_perception_preset!r}",
        )
    if critic_perception_cfg is not None:
        _validate_perception_encoder_input_contract(
            critic_perception_cfg,
            context="critic perception",
        )

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
    if critic_perception_cfg is not None:
        if not critic_perception_obs_key:
            raise ValueError("distill.critic_perception_obs_key must be non-empty when critic_perception_preset is set.")
        if critic_perception_obs_key == "perception_obs":
            raise ValueError("distill.critic_perception_obs_key must differ from actor perception_obs.")
        if teacher_perception_obs_key is not None and critic_perception_obs_key == teacher_perception_obs_key:
            raise ValueError(
                "distill.critic_perception_obs_key must differ from distill.teacher_perception_obs_key."
            )
        observation = _add_named_perception_group(
            observation,
            group_name=critic_perception_obs_key,
            func="holosoma.managers.observation.terms.perception:critic_perception_obs",
        )

    student_policy_injection_enabled = student_perception_enabled and config.perception.inject_into_policy_modules
    if not student_policy_injection_enabled and critic_perception_cfg is None:
        return dataclasses.replace(config, observation=observation)
    algo = _add_perception_modules(config, critic_perception_cfg=critic_perception_cfg, critic_obs_key=critic_perception_obs_key)
    return dataclasses.replace(config, observation=observation, algo=algo)


def _validate_perception_encoder_input_contract(perception_cfg, *, context: str) -> None:
    encoder_type = str(getattr(perception_cfg, "encoder_type", "")).strip().lower()
    if not encoder_type.startswith("defm_"):
        return
    if getattr(perception_cfg, "output_mode", None) != "camera_depth":
        raise ValueError(f"{context} uses {encoder_type}, which requires output_mode='camera_depth'.")
    if bool(getattr(perception_cfg, "camera_warp_normalize", False)):
        raise ValueError(
            f"{context} uses {encoder_type}, which requires metric depth in meters; "
            "camera_warp_normalize must be False."
        )
    pretrained = getattr(perception_cfg, "encoder_pretrained", None)
    if not isinstance(pretrained, bool):
        raise ValueError(f"{context} {encoder_type} encoder_pretrained must be a boolean.")
    pretrained_path = getattr(perception_cfg, "encoder_pretrained_path", None)
    pretrained_sha256 = getattr(perception_cfg, "encoder_pretrained_sha256", None)
    if pretrained:
        if not isinstance(pretrained_path, str) or not pretrained_path.strip():
            raise ValueError(
                f"{context} {encoder_type} requires a local encoder_pretrained_path; "
                "implicit model downloads are forbidden."
            )
        if (
            not isinstance(pretrained_sha256, str)
            or len(pretrained_sha256) != 64
            or any(char not in "0123456789abcdef" for char in pretrained_sha256)
        ):
            raise ValueError(
                f"{context} {encoder_type} requires a 64-character lowercase "
                "encoder_pretrained_sha256."
            )
    elif pretrained_path is not None or pretrained_sha256 is not None:
        raise ValueError(
            f"{context} {encoder_type} disables pretrained weights but still declares a path/SHA."
        )


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


def _get_critic_perception_settings(config: ExperimentConfig):
    algo_cfg = getattr(config.algo, "config", None)
    distill_cfg = getattr(algo_cfg, "distill", None) if algo_cfg is not None else None
    preset_name = getattr(distill_cfg, "critic_perception_preset", None) if distill_cfg is not None else None
    if preset_name is None:
        return None, None
    preset_name = str(preset_name).strip()
    if not preset_name or preset_name.lower() == "none":
        return None, None
    if preset_name not in holosoma.config_values.perception.DEFAULTS:
        raise ValueError(f"Unknown distill.critic_perception_preset: {preset_name}")
    obs_key = getattr(distill_cfg, "critic_perception_obs_key", None)
    obs_key = str(obs_key).strip() if obs_key is not None else "critic_perception_obs"
    if not obs_key:
        obs_key = "critic_perception_obs"
    return dataclasses.replace(holosoma.config_values.perception.DEFAULTS[preset_name]), obs_key


def _add_perception_modules(
    config: ExperimentConfig,
    *,
    critic_perception_cfg,
    critic_obs_key: str | None,
) -> object:
    algo_cfg = config.algo
    algo_config = getattr(algo_cfg, "config", None)
    module_dict = getattr(algo_config, "module_dict", None) if algo_config is not None else None
    if module_dict is None:
        return algo_cfg

    student_policy_injection_enabled = config.perception is not None and config.perception.inject_into_policy_modules
    actor_cfg = (
        _update_module_config(
            module_dict.actor,
            perception_cfg=config.perception,
            perception_obs_key="perception_obs",
        )
        if student_policy_injection_enabled
        else module_dict.actor
    )

    resolved_critic_cfg = critic_perception_cfg
    resolved_critic_key = critic_obs_key
    if resolved_critic_cfg is None and student_policy_injection_enabled and config.perception.inject_into_critic_modules:
        resolved_critic_cfg = config.perception
        resolved_critic_key = "perception_obs"
    critic_cfg = (
        _update_module_config(
            module_dict.critic,
            perception_cfg=resolved_critic_cfg,
            perception_obs_key=resolved_critic_key or "perception_obs",
        )
        if resolved_critic_cfg is not None
        else module_dict.critic
    )
    module_dict = dataclasses.replace(module_dict, actor=actor_cfg, critic=critic_cfg)
    algo_config = dataclasses.replace(algo_config, module_dict=module_dict)
    return dataclasses.replace(algo_cfg, config=algo_config)


def _update_module_config(
    module_cfg: ModuleConfig,
    *,
    perception_cfg,
    perception_obs_key: str,
) -> ModuleConfig:
    input_dim = [name for name in module_cfg.input_dim if name != perception_obs_key]

    module_type = module_cfg.type
    layer_cfg = module_cfg.layer_config
    encoder_fusion = getattr(perception_cfg, "encoder_fusion", "extra_input_to_hidden")
    if encoder_fusion not in {"extra_input_to_hidden", "concat"}:
        raise ValueError(f"Unsupported perception encoder_fusion: {encoder_fusion}")
    use_extra = (
        encoder_fusion == "extra_input_to_hidden"
        and perception_cfg.encoder_type != "time_gru"
        and module_type not in {"TransformerEncoder", "TransformerObsTokenEncoder", "TerrainTransformerObsTokenEncoder"}
    )
    perception_height, perception_width = _resolve_perception_obs_hw(perception_cfg)
    layer_cfg = dataclasses.replace(
        layer_cfg,
        extra_input_to_hidden=use_extra,
        perception_input_name=perception_obs_key,
        perception_output_dim=perception_cfg.encoder_output_dim,
        perception_encoder_type=perception_cfg.encoder_type,
        perception_input_height=perception_height,
        perception_input_width=perception_width,
        perception_pretrained=perception_cfg.encoder_pretrained,
        perception_pretrained_path=perception_cfg.encoder_pretrained_path,
        perception_pretrained_sha256=getattr(perception_cfg, "encoder_pretrained_sha256", None),
        perception_freeze_backbone=perception_cfg.encoder_freeze_backbone,
        perception_target_size=perception_cfg.encoder_target_size,
        perception_patch_size=perception_cfg.encoder_patch_size,
    )

    if module_type in {"MLP", "FlowMLP"}:
        module_type = "MLPPerceptionEncoder" if module_type == "MLP" else "FlowMLPPerceptionEncoder"
        module_inputs = tuple(name for name in input_dim if name != "perception_obs")
        layer_cfg = dataclasses.replace(
            layer_cfg,
            module_input_name=module_inputs,
            encoder_input_name="",
            encoder_obs_token_name=None,
        )

    return dataclasses.replace(module_cfg, type=module_type, input_dim=input_dim, layer_config=layer_cfg)


def _resolve_perception_obs_hw(perception_cfg) -> tuple[int | None, int | None]:
    if perception_cfg.output_mode == "camera_depth":
        height = int(perception_cfg.camera_height or perception_cfg.grid_size)
        width = int(perception_cfg.camera_width or perception_cfg.grid_size)
        if perception_cfg.camera_warp_preprocess:
            resize = perception_cfg.camera_warp_resize
            if resize is not None:
                resize_h, resize_w = resize
                return int(resize_h), int(resize_w)
            height = max(1, height - int(perception_cfg.camera_warp_crop_top) - int(perception_cfg.camera_warp_crop_bottom))
            width = max(1, width - int(perception_cfg.camera_warp_crop_left) - int(perception_cfg.camera_warp_crop_right))
        return height, width
    if perception_cfg.output_mode == "heightmap":
        grid_size = int(perception_cfg.grid_size)
        return grid_size, grid_size
    return None, None
