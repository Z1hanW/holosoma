from __future__ import annotations

import os
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal
from loguru import logger

from holosoma.config_types.algo import ModuleConfig

from .modules import BaseModule, PerceptionTimeGRU


def _debug_actor_log(message: str, *args) -> None:
    if os.environ.get("HOLOSOMA_DEBUG_ACTOR", "").lower() in ("", "0", "false", "no"):
        return
    rank = os.environ.get("RANK", "?")
    if os.environ.get("HOLOSOMA_DEBUG_ACTOR_ALL", "").lower() in ("", "0", "false", "no") and rank not in (
        "0",
        "?",
    ):
        return
    logger.info("HeartbeatActor rank {} pid {} " + message, rank, os.getpid(), *args)


def _tensor_debug_summary(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "None"
    is_inference = tensor.is_inference() if hasattr(tensor, "is_inference") else False
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"contiguous={tensor.is_contiguous()} requires_grad={tensor.requires_grad} "
        f"inference={is_inference}"
    )


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict,
        module_config_dict: ModuleConfig,
        num_actions,
        init_noise_std,
        history_length: dict[str, int],
    ):
        super().__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict, history_length)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = module_config_dict.min_noise_std
        self.min_mean_noise_std = module_config_dict.min_mean_noise_std
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        print(f"Actor Module: {self.actor_module.module}")

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict.output_dim):
            if output_dim == "robot_action_dim":
                module_config_dict.output_dim[idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _safe_std(self) -> torch.Tensor:
        """Return a numerically safe standard deviation tensor for Normal policy."""
        std = torch.nan_to_num(
            self.std,
            nan=1.0,
            posinf=10.0,
            neginf=0.0,
        )
        # Always keep scale strictly positive for torch.distributions.Normal.
        return torch.clamp(std, min=1e-6)

    @staticmethod
    def _expand_std_like(std: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast 1D action std to match policy mean shape without value-dependent ops."""
        if std.ndim == 1:
            return std.unsqueeze(0).expand_as(ref)
        return std.expand_as(ref)

    @staticmethod
    def _sanitize_scale(scale: torch.Tensor) -> torch.Tensor:
        """Guarantee a valid Normal scale tensor (finite and strictly positive)."""
        scale = torch.nan_to_num(scale, nan=1e-3, posinf=10.0, neginf=1e-3)
        scale = torch.abs(scale)
        return torch.clamp(scale, min=1e-6)

    def update_distribution(self, actor_obs, extra_input: torch.Tensor | None = None):
        _debug_actor_log(
            "update_distribution begin actor_obs={} extra_input={}",
            tuple(actor_obs.shape),
            None if extra_input is None else tuple(extra_input.shape),
        )
        mean = self.actor(actor_obs, extra_input=extra_input)
        _debug_actor_log("update_distribution actor forward finished mean={}", tuple(mean.shape))
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e3, neginf=-1e3)
        _debug_actor_log("update_distribution nan_to_num finished")
        safe_std = self._safe_std()
        if self.min_noise_std:
            clamped_std = torch.clamp(safe_std, min=self.min_noise_std)
            scale = self._sanitize_scale(self._expand_std_like(clamped_std, mean))
            self.distribution = Normal(mean, scale)
        elif self.min_mean_noise_std:
            current_mean = safe_std.mean()
            if current_mean < self.min_mean_noise_std:
                scale_up = self.min_mean_noise_std / (current_mean + 1e-6)
                clamped_std = safe_std * scale_up
            else:
                clamped_std = safe_std
            scale = self._sanitize_scale(self._expand_std_like(clamped_std, mean))
            self.distribution = Normal(mean, scale)
        else:
            scale = self._sanitize_scale(self._expand_std_like(safe_std, mean))
            self.distribution = Normal(mean, scale)
        _debug_actor_log("update_distribution normal built loc={} scale={}", tuple(mean.shape), tuple(scale.shape))

    def _sanitize_distribution(self):
        # Defensive guard: rebuild distribution with sanitized scale if any corruption remains.
        if self.distribution is not None:
            _debug_actor_log("sanitize_distribution begin")
            loc = torch.nan_to_num(self.distribution.loc, nan=0.0, posinf=1e3, neginf=-1e3)
            scale = self._sanitize_scale(self.distribution.scale)
            self.distribution = Normal(loc, scale)
            _debug_actor_log("sanitize_distribution finished")

    def update_distribution_from_policy_state(self, policy_state_dict):
        extra_input = policy_state_dict.get("extra_actor_input")
        self.update_distribution(policy_state_dict["actor_obs"], extra_input=extra_input)
        self._sanitize_distribution()

    def act(self, policy_state_dict):
        _debug_actor_log("act begin")
        extra_input = policy_state_dict.get("extra_actor_input")
        self.update_distribution(policy_state_dict["actor_obs"], extra_input=extra_input)
        self._sanitize_distribution()
        _debug_actor_log("act sample begin")
        sample = self.distribution.sample()
        _debug_actor_log("act sample finished sample={}", tuple(sample.shape))
        return sample

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict):
        extra_input = policy_state_dict.get("extra_actor_input")
        actor_obs = policy_state_dict["actor_obs"]
        _debug_actor_log(
            "act_inference base begin actor_obs={} extra_input={} grad_enabled={} inference_mode={}",
            tuple(actor_obs.shape),
            None if extra_input is None else tuple(extra_input.shape),
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
        )
        actions = self.actor(actor_obs, extra_input=extra_input)
        _debug_actor_log("act_inference base actor forward finished actions={}", tuple(actions.shape))
        return actions

    @property
    def supports_flow_matching(self) -> bool:
        return bool(getattr(self.actor_module, "supports_flow_matching", False))

    def flow_matching_loss(self, policy_state_dict, target_actions, loss_fn=torch.nn.functional.mse_loss):
        if not self.supports_flow_matching:
            raise ValueError("flow_matching_loss requested for a non-flow actor.")
        extra_input = policy_state_dict.get("extra_actor_input")
        return self.actor_module.flow_matching_loss(
            policy_state_dict["actor_obs"],
            target_actions,
            extra_input=extra_input,
            loss_fn=loss_fn,
        )

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")


class PPOCritic(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, history_length: dict[str, int]):
        super().__init__()
        self.critic_module = BaseModule(obs_dim_dict, module_config_dict, history_length)
        print(f"Critic Module: {self.critic_module.module}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        extra_input = policy_state_dict.get("extra_critic_input")
        return self.critic(critic_obs, extra_input=extra_input)

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass


class PPOActorEncoder(PPOActor):
    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length: dict[str, int]):
        super().__init__(obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length)
        self.module_type = module_config_dict.type
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name
        self.encoder_obs_token_name = module_config_dict.layer_config.encoder_obs_token_name
        self.perception_input_name = module_config_dict.layer_config.perception_input_name
        self.perception_encoder_type = getattr(module_config_dict.layer_config, "perception_encoder_type", "gated_linear")
        if self.perception_encoder_type == "gru":
            self.perception_encoder_type = "time_gru"
        self.perception_time_gru: PerceptionTimeGRU | None = None
        if self.perception_input_name and self.perception_encoder_type == "time_gru":
            if self.perception_input_name in obs_dim_dict:
                perception_dim = obs_dim_dict[self.perception_input_name]
            else:
                raise ValueError(f"Perception obs '{self.perception_input_name}' not found for time_gru.")
            output_dim = module_config_dict.layer_config.perception_output_dim or perception_dim
            self.perception_time_gru = PerceptionTimeGRU(
                perception_dim,
                output_dim,
                module_config_dict.layer_config,
            )

    def reset(self, dones=None):
        if self.perception_time_gru is not None and dones is not None:
            self.perception_time_gru.reset(dones)

    def _get_perception_obs(
        self,
        actor_obs: torch.Tensor,
        policy_state_dict: dict | None = None,
        *,
        source: str,
    ) -> torch.Tensor:
        if not self.perception_input_name:
            raise ValueError(f"{source} requested perception obs, but perception_input_name is not configured.")
        if self.perception_input_name in self.actor_module.input_indices_dict:
            perception_obs = actor_obs[..., self.actor_module.input_indices_dict[self.perception_input_name]]
        elif policy_state_dict is not None and self.perception_input_name in policy_state_dict:
            perception_obs = policy_state_dict[self.perception_input_name]
        else:
            raise ValueError(f"Perception obs '{self.perception_input_name}' not provided for actor.")
        _debug_actor_log(
            "{} perception obs selected key={} {}",
            source,
            self.perception_input_name,
            _tensor_debug_summary(perception_obs),
        )
        if hasattr(perception_obs, "is_inference") and perception_obs.is_inference():
            perception_obs = perception_obs.clone()
            _debug_actor_log("{} perception obs cloned from inference {}", source, _tensor_debug_summary(perception_obs))
        return perception_obs

    def _get_terrain_transformer_input(
        self, actor_obs: torch.Tensor, policy_state_dict: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.encoder_obs_token_name is None:
            raise ValueError("TerrainTransformerObsTokenEncoder requires encoder_obs_token_name.")

        proprio_token = actor_obs[..., self.actor_module.input_indices_dict[self.encoder_obs_token_name]]
        target_tokens = (
            actor_obs[..., self.actor_module.input_indices_dict[self.encoder_input_name]]
            if self.encoder_input_name
            else None
        )
        depth_token = self._get_perception_obs(actor_obs, policy_state_dict, source="TerrainTransformerObsTokenEncoder")
        perception_encoder = getattr(self.actor_module, "perception_encoder", None)
        if self.perception_encoder_type == "time_gru":
            if self.perception_time_gru is None:
                raise ValueError("time_gru enabled but perception_time_gru is not initialized.")
            depth_token = self.perception_time_gru.step(depth_token)
        elif perception_encoder is not None:
            depth_token = perception_encoder(depth_token)
        if hasattr(depth_token, "is_inference") and depth_token.is_inference():
            depth_token = depth_token.clone()

        self.actor_encoder_obs = self.actor_module.encoder(proprio_token, depth_token, target_tokens)
        parts = [self.actor_encoder_obs]

        if self.module_input_name:
            self.actor_state_obs = torch.cat(
                [
                    actor_obs[..., self.actor_module.input_indices_dict[actor_input_name]]
                    for actor_input_name in self.module_input_name
                ],
                -1,
            )
            parts.append(self.actor_state_obs)

        input_actor = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return input_actor, None

    def _get_input(self, actor_obs: torch.Tensor, policy_state_dict: dict | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _debug_actor_log(
            "_get_input enter module_type={} module_inputs={} encoder_input={} perception_key={} perception_encoder_type={} actor_obs={}",
            self.module_type,
            self.module_input_name,
            self.encoder_input_name,
            self.perception_input_name,
            self.perception_encoder_type,
            _tensor_debug_summary(actor_obs),
        )
        if actor_obs.shape[-1] != self.actor_module.input_dim:
            raise ValueError(f"Actor Obs must be {self.actor_module.input_dim}, got {actor_obs.shape[-1]}")
        if self.module_type == "TerrainTransformerObsTokenEncoder":
            return self._get_terrain_transformer_input(actor_obs, policy_state_dict)

        _debug_actor_log("_get_input encoder selection begin")
        self.encoder_obs = (
            actor_obs[..., self.actor_module.input_indices_dict[self.encoder_input_name]]
            if self.encoder_input_name
            else None
        )
        if self.encoder_obs_token_name:
            obs_token = actor_obs[..., self.actor_module.input_indices_dict[self.encoder_obs_token_name]]
            self.actor_encoder_obs = (
                self.actor_module.encoder(obs_token, self.encoder_obs)
                if self.actor_module.encoder is not None
                else obs_token
            )
        else:
            self.actor_encoder_obs = (
                self.actor_module.encoder(self.encoder_obs)
                if self.actor_module.encoder is not None and self.encoder_obs is not None
                else self.encoder_obs
            )
        _debug_actor_log("_get_input encoder selection finished actor_encoder_obs={}", _tensor_debug_summary(self.actor_encoder_obs))

        parts = []
        if self.actor_encoder_obs is not None:
            parts.append(self.actor_encoder_obs)
        perception_encoder = getattr(self.actor_module, "perception_encoder", None)
        perception_embed = None
        external_extra = policy_state_dict.get("extra_actor_input") if policy_state_dict else None
        _debug_actor_log(
            "_get_input perception branch check perception_encoder={} external_extra={}",
            type(perception_encoder).__name__ if perception_encoder is not None else None,
            _tensor_debug_summary(external_extra),
        )
        if self.perception_input_name:
            if self.perception_encoder_type == "time_gru":
                if external_extra is not None:
                    perception_embed = external_extra
                else:
                    if self.perception_time_gru is None:
                        raise ValueError("time_gru enabled but perception_time_gru is not initialized.")
                    if policy_state_dict is not None and self.perception_input_name in policy_state_dict:
                        perception_obs = policy_state_dict[self.perception_input_name]
                    else:
                        raise ValueError(f"Perception obs '{self.perception_input_name}' not provided for actor.")
                    if hasattr(perception_obs, "is_inference") and perception_obs.is_inference():
                        perception_obs = perception_obs.clone()
                    perception_embed = self.perception_time_gru.step(perception_obs)
            elif perception_encoder is not None:
                _debug_actor_log("_get_input perception obs fetch begin")
                perception_obs = self._get_perception_obs(actor_obs, policy_state_dict, source="actor")
                if not perception_obs.is_contiguous():
                    _debug_actor_log("_get_input perception obs contiguous copy begin {}", _tensor_debug_summary(perception_obs))
                    perception_obs = perception_obs.contiguous()
                    _debug_actor_log("_get_input perception obs contiguous copy finished {}", _tensor_debug_summary(perception_obs))
                _debug_actor_log(
                    "_get_input perception encoder begin encoder={} obs={}",
                    type(perception_encoder).__name__,
                    _tensor_debug_summary(perception_obs),
                )
                perception_embed = perception_encoder(perception_obs)
                _debug_actor_log("_get_input perception encoder finished embed={}", _tensor_debug_summary(perception_embed))
                if hasattr(perception_embed, "is_inference") and perception_embed.is_inference():
                    perception_embed = perception_embed.clone()
                    _debug_actor_log("_get_input perception embed cloned {}", _tensor_debug_summary(perception_embed))

        if self.module_input_name:
            _debug_actor_log("_get_input actor_state cat begin")
            self.actor_state_obs = torch.cat(
                [
                    actor_obs[..., self.actor_module.input_indices_dict[actor_input_name]]
                    for actor_input_name in self.module_input_name
                ],
                -1,
            )
            _debug_actor_log("_get_input actor_state cat finished {}", _tensor_debug_summary(self.actor_state_obs))
            parts.append(self.actor_state_obs)

        supports_extra = getattr(self.actor_module.module, "supports_extra_input", False)
        if perception_embed is not None and not supports_extra:
            parts.append(perception_embed)

        if len(parts) == 1:
            input_actor = parts[0]
        else:
            _debug_actor_log("_get_input final cat begin parts={}", [_tensor_debug_summary(part) for part in parts])
            input_actor = torch.cat(parts, dim=-1)
            _debug_actor_log("_get_input final cat finished {}", _tensor_debug_summary(input_actor))

        extra_input = perception_embed if supports_extra else None
        if external_extra is not None and external_extra is not perception_embed:
            if hasattr(external_extra, "is_inference") and external_extra.is_inference():
                external_extra = external_extra.clone()
            if supports_extra:
                _debug_actor_log(
                    "_get_input external extra cat begin extra_input={} external_extra={}",
                    _tensor_debug_summary(extra_input),
                    _tensor_debug_summary(external_extra),
                )
                extra_input = external_extra if extra_input is None else torch.cat([extra_input, external_extra], dim=-1)
                _debug_actor_log("_get_input external extra cat finished {}", _tensor_debug_summary(extra_input))
        return input_actor, extra_input

    def act(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        _debug_actor_log("encoder act _get_input begin actor_obs={}", tuple(actor_obs.shape))
        input_actor, extra_input = self._get_input(actor_obs, policy_state_dict)
        _debug_actor_log(
            "encoder act _get_input finished input_actor={} extra_input={}",
            tuple(input_actor.shape),
            None if extra_input is None else tuple(extra_input.shape),
        )
        return super().act({"actor_obs": input_actor, "extra_actor_input": extra_input})

    def update_distribution_from_policy_state(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        _debug_actor_log("encoder update _get_input begin actor_obs={}", tuple(actor_obs.shape))
        input_actor, extra_input = self._get_input(actor_obs, policy_state_dict)
        _debug_actor_log(
            "encoder update _get_input finished input_actor={} extra_input={}",
            tuple(input_actor.shape),
            None if extra_input is None else tuple(extra_input.shape),
        )
        return super().update_distribution_from_policy_state(
            {"actor_obs": input_actor, "extra_actor_input": extra_input}
        )

    def act_inference(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        _debug_actor_log(
            "encoder act_inference _get_input begin actor_obs={} grad_enabled={} inference_mode={}",
            tuple(actor_obs.shape),
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
        )
        input_actor, extra_input = self._get_input(actor_obs, policy_state_dict)
        _debug_actor_log(
            "encoder act_inference _get_input finished input_actor={} extra_input={}",
            tuple(input_actor.shape),
            None if extra_input is None else tuple(extra_input.shape),
        )
        return super().act_inference({"actor_obs": input_actor, "extra_actor_input": extra_input})

    def flow_matching_loss(self, policy_state_dict, target_actions, loss_fn=torch.nn.functional.mse_loss):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor, extra_input = self._get_input(actor_obs, policy_state_dict)
        return super().flow_matching_loss(
            {"actor_obs": input_actor, "extra_actor_input": extra_input},
            target_actions,
            loss_fn=loss_fn,
        )


class PPOCriticEncoder(PPOCritic):
    def __init__(self, obs_dim_dict, module_config_dict, history_length: dict[str, int]):
        super().__init__(obs_dim_dict, module_config_dict, history_length)
        self.module_type = module_config_dict.type
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name
        self.encoder_obs_token_name = module_config_dict.layer_config.encoder_obs_token_name
        self.perception_input_name = module_config_dict.layer_config.perception_input_name
        self.perception_encoder_type = getattr(module_config_dict.layer_config, "perception_encoder_type", "gated_linear")
        if self.perception_encoder_type == "gru":
            self.perception_encoder_type = "time_gru"
        self.perception_time_gru: PerceptionTimeGRU | None = None
        if self.perception_input_name and self.perception_encoder_type == "time_gru":
            if self.perception_input_name in obs_dim_dict:
                perception_dim = obs_dim_dict[self.perception_input_name]
            else:
                raise ValueError(f"Perception obs '{self.perception_input_name}' not found for time_gru.")
            output_dim = module_config_dict.layer_config.perception_output_dim or perception_dim
            self.perception_time_gru = PerceptionTimeGRU(
                perception_dim,
                output_dim,
                module_config_dict.layer_config,
            )

    def reset(self, dones=None):
        if self.perception_time_gru is not None and dones is not None:
            self.perception_time_gru.reset(dones)

    def _get_perception_obs(
        self,
        critic_obs: torch.Tensor,
        policy_state_dict: dict | None = None,
        *,
        source: str,
    ) -> torch.Tensor:
        if not self.perception_input_name:
            raise ValueError(f"{source} requested perception obs, but perception_input_name is not configured.")
        if self.perception_input_name in self.critic_module.input_indices_dict:
            perception_obs = critic_obs[..., self.critic_module.input_indices_dict[self.perception_input_name]]
        elif policy_state_dict is not None and self.perception_input_name in policy_state_dict:
            perception_obs = policy_state_dict[self.perception_input_name]
        else:
            raise ValueError(f"Perception obs '{self.perception_input_name}' not provided for critic.")
        if hasattr(perception_obs, "is_inference") and perception_obs.is_inference():
            perception_obs = perception_obs.clone()
        return perception_obs

    def _get_terrain_transformer_input(
        self, critic_obs: torch.Tensor, policy_state_dict: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.encoder_obs_token_name is None:
            raise ValueError("TerrainTransformerObsTokenEncoder requires encoder_obs_token_name.")

        proprio_token = critic_obs[..., self.critic_module.input_indices_dict[self.encoder_obs_token_name]]
        target_tokens = (
            critic_obs[..., self.critic_module.input_indices_dict[self.encoder_input_name]]
            if self.encoder_input_name
            else None
        )
        depth_token = self._get_perception_obs(critic_obs, policy_state_dict, source="TerrainTransformerObsTokenEncoder")
        perception_encoder = getattr(self.critic_module, "perception_encoder", None)
        if self.perception_encoder_type == "time_gru":
            if self.perception_time_gru is None:
                raise ValueError("time_gru enabled but perception_time_gru is not initialized.")
            depth_token = self.perception_time_gru.step(depth_token)
        elif perception_encoder is not None:
            depth_token = perception_encoder(depth_token)
        if hasattr(depth_token, "is_inference") and depth_token.is_inference():
            depth_token = depth_token.clone()

        self.critic_encoder_obs = self.critic_module.encoder(proprio_token, depth_token, target_tokens)
        parts = [self.critic_encoder_obs]

        if self.module_input_name:
            self.critic_state_obs = torch.cat(
                [
                    critic_obs[..., self.critic_module.input_indices_dict[critic_input_name]]
                    for critic_input_name in self.module_input_name
                ],
                -1,
            )
            parts.append(self.critic_state_obs)

        input_critic = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return input_critic, None

    def _get_input(self, critic_obs: torch.Tensor, policy_state_dict: dict | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if critic_obs.shape[-1] != self.critic_module.input_dim:
            raise ValueError(f"Critic Obs must be {self.critic_module.input_dim}, got {critic_obs.shape[-1]}")
        if self.module_type == "TerrainTransformerObsTokenEncoder":
            return self._get_terrain_transformer_input(critic_obs, policy_state_dict)

        self.encoder_obs = (
            critic_obs[..., self.critic_module.input_indices_dict[self.encoder_input_name]]
            if self.encoder_input_name
            else None
        )
        if self.encoder_obs_token_name:
            obs_token = critic_obs[..., self.critic_module.input_indices_dict[self.encoder_obs_token_name]]
            self.critic_encoder_obs = (
                self.critic_module.encoder(obs_token, self.encoder_obs)
                if self.critic_module.encoder is not None
                else obs_token
            )
        else:
            self.critic_encoder_obs = (
                self.critic_module.encoder(self.encoder_obs)
                if self.critic_module.encoder is not None and self.encoder_obs is not None
                else self.encoder_obs
            )

        parts = []
        if self.critic_encoder_obs is not None:
            parts.append(self.critic_encoder_obs)

        perception_encoder = getattr(self.critic_module, "perception_encoder", None)
        perception_embed = None
        external_extra = policy_state_dict.get("extra_critic_input") if policy_state_dict else None
        if self.perception_input_name:
            if self.perception_encoder_type == "time_gru":
                if external_extra is not None:
                    perception_embed = external_extra
                else:
                    if self.perception_time_gru is None:
                        raise ValueError("time_gru enabled but perception_time_gru is not initialized.")
                    if policy_state_dict is not None and self.perception_input_name in policy_state_dict:
                        perception_obs = policy_state_dict[self.perception_input_name]
                    else:
                        raise ValueError(f"Perception obs '{self.perception_input_name}' not provided for critic.")
                    if hasattr(perception_obs, "is_inference") and perception_obs.is_inference():
                        perception_obs = perception_obs.clone()
                    perception_embed = self.perception_time_gru.step(perception_obs)
            elif perception_encoder is not None:
                perception_obs = self._get_perception_obs(critic_obs, policy_state_dict, source="critic")
                perception_embed = perception_encoder(perception_obs)
                if hasattr(perception_embed, "is_inference") and perception_embed.is_inference():
                    perception_embed = perception_embed.clone()

        if self.module_input_name:
            self.critic_state_obs = torch.cat(
                [
                    critic_obs[..., self.critic_module.input_indices_dict[critic_input_name]]
                    for critic_input_name in self.module_input_name
                ],
                -1,
            )
            parts.append(self.critic_state_obs)

        supports_extra = getattr(self.critic_module.module, "supports_extra_input", False)
        if perception_embed is not None and not supports_extra:
            parts.append(perception_embed)

        if len(parts) == 1:
            input_critic = parts[0]
        else:
            input_critic = torch.cat(parts, dim=-1)

        extra_input = perception_embed if supports_extra else None
        if external_extra is not None and external_extra is not perception_embed:
            if hasattr(external_extra, "is_inference") and external_extra.is_inference():
                external_extra = external_extra.clone()
            if supports_extra:
                extra_input = external_extra if extra_input is None else torch.cat([extra_input, external_extra], dim=-1)
        return input_critic, extra_input

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        input_critic, extra_input = self._get_input(critic_obs, policy_state_dict)
        return super().evaluate({"critic_obs": input_critic, "extra_critic_input": extra_input})
