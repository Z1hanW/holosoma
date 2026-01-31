from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from holosoma.config_types.algo import ModuleConfig

from .modules import BaseModule


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

    def update_distribution(self, actor_obs, extra_input: torch.Tensor | None = None):
        mean = self.actor(actor_obs, extra_input=extra_input)
        if self.min_noise_std:
            clamped_std = torch.clamp(self.std, min=self.min_noise_std)
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        elif self.min_mean_noise_std:
            current_mean = self.std.mean()
            if current_mean < self.min_mean_noise_std:
                scale_up = self.min_mean_noise_std / (current_mean + 1e-6)
                clamped_std = self.std * scale_up
            else:
                clamped_std = self.std
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        else:
            self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, policy_state_dict):
        extra_input = policy_state_dict.get("extra_actor_input")
        self.update_distribution(policy_state_dict["actor_obs"], extra_input=extra_input)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict):
        extra_input = policy_state_dict.get("extra_actor_input")
        return self.actor(policy_state_dict["actor_obs"], extra_input=extra_input)

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
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name
        self.encoder_obs_token_name = module_config_dict.layer_config.encoder_obs_token_name
        self.perception_input_name = module_config_dict.layer_config.perception_input_name

    def _get_input(self, actor_obs: torch.Tensor, policy_state_dict: dict | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if actor_obs.shape[-1] != self.actor_module.input_dim:
            raise ValueError(f"Actor Obs must be {self.actor_module.input_dim}, got {actor_obs.shape[-1]}")
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

        parts = []
        if self.actor_encoder_obs is not None:
            parts.append(self.actor_encoder_obs)
        perception_encoder = getattr(self.actor_module, "perception_encoder", None)
        perception_embed = None
        if self.perception_input_name:
            if perception_encoder is not None:
                if self.perception_input_name in self.actor_module.input_indices_dict:
                    perception_obs = actor_obs[..., self.actor_module.input_indices_dict[self.perception_input_name]]
                elif policy_state_dict is not None and self.perception_input_name in policy_state_dict:
                    perception_obs = policy_state_dict[self.perception_input_name]
                else:
                    raise ValueError(f"Perception obs '{self.perception_input_name}' not provided for actor.")
                if hasattr(perception_obs, "is_inference") and perception_obs.is_inference():
                    perception_obs = perception_obs.clone()
                perception_embed = perception_encoder(perception_obs)
                if hasattr(perception_embed, "is_inference") and perception_embed.is_inference():
                    perception_embed = perception_embed.clone()

        if self.module_input_name:
            self.actor_state_obs = torch.cat(
                [
                    actor_obs[..., self.actor_module.input_indices_dict[actor_input_name]]
                    for actor_input_name in self.module_input_name
                ],
                -1,
            )
            parts.append(self.actor_state_obs)

        supports_extra = getattr(self.actor_module.module, "supports_extra_input", False)
        if perception_embed is not None and not supports_extra:
            parts.append(perception_embed)

        if len(parts) == 1:
            input_actor = parts[0]
        else:
            input_actor = torch.cat(parts, dim=-1)

        extra_input = perception_embed if supports_extra else None
        return input_actor, extra_input

    def act(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor, extra_input = self._get_input(actor_obs, policy_state_dict)
        return super().act({"actor_obs": input_actor, "extra_actor_input": extra_input})

    def act_inference(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor, extra_input = self._get_input(actor_obs, policy_state_dict)
        return super().act_inference({"actor_obs": input_actor, "extra_actor_input": extra_input})


class PPOCriticEncoder(PPOCritic):
    def __init__(self, obs_dim_dict, module_config_dict, history_length: dict[str, int]):
        super().__init__(obs_dim_dict, module_config_dict, history_length)
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name
        self.encoder_obs_token_name = module_config_dict.layer_config.encoder_obs_token_name
        self.perception_input_name = module_config_dict.layer_config.perception_input_name

    def _get_input(self, critic_obs: torch.Tensor, policy_state_dict: dict | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if critic_obs.shape[-1] != self.critic_module.input_dim:
            raise ValueError(f"Critic Obs must be {self.critic_module.input_dim}, got {critic_obs.shape[-1]}")
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
        if self.perception_input_name:
            if perception_encoder is not None:
                if self.perception_input_name in self.critic_module.input_indices_dict:
                    perception_obs = critic_obs[..., self.critic_module.input_indices_dict[self.perception_input_name]]
                elif policy_state_dict is not None and self.perception_input_name in policy_state_dict:
                    perception_obs = policy_state_dict[self.perception_input_name]
                else:
                    raise ValueError(f"Perception obs '{self.perception_input_name}' not provided for critic.")
                if hasattr(perception_obs, "is_inference") and perception_obs.is_inference():
                    perception_obs = perception_obs.clone()
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
        return input_critic, extra_input

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        input_critic, extra_input = self._get_input(critic_obs, policy_state_dict)
        return super().evaluate({"critic_obs": input_critic, "extra_critic_input": extra_input})
