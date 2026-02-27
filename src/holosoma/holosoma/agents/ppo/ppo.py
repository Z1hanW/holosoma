from __future__ import annotations

import dataclasses
import itertools
import os
from typing import TypedDict

import torch
import torch.nn.functional as F
from loguru import logger
from rich.console import Console
from torch import nn
from torch.nn.parameter import UninitializedParameter
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.config_types.algo import PPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.normalization import EmpiricalNormalization

console = Console()


class Minibatch(TypedDict):
    """A minibatch of data for training a PPO agent."""

    actor_obs: torch.Tensor
    """The observation of the actor.

    Shape: (mini_batch_size, actor_obs_dim), dtype: torch.float32
    """

    critic_obs: torch.Tensor
    """The observation of the critic.

    Shape: (mini_batch_size, critic_obs_dim), dtype: torch.float32
    """

    actions: torch.Tensor
    """The actions taken by the agent.

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    rewards: torch.Tensor
    """The rewards received from the environment.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    dones: torch.Tensor
    """Whether each episode is done after taking the action.

    Shape: (mini_batch_size, 1), dtype: torch.bool
    """

    values: torch.Tensor
    """The value estimates from the critic.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    returns: torch.Tensor
    """The computed (unnormalized) returns for each step.

    The returns are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    advantages: torch.Tensor
    """The computed (normalized) advantages for each step.

    The advantages are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    actions_log_prob: torch.Tensor
    """The log probabilities of the actions.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    action_mean: torch.Tensor
    """The mean of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    action_sigma: torch.Tensor
    """The standard deviation of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """



class PPO(BaseAlgo):
    config: PPOConfig

    def __init__(self, env: BaseTask, config: PPOConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
        super().__init__(env, config, device, multi_gpu_cfg)
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self._init_config()

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        _ = self.env.reset_all()

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        # Observation manager system - history is defined per-module in module_dict
        assert self.env.observation_manager is not None
        self.algo_history_length_dict = {
            "actor_obs": self.env.observation_manager.cfg.groups["actor_obs"].history_length,
            "critic_obs": self.env.observation_manager.cfg.groups["critic_obs"].history_length,
        }

        self.num_act = self.env.robot_config.actions_dim

        self.actor_learning_rate = self.config.actor_learning_rate
        self.max_actor_learning_rate = self.config.max_actor_learning_rate or max(self.actor_learning_rate, 1e-2)
        self.min_actor_learning_rate = self.config.min_actor_learning_rate or min(self.actor_learning_rate, 1e-5)
        self.critic_learning_rate = self.config.critic_learning_rate
        self.max_critic_learning_rate = self.config.max_critic_learning_rate or max(self.critic_learning_rate, 1e-2)
        self.min_critic_learning_rate = self.config.min_critic_learning_rate or min(self.critic_learning_rate, 1e-5)

        # Observation related Config
        self.use_symmetry = self.config.use_symmetry
        self._init_obs_keys()
        self._init_obs_slices()
        self._setup_obs_normalizers()
        self.distill_enabled = False
        self.distill_mode = "mse"
        self.dagger_enabled = False
        self.distill_loss_coef = 0.0
        self.bc_loss_coef = 0.0
        self.clip_teacher_actions = False
        self.clip_actions_threshold = 0.0
        self.take_teacher_actions = False
        self.switch_to_rl_after = -1
        self.use_multi_teacher = False
        self.multi_teacher_select_obs_var = "teacher_checkpoint_index"
        self.ppo_start_epoch = -1
        self.dagger_end_epoch = -1
        self.dagger_loss_coef = 1.0
        self.use_ppo_dagger_schedule = False
        self.ppo_coeff = 1.0
        self.distill_loss_fn = F.mse_loss
        self.dagger_ignore_zero_teacher_actions = True
        self.dagger_match_std = False
        self.teacher_actor = None
        self.teacher_actors: list[nn.Module] = []
        self.teacher_actor_obs_normalizers: dict[str, nn.Module] = {}
        self.teacher_actor_obs_normalizers_list: list[dict[str, nn.Module]] = []

    def _build_obs_slices(self, keys: list[str]) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        start = 0
        for key in keys:
            dim = self.algo_obs_dim_dict[key]
            slices[key] = slice(start, start + dim)
            start += dim
        return slices

    def _init_obs_slices(self) -> None:
        self.actor_obs_slices = self._build_obs_slices(self.actor_obs_keys)
        self.critic_obs_slices = self._build_obs_slices(self.critic_obs_keys)
        self.teacher_obs_keys = list(self.actor_obs_keys)
        self.teacher_obs_slices = dict(self.actor_obs_slices)
        self.teacher_obs_dim = self._get_obs_dim(self.teacher_obs_keys)

    def _setup_obs_normalizers(self) -> None:
        self.actor_obs_normalizers = self._build_group_normalizers(self.actor_obs_keys, self.config.normalize_actor_obs)
        self.critic_obs_normalizers = self._build_group_normalizers(self.critic_obs_keys, self.config.normalize_critic_obs)

    def _build_group_normalizers(self, keys: list[str], enabled: bool) -> dict[str, nn.Module]:
        normalizers: dict[str, nn.Module] = {}
        for key in keys:
            dim = self.algo_obs_dim_dict[key]
            if enabled:
                normalizers[key] = EmpiricalNormalization(
                    shape=(dim,),
                    device=self.device,
                    eps=self.config.obs_normalizer_eps,
                    until=self.config.obs_normalizer_until,
                )
            else:
                normalizers[key] = nn.Identity()
        return normalizers

    def _apply_obs_normalizer(self, normalizer: nn.Module, obs: torch.Tensor, update: bool) -> torch.Tensor:
        if isinstance(normalizer, EmpiricalNormalization):
            return normalizer(obs, update=update)
        return normalizer(obs)

    def _normalize_concat_obs(
        self,
        obs: torch.Tensor,
        keys: list[str],
        slices: dict[str, slice],
        normalizers: dict[str, nn.Module],
        *,
        update: bool,
    ) -> torch.Tensor:
        parts = []
        for key in keys:
            part = obs[..., slices[key]]
            part = self._apply_obs_normalizer(normalizers[key], part, update)
            parts.append(part)
        return torch.cat(parts, dim=-1)

    def _normalize_actor_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        return self._normalize_concat_obs(
            obs, self.actor_obs_keys, self.actor_obs_slices, self.actor_obs_normalizers, update=update
        )

    def _normalize_critic_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        return self._normalize_concat_obs(
            obs, self.critic_obs_keys, self.critic_obs_slices, self.critic_obs_normalizers, update=update
        )

    def _normalize_teacher_actor_obs(
        self, obs: torch.Tensor, normalizers: dict[str, nn.Module] | None = None
    ) -> torch.Tensor:
        if not self.distill_enabled:
            return obs
        if obs.shape[-1] != self.teacher_obs_dim:
            raise ValueError(
                f"Teacher obs dim mismatch: expected {self.teacher_obs_dim}, got {obs.shape[-1]}"
            )
        if normalizers is None:
            normalizers = self.teacher_actor_obs_normalizers
        return self._normalize_concat_obs(
            obs,
            self.teacher_obs_keys,
            self.teacher_obs_slices,
            normalizers,
            update=False,
        )

    def _get_actor_std_for_loss(self, actor: nn.Module) -> torch.Tensor:
        std = actor.std
        min_noise_std = getattr(actor, "min_noise_std", None)
        min_mean_noise_std = getattr(actor, "min_mean_noise_std", None)
        if min_noise_std:
            return torch.clamp(std, min=min_noise_std)
        if min_mean_noise_std:
            current_mean = std.mean()
            if current_mean < min_mean_noise_std:
                scale_up = min_mean_noise_std / (current_mean + 1e-6)
                return std * scale_up
        return std

    def _init_obs_keys(self):
        self.actor_obs_keys = self.config.module_dict.actor.input_dim
        self.critic_obs_keys = self.config.module_dict.critic.input_dim
        self.actor_perception_key = self.config.module_dict.actor.layer_config.perception_input_name or ""
        self.critic_perception_key = self.config.module_dict.critic.layer_config.perception_input_name or ""
        if self.actor_perception_key and self.actor_perception_key not in self.algo_obs_dim_dict:
            raise ValueError(f"Actor perception key '{self.actor_perception_key}' not found in observation manager.")
        if self.critic_perception_key and self.critic_perception_key not in self.algo_obs_dim_dict:
            raise ValueError(f"Critic perception key '{self.critic_perception_key}' not found in observation manager.")

    def setup(self):
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        logger.info("Setting up Storage")
        self._setup_storage()

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

    def _setup_models_and_optimizer(self):
        self.actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self.critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self.use_time_gru = bool(
            getattr(self.actor, "perception_time_gru", None) is not None
            or getattr(self.critic, "perception_time_gru", None) is not None
        )

        self._setup_distillation()

        if self.use_symmetry:
            self.symmetry_utils = SymmetryUtils(self.env)

        # Synchronize model weights across GPUs after initialization
        if self.is_multi_gpu:
            self._synchronize_model_weights()

        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = instantiate(
            self.config.critic_optimizer, params=self.critic.parameters(), lr=self.critic_learning_rate
        )

    def _build_teacher_actor_config(self, obs_keys: list[str]):
        actor_cfg = self.config.module_dict.actor
        if list(actor_cfg.input_dim) == list(obs_keys):
            return actor_cfg
        layer_cfg = actor_cfg.layer_config
        excluded_inputs = set()
        if layer_cfg.encoder_input_name:
            excluded_inputs.add(layer_cfg.encoder_input_name)
        if layer_cfg.encoder_obs_token_name:
            excluded_inputs.add(layer_cfg.encoder_obs_token_name)
        if layer_cfg.perception_input_name:
            excluded_inputs.add(layer_cfg.perception_input_name)
        module_inputs = tuple(name for name in obs_keys if name not in excluded_inputs)
        layer_cfg = dataclasses.replace(layer_cfg, module_input_name=module_inputs)
        if layer_cfg.encoder_input_name and layer_cfg.encoder_input_name not in obs_keys:
            layer_cfg = dataclasses.replace(layer_cfg, encoder_input_name="")
        if layer_cfg.encoder_obs_token_name and layer_cfg.encoder_obs_token_name not in obs_keys:
            layer_cfg = dataclasses.replace(layer_cfg, encoder_obs_token_name=None)
        removed_perception_input = False
        if layer_cfg.perception_input_name and layer_cfg.perception_input_name not in obs_keys:
            layer_cfg = dataclasses.replace(layer_cfg, perception_input_name="")
            removed_perception_input = True

        actor_type = actor_cfg.type
        # Teacher checkpoints can be non-perception models while student actor is perception-enabled.
        # If perception input is removed for teacher obs keys, fall back to plain MLP to keep teacher load valid.
        if actor_type == "MLPPerceptionEncoder" and not layer_cfg.perception_input_name:
            actor_type = "MLP"
        if removed_perception_input and layer_cfg.extra_input_to_hidden:
            layer_cfg = dataclasses.replace(layer_cfg, extra_input_to_hidden=False)

        return dataclasses.replace(actor_cfg, type=actor_type, input_dim=list(obs_keys), layer_config=layer_cfg)

    def _load_teacher_actor(
        self, ckpt_path: str, obs_keys: list[str] | None = None
    ) -> tuple[nn.Module, dict[str, nn.Module]]:
        if ckpt_path.startswith("wandb://"):
            from holosoma.utils.eval_utils import load_checkpoint  # noqa: PLC0415

            ckpt_path = str(load_checkpoint(ckpt_path, str(self.log_dir)))

        teacher_state = torch.load(ckpt_path, map_location=self.device)
        teacher_obs_keys = obs_keys if obs_keys is not None else self.actor_obs_keys
        teacher_actor_cfg = self._build_teacher_actor_config(teacher_obs_keys)
        teacher_actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=teacher_actor_cfg,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        allow_non_strict = False
        if hasattr(teacher_actor, "actor_module"):
            allow_non_strict = getattr(teacher_actor.actor_module.module, "supports_extra_input", False)
        try:
            teacher_actor.load_state_dict(teacher_state["actor_model_state_dict"])
        except RuntimeError:
            if not allow_non_strict:
                raise
            logger.warning("Strict teacher load failed; retrying with strict=False for extra-input modules.")
            teacher_actor.load_state_dict(teacher_state["actor_model_state_dict"], strict=False)
        teacher_actor.eval()
        for param in teacher_actor.parameters():
            if isinstance(param, UninitializedParameter):
                continue
            param.requires_grad_(False)

        teacher_normalizers = self._build_group_normalizers(teacher_obs_keys, self.config.normalize_actor_obs)
        actor_norm_state = teacher_state.get("actor_obs_normalizer_state")
        if isinstance(actor_norm_state, dict):
            for key, state in actor_norm_state.items():
                if state is not None and key in teacher_normalizers:
                    teacher_normalizers[key].load_state_dict(state)
        for normalizer in teacher_normalizers.values():
            normalizer.eval()
        return teacher_actor, teacher_normalizers

    def _setup_distillation(self) -> None:
        distill_cfg = self.config.distill
        self.distill_mode = getattr(distill_cfg, "mode", "mse")
        self.distill_enabled = False
        self.dagger_enabled = False
        self.distill_loss_coef = float(distill_cfg.loss_coef)
        self.bc_loss_coef = (
            float(distill_cfg.bc_loss_coef)
            if distill_cfg.bc_loss_coef is not None
            else float(distill_cfg.loss_coef)
        )
        self.clip_teacher_actions = bool(distill_cfg.clip_teacher_actions)
        self.clip_actions_threshold = float(distill_cfg.clip_actions_threshold)
        self.take_teacher_actions = bool(distill_cfg.take_teacher_actions)
        self.switch_to_rl_after = int(distill_cfg.switch_to_rl_after)
        self.use_multi_teacher = bool(distill_cfg.use_multi_teacher)
        self.multi_teacher_select_obs_var = str(distill_cfg.multi_teacher_select_obs_var)
        self.ppo_start_epoch = int(getattr(distill_cfg, "ppo_start_epoch", -1))
        self.dagger_end_epoch = int(getattr(distill_cfg, "dagger_end_epoch", -1))
        self.dagger_loss_coef = float(getattr(distill_cfg, "dagger_loss_coef", 1.0))
        self.use_ppo_dagger_schedule = self.ppo_start_epoch >= 0 and self.dagger_end_epoch > self.ppo_start_epoch
        self.ppo_coeff = 0.0 if self.use_ppo_dagger_schedule else 1.0
        loss_type = str(getattr(distill_cfg, "distill_loss_type", "mse")).strip().lower()
        if loss_type == "mse":
            self.distill_loss_fn = F.mse_loss
        elif loss_type == "huber":
            self.distill_loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown distill_loss_type: {loss_type}")
        self.dagger_ignore_zero_teacher_actions = bool(
            getattr(distill_cfg, "dagger_ignore_zero_teacher_actions", True)
        )
        self.dagger_match_std = bool(getattr(distill_cfg, "dagger_match_std", False))

        self.teacher_actor = None
        self.teacher_actors = []
        self.teacher_actor_obs_normalizers = {}
        self.teacher_actor_obs_normalizers_list = []

        teacher_obs_keys = distill_cfg.teacher_obs_keys or self.actor_obs_keys
        if isinstance(teacher_obs_keys, str):
            cleaned = teacher_obs_keys.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            teacher_obs_keys = [
                item.strip().strip("'").strip('"')
                for item in cleaned.split(",")
                if item.strip()
            ]
        if not teacher_obs_keys:
            raise ValueError("Distillation teacher_obs_keys is empty.")
        missing_keys = [key for key in teacher_obs_keys if key not in self.algo_obs_dim_dict]
        if missing_keys:
            raise ValueError(f"Teacher obs keys not found in observation manager: {missing_keys}")
        self.teacher_obs_keys = list(teacher_obs_keys)
        self.teacher_obs_slices = self._build_obs_slices(self.teacher_obs_keys)
        self.teacher_obs_dim = self._get_obs_dim(self.teacher_obs_keys)

        teacher_checkpoint = distill_cfg.policy_to_clone or distill_cfg.teacher_checkpoint

        if self.distill_mode == "dagger":
            if not teacher_checkpoint:
                return
            if self.bc_loss_coef <= 0.0 and self.switch_to_rl_after <= 0 and not self.use_ppo_dagger_schedule:
                return

            teacher_paths = teacher_checkpoint if isinstance(teacher_checkpoint, list) else [teacher_checkpoint]
            if self.use_multi_teacher:
                if not teacher_paths:
                    raise ValueError("use_multi_teacher=True requires a non-empty policy_to_clone list.")
            elif len(teacher_paths) != 1:
                raise ValueError("Multiple teacher checkpoints provided but use_multi_teacher is False.")

            for path in teacher_paths:
                teacher_actor, teacher_normalizers = self._load_teacher_actor(path, obs_keys=self.teacher_obs_keys)
                if self.use_multi_teacher:
                    self.teacher_actors.append(teacher_actor)
                    self.teacher_actor_obs_normalizers_list.append(teacher_normalizers)
                else:
                    self.teacher_actor = teacher_actor
                    self.teacher_actor_obs_normalizers = teacher_normalizers

            if self.bc_loss_coef > 0.0 or self.switch_to_rl_after > 0 or self.use_ppo_dagger_schedule:
                self.distill_enabled = True
                self.dagger_enabled = True
            return

        if not distill_cfg.enabled:
            return
        if not teacher_checkpoint:
            raise ValueError("Teacher checkpoint is required for distillation.")
        if isinstance(teacher_checkpoint, list):
            raise ValueError("Single-teacher mode expects a single teacher checkpoint.")

        self.teacher_actor, self.teacher_actor_obs_normalizers = self._load_teacher_actor(
            teacher_checkpoint, obs_keys=self.teacher_obs_keys
        )
        if distill_cfg.enabled:
            self.distill_enabled = True

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        """Compute total observation dimension for given observation keys."""
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            # Note: algo_obs_dim_dict from observation_manager.get_obs_dims() already includes history
            obs_dim += key_dim
        return obs_dim

    def _get_zero_input(self):
        """
        Create a dummy (all-zero) input for the actor.

        During training, we cannot use the logic in `self.get_example_obs()`, since it resets environments mid-rollout.
        """
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        return torch.zeros(1, actor_obs_dim, device=self.device)

    def _get_zero_perception_input(self) -> torch.Tensor | None:
        if not self.actor_perception_key:
            return None
        perception_dim = self.algo_obs_dim_dict[self.actor_perception_key]
        return torch.zeros(1, perception_dim, device=self.device)

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        print(f"Registering key: actor_obs with shape: {actor_obs_dim}")
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)

        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        print(f"Registering key: critic_obs with shape: {critic_obs_dim}")
        self.storage.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)

        # Register others based on Minibatch structure
        minibatch_keys = [
            ("actions", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
            ("values", (1,), torch.float),
            ("returns", (1,), torch.float),
            ("advantages", (1,), torch.float),
            ("actions_log_prob", (1,), torch.float),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
        ]
        for key, shape, dtype in minibatch_keys:
            self.storage.register(key, shape=shape, dtype=dtype)
        if self.dagger_enabled:
            self.storage.register("teacher_actions", shape=(self.num_act,), dtype=torch.float)
            if self.use_multi_teacher:
                self.storage.register("teacher_indices", shape=(1,), dtype=torch.long)
        perception_keys = {key for key in [self.actor_perception_key, self.critic_perception_key] if key}
        for key in perception_keys:
            self.storage.register(key, shape=(self.algo_obs_dim_dict[key],), dtype=torch.float)

    def _eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        for normalizer in self.actor_obs_normalizers.values():
            normalizer.eval()
        for normalizer in self.critic_obs_normalizers.values():
            normalizer.eval()
        if self.teacher_actor is not None:
            self.teacher_actor.eval()
            for normalizer in self.teacher_actor_obs_normalizers.values():
                normalizer.eval()
        if self.teacher_actors:
            for teacher_actor, normalizers in zip(self.teacher_actors, self.teacher_actor_obs_normalizers_list):
                teacher_actor.eval()
                for normalizer in normalizers.values():
                    normalizer.eval()

    def _train_mode(self):
        self.actor.train()
        self.critic.train()
        for normalizer in self.actor_obs_normalizers.values():
            normalizer.train()
        for normalizer in self.critic_obs_normalizers.values():
            normalizer.train()
        if self.teacher_actor is not None:
            self.teacher_actor.eval()
            for normalizer in self.teacher_actor_obs_normalizers.values():
                normalizer.eval()
        if self.teacher_actors:
            for teacher_actor, normalizers in zip(self.teacher_actors, self.teacher_actor_obs_normalizers_list):
                teacher_actor.eval()
                for normalizer in normalizers.values():
                    normalizer.eval()

    def learn(self):
        self._train_mode()

        obs_dict = self.env.reset_all()

        # Initialize environments with different episode length buffers
        # Must happen AFTER reset_all() to avoid being overwritten by reset
        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        for it in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.config.num_learning_iterations,
        ):
            self.current_learning_iteration = it

            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0:
                if self.is_multi_gpu and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                if self.is_main_process:
                    self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                    self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx"))
                if self.is_multi_gpu and torch.distributed.is_initialized():
                    torch.distributed.barrier()

        if self.is_multi_gpu and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))
        if self.is_multi_gpu and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _select_teacher_actions(
        self, teacher_obs_raw: torch.Tensor, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_multi_teacher:
            if self.multi_teacher_select_obs_var not in obs_dict:
                raise ValueError(
                    f"Multi-teacher enabled but observation '{self.multi_teacher_select_obs_var}' not found."
                )
            teacher_indices = obs_dict[self.multi_teacher_select_obs_var].view(-1).long()
            teacher_actions = torch.zeros((teacher_obs_raw.shape[0], self.num_act), device=teacher_obs_raw.device)
            for idx, (teacher_actor, normalizers) in enumerate(
                zip(self.teacher_actors, self.teacher_actor_obs_normalizers_list)
            ):
                mask = teacher_indices == idx
                if not mask.any():
                    continue
                teacher_obs = self._normalize_teacher_actor_obs(teacher_obs_raw[mask], normalizers=normalizers)
                teacher_actions[mask] = teacher_actor.act({"actor_obs": teacher_obs})
            return teacher_actions, teacher_indices

        assert self.teacher_actor is not None, "Teacher actor is not initialized."
        teacher_obs = self._normalize_teacher_actor_obs(teacher_obs_raw)
        teacher_actions = self.teacher_actor.act({"actor_obs": teacher_obs})
        return teacher_actions, None

    def _adjust_ppo_dagger_coeff(self, current_epoch: int) -> None:
        """Far-tracking style PPO/DAgger mixing schedule.

        - epoch < ppo_start_epoch: ppo_coeff = 0.0
        - epoch >= dagger_end_epoch: ppo_coeff = 0.9
        - otherwise: linear ramp to 0.9
        """
        if not self.use_ppo_dagger_schedule:
            self.ppo_coeff = 1.0
            return

        if current_epoch < self.ppo_start_epoch:
            self.ppo_coeff = 0.0
            return
        if current_epoch >= self.dagger_end_epoch:
            self.ppo_coeff = 0.9
            return

        total_epochs = max(1, self.dagger_end_epoch - self.ppo_start_epoch)
        ppo_epochs = max(0, current_epoch - self.ppo_start_epoch)
        self.ppo_coeff = min(float(ppo_epochs) / float(total_epochs), 0.9)

    def _rollout_step(self, obs_dict):
        with torch.no_grad():
            for _ in range(self.config.num_steps_per_env):
                # Environment step
                actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs_raw = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)

                actor_obs = self._normalize_actor_obs(actor_obs_raw, update=True)
                critic_obs = self._normalize_critic_obs(critic_obs_raw, update=True)

                actor_policy_state = {"actor_obs": actor_obs}
                if self.actor_perception_key:
                    actor_policy_state[self.actor_perception_key] = obs_dict[self.actor_perception_key]
                actions = self.actor.act(actor_policy_state)

                critic_policy_state = {"critic_obs": critic_obs}
                if self.critic_perception_key:
                    critic_policy_state[self.critic_perception_key] = obs_dict[self.critic_perception_key]
                values = self.critic.evaluate(critic_policy_state).detach()

                teacher_actions = None
                teacher_indices = None
                actions_to_step = actions
                if self.dagger_enabled and (self.bc_loss_coef > 0.0 or self.use_ppo_dagger_schedule):
                    if self.teacher_obs_keys == self.actor_obs_keys:
                        teacher_obs_raw = actor_obs_raw
                    else:
                        teacher_obs_raw = torch.cat([obs_dict[k] for k in self.teacher_obs_keys], dim=1)
                    teacher_actions, teacher_indices = self._select_teacher_actions(teacher_obs_raw, obs_dict)
                    if self.take_teacher_actions:
                        actions_to_step = teacher_actions

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions_to_step})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # Compute bootstrap value for timeouts
                final_rewards = torch.zeros_like(rewards)
                if infos["time_outs"].any():
                    final_critic_obs = torch.cat([infos["final_observations"][k] for k in self.critic_obs_keys], dim=1)
                    final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=True)
                    final_policy_state = {"critic_obs": final_critic_obs}
                    if (
                        self.critic_perception_key
                        and self.critic_perception_key in infos["final_observations"]
                    ):
                        final_policy_state[self.critic_perception_key] = infos["final_observations"][
                            self.critic_perception_key
                        ]
                    final_values = self.critic.evaluate(final_policy_state).detach()
                    final_rewards += self.config.gamma * torch.squeeze(
                        final_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
                    )

                storage_kwargs = {
                    "actor_obs": actor_obs_raw,
                    "critic_obs": critic_obs_raw,
                    "actions": actions,
                    "values": values,
                    "actions_log_prob": self.actor.get_actions_log_prob(actions).detach().unsqueeze(1),
                    "action_mean": self.actor.action_mean.detach(),
                    "action_sigma": self.actor.action_std.detach(),
                    "rewards": (rewards + final_rewards).view(-1, 1),
                    "dones": dones.view(-1, 1),
                    "teacher_actions": teacher_actions.detach()
                    if teacher_actions is not None
                    else torch.zeros_like(actions),
                    "teacher_indices": teacher_indices.view(-1, 1)
                    if teacher_indices is not None
                    else torch.zeros(actions.shape[0], 1, device=actions.device, dtype=torch.long),
                }
                if self.actor_perception_key:
                    storage_kwargs[self.actor_perception_key] = obs_dict[self.actor_perception_key]
                if self.critic_perception_key and self.critic_perception_key != self.actor_perception_key:
                    storage_kwargs[self.critic_perception_key] = obs_dict[self.critic_perception_key]
                self.storage.add(**storage_kwargs)

                # Reset actor and critic for completed envs
                self.actor.reset(dones)
                self.critic.reset(dones)
                if self.dagger_enabled:
                    if self.use_multi_teacher:
                        for teacher_actor in self.teacher_actors:
                            teacher_actor.reset(dones)
                    elif self.teacher_actor is not None:
                        self.teacher_actor.reset(dones)

                if self.log_dir is not None:
                    # Update episode stats using logging helper
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

            # Return / Advantage computation
            last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
            last_policy_state = {"critic_obs": last_critic_obs}
            if self.critic_perception_key and self.critic_perception_key in obs_dict:
                last_policy_state[self.critic_perception_key] = obs_dict[self.critic_perception_key]
            last_values = self.critic.evaluate(last_policy_state).detach().to(self.device)
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
            )

            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        return obs_dict

    def _compute_returns_and_advantages(self, last_values, values, dones, rewards):
        advantage = 0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.config.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.config.gamma * self.config.lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values

        if self.is_multi_gpu:
            advantages = self._normalize_advantages_multi_gpu(advantages)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _training_step(self) -> dict[str, float]:
        if self.dagger_enabled and self.use_ppo_dagger_schedule:
            self._adjust_ppo_dagger_coeff(self.current_learning_iteration)
        if self.dagger_enabled and (not self.use_ppo_dagger_schedule) and self.switch_to_rl_after > 0:
            if self.current_learning_iteration == self.switch_to_rl_after:
                self.bc_loss_coef = 0.0
        if self.use_time_gru:
            generator = self.storage.sequence_mini_batch_generator(
                self.config.num_mini_batches, self.config.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)

        minibatch: Minibatch
        loss_dict = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
        for minibatch in generator:
            loss_dict = self._update_algo_step(minibatch, loss_dict)

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for key in loss_dict:
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _update_algo_step(self, minibatch: Minibatch, loss_dict: dict[str, float]):
        ppo_loss_dict = self._compute_ppo_loss(minibatch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        ppo_loss = ppo_loss_dict["actor_loss"] + ppo_loss_dict["critic_loss"]
        ppo_loss.backward()

        if self.is_multi_gpu:
            self._reduce_parameters()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict["Value"] += ppo_loss_dict.pop("value_loss").item()
        loss_dict["Surrogate"] += ppo_loss_dict.pop("surrogate_loss").item()
        loss_dict["Entropy"] += ppo_loss_dict.pop("entropy_loss").item()
        loss_dict["KL"] += ppo_loss_dict.pop("kl_mean").item()
        for key, loss in ppo_loss_dict.items():
            if key not in loss_dict:
                loss_dict[key] = 0.0
            loss_value = loss.item() if torch.is_tensor(loss) else loss
            loss_dict[key] += loss_value
        return loss_dict

    def _compute_ppo_loss(self, minibatch: Minibatch):
        if self.use_time_gru:
            return self._compute_ppo_loss_sequence(minibatch)
        raw_actor_obs = minibatch["actor_obs"]
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]
        old_actions_log_prob_batch = minibatch["actions_log_prob"]
        old_mu_batch = minibatch["action_mean"]
        old_sigma_batch = minibatch["action_sigma"]
        actor_perception_obs = (
            minibatch.get(self.actor_perception_key) if self.actor_perception_key else None
        )
        critic_perception_obs = (
            minibatch.get(self.critic_perception_key) if self.critic_perception_key else None
        )

        # Symmetry augmentation
        original_batch_size = actions_batch.shape[0]
        if self.use_symmetry:
            actor_obs = self.symmetry_utils.augment_observations(
                obs=raw_actor_obs,
                env=self.env,
                obs_list=self.actor_obs_keys,
            )
            critic_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["critic_obs"],
                env=self.env,
                obs_list=self.critic_obs_keys,
            )
            actions_batch = self.symmetry_utils.augment_actions(
                actions=actions_batch,
            )
            num_aug = int(actor_obs.shape[0] / original_batch_size)
            old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
            target_values_batch = target_values_batch.repeat(num_aug, 1)
            advantages_batch = advantages_batch.repeat(num_aug, 1)
            returns_batch = returns_batch.repeat(num_aug, 1)
            if actor_perception_obs is not None:
                actor_perception_obs = actor_perception_obs.repeat(num_aug, 1)
            if critic_perception_obs is not None:
                critic_perception_obs = critic_perception_obs.repeat(num_aug, 1)
        else:
            actor_obs = minibatch["actor_obs"]
            critic_obs = minibatch["critic_obs"]

        if actor_perception_obs is not None and actor_perception_obs.is_inference():
            actor_perception_obs = actor_perception_obs.clone()
        if critic_perception_obs is not None and critic_perception_obs.is_inference():
            critic_perception_obs = critic_perception_obs.clone()

        actor_obs = self._normalize_actor_obs(actor_obs, update=True)
        critic_obs = self._normalize_critic_obs(critic_obs, update=True)

        actor_policy_state = {"actor_obs": actor_obs}
        if actor_perception_obs is not None:
            actor_policy_state[self.actor_perception_key] = actor_perception_obs
        self.actor.act(actor_policy_state)

        critic_policy_state = {"critic_obs": critic_obs}
        if critic_perception_obs is not None:
            critic_policy_state[self.critic_perception_key] = critic_perception_obs
        value_batch = self.critic.evaluate(critic_policy_state)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self.actor.entropy[:original_batch_size]

        kl_mean = torch.tensor(0.0, device=self.device)
        update_kl = not (self.dagger_enabled and self.use_ppo_dagger_schedule and self.ppo_coeff <= 0.1)
        if self.config.desired_kl is not None and self.config.schedule == "adaptive" and update_kl:
            # Compute the KL divergence between the old and new action distributions
            kl_mean = self._compute_kl_div(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)
            self._update_learning_rate(kl_mean)

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        if self.use_symmetry and (self.config.symmetry_actor_coef > 0.0 or self.config.symmetry_critic_coef > 0.0):
            mean_policy_state = {"actor_obs": actor_obs.detach().clone()}
            if actor_perception_obs is not None:
                mean_policy_state[self.actor_perception_key] = actor_perception_obs.detach().clone()
            mean_actions_batch = self.actor.act_inference(mean_policy_state)
            mean_actions_for_original_batch, mean_actions_for_symmetry_batch = (
                mean_actions_batch[:original_batch_size],
                mean_actions_batch[original_batch_size:],
            )
            mean_symmetry_actions_batch = self.symmetry_utils.augment_actions(
                actions=mean_actions_for_original_batch,
            )[original_batch_size:]
            symmetry_actor_loss = F.mse_loss(
                mean_actions_for_symmetry_batch,
                mean_symmetry_actions_batch,
            )

            # Symmetry critic loss
            symmetry_critic_loss = F.mse_loss(
                value_batch[:original_batch_size],
                value_batch[original_batch_size:],
            )
        else:
            symmetry_actor_loss = torch.tensor(0.0, device=self.device)
            symmetry_critic_loss = torch.tensor(0.0, device=self.device)

        entropy_loss = entropy_batch.mean()
        actor_loss_base = (
            surrogate_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.symmetry_actor_coef * symmetry_actor_loss
        )

        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        actor_loss = actor_loss_base
        distill_loss = torch.tensor(0.0, device=self.device)
        bc_loss = torch.tensor(0.0, device=self.device)
        if self.distill_mode == "dagger" and self.dagger_enabled and (
            self.bc_loss_coef > 0.0 or self.use_ppo_dagger_schedule
        ):
            teacher_actions_batch = minibatch.get("teacher_actions")
            if teacher_actions_batch is None:
                raise ValueError("Dagger enabled but teacher_actions are missing from rollout storage.")
            teacher_actions_batch = teacher_actions_batch[:original_batch_size]
            if self.clip_teacher_actions:
                teacher_actions_batch = torch.clamp(
                    teacher_actions_batch, -self.clip_actions_threshold, self.clip_actions_threshold
                )

            distill_per_elem = self.distill_loss_fn(mu_batch, teacher_actions_batch, reduction="none")
            if distill_per_elem.ndim > 1:
                distill_per_sample = distill_per_elem.mean(dim=-1)
            else:
                distill_per_sample = distill_per_elem

            if self.dagger_ignore_zero_teacher_actions:
                expert_terminate = torch.all(teacher_actions_batch == 0.0, dim=-1)
                if (~expert_terminate).any():
                    bc_loss = distill_per_sample[~expert_terminate].mean()
                else:
                    bc_loss = torch.tensor(0.0, device=self.device)
            else:
                bc_loss = distill_per_sample.mean()

            if self.dagger_match_std:
                if self.use_multi_teacher:
                    teacher_indices = minibatch.get("teacher_indices")
                    if teacher_indices is None:
                        raise ValueError("Multi-teacher enabled but teacher_indices are missing from rollout storage.")
                    teacher_indices = teacher_indices.view(-1)[:original_batch_size]
                    sigma_teacher = torch.zeros_like(sigma_batch)
                    for idx, teacher_actor in enumerate(self.teacher_actors):
                        mask = teacher_indices == idx
                        if mask.any():
                            sigma_teacher[mask] = self._get_actor_std_for_loss(teacher_actor).detach()
                else:
                    assert self.teacher_actor is not None, "Teacher actor is not initialized."
                    sigma_teacher = self._get_actor_std_for_loss(self.teacher_actor).detach()
                    sigma_teacher = sigma_teacher.unsqueeze(0).expand_as(sigma_batch)
                bc_loss = bc_loss + (sigma_batch - sigma_teacher).pow(2).sum(dim=-1).mean()

            if self.use_ppo_dagger_schedule:
                dagger_weight = self.dagger_loss_coef * (1.0 - self.ppo_coeff)
                actor_loss = self.ppo_coeff * actor_loss_base + dagger_weight * bc_loss
            elif self.bc_loss_coef > 0.0:
                actor_loss = (1.0 - self.bc_loss_coef) * actor_loss_base + self.bc_loss_coef * bc_loss
        elif self.distill_enabled:
            assert self.teacher_actor is not None, "Distillation enabled but teacher actor is not initialized."
            teacher_obs = self._normalize_teacher_actor_obs(raw_actor_obs)
            with torch.inference_mode():
                teacher_actions = self.teacher_actor.act_inference({"actor_obs": teacher_obs})
            distill_loss = F.mse_loss(mu_batch, teacher_actions)
            actor_loss = actor_loss + self.distill_loss_coef * distill_loss

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "distill_loss": distill_loss,
            "bc_loss": bc_loss,
            "kl_mean": kl_mean,
        }

    def _encode_perception_sequence(
        self, encoder: nn.Module, obs_seq: torch.Tensor, dones_seq: torch.Tensor | None
    ) -> torch.Tensor:
        if not hasattr(encoder, "forward_sequence"):
            raise ValueError("Perception encoder does not support sequence encoding.")
        return encoder.forward_sequence(obs_seq, dones_seq=dones_seq)

    def _compute_ppo_loss_sequence(self, minibatch: Minibatch):
        # Sequence shapes: [T, B, ...]
        raw_actor_obs = minibatch["actor_obs"]
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]
        old_actions_log_prob_batch = minibatch["actions_log_prob"]
        old_mu_batch = minibatch["action_mean"]
        old_sigma_batch = minibatch["action_sigma"]
        actor_perception_obs = minibatch.get(self.actor_perception_key) if self.actor_perception_key else None
        critic_perception_obs = minibatch.get(self.critic_perception_key) if self.critic_perception_key else None
        dones_seq = minibatch.get("dones")

        # Flatten time/env for normalization
        t_steps, batch = actions_batch.shape[0], actions_batch.shape[1]
        actor_obs_flat = raw_actor_obs.flatten(0, 1)
        critic_obs_flat = minibatch["critic_obs"].flatten(0, 1)

        actor_obs_flat = self._normalize_actor_obs(actor_obs_flat, update=True)
        critic_obs_flat = self._normalize_critic_obs(critic_obs_flat, update=True)

        actor_obs = actor_obs_flat.view(t_steps, batch, -1)
        critic_obs = critic_obs_flat.view(t_steps, batch, -1)

        # Encode perception sequences
        if actor_perception_obs is None or critic_perception_obs is None:
            raise ValueError("time_gru requires perception_obs for both actor and critic.")

        if hasattr(actor_perception_obs, "is_inference") and actor_perception_obs.is_inference():
            actor_perception_obs = actor_perception_obs.clone()
        if hasattr(critic_perception_obs, "is_inference") and critic_perception_obs.is_inference():
            critic_perception_obs = critic_perception_obs.clone()

        actor_embed_seq = self._encode_perception_sequence(
            self.actor.perception_time_gru, actor_perception_obs, dones_seq
        )
        critic_embed_seq = self._encode_perception_sequence(
            self.critic.perception_time_gru, critic_perception_obs, dones_seq
        )

        # Flatten for PPO loss
        actor_embed_flat = actor_embed_seq.flatten(0, 1)
        critic_embed_flat = critic_embed_seq.flatten(0, 1)
        actions_flat = actions_batch.flatten(0, 1)
        target_values_flat = target_values_batch.flatten(0, 1)
        returns_flat = returns_batch.flatten(0, 1)
        advantages_flat = advantages_batch.flatten(0, 1)
        old_actions_log_prob_flat = old_actions_log_prob_batch.flatten(0, 1)
        old_mu_flat = old_mu_batch.flatten(0, 1)
        old_sigma_flat = old_sigma_batch.flatten(0, 1)

        # Symmetry augmentation
        original_batch_size = actions_flat.shape[0]
        if self.use_symmetry:
            actor_obs_aug = self.symmetry_utils.augment_observations(
                obs=actor_obs_flat,
                env=self.env,
                obs_list=self.actor_obs_keys,
            )
            critic_obs_aug = self.symmetry_utils.augment_observations(
                obs=critic_obs_flat,
                env=self.env,
                obs_list=self.critic_obs_keys,
            )
            actions_flat = self.symmetry_utils.augment_actions(actions=actions_flat)
            num_aug = int(actor_obs_aug.shape[0] / original_batch_size)
            old_actions_log_prob_flat = old_actions_log_prob_flat.repeat(num_aug, 1)
            returns_flat = returns_flat.repeat(num_aug, 1)
            advantages_flat = advantages_flat.repeat(num_aug, 1)
            target_values_flat = target_values_flat.repeat(num_aug, 1)
            old_mu_flat = old_mu_flat.repeat(num_aug, 1)
            old_sigma_flat = old_sigma_flat.repeat(num_aug, 1)
            actor_embed_flat = actor_embed_flat.repeat(num_aug, 1)
            critic_embed_flat = critic_embed_flat.repeat(num_aug, 1)
            actor_obs_flat = actor_obs_aug
            critic_obs_flat = critic_obs_aug

        actor_policy_state = {"actor_obs": actor_obs_flat, "extra_actor_input": actor_embed_flat}
        self.actor.act(actor_policy_state)

        critic_policy_state = {"critic_obs": critic_obs_flat, "extra_critic_input": critic_embed_flat}
        value_batch = self.critic.evaluate(critic_policy_state)

        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_flat)
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self.actor.entropy[:original_batch_size]

        kl_mean = torch.tensor(0.0, device=self.device)
        update_kl = not (self.dagger_enabled and self.use_ppo_dagger_schedule and self.ppo_coeff <= 0.1)
        if self.config.desired_kl is not None and self.config.schedule == "adaptive" and update_kl:
            kl_mean = self._compute_kl_div(old_mu_flat, old_sigma_flat, mu_batch, sigma_batch)
            self._update_learning_rate(kl_mean)

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_flat))
        surrogate = -torch.squeeze(advantages_flat) * ratio
        surrogate_clipped = -torch.squeeze(advantages_flat) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_clipped = target_values_flat + (value_batch - target_values_flat).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_flat).pow(2)
        value_losses_clipped = (value_clipped - returns_flat).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        if self.use_symmetry and (self.config.symmetry_actor_coef > 0.0 or self.config.symmetry_critic_coef > 0.0):
            mean_policy_state = {
                "actor_obs": actor_obs_flat.detach().clone(),
                "extra_actor_input": actor_embed_flat.detach().clone(),
            }
            mean_actions_batch = self.actor.act_inference(mean_policy_state)
            mean_actions_for_original_batch, mean_actions_for_symmetry_batch = (
                mean_actions_batch[:original_batch_size],
                mean_actions_batch[original_batch_size:],
            )
            mean_symmetry_actions_batch = self.symmetry_utils.augment_actions(
                actions=mean_actions_for_original_batch,
            )[original_batch_size:]
            symmetry_actor_loss = F.mse_loss(
                mean_actions_for_symmetry_batch,
                mean_symmetry_actions_batch,
            )

            symmetry_critic_loss = F.mse_loss(
                value_batch[:original_batch_size],
                value_batch[original_batch_size:],
            )
        else:
            symmetry_actor_loss = torch.tensor(0.0, device=self.device)
            symmetry_critic_loss = torch.tensor(0.0, device=self.device)

        entropy_loss = entropy_batch.mean()
        actor_loss_base = (
            surrogate_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.symmetry_actor_coef * symmetry_actor_loss
        )
        actor_loss = actor_loss_base
        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        distill_loss = torch.tensor(0.0, device=self.device)
        bc_loss = torch.tensor(0.0, device=self.device)
        if self.distill_enabled or (self.distill_mode == "dagger" and self.dagger_enabled):
            raise ValueError("Distillation is not supported in time_gru mode.")

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "distill_loss": distill_loss,
            "bc_loss": bc_loss,
            "kl_mean": kl_mean,
        }

    def _compute_kl_div(self, old_mu_batch, old_sigma_batch, mu_batch, sigma_batch) -> torch.Tensor:
        with torch.inference_mode():
            # Compute the KL divergence between the old and new action distributions
            old_dist = Normal(old_mu_batch, old_sigma_batch)
            new_dist = Normal(mu_batch, sigma_batch)
            kl = kl_divergence(old_dist, new_dist).sum(-1)
            kl_mean = torch.mean(kl)

            # Reduce the KL divergence across all GPUs
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
        return kl_mean

    def _update_learning_rate(self, kl_mean: torch.Tensor):
        if kl_mean > self.config.desired_kl * 2.0:
            self.actor_learning_rate = max(self.min_actor_learning_rate, self.actor_learning_rate / 1.5)
            self.critic_learning_rate = max(self.min_critic_learning_rate, self.critic_learning_rate / 1.5)
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
            self.actor_learning_rate = min(self.max_actor_learning_rate, self.actor_learning_rate * 1.5)
            self.critic_learning_rate = min(self.max_critic_learning_rate, self.critic_learning_rate * 1.5)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.critic_learning_rate

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            allow_non_strict = any(
                getattr(module, "supports_extra_input", False)
                for module in [
                    getattr(self.actor, "actor_module", None).module if hasattr(self.actor, "actor_module") else None,
                    getattr(self.critic, "critic_module", None).module if hasattr(self.critic, "critic_module") else None,
                ]
                if module is not None
            )
            try:
                self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
                self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            except RuntimeError as exc:
                if not allow_non_strict:
                    raise
                logger.warning("Strict checkpoint load failed; retrying with strict=False for extra-input modules.")
                actor_keys = self.actor.load_state_dict(loaded_dict["actor_model_state_dict"], strict=False)
                critic_keys = self.critic.load_state_dict(loaded_dict["critic_model_state_dict"], strict=False)
                if actor_keys.missing_keys or actor_keys.unexpected_keys:
                    logger.warning(
                        f"Actor non-strict load: missing={actor_keys.missing_keys}, "
                        f"unexpected={actor_keys.unexpected_keys}"
                    )
                if critic_keys.missing_keys or critic_keys.unexpected_keys:
                    logger.warning(
                        f"Critic non-strict load: missing={critic_keys.missing_keys}, "
                        f"unexpected={critic_keys.unexpected_keys}"
                    )
            actor_norm_state = loaded_dict.get("actor_obs_normalizer_state")
            critic_norm_state = loaded_dict.get("critic_obs_normalizer_state")
            if isinstance(actor_norm_state, dict):
                for key, state in actor_norm_state.items():
                    if state is not None and key in self.actor_obs_normalizers:
                        self.actor_obs_normalizers[key].load_state_dict(state)
            if isinstance(critic_norm_state, dict):
                for key, state in critic_norm_state.items():
                    if state is not None and key in self.critic_obs_normalizers:
                        self.critic_obs_normalizers[key].load_state_dict(state)
            if self.config.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
                self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"]["param_groups"][0]["lr"]
                logger.info("Optimizer loaded from checkpoint")
            self.current_learning_iteration = loaded_dict["iter"]
            self._restore_env_state(loaded_dict.get("env_state"))
            return loaded_dict.get("infos")
        return None

    def save(self, path, infos=None):
        def normalizer_states(normalizers: dict[str, nn.Module]):
            states: dict[str, dict | None] = {}
            for key, normalizer in normalizers.items():
                states[key] = normalizer.state_dict() if hasattr(normalizer, "state_dict") else None
            return states

        checkpoint_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "critic_model_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "actor_obs_normalizer_state": normalizer_states(self.actor_obs_normalizers),
            "critic_obs_normalizer_state": normalizer_states(self.critic_obs_normalizers),
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def export(self, onnx_file_path: str):
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `PPO.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self._eval_mode()

        # Save the .onnx file to filesystem
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            example_obs_dict = {"actor_obs": self._get_zero_input()}
            zero_perception = self._get_zero_perception_input()
            if zero_perception is not None:
                example_obs_dict[self.actor_perception_key] = zero_perception
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict=example_obs_dict,
            )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        # Extract URDF text from the robot config
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        # Upload the .onnx file to wandb
        self.logging_helper.save_to_wandb(onnx_file_path)

        # Restore original training state
        if was_training:
            self._train_mode()

    def _post_epoch_logging(self, it, loss_dict):
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self.actor.std.mean().item(),
            },
        }
        motion_command = None
        if self.env.command_manager is not None:
            motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is not None:
            train_logs = extra_log_dicts.setdefault("Train", {})
            motion_total = float(motion_command.motion.time_step_total)
            train_logs["mean_episode_length_motion_total"] = motion_total
            train_logs["mean_episode_length_motion_total/time"] = motion_total
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate
        # Use logging helper
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    def _reduce_parameters(self):
        grads = [
            param.grad.view(-1)
            for model in [self.actor, self.critic]
            for param in model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for model in [self.actor, self.critic]:
            for param in model.parameters():
                if param.grad is not None:
                    numel = param.numel()
                    param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad))
                    offset += numel

    def _synchronize_model_weights(self):
        """Synchronize actor and critic weights across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast critic weights from rank 0 to all other ranks
        for param in self.critic.parameters():
            torch.distributed.broadcast(param.data, src=0)

        logger.info(f"Synchronized model weights across {self.gpu_world_size} GPUs")

    def _normalize_advantages_multi_gpu(self, advantages):
        local_stats = torch.stack(
            [
                advantages.mean(),
                (advantages**2).mean(),
            ]
        )
        torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)

        global_mean = local_stats[0] / self.gpu_world_size
        global_sq_mean = local_stats[1] / self.gpu_world_size
        global_variance = global_sq_mean - global_mean**2
        global_std = torch.sqrt(global_variance + 1e-8)

        return (advantages - global_mean) / global_std

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    @property
    def actor_onnx_wrapper(self):
        class ActorWrapper(nn.Module):
            def __init__(self, actor, normalizers, keys, slices, perception_key):
                super().__init__()
                self.actor = actor
                self.normalizers = normalizers
                self.keys = keys
                self.slices = slices
                self.perception_key = perception_key

            def forward(self, actor_obs, perception_obs=None):
                parts = []
                for key in self.keys:
                    part = actor_obs[..., self.slices[key]]
                    normalizer = self.normalizers[key]
                    if isinstance(normalizer, EmpiricalNormalization):
                        part = normalizer(part, update=False)
                    else:
                        part = normalizer(part)
                    parts.append(part)
                actor_obs = torch.cat(parts, dim=-1)
                policy_state = {"actor_obs": actor_obs}
                if self.perception_key and perception_obs is not None:
                    policy_state[self.perception_key] = perception_obs
                return self.actor.act_inference(policy_state)

        return ActorWrapper(
            self.actor,
            self.actor_obs_normalizers,
            self.actor_obs_keys,
            self.actor_obs_slices,
            self.actor_perception_key,
        )

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.env.reset_all()
        example = {
            "actor_obs": torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1),
        }
        if self.actor_perception_key and self.actor_perception_key in obs_dict:
            example[self.actor_perception_key] = obs_dict[self.actor_perception_key]
        return example

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        self.eval_policy = self.get_inference_policy()

        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        critic_obs = torch.cat([actor_state["obs"][k] for k in self.critic_obs_keys], dim=1)
        actor_state["obs"]["critic_obs"] = critic_obs

        actor_state = self._pre_eval_env_step(actor_state)

        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        actor_obs_raw = torch.cat([actor_state["obs"][k] for k in self.actor_obs_keys], dim=1)
        actor_obs = actor_obs_raw
        actor_obs = self._normalize_actor_obs(actor_obs, update=False)
        policy_state = {"actor_obs": actor_obs}
        if self.actor_perception_key and self.actor_perception_key in actor_state["obs"]:
            policy_state[self.actor_perception_key] = actor_state["obs"][self.actor_perception_key]
        actions = self.eval_policy(policy_state)
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def get_inference_policy(self, device=None):
        self.actor.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor.to(device)
            for normalizer in self.actor_obs_normalizers.values():
                normalizer.to(device)
        for normalizer in self.actor_obs_normalizers.values():
            normalizer.eval()

        def _policy(obs_dict):
            actor_obs = obs_dict["actor_obs"]
            actor_obs = self._normalize_actor_obs(actor_obs, update=False)
            policy_state = {"actor_obs": actor_obs}
            if self.actor_perception_key and self.actor_perception_key in obs_dict:
                policy_state[self.actor_perception_key] = obs_dict[self.actor_perception_key]
            return self.actor.act_inference(policy_state)

        return _policy
