from __future__ import annotations

import dataclasses
import itertools
import json
import os
from pathlib import Path
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
from holosoma.config_types.algo import LayerConfig, ModuleConfig, PPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.normalization import EmpiricalNormalization
from holosoma.utils.step_timing import StepTiming, compact_timing_summary, env_int

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
        self.algo_timing = StepTiming.from_env(device=self.device)
        self._step_timing_interval = max(1, env_int("HOLOSOMA_STEP_TIMING_INTERVAL", default=1))
        self._last_algo_step_timing: dict[str, dict[str, float]] = {}
        self._last_env_step_timing: dict[str, dict[str, float]] = {}
        self.gpu_local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(self.gpu_world_size)))
        self._hierarchical_grad_reduce_ready = False
        self._hierarchical_grad_reduce_available = False
        self._hierarchical_local_group = None
        self._hierarchical_local_barrier_group = None
        self._hierarchical_leader_group = None
        self._hierarchical_leader_gloo_group = None
        self._hierarchical_local_leader_rank = 0
        self._hierarchical_is_leader_rank = False
        self._gloo_grad_reduce_ready = False
        self._gloo_grad_reduce_group = None
        self._gloo_barrier_ready = False
        self._gloo_barrier_group = None
        if self.algo_timing.enabled and self.is_main_process:
            logger.info(
                "Step timing enabled (sync_cuda={}, interval={})",
                self.algo_timing.sync_cuda,
                self._step_timing_interval,
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
        self.teacher_action_mix_ratio = 0.0
        self.teacher_action_mix_ratio_start: float | None = None
        self.teacher_action_mix_ratio_end: float | None = None
        self.teacher_action_mix_ratio_end_iteration = -1
        self.use_teacher_action_mix_schedule = False
        self.switch_to_rl_after = -1
        self.use_multi_teacher = False
        self.multi_teacher_select_obs_var = "teacher_checkpoint_index"
        self.ppo_start_epoch = -1
        self.dagger_end_epoch = -1
        self.ppo_target_coeff = 0.9
        self.ppo_start_coeff = 0.0
        self.ppo_start_noise_std: float | None = None
        self.ppo_start_noise_std_until_coeff = 0.1
        self._ppo_start_noise_std_cap_announced = False
        self.dagger_loss_coef = 10.0
        self.use_ppo_dagger_schedule = False
        self.ppo_coeff = 1.0
        self.distill_loss_fn = F.mse_loss
        self.dagger_ignore_zero_teacher_actions = True
        self.dagger_ignore_episode_initial_steps = 0
        self.dagger_match_std = False
        self.strict_teacher_load = True
        self.teacher_perception_obs_key = ""
        self.teacher_actor = None
        self.teacher_actors: list[nn.Module] = []
        self.teacher_actor_obs_normalizers: dict[str, nn.Module] = {}
        self.teacher_actor_obs_normalizers_list: list[dict[str, nn.Module]] = []
        self.fixed_bc_eval_num_samples = 0
        self.fixed_bc_eval_log_interval = 1
        self._fixed_bc_eval_ready = False
        self._fixed_bc_eval_size = 0
        self._fixed_bc_eval_actor_obs_parts: list[torch.Tensor] = []
        self._fixed_bc_eval_teacher_actions_parts: list[torch.Tensor] = []
        self._fixed_bc_eval_actor_perception_parts: list[torch.Tensor] = []
        self._fixed_bc_eval_dataset: dict[str, torch.Tensor] = {}

    def _reset_step_timing(self) -> None:
        if not self.algo_timing.enabled:
            return
        self.algo_timing.reset()
        env_timing = getattr(self.env, "step_timing", None)
        if env_timing is not None:
            env_timing.reset()

    def _capture_step_timing(self) -> None:
        if not self.algo_timing.enabled:
            return
        self._last_algo_step_timing = self.algo_timing.snapshot(reset=False)
        env_timing = getattr(self.env, "step_timing", None)
        if env_timing is None:
            self._last_env_step_timing = {}
        else:
            self._last_env_step_timing = env_timing.snapshot(reset=False)

    def _emit_step_timing_summary(self, it: int) -> None:
        if not self.algo_timing.enabled or not self.is_main_process:
            return
        if it % self._step_timing_interval != 0:
            return
        algo_order = (
            "iteration/rollout",
            "iteration/training_step",
            "rollout/env_step",
            "rollout/teacher_actions",
            "rollout/actor_forward",
            "rollout/critic_forward",
            "rollout/returns",
            "training/update_algo_step",
        )
        env_order = (
            "env_step_total",
            "physics",
            "physics/apply_force",
            "physics/simulate_step",
            "physics/sim/write_data_to_sim",
            "physics/sim/write_robot_to_sim",
            "physics/sim/robot/apply_actuator_model",
            "physics/sim/robot/set_dof_forces",
            "physics/sim/write_nonrobot_to_sim",
            "physics/sim/step",
            "physics/sim/scene_update",
            "physics/sim/update_dof_refs",
            "post/perception",
            "post/reward",
            "post/reward/term/offline_contact_guidance",
            "post/log_update",
            "post/log_update/update_log_dict",
            "post/log_update/motion_metrics",
            "post/tasks",
            "post/tasks/command_manager",
            "post/tasks/motion/contact_prior",
            "post/tasks/motion/future_targets",
            "post/tasks/motion/relative_body_pose",
            "post/observations",
            "post/reset_envs",
            "pre_physics",
        )
        logger.info(
            "StepTiming iter={} algo {}",
            it,
            compact_timing_summary(self._last_algo_step_timing, algo_order, max_extra=4),
        )
        logger.info(
            "StepTiming iter={} env {}",
            it,
            compact_timing_summary(self._last_env_step_timing, env_order, max_extra=4),
        )

    def _add_step_timing_logs(self, extra_log_dicts: dict[str, dict[str, float]]) -> None:
        if not self.algo_timing.enabled:
            return
        timing_logs = extra_log_dicts.setdefault("Timing", {})
        selected = (
            ("algo", self._last_algo_step_timing, "iteration/rollout"),
            ("algo", self._last_algo_step_timing, "iteration/training_step"),
            ("algo", self._last_algo_step_timing, "rollout/env_step"),
            ("algo", self._last_algo_step_timing, "rollout/teacher_actions"),
            ("algo", self._last_algo_step_timing, "rollout/actor_forward"),
            ("algo", self._last_algo_step_timing, "rollout/critic_forward"),
            ("algo", self._last_algo_step_timing, "rollout/returns"),
            ("algo", self._last_algo_step_timing, "training/update_algo_step"),
            ("env", self._last_env_step_timing, "env_step_total"),
            ("env", self._last_env_step_timing, "physics"),
            ("env", self._last_env_step_timing, "post/perception"),
            ("env", self._last_env_step_timing, "post/reward"),
            ("env", self._last_env_step_timing, "post/log_update"),
            ("env", self._last_env_step_timing, "post/log_update/update_log_dict"),
            ("env", self._last_env_step_timing, "post/log_update/motion_metrics"),
            ("env", self._last_env_step_timing, "post/tasks"),
            ("env", self._last_env_step_timing, "post/tasks/command_manager"),
            ("env", self._last_env_step_timing, "post/tasks/motion/contact_prior"),
            ("env", self._last_env_step_timing, "post/tasks/motion/future_targets"),
            ("env", self._last_env_step_timing, "post/tasks/motion/relative_body_pose"),
            ("env", self._last_env_step_timing, "post/observations"),
            ("env", self._last_env_step_timing, "post/reset_envs"),
        )
        for prefix, snapshot, name in selected:
            stats = snapshot.get(name)
            if stats is None:
                continue
            safe_name = name.replace("/", "_")
            timing_logs[f"{prefix}_{safe_name}_sum_ms"] = float(stats.get("sum_ms", 0.0))
            timing_logs[f"{prefix}_{safe_name}_mean_ms"] = float(stats.get("mean_ms", 0.0))

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
        if not self.config.normalize_actor_obs:
            return obs
        return self._normalize_concat_obs(
            obs, self.actor_obs_keys, self.actor_obs_slices, self.actor_obs_normalizers, update=update
        )

    def _normalize_critic_obs(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if not self.config.normalize_critic_obs:
            return obs
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
        if all(not isinstance(normalizer, EmpiricalNormalization) for normalizer in normalizers.values()):
            return obs
        return self._normalize_concat_obs(
            obs,
            self.teacher_obs_keys,
            self.teacher_obs_slices,
            normalizers,
            update=False,
        )

    def _get_actor_std_for_loss(self, actor: nn.Module) -> torch.Tensor:
        std = torch.nan_to_num(
            actor.std,
            nan=self.config.init_noise_std,
            posinf=10.0,
            neginf=0.0,
        )
        std = torch.clamp(std, min=1e-6)
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

    def _configure_active_observation_groups(self) -> None:
        observation_manager = getattr(self.env, "observation_manager", None)
        if observation_manager is None or not hasattr(observation_manager, "set_active_groups"):
            return

        disabled = os.environ.get("HOLOSOMA_DISABLE_ACTIVE_OBS_GROUP_FILTER", "").strip().lower()
        if disabled in ("1", "true", "yes", "on"):
            observation_manager.set_active_groups(None)
            return

        required: list[str] = []

        def add_group(group_name: str | None) -> None:
            if not group_name:
                return
            if group_name not in required:
                required.append(group_name)

        def add_groups(group_names: list[str] | tuple[str, ...] | None) -> None:
            if not group_names:
                return
            for group_name in group_names:
                add_group(group_name)

        add_groups(self.actor_obs_keys)
        add_groups(self.critic_obs_keys)
        add_groups(self.teacher_obs_keys)
        add_group(self.actor_perception_key)
        add_group(self.critic_perception_key)
        add_group(self.teacher_perception_obs_key)
        if self.use_multi_teacher:
            add_group(self.multi_teacher_select_obs_var)

        observation_manager.set_active_groups(required)
        if self.is_main_process:
            total = len(getattr(observation_manager.cfg, "groups", {}))
            logger.info("Active PPO observation groups: {} / {} {}", len(required), total, required)

    def setup(self):
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} PPO.setup models/optimizers ready", self.gpu_global_rank)
        self._configure_active_observation_groups()
        logger.info("Setting up Storage")
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} PPO.setup storage begin", self.gpu_global_rank)
        self._setup_storage()
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} PPO.setup storage finished", self.gpu_global_rank)

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

    def _setup_models_and_optimizer(self):
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup actor/critic begin", self.gpu_global_rank)
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
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup actor/critic finished", self.gpu_global_rank)
        self.use_time_gru = bool(
            getattr(self.actor, "perception_time_gru", None) is not None
            or getattr(self.critic, "perception_time_gru", None) is not None
        )

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup distillation begin", self.gpu_global_rank)
        self._setup_distillation()
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} setup distillation finished", self.gpu_global_rank)

        if self.use_symmetry:
            self.symmetry_utils = SymmetryUtils(self.env)

        # Synchronize model weights across GPUs after initialization
        if self.is_multi_gpu:
            if debug_heartbeat:
                logger.info("Heartbeat: rank {} model weight sync begin", self.gpu_global_rank)
            self._synchronize_model_weights()
            if debug_heartbeat:
                logger.info("Heartbeat: rank {} model weight sync finished", self.gpu_global_rank)

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} optimizer setup begin", self.gpu_global_rank)
        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = instantiate(
            self.config.critic_optimizer, params=self.critic.parameters(), lr=self.critic_learning_rate
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} optimizer setup finished", self.gpu_global_rank)

    def _build_teacher_actor_config(self, obs_keys: list[str], base_actor_cfg: ModuleConfig | None = None):
        actor_cfg = base_actor_cfg or self.config.module_dict.actor
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
        if layer_cfg.perception_input_name:
            resolved_perception_key = ""
            if self.teacher_perception_obs_key:
                resolved_perception_key = self.teacher_perception_obs_key
            elif layer_cfg.perception_input_name in self.algo_obs_dim_dict:
                resolved_perception_key = layer_cfg.perception_input_name
            if resolved_perception_key:
                layer_cfg = dataclasses.replace(layer_cfg, perception_input_name=resolved_perception_key)
            else:
                layer_cfg = dataclasses.replace(layer_cfg, perception_input_name="")
                removed_perception_input = True

        actor_type = actor_cfg.type
        # In strict mode we do not auto-fallback teacher architecture on obs mismatch.
        if self.strict_teacher_load and actor_type == "MLPPerceptionEncoder" and not layer_cfg.perception_input_name:
            raise ValueError(
                "Teacher checkpoint expects perception input, but current teacher_obs_keys remove it. "
                "Set matching teacher_obs_keys (e.g. legacy group) or disable strict_teacher_load explicitly."
            )
        # Backward-compatible fallback for non-strict mode only.
        if (not self.strict_teacher_load) and actor_type == "MLPPerceptionEncoder" and not layer_cfg.perception_input_name:
            actor_type = "MLP"
        if removed_perception_input and layer_cfg.extra_input_to_hidden:
            layer_cfg = dataclasses.replace(layer_cfg, extra_input_to_hidden=False)

        return dataclasses.replace(actor_cfg, type=actor_type, input_dim=list(obs_keys), layer_config=layer_cfg)

    def _extract_teacher_actor_config(self, teacher_state: dict) -> ModuleConfig | None:
        exp_cfg = teacher_state.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            return None
        try:
            actor_cfg_raw = exp_cfg["algo"]["config"]["module_dict"]["actor"]
        except (KeyError, TypeError):
            return None
        if not isinstance(actor_cfg_raw, dict):
            return None
        layer_cfg_raw = actor_cfg_raw.get("layer_config")
        if not isinstance(layer_cfg_raw, dict):
            return None
        layer_kwargs = dict(layer_cfg_raw)
        module_input_name = layer_kwargs.get("module_input_name")
        if isinstance(module_input_name, list):
            layer_kwargs["module_input_name"] = tuple(module_input_name)
        try:
            layer_cfg = LayerConfig(**layer_kwargs)
            actor_cfg = ModuleConfig(
                type=str(actor_cfg_raw.get("type", "MLP")),
                input_dim=list(actor_cfg_raw.get("input_dim", [])),
                output_dim=list(actor_cfg_raw.get("output_dim", [])),
                layer_config=layer_cfg,
                min_noise_std=actor_cfg_raw.get("min_noise_std"),
                min_mean_noise_std=actor_cfg_raw.get("min_mean_noise_std"),
            )
            return actor_cfg
        except Exception as exc:
            logger.warning(f"Failed to parse teacher actor config from checkpoint; falling back to runtime config. {exc}")
            return None

    def _validate_teacher_checkpoint_runtime_config(
        self,
        teacher_state: dict,
        *,
        obs_keys: list[str],
        teacher_actor_cfg: ModuleConfig,
    ) -> None:
        if not self.strict_teacher_load:
            return

        exp_cfg = teacher_state.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            return

        checkpoint_groups = exp_cfg.get("observation", {}).get("groups", {})
        runtime_groups = getattr(getattr(self.env, "observation_manager", None), "cfg", None)
        runtime_groups = getattr(runtime_groups, "groups", {}) if runtime_groups is not None else {}

        mismatches: list[str] = []
        matched_groups: list[str] = []
        for obs_key in obs_keys:
            checkpoint_group = checkpoint_groups.get(obs_key) if isinstance(checkpoint_groups, dict) else None
            if not isinstance(checkpoint_group, dict):
                continue

            runtime_group = runtime_groups.get(obs_key) if isinstance(runtime_groups, dict) else None
            if runtime_group is None:
                mismatches.append(f"{obs_key}: missing runtime observation group")
                continue

            checkpoint_history = checkpoint_group.get("history_length")
            runtime_history = getattr(runtime_group, "history_length", None)
            if checkpoint_history != runtime_history:
                mismatches.append(
                    f"{obs_key}: history_length checkpoint={checkpoint_history} runtime={runtime_history}"
                )

            checkpoint_terms = checkpoint_group.get("terms", {})
            checkpoint_term_names = list(checkpoint_terms.keys()) if isinstance(checkpoint_terms, dict) else []
            runtime_terms = getattr(runtime_group, "terms", {})
            runtime_term_names = list(runtime_terms.keys()) if isinstance(runtime_terms, dict) else []
            if checkpoint_term_names != runtime_term_names:
                mismatches.append(
                    f"{obs_key}: terms checkpoint={checkpoint_term_names} runtime={runtime_term_names}"
                )
            matched_groups.append(obs_key)

        try:
            checkpoint_actor_cfg = exp_cfg["algo"]["config"]["module_dict"]["actor"]
        except (KeyError, TypeError):
            checkpoint_actor_cfg = {}
        checkpoint_actor_inputs = (
            checkpoint_actor_cfg.get("input_dim") if isinstance(checkpoint_actor_cfg, dict) else None
        )
        if isinstance(checkpoint_actor_inputs, tuple):
            checkpoint_actor_inputs = list(checkpoint_actor_inputs)
        if checkpoint_actor_inputs and list(checkpoint_actor_inputs) != list(obs_keys):
            logger.warning(
                "Teacher checkpoint actor input keys differ from runtime teacher_obs_keys: "
                "checkpoint={} runtime={}. Strict state_dict loading still enforces tensor compatibility; "
                "use an exact compatibility observation group only when the runtime group preserves the "
                "checkpoint term order and history.",
                checkpoint_actor_inputs,
                obs_keys,
            )

        checkpoint_layer_cfg = (
            checkpoint_actor_cfg.get("layer_config", {}) if isinstance(checkpoint_actor_cfg, dict) else {}
        )
        checkpoint_perception_key = ""
        if isinstance(checkpoint_layer_cfg, dict):
            checkpoint_perception_key = str(checkpoint_layer_cfg.get("perception_input_name", "") or "")
        runtime_perception_key = str(teacher_actor_cfg.layer_config.perception_input_name or "")
        if bool(checkpoint_perception_key) != bool(runtime_perception_key):
            mismatches.append(
                "teacher perception input presence mismatch: "
                f"checkpoint={checkpoint_perception_key or '<none>'} "
                f"runtime={runtime_perception_key or '<none>'}"
            )

        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                "Teacher checkpoint/runtime observation config mismatch under strict_teacher_load. "
                f"{details}"
            )

        if matched_groups:
            logger.info(
                "Teacher checkpoint observation config matches runtime for groups: {}",
                ", ".join(matched_groups),
            )

    def _load_teacher_actor(
        self, ckpt_path: str, obs_keys: list[str] | None = None
    ) -> tuple[nn.Module, dict[str, nn.Module]]:
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        if ckpt_path.startswith("wandb://"):
            from holosoma.utils.eval_utils import load_checkpoint  # noqa: PLC0415

            teacher_cache_dir = self.log_dir / ".teacher_ckpt_cache" / f"rank_{self.gpu_global_rank}"
            ckpt_path = str(load_checkpoint(ckpt_path, str(teacher_cache_dir)))

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher torch.load begin {}", self.gpu_global_rank, ckpt_path)
        teacher_state = torch.load(ckpt_path, map_location=self.device)
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher torch.load finished", self.gpu_global_rank)
        teacher_obs_keys = obs_keys if obs_keys is not None else self.actor_obs_keys
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher config build begin", self.gpu_global_rank)
        teacher_actor_base_cfg = self._extract_teacher_actor_config(teacher_state)
        teacher_actor_cfg = self._build_teacher_actor_config(teacher_obs_keys, base_actor_cfg=teacher_actor_base_cfg)
        self._validate_teacher_checkpoint_runtime_config(
            teacher_state,
            obs_keys=teacher_obs_keys,
            teacher_actor_cfg=teacher_actor_cfg,
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher actor module build begin", self.gpu_global_rank)
        teacher_actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=teacher_actor_cfg,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher state_dict load begin", self.gpu_global_rank)
        try:
            teacher_actor.load_state_dict(teacher_state["actor_model_state_dict"])
        except RuntimeError:
            if self.strict_teacher_load:
                raise
            allow_non_strict = False
            if hasattr(teacher_actor, "actor_module"):
                allow_non_strict = getattr(teacher_actor.actor_module.module, "supports_extra_input", False)
            if not allow_non_strict:
                raise
            logger.warning("Strict teacher load failed; retrying with strict=False for extra-input modules.")
            teacher_actor.load_state_dict(teacher_state["actor_model_state_dict"], strict=False)
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher state_dict load finished", self.gpu_global_rank)
        teacher_actor.eval()
        for param in teacher_actor.parameters():
            if isinstance(param, UninitializedParameter):
                continue
            param.requires_grad_(False)

        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher normalizers build begin", self.gpu_global_rank)
        teacher_normalizers = self._build_group_normalizers(teacher_obs_keys, self.config.normalize_actor_obs)
        actor_norm_state = teacher_state.get("actor_obs_normalizer_state")
        if isinstance(actor_norm_state, dict):
            for key, state in actor_norm_state.items():
                if state is not None and key in teacher_normalizers:
                    teacher_normalizers[key].load_state_dict(state)
        for normalizer in teacher_normalizers.values():
            normalizer.eval()
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} teacher load finished", self.gpu_global_rank)
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
        self.teacher_use_stochastic_actions = bool(getattr(distill_cfg, "teacher_use_stochastic_actions", False))
        self.teacher_action_mix_ratio = float(getattr(distill_cfg, "teacher_action_mix_ratio", 0.0))
        if not (0.0 <= self.teacher_action_mix_ratio <= 1.0):
            raise ValueError(
                f"distill.teacher_action_mix_ratio must be in [0.0, 1.0], got {self.teacher_action_mix_ratio}."
            )
        teacher_action_mix_ratio_start = getattr(distill_cfg, "teacher_action_mix_ratio_start", None)
        teacher_action_mix_ratio_end = getattr(distill_cfg, "teacher_action_mix_ratio_end", None)
        self.teacher_action_mix_ratio_end_iteration = int(
            getattr(distill_cfg, "teacher_action_mix_ratio_end_iteration", -1)
        )
        if (teacher_action_mix_ratio_start is None) != (teacher_action_mix_ratio_end is None):
            raise ValueError(
                "distill.teacher_action_mix_ratio_start and distill.teacher_action_mix_ratio_end must be set together."
            )
        if teacher_action_mix_ratio_start is not None and teacher_action_mix_ratio_end is not None:
            self.teacher_action_mix_ratio_start = float(teacher_action_mix_ratio_start)
            self.teacher_action_mix_ratio_end = float(teacher_action_mix_ratio_end)
            if not (0.0 <= self.teacher_action_mix_ratio_start <= 1.0):
                raise ValueError(
                    "distill.teacher_action_mix_ratio_start must be in [0.0, 1.0], "
                    f"got {self.teacher_action_mix_ratio_start}."
                )
            if not (0.0 <= self.teacher_action_mix_ratio_end <= 1.0):
                raise ValueError(
                    f"distill.teacher_action_mix_ratio_end must be in [0.0, 1.0], got {self.teacher_action_mix_ratio_end}."
                )
            if self.teacher_action_mix_ratio_end_iteration < 0:
                raise ValueError(
                    "distill.teacher_action_mix_ratio_end_iteration must be >= 0 when teacher-action mix scheduling is enabled."
                )
            self.use_teacher_action_mix_schedule = True
            self.teacher_action_mix_ratio = self.teacher_action_mix_ratio_start
        self.switch_to_rl_after = int(distill_cfg.switch_to_rl_after)
        self.use_multi_teacher = bool(distill_cfg.use_multi_teacher)
        self.multi_teacher_select_obs_var = str(distill_cfg.multi_teacher_select_obs_var)
        self.ppo_start_epoch = int(getattr(distill_cfg, "ppo_start_epoch", -1))
        self.dagger_end_epoch = int(getattr(distill_cfg, "dagger_end_epoch", -1))
        self.ppo_target_coeff = float(getattr(distill_cfg, "ppo_target_coeff", 0.9))
        if not (0.0 <= self.ppo_target_coeff <= 1.0):
            raise ValueError(f"distill.ppo_target_coeff must be in [0.0, 1.0], got {self.ppo_target_coeff}.")
        self.ppo_start_coeff = float(getattr(distill_cfg, "ppo_start_coeff", 0.0))
        if not (0.0 <= self.ppo_start_coeff <= 1.0):
            raise ValueError(f"distill.ppo_start_coeff must be in [0.0, 1.0], got {self.ppo_start_coeff}.")
        if self.ppo_start_coeff > self.ppo_target_coeff:
            raise ValueError(
                "distill.ppo_start_coeff must be <= distill.ppo_target_coeff, "
                f"got {self.ppo_start_coeff} > {self.ppo_target_coeff}."
            )
        raw_start_noise_std = getattr(distill_cfg, "ppo_start_noise_std", None)
        self.ppo_start_noise_std = None if raw_start_noise_std is None else float(raw_start_noise_std)
        if self.ppo_start_noise_std is not None and self.ppo_start_noise_std <= 0.0:
            raise ValueError(
                "distill.ppo_start_noise_std must be > 0.0 when set, "
                f"got {self.ppo_start_noise_std}."
            )
        self.ppo_start_noise_std_until_coeff = float(
            getattr(distill_cfg, "ppo_start_noise_std_until_coeff", 0.1)
        )
        if not (0.0 <= self.ppo_start_noise_std_until_coeff <= 1.0):
            raise ValueError(
                "distill.ppo_start_noise_std_until_coeff must be in [0.0, 1.0], "
                f"got {self.ppo_start_noise_std_until_coeff}."
            )
        self.ppo_schedule_step_epochs = int(getattr(distill_cfg, "ppo_schedule_step_epochs", 0))
        if self.ppo_schedule_step_epochs < 0:
            raise ValueError(
                "distill.ppo_schedule_step_epochs must be >= 0, "
                f"got {self.ppo_schedule_step_epochs}."
            )
        self.dagger_loss_coef = float(getattr(distill_cfg, "dagger_loss_coef", 10.0))
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
        self.dagger_ignore_episode_initial_steps = max(
            0, int(getattr(distill_cfg, "dagger_ignore_episode_initial_steps", 0))
        )
        self.dagger_match_std = bool(getattr(distill_cfg, "dagger_match_std", False))
        self.strict_teacher_load = bool(getattr(distill_cfg, "strict_teacher_load", True))
        self.fixed_bc_eval_num_samples = max(0, int(getattr(distill_cfg, "fixed_bc_eval_num_samples", 0)))
        self.fixed_bc_eval_log_interval = max(1, int(getattr(distill_cfg, "fixed_bc_eval_log_interval", 1)))
        teacher_checkpoint = distill_cfg.policy_to_clone or distill_cfg.teacher_checkpoint
        if self.distill_mode == "dagger":
            if not teacher_checkpoint:
                return
            if self.bc_loss_coef <= 0.0 and self.switch_to_rl_after <= 0 and not self.use_ppo_dagger_schedule:
                return
        elif not distill_cfg.enabled:
            return

        teacher_perception_obs_key = getattr(distill_cfg, "teacher_perception_obs_key", None)
        self.teacher_perception_obs_key = str(teacher_perception_obs_key).strip() if teacher_perception_obs_key else ""
        if self.teacher_perception_obs_key and self.teacher_perception_obs_key not in self.algo_obs_dim_dict:
            raise ValueError(
                "Distillation teacher_perception_obs_key not found in observation manager: "
                f"{self.teacher_perception_obs_key}"
            )

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

        if self.distill_mode == "dagger":
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

    def _maybe_capture_fixed_bc_eval_samples(
        self,
        *,
        actor_obs_raw: torch.Tensor,
        actor_perception_obs: torch.Tensor | None,
        teacher_actions: torch.Tensor | None,
        teacher_bc_mask: torch.Tensor | None,
    ) -> None:
        if not self.is_main_process or self.fixed_bc_eval_num_samples <= 0:
            return
        if not self.dagger_enabled or teacher_actions is None or self._fixed_bc_eval_ready:
            return
        if self.actor_perception_key and actor_perception_obs is None:
            return

        valid_mask = torch.ones((teacher_actions.shape[0],), device=teacher_actions.device, dtype=torch.bool)
        if teacher_bc_mask is not None:
            valid_mask &= teacher_bc_mask.view(-1).to(dtype=torch.bool)
        if self.dagger_ignore_zero_teacher_actions:
            valid_mask &= ~torch.all(teacher_actions == 0.0, dim=-1)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return

        remaining = self.fixed_bc_eval_num_samples - self._fixed_bc_eval_size
        if remaining <= 0:
            return
        selected = valid_indices[:remaining]
        self._fixed_bc_eval_actor_obs_parts.append(actor_obs_raw[selected].detach().cpu().clone())
        self._fixed_bc_eval_teacher_actions_parts.append(teacher_actions[selected].detach().cpu().clone())
        if self.actor_perception_key:
            assert actor_perception_obs is not None
            self._fixed_bc_eval_actor_perception_parts.append(actor_perception_obs[selected].detach().cpu().clone())
        self._fixed_bc_eval_size += int(selected.numel())

        if self._fixed_bc_eval_size < self.fixed_bc_eval_num_samples:
            return

        self._fixed_bc_eval_dataset = {
            "actor_obs_raw": torch.cat(self._fixed_bc_eval_actor_obs_parts, dim=0)[: self.fixed_bc_eval_num_samples],
            "teacher_actions": torch.cat(self._fixed_bc_eval_teacher_actions_parts, dim=0)[: self.fixed_bc_eval_num_samples],
        }
        if self.actor_perception_key:
            self._fixed_bc_eval_dataset["actor_perception"] = torch.cat(
                self._fixed_bc_eval_actor_perception_parts, dim=0
            )[: self.fixed_bc_eval_num_samples]
        self._fixed_bc_eval_actor_obs_parts.clear()
        self._fixed_bc_eval_teacher_actions_parts.clear()
        self._fixed_bc_eval_actor_perception_parts.clear()
        self._fixed_bc_eval_ready = True

    @torch.no_grad()
    def _get_fixed_bc_eval_metrics(self, current_iteration: int) -> dict[str, float]:
        if not self.is_main_process or not self._fixed_bc_eval_ready:
            return {}
        if self.fixed_bc_eval_num_samples <= 0 or self.fixed_bc_eval_log_interval <= 0:
            return {}
        if current_iteration % self.fixed_bc_eval_log_interval != 0:
            return {}

        actor_training = self.actor.training
        normalizer_training = {
            key: normalizer.training for key, normalizer in self.actor_obs_normalizers.items() if hasattr(normalizer, "training")
        }
        self.actor.eval()
        for normalizer in self.actor_obs_normalizers.values():
            if hasattr(normalizer, "eval"):
                normalizer.eval()

        actor_obs_raw = self._fixed_bc_eval_dataset["actor_obs_raw"].to(self.device)
        actor_obs = self._normalize_actor_obs(actor_obs_raw, update=False)
        policy_state = {"actor_obs": actor_obs}
        if self.actor_perception_key and "actor_perception" in self._fixed_bc_eval_dataset:
            policy_state[self.actor_perception_key] = self._fixed_bc_eval_dataset["actor_perception"].to(self.device)
        student_actions = self.actor.act_inference(policy_state)
        teacher_actions = self._fixed_bc_eval_dataset["teacher_actions"].to(self.device)
        action_error = student_actions - teacher_actions

        if actor_training:
            self.actor.train()
        for key, normalizer in self.actor_obs_normalizers.items():
            if hasattr(normalizer, "train") and normalizer_training.get(key, False):
                normalizer.train()

        return {
            "fixed_bc_mu_mse": float(action_error.pow(2).mean().item()),
            "fixed_bc_num_samples": float(teacher_actions.shape[0]),
        }

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
            if (
                self.dagger_ignore_episode_initial_steps > 0
                or self._motion_command_supports_runtime_default_pose_prepend_mask()
            ):
                self.storage.register("teacher_bc_mask", shape=(1,), dtype=torch.bool)
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

    def _distributed_barrier(self) -> None:
        if not self.is_multi_gpu or not torch.distributed.is_initialized():
            return
        if os.environ.get("HOLOSOMA_GLOO_BARRIER", "").lower() in ("1", "true", "yes", "on"):
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                torch.distributed.barrier(group=gloo_group)
                return
        try:
            torch.distributed.barrier(device_ids=[int(self.gpu_local_rank)])
        except TypeError:
            torch.distributed.barrier()

    def _gloo_small_collectives_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _all_reduce_small_tensor(
        self,
        tensor: torch.Tensor,
        *,
        op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
    ) -> torch.Tensor:
        if self._gloo_small_collectives_enabled():
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                cpu_tensor = tensor.detach().cpu()
                torch.distributed.all_reduce(cpu_tensor, op=op, group=gloo_group)
                return cpu_tensor.to(device=tensor.device, dtype=tensor.dtype)
        torch.distributed.all_reduce(tensor, op=op)
        return tensor

    def _broadcast_tensor(self, tensor: torch.Tensor, *, src: int = 0) -> None:
        if self._gloo_small_collectives_enabled():
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                cpu_tensor = tensor.detach().cpu()
                torch.distributed.broadcast(cpu_tensor, src=src, group=gloo_group)
                tensor.detach().copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
                return
        torch.distributed.broadcast(tensor, src=src)

    def _synchronize_curriculum_metrics(self):
        if not self.has_curricula_enabled():
            return
        env = self._unwrap_env()
        if self._gloo_small_collectives_enabled():
            gloo_group = self._setup_gloo_barrier_group()
            if gloo_group is not None:
                env.synchronize_curriculum_state(
                    device="cpu",
                    world_size=self.gpu_world_size,
                    process_group=gloo_group,
                )
                return
        env.synchronize_curriculum_state(device=self.device, world_size=self.gpu_world_size)

    def learn(self):
        self._train_mode()

        logger.info("Entering PPO.learn at iteration {}.", self.current_learning_iteration)
        logger.info("PPO.learn initial reset_all starting.")
        obs_dict = self.env.reset_all()
        logger.info("PPO.learn initial reset_all finished with obs keys: {}.", sorted(obs_dict.keys()))
        self._reset_step_timing()

        # Initialize environments with different episode length buffers
        # Must happen AFTER reset_all() to avoid being overwritten by reset
        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
        logger.info("PPO.learn initial obs transfer to {} finished.", self.device)

        run_end_iteration = self.current_learning_iteration + self.config.num_learning_iterations
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        start_iteration = self.current_learning_iteration
        skip_initial_checkpoint = os.environ.get("HOLOSOMA_SKIP_INITIAL_CHECKPOINT", "1").lower() not in (
            "",
            "0",
            "false",
            "no",
        )
        for it in range(
            self.current_learning_iteration,
            run_end_iteration,
        ):
            self.current_learning_iteration = it
            self._reset_step_timing()
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} starting teacher/curriculum updates", it)
            self._adjust_teacher_action_mix_ratio(it)
            self._apply_ppo_start_noise_std_cap(it)
            self._sync_training_curriculum_state(
                current_iteration=it,
                total_iterations=run_end_iteration,
            )
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} finished teacher/curriculum state update", it)

            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} starting curriculum metric sync", it)
                self._synchronize_curriculum_metrics()
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} finished curriculum metric sync", it)

            if debug_heartbeat:
                logger.info("Heartbeat: iter {} starting rollout", it)
            if self.algo_timing.enabled:
                with self.algo_timing.record("iteration/rollout"):
                    with self.logging_helper.record_collection_time():
                        obs_dict = self._rollout_step(obs_dict)
            else:
                with self.logging_helper.record_collection_time():
                    obs_dict = self._rollout_step(obs_dict)
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} finished rollout", it)

            if debug_heartbeat:
                logger.info("Heartbeat: iter {} starting training_step", it)
            if self.algo_timing.enabled:
                with self.algo_timing.record("iteration/training_step"):
                    with self.logging_helper.record_learn_time():
                        loss_dict = self._training_step()
            else:
                with self.logging_helper.record_learn_time():
                    loss_dict = self._training_step()
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} finished training_step", it)
            self._capture_step_timing()
            self._emit_step_timing_summary(it)

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            should_save_checkpoint = it % self.config.save_interval == 0
            if should_save_checkpoint and skip_initial_checkpoint and it == start_iteration:
                logger.info("Skipping checkpoint save at initial iteration {}.", it)
                should_save_checkpoint = False

            if should_save_checkpoint:
                self._distributed_barrier()
                if self.is_main_process:
                    self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                    self._export_onnx_checkpoint(os.path.join(self.log_dir, f"model_{it:05d}.onnx"))
                self._distributed_barrier()

        self._distributed_barrier()
        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            onnx_path = os.path.join(
                self.log_dir,
                f"model_{self.current_learning_iteration:05d}.onnx",
            )
            self._export_onnx_checkpoint(onnx_path)
        self._distributed_barrier()

    def _should_export_onnx(self) -> bool:
        if self._experiment_config is None:
            return True
        return bool(getattr(self._experiment_config.training, "export_onnx", True))

    def _export_onnx_checkpoint(self, onnx_file_path: str) -> None:
        if not self._should_export_onnx():
            return
        try:
            self.export(onnx_file_path=onnx_file_path)
        except Exception:
            logger.exception("ONNX export failed for {}; continuing after saving the .pt checkpoint.", onnx_file_path)

    def _select_teacher_actions(
        self, teacher_obs_raw: torch.Tensor, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        def teacher_act(teacher_actor: nn.Module, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.teacher_use_stochastic_actions:
                return teacher_actor.act(policy_state)
            return teacher_actor.act_inference(policy_state)

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
                teacher_policy_state = {"actor_obs": teacher_obs}
                teacher_perception_key = str(getattr(teacher_actor, "perception_input_name", "") or "")
                if teacher_perception_key:
                    if teacher_perception_key not in obs_dict:
                        raise ValueError(
                            f"Teacher perception obs '{teacher_perception_key}' not found in current observation dict."
                        )
                    teacher_policy_state[teacher_perception_key] = obs_dict[teacher_perception_key][mask]
                teacher_actions[mask] = teacher_act(teacher_actor, teacher_policy_state)
            return teacher_actions, teacher_indices

        assert self.teacher_actor is not None, "Teacher actor is not initialized."
        teacher_obs = self._normalize_teacher_actor_obs(teacher_obs_raw)
        teacher_policy_state = {"actor_obs": teacher_obs}
        teacher_perception_key = str(getattr(self.teacher_actor, "perception_input_name", "") or "")
        if teacher_perception_key:
            if teacher_perception_key not in obs_dict:
                raise ValueError(
                    f"Teacher perception obs '{teacher_perception_key}' not found in current observation dict."
                )
            teacher_policy_state[teacher_perception_key] = obs_dict[teacher_perception_key]
        teacher_actions = teacher_act(self.teacher_actor, teacher_policy_state)
        return teacher_actions, None

    def _compute_ppo_dagger_coeff_for_epoch(self, current_epoch: int) -> float:
        if not self.use_ppo_dagger_schedule:
            return 1.0

        if current_epoch < self.ppo_start_epoch:
            return 0.0
        if current_epoch >= self.dagger_end_epoch:
            return self.ppo_target_coeff

        total_epochs = max(1, self.dagger_end_epoch - self.ppo_start_epoch)
        ppo_epochs = max(0, current_epoch - self.ppo_start_epoch)
        coeff_span = self.ppo_target_coeff - self.ppo_start_coeff
        if self.ppo_schedule_step_epochs > 0:
            step_epochs = max(1, self.ppo_schedule_step_epochs)
            total_steps = max(1, (total_epochs + step_epochs - 1) // step_epochs)
            completed_steps = max(0, ppo_epochs // step_epochs)
            progress = min(float(completed_steps) / float(total_steps), 1.0)
            return self.ppo_start_coeff + progress * coeff_span

        progress = min(float(ppo_epochs) / float(total_epochs), 1.0)
        return self.ppo_start_coeff + progress * coeff_span

    def _adjust_ppo_dagger_coeff(self, current_epoch: int) -> None:
        """PPO/DAgger curriculum mixing schedule.

        - epoch < ppo_start_epoch: ppo_coeff = 0.0
        - epoch >= dagger_end_epoch: ppo_coeff = ppo_target_coeff
        - otherwise: linear ramp from ppo_start_coeff to ppo_target_coeff, or
          staircase updates when ``ppo_schedule_step_epochs > 0``
        """
        self.ppo_coeff = self._compute_ppo_dagger_coeff_for_epoch(current_epoch)

    def _should_apply_ppo_start_noise_std_cap(self, current_epoch: int) -> bool:
        if self.ppo_start_noise_std is None or not self.use_ppo_dagger_schedule:
            return False
        if current_epoch < self.ppo_start_epoch:
            return False
        ppo_coeff = self._compute_ppo_dagger_coeff_for_epoch(current_epoch)
        if ppo_coeff <= self.ppo_start_noise_std_until_coeff + 1e-8:
            return True

        if self.ppo_schedule_step_epochs > 0:
            step_epochs = max(1, self.ppo_schedule_step_epochs)
            if self.ppo_start_coeff > 0.0:
                first_positive_tier_start = self.ppo_start_epoch
            else:
                first_positive_tier_start = self.ppo_start_epoch + step_epochs
            first_positive_tier_end = first_positive_tier_start + step_epochs
            return current_epoch < first_positive_tier_end

        return False

    def _apply_ppo_start_noise_std_cap(self, current_epoch: int) -> None:
        if not self._should_apply_ppo_start_noise_std_cap(current_epoch):
            return
        if not hasattr(self.actor, "std"):
            return
        assert self.ppo_start_noise_std is not None
        with torch.no_grad():
            std = torch.nan_to_num(
                self.actor.std.data,
                nan=self.config.init_noise_std,
                posinf=10.0,
                neginf=0.0,
            )
            lower = 1e-6
            min_noise_std = getattr(self.actor, "min_noise_std", None)
            if min_noise_std:
                lower = max(lower, float(min_noise_std))
            cap = max(float(self.ppo_start_noise_std), lower)
            capped_std = torch.clamp(std, min=lower, max=cap)
            did_cap = bool(torch.any(capped_std < std).item())
            self.actor.std.data.copy_(capped_std)

        if did_cap and not self._ppo_start_noise_std_cap_announced:
            logger.info(
                "Capped actor noise std for PPO start: mean {:.6f} -> {:.6f} "
                "(cap={}, until_ppo_coeff={}).",
                float(std.mean().item()),
                float(capped_std.mean().item()),
                self.ppo_start_noise_std,
                self.ppo_start_noise_std_until_coeff,
            )
            self._ppo_start_noise_std_cap_announced = True

    def _adjust_teacher_action_mix_ratio(self, current_iteration: int) -> None:
        if not self.use_teacher_action_mix_schedule:
            return
        assert self.teacher_action_mix_ratio_start is not None
        assert self.teacher_action_mix_ratio_end is not None
        if self.teacher_action_mix_ratio_end_iteration <= 0:
            self.teacher_action_mix_ratio = self.teacher_action_mix_ratio_end
            return
        alpha = min(max(float(current_iteration), 0.0) / float(self.teacher_action_mix_ratio_end_iteration), 1.0)
        self.teacher_action_mix_ratio = (
            self.teacher_action_mix_ratio_start
            + (self.teacher_action_mix_ratio_end - self.teacher_action_mix_ratio_start) * alpha
        )

    def _sync_training_curriculum_state(self, *, current_iteration: int, total_iterations: int) -> None:
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None:
            return
        motion_command = command_manager.get_state("motion_command")
        if motion_command is None or not hasattr(motion_command, "set_training_iteration"):
            return
        motion_command.set_training_iteration(current_iteration, total_iterations=total_iterations)

    def _motion_command_supports_runtime_default_pose_prepend_mask(self) -> bool:
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None:
            return False
        motion_command = command_manager.get_state("motion_command")
        return motion_command is not None and hasattr(motion_command, "get_runtime_default_pose_prepend_mask")

    def _use_deterministic_student_actions(self) -> bool:
        """Use mean actions during pure BC phases to reduce rollout noise drift."""
        if not self.dagger_enabled:
            return False
        if self.use_ppo_dagger_schedule:
            return self.ppo_coeff <= 0.0
        return self.bc_loss_coef >= 1.0

    def _actor_uses_flow_matching(self) -> bool:
        return bool(getattr(self.actor, "supports_flow_matching", False))

    def _rollout_step(self, obs_dict):
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        timing = self.algo_timing if self.algo_timing.enabled else None
        with torch.no_grad():
            for rollout_step in range(self.config.num_steps_per_env):
                # Environment step
                if timing is not None:
                    with timing.record("rollout/obs_cat"):
                        actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                        critic_obs_raw = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                    with timing.record("rollout/obs_normalize"):
                        actor_obs = self._normalize_actor_obs(actor_obs_raw, update=True)
                        critic_obs = self._normalize_critic_obs(critic_obs_raw, update=True)
                else:
                    actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                    critic_obs_raw = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)

                    actor_obs = self._normalize_actor_obs(actor_obs_raw, update=True)
                    critic_obs = self._normalize_critic_obs(critic_obs_raw, update=True)

                # Keep perception aligned with the same pre-step state/action sample.
                actor_perception_obs_current = (
                    obs_dict[self.actor_perception_key] if self.actor_perception_key else None
                )
                critic_perception_obs_current = None
                if self.critic_perception_key:
                    critic_perception_obs_current = obs_dict[self.critic_perception_key]

                actor_policy_state = {"actor_obs": actor_obs}
                if actor_perception_obs_current is not None:
                    actor_policy_state[self.actor_perception_key] = actor_perception_obs_current
                if timing is not None:
                    with timing.record("rollout/actor_forward"):
                        actions = self.actor.act(actor_policy_state)
                        if self._use_deterministic_student_actions():
                            actions = self.actor.action_mean.detach()
                else:
                    actions = self.actor.act(actor_policy_state)
                    if self._use_deterministic_student_actions():
                        actions = self.actor.action_mean.detach()

                critic_policy_state = {"critic_obs": critic_obs}
                if critic_perception_obs_current is not None:
                    critic_policy_state[self.critic_perception_key] = critic_perception_obs_current
                if timing is not None:
                    with timing.record("rollout/critic_forward"):
                        values = self.critic.evaluate(critic_policy_state).detach()
                else:
                    values = self.critic.evaluate(critic_policy_state).detach()

                teacher_bc_mask_current = None
                if (
                    self.dagger_ignore_episode_initial_steps > 0
                    or self._motion_command_supports_runtime_default_pose_prepend_mask()
                ):
                    teacher_bc_mask_current = torch.ones((actions.shape[0], 1), device=actions.device, dtype=torch.bool)
                    motion_command = None
                    if self.env.command_manager is not None:
                        motion_command = self.env.command_manager.get_state("motion_command")
                    if self.dagger_ignore_episode_initial_steps > 0:
                        episode_length_buf = getattr(self.env, "episode_length_buf", None)
                        if episode_length_buf is not None:
                            teacher_bc_mask_current &= (
                                episode_length_buf >= self.dagger_ignore_episode_initial_steps
                            ).unsqueeze(1)
                    if motion_command is not None and hasattr(motion_command, "get_runtime_default_pose_prepend_mask"):
                        teacher_bc_mask_current &= (~motion_command.get_runtime_default_pose_prepend_mask()).unsqueeze(1)

                teacher_actions = None
                teacher_indices = None
                actions_to_step = actions
                if self.dagger_enabled and (self.bc_loss_coef > 0.0 or self.use_ppo_dagger_schedule):
                    if timing is not None:
                        with timing.record("rollout/teacher_obs_cat"):
                            if self.teacher_obs_keys == self.actor_obs_keys:
                                teacher_obs_raw = actor_obs_raw
                            else:
                                teacher_obs_raw = torch.cat([obs_dict[k] for k in self.teacher_obs_keys], dim=1)
                        with timing.record("rollout/teacher_actions"):
                            teacher_actions, teacher_indices = self._select_teacher_actions(teacher_obs_raw, obs_dict)
                        with timing.record("rollout/teacher_mix"):
                            self._maybe_capture_fixed_bc_eval_samples(
                                actor_obs_raw=actor_obs_raw,
                                actor_perception_obs=actor_perception_obs_current,
                                teacher_actions=teacher_actions,
                                teacher_bc_mask=teacher_bc_mask_current,
                            )
                            if self.teacher_action_mix_ratio > 0.0:
                                teacher_mask = (
                                    torch.rand((actions.shape[0], 1), device=actions.device)
                                    < self.teacher_action_mix_ratio
                                )
                                actions_to_step = torch.where(teacher_mask, teacher_actions, actions)
                            elif self.take_teacher_actions:
                                actions_to_step = teacher_actions
                    else:
                        if self.teacher_obs_keys == self.actor_obs_keys:
                            teacher_obs_raw = actor_obs_raw
                        else:
                            teacher_obs_raw = torch.cat([obs_dict[k] for k in self.teacher_obs_keys], dim=1)
                        teacher_actions, teacher_indices = self._select_teacher_actions(teacher_obs_raw, obs_dict)
                        self._maybe_capture_fixed_bc_eval_samples(
                            actor_obs_raw=actor_obs_raw,
                            actor_perception_obs=actor_perception_obs_current,
                            teacher_actions=teacher_actions,
                            teacher_bc_mask=teacher_bc_mask_current,
                        )
                        if self.teacher_action_mix_ratio > 0.0:
                            teacher_mask = (
                                torch.rand((actions.shape[0], 1), device=actions.device) < self.teacher_action_mix_ratio
                            )
                            actions_to_step = torch.where(teacher_mask, teacher_actions, actions)
                        elif self.take_teacher_actions:
                            actions_to_step = teacher_actions

                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rollout_step {}/{} before env.step",
                        self.current_learning_iteration,
                        rollout_step + 1,
                        self.config.num_steps_per_env,
                    )
                if timing is not None:
                    with timing.record("rollout/env_step"):
                        obs_dict, rewards, dones, infos = self.env.step({"actions": actions_to_step})
                else:
                    obs_dict, rewards, dones, infos = self.env.step({"actions": actions_to_step})
                if debug_heartbeat:
                    timeout_count = 0
                    if isinstance(infos, dict) and "time_outs" in infos and infos["time_outs"] is not None:
                        timeout_count = int(infos["time_outs"].sum().item())
                    logger.info(
                        "Heartbeat: iter {} rollout_step {}/{} after env.step (done_envs={}, timeout_envs={})",
                        self.current_learning_iteration,
                        rollout_step + 1,
                        self.config.num_steps_per_env,
                        int(dones.sum().item()),
                        timeout_count,
                    )

                if timing is not None:
                    with timing.record("rollout/device_transfer"):
                        for obs_key in obs_dict:
                            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                        rewards, dones = rewards.to(self.device), dones.to(self.device)
                else:
                    for obs_key in obs_dict:
                        obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                # Compute bootstrap value for timeouts
                final_rewards = torch.zeros_like(rewards)
                if timing is not None:
                    with timing.record("rollout/final_timeout_bootstrap"):
                        if infos["time_outs"].any():
                            final_critic_obs = torch.cat(
                                [infos["final_observations"][k] for k in self.critic_obs_keys], dim=1
                            )
                            # Timeout final observations are rank-local and conditional. Updating distributed
                            # normalizers here would desynchronize all_reduce order across ranks.
                            final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
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
                else:
                    if infos["time_outs"].any():
                        final_critic_obs = torch.cat([infos["final_observations"][k] for k in self.critic_obs_keys], dim=1)
                        # Timeout final observations are rank-local and conditional. Updating distributed
                        # normalizers here would desynchronize all_reduce order across ranks.
                        final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
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

                if timing is not None:
                    with timing.record("rollout/storage_add"):
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
                        if teacher_bc_mask_current is not None:
                            storage_kwargs["teacher_bc_mask"] = teacher_bc_mask_current
                        if actor_perception_obs_current is not None:
                            storage_kwargs[self.actor_perception_key] = actor_perception_obs_current
                        if (
                            critic_perception_obs_current is not None
                            and self.critic_perception_key != self.actor_perception_key
                        ):
                            storage_kwargs[self.critic_perception_key] = critic_perception_obs_current
                        self.storage.add(**storage_kwargs)
                else:
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
                    if teacher_bc_mask_current is not None:
                        storage_kwargs["teacher_bc_mask"] = teacher_bc_mask_current
                    if actor_perception_obs_current is not None:
                        storage_kwargs[self.actor_perception_key] = actor_perception_obs_current
                    if (
                        critic_perception_obs_current is not None
                        and self.critic_perception_key != self.actor_perception_key
                    ):
                        storage_kwargs[self.critic_perception_key] = critic_perception_obs_current
                    self.storage.add(**storage_kwargs)

                # Reset actor and critic for completed envs
                if timing is not None:
                    with timing.record("rollout/model_reset"):
                        self.actor.reset(dones)
                        self.critic.reset(dones)
                        if self.dagger_enabled:
                            if self.use_multi_teacher:
                                for teacher_actor in self.teacher_actors:
                                    teacher_actor.reset(dones)
                            elif self.teacher_actor is not None:
                                self.teacher_actor.reset(dones)
                else:
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
                    if timing is not None:
                        with timing.record("rollout/episode_stats"):
                            self.logging_helper.update_episode_stats(rewards, dones, infos)
                    else:
                        self.logging_helper.update_episode_stats(rewards, dones, infos)

            # Return / Advantage computation
            if timing is not None:
                with timing.record("rollout/returns"):
                    last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                    last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
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
            else:
                last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
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
        timing = self.algo_timing if self.algo_timing.enabled else None
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        rank_label = f"{getattr(self, 'gpu_global_rank', 0)}/{getattr(self, 'gpu_world_size', 1)}"
        minibatch_keys = {
            "actor_obs",
            "critic_obs",
            "actions",
            "values",
            "advantages",
            "returns",
            "actions_log_prob",
            "action_mean",
            "action_sigma",
        }
        if self.dagger_enabled:
            minibatch_keys.add("teacher_actions")
            minibatch_keys.add("teacher_indices")
            minibatch_keys.add("teacher_bc_mask")
        if self.actor_perception_key:
            minibatch_keys.add(self.actor_perception_key)
        if self.critic_perception_key:
            minibatch_keys.add(self.critic_perception_key)
        if self.use_time_gru:
            minibatch_keys.add("dones")
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} training_step enter (epochs={}, mini_batches={})",
                self.current_learning_iteration,
                rank_label,
                self.config.num_learning_epochs,
                self.config.num_mini_batches,
            )
        if self.dagger_enabled and self.use_ppo_dagger_schedule:
            self._adjust_ppo_dagger_coeff(self.current_learning_iteration)
        if self.dagger_enabled and (not self.use_ppo_dagger_schedule) and self.switch_to_rl_after > 0:
            if self.current_learning_iteration == self.switch_to_rl_after:
                self.bc_loss_coef = 0.0
        if timing is not None:
            with timing.record("training/generator_setup"):
                if self.use_time_gru:
                    generator = self.storage.sequence_mini_batch_generator(
                        self.config.num_mini_batches, self.config.num_learning_epochs
                    )
                else:
                    generator = self.storage.mini_batch_generator(
                        self.config.num_mini_batches, self.config.num_learning_epochs, keys=minibatch_keys
                    )
        else:
            if self.use_time_gru:
                generator = self.storage.sequence_mini_batch_generator(
                    self.config.num_mini_batches, self.config.num_learning_epochs
                )
            else:
                generator = self.storage.mini_batch_generator(
                    self.config.num_mini_batches, self.config.num_learning_epochs, keys=minibatch_keys
                )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} training_step generator ready",
                self.current_learning_iteration,
                rank_label,
            )

        minibatch: Minibatch
        loss_dict = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
        minibatch_idx = 0
        for minibatch in generator:
            minibatch_idx += 1
            self._debug_current_minibatch_idx = minibatch_idx
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} minibatch {} starting update",
                    self.current_learning_iteration,
                    rank_label,
                    minibatch_idx,
                )
            if timing is not None:
                with timing.record("training/update_algo_step"):
                    loss_dict = self._update_algo_step(minibatch, loss_dict)
            else:
                loss_dict = self._update_algo_step(minibatch, loss_dict)
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} minibatch {} finished update",
                    self.current_learning_iteration,
                    rank_label,
                    minibatch_idx,
                )

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for key in loss_dict:
            loss_dict[key] /= num_updates
        try:
            loss_dict["teacher_bc_mask_fraction"] = float(self.storage["teacher_bc_mask"].float().mean().item())
        except KeyError:
            pass
        if timing is not None:
            with timing.record("training/storage_clear"):
                self.storage.clear()
        else:
            self.storage.clear()
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} training_step exit after {} minibatches",
                self.current_learning_iteration,
                rank_label,
                minibatch_idx,
            )
        return loss_dict

    @staticmethod
    def _loss_to_float(loss: torch.Tensor | float | int) -> float:
        if torch.is_tensor(loss):
            loss = loss.detach()
            if loss.numel() != 1:
                loss = loss.mean()
            return float(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0).item())
        loss_value = float(loss)
        if loss_value != loss_value or loss_value in (float("inf"), float("-inf")):
            return 0.0
        return loss_value

    @staticmethod
    def _loss_is_finite(loss: torch.Tensor | float | int) -> bool:
        if torch.is_tensor(loss):
            return bool(torch.isfinite(loss).all())
        loss_value = float(loss)
        return loss_value == loss_value and loss_value not in (float("inf"), float("-inf"))

    def _accumulate_loss_dict(self, loss_dict: dict[str, float], ppo_loss_dict: dict[str, torch.Tensor]):
        if os.environ.get("HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return loss_dict
        loss_dict["Value"] += self._loss_to_float(ppo_loss_dict.get("value_loss", 0.0))
        loss_dict["Surrogate"] += self._loss_to_float(ppo_loss_dict.get("surrogate_loss", 0.0))
        loss_dict["Entropy"] += self._loss_to_float(ppo_loss_dict.get("entropy_loss", 0.0))
        loss_dict["KL"] += self._loss_to_float(ppo_loss_dict.get("kl_mean", 0.0))
        reserved = {"value_loss", "surrogate_loss", "entropy_loss", "kl_mean"}
        for key, loss in ppo_loss_dict.items():
            if key in reserved:
                continue
            if key not in loss_dict:
                loss_dict[key] = 0.0
            loss_dict[key] += self._loss_to_float(loss)
        return loss_dict

    def _sanitize_actor_std(self):
        if not hasattr(self.actor, "std"):
            return
        with torch.no_grad():
            std = torch.nan_to_num(
                self.actor.std.data,
                nan=self.config.init_noise_std,
                posinf=10.0,
                neginf=0.0,
            )
            min_noise_std = getattr(self.actor, "min_noise_std", None)
            if min_noise_std:
                std = torch.clamp(std, min=min_noise_std)
            else:
                std = torch.clamp(std, min=1e-6)
            self.actor.std.data.copy_(std)

    def _has_non_finite_gradients(self) -> bool:
        for param in itertools.chain(self.actor.parameters(), self.critic.parameters()):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
        return False

    def _update_algo_step(self, minibatch: Minibatch, loss_dict: dict[str, float]):
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        rank_label = f"{getattr(self, 'gpu_global_rank', 0)}/{getattr(self, 'gpu_world_size', 1)}"
        supervised_actor_only_step = os.environ.get("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ) or os.environ.get("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "").lower() in ("1", "true", "yes", "on")
        stream_supervised_actor_backward = supervised_actor_only_step and os.environ.get(
            "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", ""
        ).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if stream_supervised_actor_backward:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad(set_to_none=True)
        self._stream_supervised_actor_backward = stream_supervised_actor_backward
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update compute_loss begin (grad_enabled={} inference_mode={})",
                self.current_learning_iteration,
                rank_label,
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        try:
            with torch.inference_mode(False), torch.enable_grad():
                ppo_loss_dict = self._compute_ppo_loss(minibatch)
        finally:
            self._stream_supervised_actor_backward = False
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update compute_loss finished (grad_enabled={} inference_mode={})",
                self.current_learning_iteration,
                rank_label,
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
            )
        backward_already_done = bool(ppo_loss_dict.pop("_backward_already_done", False))
        actor_loss = ppo_loss_dict["actor_loss"]
        critic_loss = ppo_loss_dict["critic_loss"]

        skip_loss_finite_check = os.environ.get("HOLOSOMA_SKIP_LOSS_FINITE_CHECK", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        local_loss_finite = True
        loss_finite = True
        if not skip_loss_finite_check:
            local_loss_finite = self._loss_is_finite(actor_loss) and self._loss_is_finite(critic_loss)
            loss_finite = local_loss_finite
        if self.is_multi_gpu and torch.distributed.is_initialized() and not skip_loss_finite_check:
            finite_flag = torch.tensor(1 if local_loss_finite else 0, device=self.device, dtype=torch.int32)
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update finite all_reduce begin local={}",
                    self.current_learning_iteration,
                    rank_label,
                    local_loss_finite,
                )
            finite_flag = self._all_reduce_small_tensor(finite_flag, op=torch.distributed.ReduceOp.MIN)
            loss_finite = bool(finite_flag.item())
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update finite all_reduce finished global={}",
                    self.current_learning_iteration,
                    rank_label,
                    loss_finite,
                )
        elif debug_heartbeat and skip_loss_finite_check:
            logger.info(
                "Heartbeat: iter {} rank {} update finite check skipped",
                self.current_learning_iteration,
                rank_label,
            )

        if not loss_finite:
            if local_loss_finite:
                logger.warning("Skipping optimizer step because another rank reported non-finite loss.")
            else:
                logger.warning(
                    "Skipping optimizer step due to non-finite loss "
                    f"(actor={self._loss_to_float(actor_loss):.6f}, "
                    f"critic={self._loss_to_float(critic_loss):.6f})."
                )
            self._sanitize_actor_std()
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            return self._accumulate_loss_dict(loss_dict, ppo_loss_dict)

        if not backward_already_done:
            self.actor_optimizer.zero_grad()
            if supervised_actor_only_step:
                self.critic_optimizer.zero_grad(set_to_none=True)
            else:
                self.critic_optimizer.zero_grad()

        ppo_loss = actor_loss if supervised_actor_only_step else actor_loss + critic_loss
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update backward begin",
                self.current_learning_iteration,
                rank_label,
            )
        if backward_already_done:
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update backward skipped already_streamed",
                    self.current_learning_iteration,
                    rank_label,
                )
        else:
            ppo_loss.backward()
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update backward finished",
                self.current_learning_iteration,
                rank_label,
            )

        if self.is_multi_gpu:
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update grad all_reduce begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            self._reduce_parameters(include_critic=not supervised_actor_only_step)
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update grad all_reduce finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            if os.environ.get("HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update grad all_reduce cuda sync begin",
                        self.current_learning_iteration,
                        rank_label,
                    )
                torch.cuda.synchronize(self.device)
                if debug_heartbeat:
                    logger.info(
                        "Heartbeat: iter {} rank {} update grad all_reduce cuda sync finished",
                        self.current_learning_iteration,
                        rank_label,
                    )

        skip_grad_finite_check = os.environ.get("HOLOSOMA_SKIP_GRAD_FINITE_CHECK", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update grad finite check begin",
                self.current_learning_iteration,
                rank_label,
            )
        has_non_finite_gradients = False if skip_grad_finite_check else self._has_non_finite_gradients()
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update grad finite check finished result={} skipped={}",
                self.current_learning_iteration,
                rank_label,
                has_non_finite_gradients,
                skip_grad_finite_check,
            )
        if has_non_finite_gradients:
            logger.warning("Skipping optimizer step due to non-finite gradients.")
            self._sanitize_actor_std()
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            return self._accumulate_loss_dict(loss_dict, ppo_loss_dict)

        # Gradient step
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update actor grad clip begin",
                self.current_learning_iteration,
                rank_label,
            )
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update actor grad clip finished",
                self.current_learning_iteration,
                rank_label,
            )
            logger.info(
                "Heartbeat: iter {} rank {} update critic grad clip begin",
                self.current_learning_iteration,
                rank_label,
            )
        if supervised_actor_only_step:
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update critic grad clip skipped actor-only",
                    self.current_learning_iteration,
                    rank_label,
                )
        else:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update critic grad clip finished",
                self.current_learning_iteration,
                rank_label,
            )

        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update actor optimizer step begin",
                self.current_learning_iteration,
                rank_label,
            )
        self.actor_optimizer.step()
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update actor optimizer step finished",
                self.current_learning_iteration,
                rank_label,
            )
            logger.info(
                "Heartbeat: iter {} rank {} update critic optimizer step begin",
                self.current_learning_iteration,
                rank_label,
            )
        if supervised_actor_only_step:
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update critic optimizer step skipped actor-only",
                    self.current_learning_iteration,
                    rank_label,
                )
        else:
            self.critic_optimizer.step()
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update critic optimizer step finished",
                self.current_learning_iteration,
                rank_label,
            )
        self._apply_ppo_start_noise_std_cap(self.current_learning_iteration)
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} update optimizer step finished",
                self.current_learning_iteration,
                rank_label,
            )
        if os.environ.get("HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update optimizer cuda sync begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            torch.cuda.synchronize(self.device)
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} update optimizer cuda sync finished",
                    self.current_learning_iteration,
                    rank_label,
                )

        return self._accumulate_loss_dict(loss_dict, ppo_loss_dict)

    def _compute_ppo_loss(self, minibatch: Minibatch):
        if self.use_time_gru:
            return self._compute_ppo_loss_sequence(minibatch)
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        rank_label = f"{self.gpu_global_rank}/{self.gpu_world_size}" if self.is_multi_gpu else str(self.gpu_global_rank)
        def _clone_if_inference_tensor(value):
            if isinstance(value, torch.Tensor) and value.is_inference():
                return value.clone()
            return value

        raw_actor_obs = _clone_if_inference_tensor(minibatch["actor_obs"])
        actions_batch = _clone_if_inference_tensor(minibatch["actions"])
        target_values_batch = _clone_if_inference_tensor(minibatch["values"])
        advantages_batch = _clone_if_inference_tensor(minibatch["advantages"])
        returns_batch = _clone_if_inference_tensor(minibatch["returns"])
        old_actions_log_prob_batch = _clone_if_inference_tensor(minibatch["actions_log_prob"])
        old_mu_batch = _clone_if_inference_tensor(minibatch["action_mean"])
        old_sigma_batch = _clone_if_inference_tensor(minibatch["action_sigma"])
        actor_perception_obs = (
            _clone_if_inference_tensor(minibatch.get(self.actor_perception_key)) if self.actor_perception_key else None
        )
        critic_perception_obs = (
            _clone_if_inference_tensor(minibatch.get(self.critic_perception_key)) if self.critic_perception_key else None
        )
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} loss begin actor_obs={} actions={} actor_flow={}",
                self.current_learning_iteration,
                rank_label,
                tuple(raw_actor_obs.shape),
                tuple(actions_batch.shape),
                self._actor_uses_flow_matching(),
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
            actor_obs = raw_actor_obs
            critic_obs = _clone_if_inference_tensor(minibatch["critic_obs"])

        if actor_perception_obs is not None and actor_perception_obs.is_inference():
            actor_perception_obs = actor_perception_obs.clone()
        if critic_perception_obs is not None and critic_perception_obs.is_inference():
            critic_perception_obs = critic_perception_obs.clone()

        actor_obs = self._normalize_actor_obs(actor_obs, update=True)
        critic_obs = self._normalize_critic_obs(critic_obs, update=True)
        actor_obs = _clone_if_inference_tensor(actor_obs)
        critic_obs = _clone_if_inference_tensor(critic_obs)
        if debug_heartbeat:
            logger.info(
                "Heartbeat: iter {} rank {} loss normalized obs actor={} critic={}",
                self.current_learning_iteration,
                rank_label,
                tuple(actor_obs.shape),
                tuple(critic_obs.shape),
            )

        supervised_dagger_only = os.environ.get("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if supervised_dagger_only and self.distill_mode == "dagger" and self.dagger_enabled:
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            teacher_actions_batch = minibatch.get("teacher_actions")
            if teacher_actions_batch is None:
                raise ValueError("Dagger supervised-only mode requires teacher_actions in rollout storage.")
            teacher_actions_batch = _clone_if_inference_tensor(teacher_actions_batch[:original_batch_size])
            if self.clip_teacher_actions:
                teacher_actions_batch = torch.clamp(
                    teacher_actions_batch, -self.clip_actions_threshold, self.clip_actions_threshold
                )

            actor_policy_state = {"actor_obs": actor_obs[:original_batch_size]}
            if actor_perception_obs is not None:
                actor_policy_state[self.actor_perception_key] = actor_perception_obs[:original_batch_size]
            teacher_bc_mask_batch = minibatch.get("teacher_bc_mask")
            if teacher_bc_mask_batch is not None:
                teacher_bc_mask_batch = _clone_if_inference_tensor(
                    teacher_bc_mask_batch[:original_batch_size]
                ).view(-1)
            if self.use_ppo_dagger_schedule:
                lambda_ppo = max(0.0, min(1.0, float(self.ppo_coeff)))
                dagger_weight = self.dagger_loss_coef * (1.0 - lambda_ppo)
            elif self.bc_loss_coef > 0.0:
                dagger_weight = self.bc_loss_coef
            else:
                dagger_weight = self.dagger_loss_coef

            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference begin",
                    self.current_learning_iteration,
                    rank_label,
                )
            bc_loss = None
            backward_already_done = False
            if self._actor_uses_flow_matching():
                distill_per_sample = self.actor.flow_matching_loss(
                    actor_policy_state,
                    teacher_actions_batch,
                    loss_fn=self.distill_loss_fn,
                )
            else:
                actor_microbatch_size = int(os.environ.get("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "0") or 0)
                if actor_microbatch_size > 0 and original_batch_size > actor_microbatch_size:
                    if debug_heartbeat:
                        logger.info(
                            "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch size={} batch={}",
                            self.current_learning_iteration,
                            rank_label,
                            actor_microbatch_size,
                            original_batch_size,
                        )
                    stream_microbatch_backward = bool(getattr(self, "_stream_supervised_actor_backward", False))
                    distill_weighted_sum = torch.zeros((), device=self.device)
                    valid_count = torch.zeros((), device=self.device)
                    log_all_microbatches = os.environ.get("HOLOSOMA_DEBUG_MICROBATCH_ALL", "").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    sync_after_microbatch_forward = os.environ.get(
                        "HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD", ""
                    ).lower() in ("1", "true", "yes", "on")
                    if stream_microbatch_backward:
                        with torch.no_grad():
                            for micro_start in range(0, original_batch_size, actor_microbatch_size):
                                micro_end = min(micro_start + actor_microbatch_size, original_batch_size)
                                teacher_actions_micro = teacher_actions_batch[micro_start:micro_end]
                                valid_mask_micro = torch.ones(
                                    (micro_end - micro_start,), device=self.device, dtype=torch.bool
                                )
                                if teacher_bc_mask_batch is not None:
                                    valid_mask_micro &= teacher_bc_mask_batch[micro_start:micro_end].to(
                                        dtype=torch.bool
                                    )
                                if self.dagger_ignore_zero_teacher_actions:
                                    valid_mask_micro &= ~torch.all(teacher_actions_micro == 0.0, dim=-1)
                                valid_count = valid_count + valid_mask_micro.to(
                                    dtype=teacher_actions_batch.dtype
                                ).sum()
                            valid_count = torch.clamp(valid_count, min=1.0)
                    for micro_start in range(0, original_batch_size, actor_microbatch_size):
                        micro_end = min(micro_start + actor_microbatch_size, original_batch_size)
                        micro_policy_state = {
                            key: value[micro_start:micro_end]
                            for key, value in actor_policy_state.items()
                        }
                        if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                            logger.info(
                                "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch begin {}:{} grad_enabled={} inference_mode={}",
                                self.current_learning_iteration,
                                rank_label,
                                micro_start,
                                micro_end,
                                torch.is_grad_enabled(),
                                torch.is_inference_mode_enabled(),
                            )
                        with torch.inference_mode(False), torch.enable_grad():
                            student_actions_micro = self.actor.act_inference(micro_policy_state)
                        if sync_after_microbatch_forward:
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch cuda sync begin {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                            torch.cuda.synchronize(self.device)
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch cuda sync finished {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                        if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                            logger.info(
                                "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch finished {}:{}",
                                self.current_learning_iteration,
                                rank_label,
                                micro_start,
                                micro_end,
                            )
                        teacher_actions_micro = teacher_actions_batch[micro_start:micro_end]
                        distill_per_elem_micro = self.distill_loss_fn(
                            student_actions_micro, teacher_actions_micro, reduction="none"
                        )
                        if distill_per_elem_micro.ndim > 1:
                            distill_per_sample_micro = distill_per_elem_micro.mean(dim=-1)
                        else:
                            distill_per_sample_micro = distill_per_elem_micro
                        valid_mask_micro = torch.ones_like(distill_per_sample_micro, dtype=torch.bool)
                        if teacher_bc_mask_batch is not None:
                            valid_mask_micro &= teacher_bc_mask_batch[micro_start:micro_end].to(dtype=torch.bool)
                        if self.dagger_ignore_zero_teacher_actions:
                            valid_mask_micro &= ~torch.all(teacher_actions_micro == 0.0, dim=-1)
                        valid_weight_micro = valid_mask_micro.to(dtype=distill_per_sample_micro.dtype)
                        distill_weighted_sum_micro = (
                            distill_per_sample_micro * valid_weight_micro
                        ).sum()
                        if stream_microbatch_backward:
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch backward begin {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                            (dagger_weight * distill_weighted_sum_micro / valid_count).backward()
                            if debug_heartbeat and (micro_start == 0 or log_all_microbatches):
                                logger.info(
                                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference microbatch backward finished {}:{}",
                                    self.current_learning_iteration,
                                    rank_label,
                                    micro_start,
                                    micro_end,
                                )
                            distill_weighted_sum = distill_weighted_sum + distill_weighted_sum_micro.detach()
                        else:
                            distill_weighted_sum = distill_weighted_sum + distill_weighted_sum_micro
                            valid_count = valid_count + valid_weight_micro.sum()
                    bc_loss = distill_weighted_sum / torch.clamp(valid_count, min=1.0)
                    backward_already_done = stream_microbatch_backward
                else:
                    with torch.inference_mode(False), torch.enable_grad():
                        student_actions = self.actor.act_inference(actor_policy_state)
                    distill_per_elem = self.distill_loss_fn(student_actions, teacher_actions_batch, reduction="none")
                    if distill_per_elem.ndim > 1:
                        distill_per_sample = distill_per_elem.mean(dim=-1)
                    else:
                        distill_per_sample = distill_per_elem
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor.inference finished",
                    self.current_learning_iteration,
                    rank_label,
                )

            if bc_loss is None:
                valid_mask = torch.ones_like(distill_per_sample, dtype=torch.bool)
                if teacher_bc_mask_batch is not None:
                    valid_mask &= teacher_bc_mask_batch.to(dtype=torch.bool)
                if self.dagger_ignore_zero_teacher_actions:
                    valid_mask &= ~torch.all(teacher_actions_batch == 0.0, dim=-1)

                valid_weight = valid_mask.to(dtype=distill_per_sample.dtype)
                valid_count = torch.clamp(valid_weight.sum(), min=1.0)
                bc_loss = (distill_per_sample * valid_weight).sum() / valid_count

            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger weight begin",
                    self.current_learning_iteration,
                    rank_label,
                )

            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor_loss begin weight={}",
                    self.current_learning_iteration,
                    rank_label,
                    dagger_weight,
                )
            actor_loss = dagger_weight * bc_loss
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger actor_loss finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            zero = 0.0
            if debug_heartbeat:
                logger.info(
                    "Heartbeat: iter {} rank {} loss supervised_dagger finished",
                    self.current_learning_iteration,
                    rank_label,
                )
            return {
                "actor_loss": actor_loss,
                "critic_loss": zero,
                "symmetry_actor_loss": zero,
                "symmetry_critic_loss": zero,
                "value_loss": zero,
                "surrogate_loss": zero,
                "entropy_loss": zero,
                "distill_loss": bc_loss,
                "bc_loss": bc_loss,
                "ppo_coeff": float(self.ppo_coeff),
                "dagger_weight": dagger_weight,
                "kl_mean": zero,
                "_backward_already_done": backward_already_done,
            }

        actor_policy_state = {"actor_obs": actor_obs}
        if actor_perception_obs is not None:
            actor_policy_state[self.actor_perception_key] = actor_perception_obs
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss actor.act begin", self.current_learning_iteration, rank_label)
        self.actor.update_distribution_from_policy_state(actor_policy_state)
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss actor.act finished", self.current_learning_iteration, rank_label)

        critic_policy_state = {"critic_obs": critic_obs}
        if critic_perception_obs is not None:
            critic_policy_state[self.critic_perception_key] = critic_perception_obs
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss critic.evaluate begin", self.current_learning_iteration, rank_label)
        value_batch = self.critic.evaluate(critic_policy_state)
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss critic.evaluate finished", self.current_learning_iteration, rank_label)
            logger.info("Heartbeat: iter {} rank {} loss action log_prob begin", self.current_learning_iteration, rank_label)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self.actor.entropy[:original_batch_size]
        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss action log_prob finished", self.current_learning_iteration, rank_label)

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
        actor_loss_base = surrogate_loss - self.config.entropy_coef * entropy_loss
        actor_regularizer = self.config.symmetry_actor_coef * symmetry_actor_loss

        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        actor_loss = actor_loss_base + actor_regularizer
        distill_loss = torch.tensor(0.0, device=self.device)
        bc_loss = torch.tensor(0.0, device=self.device)
        dagger_weight = torch.tensor(0.0, device=self.device)
        if self.distill_mode == "dagger" and self.dagger_enabled and (
            self.bc_loss_coef > 0.0 or self.use_ppo_dagger_schedule
        ):
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss dagger begin", self.current_learning_iteration, rank_label)
            teacher_actions_batch = minibatch.get("teacher_actions")
            if teacher_actions_batch is None:
                raise ValueError("Dagger enabled but teacher_actions are missing from rollout storage.")
            teacher_actions_batch = teacher_actions_batch[:original_batch_size]
            if self.clip_teacher_actions:
                teacher_actions_batch = torch.clamp(
                    teacher_actions_batch, -self.clip_actions_threshold, self.clip_actions_threshold
                )

            if self._actor_uses_flow_matching():
                distill_actor_policy_state = {"actor_obs": actor_obs[:original_batch_size]}
                if actor_perception_obs is not None:
                    distill_actor_policy_state[self.actor_perception_key] = actor_perception_obs[:original_batch_size]
                distill_per_sample = self.actor.flow_matching_loss(
                    distill_actor_policy_state,
                    teacher_actions_batch,
                    loss_fn=self.distill_loss_fn,
                )
            else:
                distill_per_elem = self.distill_loss_fn(mu_batch, teacher_actions_batch, reduction="none")
                if distill_per_elem.ndim > 1:
                    distill_per_sample = distill_per_elem.mean(dim=-1)
                else:
                    distill_per_sample = distill_per_elem

            valid_mask = torch.ones_like(distill_per_sample, dtype=torch.bool)
            teacher_bc_mask = minibatch.get("teacher_bc_mask")
            if teacher_bc_mask is not None:
                valid_mask &= teacher_bc_mask[:original_batch_size].view(-1).to(dtype=torch.bool)

            if self.dagger_ignore_zero_teacher_actions:
                expert_terminate = torch.all(teacher_actions_batch == 0.0, dim=-1)
                valid_mask &= ~expert_terminate

            if valid_mask.any():
                bc_loss = distill_per_sample[valid_mask].mean()
            else:
                bc_loss = torch.tensor(0.0, device=self.device)

            if self.dagger_match_std:
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} rank {} loss dagger std-match begin", self.current_learning_iteration, rank_label)
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
                sigma_loss = (sigma_batch - sigma_teacher).pow(2).sum(dim=-1)
                if valid_mask.any():
                    bc_loss = bc_loss + sigma_loss[valid_mask].mean()
                else:
                    bc_loss = bc_loss + torch.tensor(0.0, device=self.device)
                if debug_heartbeat:
                    logger.info("Heartbeat: iter {} rank {} loss dagger std-match finished", self.current_learning_iteration, rank_label)

            # In DAgger mode, distillation objective is the BC term.
            distill_loss = bc_loss

            if self.use_ppo_dagger_schedule:
                # Match far-tracking hybrid loss:
                #   L = L_ppo + dagger_loss_coef * (1 - ppo_coeff) * L_dagger
                lambda_ppo = max(0.0, min(1.0, float(self.ppo_coeff)))
                lambda_d = 1.0 - lambda_ppo
                dagger_weight = torch.tensor(self.dagger_loss_coef * lambda_d, device=self.device)
                actor_loss = lambda_ppo * actor_loss_base + actor_regularizer + dagger_weight * bc_loss
            elif self.bc_loss_coef > 0.0:
                actor_loss = (1.0 - self.bc_loss_coef) * actor_loss_base + actor_regularizer + self.bc_loss_coef * bc_loss
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss dagger finished", self.current_learning_iteration, rank_label)
        elif self.distill_enabled:
            assert self.teacher_actor is not None, "Distillation enabled but teacher actor is not initialized."
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss distill begin", self.current_learning_iteration, rank_label)
            teacher_obs = self._normalize_teacher_actor_obs(raw_actor_obs)
            with torch.inference_mode():
                teacher_actions = self.teacher_actor.act_inference({"actor_obs": teacher_obs})
            if self._actor_uses_flow_matching():
                distill_actor_policy_state = {"actor_obs": actor_obs[:original_batch_size]}
                if actor_perception_obs is not None:
                    distill_actor_policy_state[self.actor_perception_key] = actor_perception_obs[:original_batch_size]
                distill_loss = self.actor.flow_matching_loss(
                    distill_actor_policy_state,
                    teacher_actions,
                    loss_fn=self.distill_loss_fn,
                ).mean()
            else:
                distill_loss = F.mse_loss(mu_batch, teacher_actions)
            actor_loss = actor_loss + self.distill_loss_coef * distill_loss
            if debug_heartbeat:
                logger.info("Heartbeat: iter {} rank {} loss distill finished", self.current_learning_iteration, rank_label)

        if debug_heartbeat:
            logger.info("Heartbeat: iter {} rank {} loss finished", self.current_learning_iteration, rank_label)
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
            "ppo_coeff": float(self.ppo_coeff),
            "dagger_weight": dagger_weight,
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
        self.actor.update_distribution_from_policy_state(actor_policy_state)

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
            "ppo_coeff": float(self.ppo_coeff),
            "dagger_weight": 0.0,
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
                kl_mean = self._all_reduce_small_tensor(kl_mean, op=torch.distributed.ReduceOp.SUM)
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

    @staticmethod
    def _move_checkpoint_value_to_device(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: PPO._move_checkpoint_value_to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [PPO._move_checkpoint_value_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(PPO._move_checkpoint_value_to_device(item, device) for item in value)
        return value

    def _move_optimizer_state_to_device(self, optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                state[key] = self._move_checkpoint_value_to_device(value, self.device)

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location="cpu")
            logger.info("Checkpoint deserialized on CPU; restoring tensors to {}.", self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
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
                self._move_optimizer_state_to_device(self.actor_optimizer)
                self._move_optimizer_state_to_device(self.critic_optimizer)
                self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
                self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"]["param_groups"][0]["lr"]
                logger.info("Optimizer loaded from checkpoint")
            self.current_learning_iteration = loaded_dict["iter"]
            env_state = self._move_checkpoint_value_to_device(loaded_dict.get("env_state"), self.device)
            self._restore_env_state(env_state)
            self._apply_ppo_start_noise_std_cap(self.current_learning_iteration)
            return loaded_dict.get("infos")
        return None

    def load_policy_init(self, ckpt_path: str | None) -> dict | None:
        """Initialize only actor policy parameters from a checkpoint.

        This intentionally does not restore critic, optimizers, observation
        normalizers, iteration counters, or environment state.
        """
        if ckpt_path is None:
            return None
        logger.info(f"Initializing actor policy parameters from checkpoint: {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location="cpu")
        actor_state = loaded_dict.get("actor_model_state_dict")
        if not isinstance(actor_state, dict):
            raise KeyError(f"Checkpoint does not contain actor_model_state_dict: {ckpt_path}")

        current_actor_state = self.actor.state_dict()
        compatible_actor_state = {}
        unexpected_keys = []
        mismatched_shapes = []
        non_tensor_keys = []
        for key, value in actor_state.items():
            current_value = current_actor_state.get(key)
            if current_value is None:
                unexpected_keys.append(key)
                continue
            if not isinstance(value, torch.Tensor) or not isinstance(current_value, torch.Tensor):
                non_tensor_keys.append(key)
                continue
            if value.shape != current_value.shape:
                mismatched_shapes.append((key, tuple(value.shape), tuple(current_value.shape)))
                continue
            compatible_actor_state[key] = value

        if not compatible_actor_state:
            raise RuntimeError(f"No compatible actor tensors found in policy init checkpoint: {ckpt_path}")

        current_actor_state.update(compatible_actor_state)
        self.actor.load_state_dict(current_actor_state)

        if unexpected_keys or mismatched_shapes or non_tensor_keys:
            logger.warning(
                "Policy init loaded {}/{} actor tensors from {}; skipped {} unexpected key(s), "
                "{} shape mismatch(es), and {} non-tensor key(s).",
                len(compatible_actor_state),
                len(actor_state),
                ckpt_path,
                len(unexpected_keys),
                len(mismatched_shapes),
                len(non_tensor_keys),
            )
            if mismatched_shapes:
                preview = ", ".join(
                    f"{key}: checkpoint{checkpoint_shape}->current{current_shape}"
                    for key, checkpoint_shape, current_shape in mismatched_shapes[:5]
                )
                logger.warning("Policy init skipped shape mismatches: {}", preview)
            if unexpected_keys:
                logger.warning("Policy init skipped unexpected keys: {}", ", ".join(unexpected_keys[:8]))

        checkpoint_iter = loaded_dict.get("iter", loaded_dict.get("iteration", "<unknown>"))
        logger.info(
            "Loaded actor policy parameters from {}; ignored checkpoint iteration={}, critic, optimizers, "
            "normalizers, and env_state. Training will start from iteration {}.",
            ckpt_path,
            checkpoint_iter,
            self.current_learning_iteration,
        )
        return loaded_dict.get("infos")

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

        # Save a pure policy .onnx for deployment. Motion replay/reference
        # tensors belong in debug/demo tooling, not in the policy artifact.
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
        mean_noise_std_tensor = self.actor.std.detach().mean()
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self._loss_to_float(mean_noise_std_tensor),
                "mean_noise_std_is_finite": float(torch.isfinite(mean_noise_std_tensor).item()),
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
            train_logs["command_goal_training_iteration"] = float(getattr(motion_command, "_training_iteration", it) or it)
            if hasattr(motion_command, "get_clean_noisy_clip_curriculum_log_state"):
                train_logs.update(motion_command.get_clean_noisy_clip_curriculum_log_state())
        if self.dagger_enabled and self.use_ppo_dagger_schedule:
            train_logs = extra_log_dicts.setdefault("Train", {})
            train_logs["ppo_dagger_target_coeff"] = float(self.ppo_target_coeff)
            train_logs["ppo_dagger_start_coeff"] = float(self.ppo_start_coeff)
            train_logs["ppo_dagger_coeff"] = float(self.ppo_coeff)
            train_logs["ppo_dagger_bc_weight"] = float(self.dagger_loss_coef * max(0.0, 1.0 - float(self.ppo_coeff)))
            if self.ppo_start_noise_std is not None:
                train_logs["ppo_start_noise_std"] = float(self.ppo_start_noise_std)
                train_logs["ppo_start_noise_std_until_coeff"] = float(self.ppo_start_noise_std_until_coeff)
        if self.dagger_enabled:
            train_logs = extra_log_dicts.setdefault("Train", {})
            train_logs["teacher_action_mix_ratio"] = float(self.teacher_action_mix_ratio)
            if self.use_teacher_action_mix_schedule:
                train_logs["teacher_action_mix_ratio_start"] = float(self.teacher_action_mix_ratio_start)
                train_logs["teacher_action_mix_ratio_end"] = float(self.teacher_action_mix_ratio_end)
                train_logs["teacher_action_mix_ratio_end_iteration"] = float(self.teacher_action_mix_ratio_end_iteration)
            fixed_bc_eval_metrics = self._get_fixed_bc_eval_metrics(current_iteration=it)
            if fixed_bc_eval_metrics:
                extra_log_dicts.setdefault("Eval", {}).update(fixed_bc_eval_metrics)
        self._add_step_timing_logs(extra_log_dicts)
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate
        # Use logging helper
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    def _reduce_parameters(self, include_critic: bool = True):
        models = [self.actor]
        if include_critic:
            models.append(self.critic)
        params = [param for model in models for param in model.parameters()]
        if not params:
            return

        debug_grad_reduce = os.environ.get("HOLOSOMA_DEBUG_GRAD_REDUCE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        rank_label = f"{getattr(self, 'gpu_global_rank', 0)}/{getattr(self, 'gpu_world_size', 1)}"
        total_numel = sum(param.numel() for param in params)
        first_grad = next((param.grad for param in params if param.grad is not None), None)
        dtype = first_grad.dtype if first_grad is not None else params[0].dtype
        device = params[0].device
        all_grads = torch.zeros(total_numel, device=device, dtype=dtype)
        grad_mask = torch.zeros(len(params), device=device, dtype=dtype)

        offset = 0
        local_grad_param_count = 0
        local_grad_numel = 0
        first_missing_param_idx = None
        for param_idx, param in enumerate(params):
            numel = param.numel()
            if param.grad is not None:
                grad = param.grad.detach()
                all_grads[offset : offset + numel].copy_(grad.reshape(-1).to(dtype=dtype))
                grad_mask[param_idx] = 1.0
                local_grad_param_count += 1
                local_grad_numel += numel
            elif first_missing_param_idx is None:
                first_missing_param_idx = param_idx
            offset += numel

        debug_minibatch_idx = getattr(self, "_debug_current_minibatch_idx", None)
        if debug_grad_reduce:
            print(
                "GradReducePrint "
                f"iter={self.current_learning_iteration} rank={rank_label} minibatch={debug_minibatch_idx} "
                f"phase=begin include_critic={include_critic} "
                f"local_grad_params={local_grad_param_count}/{len(params)} "
                f"local_grad_numel={local_grad_numel}/{total_numel} "
                f"first_missing_param_idx={first_missing_param_idx}",
                flush=True,
            )
            logger.info(
                "GradReduce: iter {} rank {} begin include_critic={} local_grad_params={}/{} "
                "local_grad_numel={}/{} first_missing_param_idx={}",
                self.current_learning_iteration,
                rank_label,
                include_critic,
                local_grad_param_count,
                len(params),
                local_grad_numel,
                total_numel,
                first_missing_param_idx,
            )

        payload = torch.cat((all_grads, grad_mask))
        reduce_path = self._all_reduce_grad_payload(payload)
        reduced_grads = payload[:total_numel].div_(self.gpu_world_size)
        grad_counts = payload[total_numel:].detach().cpu()

        if debug_grad_reduce:
            global_grad_param_count = int((grad_counts > 0).sum().item())
            print(
                "GradReducePrint "
                f"iter={self.current_learning_iteration} rank={rank_label} minibatch={debug_minibatch_idx} "
                f"phase=finished path={reduce_path} global_grad_params={global_grad_param_count}/{len(params)}",
                flush=True,
            )
            logger.info(
                "GradReduce: iter {} rank {} reduced path={} global_grad_params={}/{}",
                self.current_learning_iteration,
                rank_label,
                reduce_path,
                global_grad_param_count,
                len(params),
            )

        offset = 0
        for param_idx, param in enumerate(params):
            numel = param.numel()
            if grad_counts[param_idx].item() > 0:
                reduced_view = reduced_grads[offset : offset + numel].view_as(param)
                if param.grad is None:
                    param.grad = torch.empty_like(param, memory_format=torch.preserve_format)
                param.grad.detach().copy_(reduced_view.to(dtype=param.grad.dtype))
            else:
                param.grad = None
            offset += numel

    def _hierarchical_grad_reduce_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _gloo_grad_reduce_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_GLOO_GRAD_REDUCE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _setup_gloo_grad_reduce_group(self):
        if self._gloo_grad_reduce_ready:
            return self._gloo_grad_reduce_group
        self._gloo_grad_reduce_ready = True
        if not self.is_multi_gpu or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
        if self._gloo_barrier_group is not None:
            self._gloo_grad_reduce_group = self._gloo_barrier_group
        else:
            self._gloo_grad_reduce_group = torch.distributed.new_group(
                ranks=list(range(self.gpu_world_size)),
                backend="gloo",
            )
        if self.is_main_process:
            logger.info("Gloo CPU gradient reduce enabled across {} ranks.", self.gpu_world_size)
        return self._gloo_grad_reduce_group

    def _setup_gloo_barrier_group(self):
        if self._gloo_barrier_ready:
            return self._gloo_barrier_group
        self._gloo_barrier_ready = True
        if not self.is_multi_gpu or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
        if self._gloo_grad_reduce_group is not None:
            self._gloo_barrier_group = self._gloo_grad_reduce_group
        else:
            self._gloo_barrier_group = torch.distributed.new_group(
                ranks=list(range(self.gpu_world_size)),
                backend="gloo",
            )
        if self.is_main_process:
            logger.info("Gloo distributed barrier enabled across {} ranks.", self.gpu_world_size)
        return self._gloo_barrier_group

    def _hierarchical_grad_reduce_cpu_leader_enabled(self) -> bool:
        return os.environ.get("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _setup_hierarchical_grad_reduce_groups(self) -> bool:
        if self._hierarchical_grad_reduce_ready:
            return self._hierarchical_grad_reduce_available

        self._hierarchical_grad_reduce_ready = True
        if (
            not self.is_multi_gpu
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or self.gpu_local_world_size <= 1
            or self.gpu_world_size <= self.gpu_local_world_size
            or self.gpu_world_size % self.gpu_local_world_size != 0
        ):
            return False

        node_count = self.gpu_world_size // self.gpu_local_world_size
        local_node_idx = self.gpu_global_rank // self.gpu_local_world_size
        for node_idx in range(node_count):
            start_rank = node_idx * self.gpu_local_world_size
            local_ranks = list(range(start_rank, start_rank + self.gpu_local_world_size))
            local_group = torch.distributed.new_group(ranks=local_ranks)
            local_barrier_group = torch.distributed.new_group(ranks=local_ranks, backend="gloo")
            if node_idx == local_node_idx:
                self._hierarchical_local_group = local_group
                self._hierarchical_local_barrier_group = local_barrier_group
                self._hierarchical_local_leader_rank = start_rank

        leader_ranks = list(range(0, self.gpu_world_size, self.gpu_local_world_size))
        self._hierarchical_leader_group = torch.distributed.new_group(ranks=leader_ranks)
        self._hierarchical_leader_gloo_group = torch.distributed.new_group(ranks=leader_ranks, backend="gloo")
        self._hierarchical_is_leader_rank = self.gpu_global_rank in leader_ranks
        self._hierarchical_grad_reduce_available = (
            self._hierarchical_local_group is not None
            and self._hierarchical_local_barrier_group is not None
            and self._hierarchical_leader_gloo_group is not None
        )
        if self.is_main_process:
            logger.info(
                "Hierarchical gradient reduce enabled: world_size={} local_world_size={} nodes={}",
                self.gpu_world_size,
                self.gpu_local_world_size,
                node_count,
            )
        return self._hierarchical_grad_reduce_available

    def _all_reduce_grad_payload(self, payload: torch.Tensor) -> str:
        if self._gloo_grad_reduce_enabled():
            gloo_group = self._setup_gloo_grad_reduce_group()
            if gloo_group is not None:
                cpu_payload = payload.detach().cpu()
                torch.distributed.all_reduce(
                    cpu_payload,
                    op=torch.distributed.ReduceOp.SUM,
                    group=gloo_group,
                )
                payload.copy_(cpu_payload.to(device=payload.device, dtype=payload.dtype))
                return "gloo_cpu"

        if self._hierarchical_grad_reduce_enabled() and self._setup_hierarchical_grad_reduce_groups():
            torch.distributed.reduce(
                payload,
                dst=self._hierarchical_local_leader_rank,
                op=torch.distributed.ReduceOp.SUM,
                group=self._hierarchical_local_group,
            )
            if self._hierarchical_is_leader_rank:
                if self._hierarchical_grad_reduce_cpu_leader_enabled():
                    cpu_payload = payload.detach().cpu()
                    torch.distributed.all_reduce(
                        cpu_payload,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self._hierarchical_leader_gloo_group,
                    )
                    payload.copy_(cpu_payload.to(device=payload.device, dtype=payload.dtype))
                else:
                    torch.distributed.all_reduce(
                        payload,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self._hierarchical_leader_group,
                    )
            torch.distributed.barrier(group=self._hierarchical_local_barrier_group)
            torch.distributed.broadcast(
                payload,
                src=self._hierarchical_local_leader_rank,
                group=self._hierarchical_local_group,
            )
            if self._hierarchical_grad_reduce_cpu_leader_enabled():
                return "hierarchical_cpu_leader"
            return "hierarchical"

        torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
        return "flat"

    def _synchronize_model_weights(self):
        """Synchronize actor and critic weights across all GPUs."""
        debug_heartbeat = os.environ.get("HOLOSOMA_DEBUG_HEARTBEAT", "").lower() not in ("", "0", "false", "no")
        # Broadcast actor weights from rank 0 to all other ranks
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} broadcast actor begin", self.gpu_global_rank)
        for param in self.actor.parameters():
            self._broadcast_tensor(param.data, src=0)
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} broadcast actor finished", self.gpu_global_rank)

        skip_critic_weight_sync = os.environ.get("HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if skip_critic_weight_sync:
            if debug_heartbeat:
                logger.info("Heartbeat: rank {} broadcast critic skipped by HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", self.gpu_global_rank)
            logger.info(f"Synchronized actor weights across {self.gpu_world_size} GPUs; skipped critic weight sync")
            return

        # Broadcast critic weights from rank 0 to all other ranks
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} broadcast critic begin", self.gpu_global_rank)
        for param in self.critic.parameters():
            self._broadcast_tensor(param.data, src=0)
        if debug_heartbeat:
            logger.info("Heartbeat: rank {} broadcast critic finished", self.gpu_global_rank)

        logger.info(f"Synchronized model weights across {self.gpu_world_size} GPUs")

    def _normalize_advantages_multi_gpu(self, advantages):
        local_stats = torch.stack(
            [
                advantages.mean(),
                (advantages**2).mean(),
            ]
        )
        local_stats = self._all_reduce_small_tensor(local_stats, op=torch.distributed.ReduceOp.SUM)

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
                self.keys = keys
                self.slices = slices
                self.perception_key = perception_key
                self.normalizers = nn.ModuleDict({key: normalizers[key] for key in keys})

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

    def _maybe_debug_eval_policy_io(
        self,
        *,
        step: int | None,
        actor_obs_raw: torch.Tensor,
        actor_obs: torch.Tensor,
        policy_state: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> None:
        debug_path = os.environ.get("HOLOSOMA_EVAL_DEBUG_PATH", "").strip()
        if not debug_path:
            return
        debug_limit = int(os.environ.get("HOLOSOMA_EVAL_DEBUG_LIMIT", "12"))
        debug_count = int(getattr(self, "_eval_debug_count", 0))
        if debug_count >= debug_limit:
            return

        path = Path(debug_path)
        if not getattr(self, "_eval_debug_initialized", False):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            self._eval_debug_initialized = True

        perception_obs = policy_state.get(self.actor_perception_key) if self.actor_perception_key else None
        torque_record: dict[str, list[float] | float | int] = {}
        action_term = None
        action_manager = getattr(self.env, "action_manager", None)
        if action_manager is not None and hasattr(action_manager, "get_term"):
            try:
                action_term = action_manager.get_term("joint_control")
            except Exception:
                action_term = None
        if action_term is not None:
            with torch.no_grad():
                actions_scaled = actions * action_term.action_scales
                control_type = self.env.robot_config.control.control_type
                if control_type == "P":
                    torques_unclipped = (
                        action_term._kp_scale
                        * action_term.p_gains
                        * (actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos)
                        - action_term._kd_scale * action_term.d_gains * self.env.simulator.dof_vel
                    )
                elif control_type == "V":
                    torques_unclipped = (
                        action_term._kp_scale * action_term.p_gains * (actions_scaled - self.env.simulator.dof_vel)
                        - action_term._kd_scale
                        * action_term.d_gains
                        * (self.env.simulator.dof_vel - action_term._prev_dof_vel)
                        / self.env.sim_dt
                    )
                elif control_type == "T":
                    torques_unclipped = actions_scaled
                else:
                    torques_unclipped = None

                if torques_unclipped is not None:
                    torques_clipped = torques_unclipped
                    if self.env.robot_config.control.clip_torques:
                        torques_clipped = torch.clip(torques_clipped, -self.env.torque_limits, self.env.torque_limits)
                    sat_ratio = torch.abs(torques_clipped) / torch.clamp(self.env.torque_limits, min=1.0e-6)
                    torque_record = {
                        "torque_unclipped_values": torques_unclipped.detach().cpu().reshape(-1).to(torch.float32).tolist(),
                        "torque_clipped_values": torques_clipped.detach().cpu().reshape(-1).to(torch.float32).tolist(),
                        "torque_sat_ratio_values": sat_ratio.detach().cpu().reshape(-1).to(torch.float32).tolist(),
                        "torque_saturated_joint_count": int(
                            torch.count_nonzero(torch.abs(torques_unclipped) >= self.env.torque_limits - 1.0e-5).item()
                        ),
                    }
        record = {
            "count": debug_count,
            "step": None if step is None else int(step),
            "actor_obs_raw_values": actor_obs_raw.detach().cpu().reshape(-1).to(torch.float32).tolist(),
            "actor_obs_norm_values": actor_obs.detach().cpu().reshape(-1).to(torch.float32).tolist(),
            "perception_obs_values": (
                None
                if perception_obs is None
                else perception_obs.detach().cpu().reshape(-1).to(torch.float32).tolist()
            ),
            "action_values": actions.detach().cpu().reshape(-1).to(torch.float32).tolist(),
        }
        simulator = getattr(self.env, "simulator", None)
        if simulator is not None:
            try:
                record["sim_time_ms"] = float(simulator.time()) * 1000.0
            except Exception:
                pass
            try:
                record["robot_root_state"] = simulator.robot_root_states[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                record["robot_dof_pos"] = simulator.dof_pos[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                record["robot_dof_vel"] = simulator.dof_vel[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
            except Exception:
                pass
            actor_states: dict[str, list[float]] = {}
            try:
                env_ids = torch.tensor([0], device=simulator.device, dtype=torch.long)
                actor_metadata = getattr(simulator, "_actor_root_metadata", {})
                if isinstance(actor_metadata, dict) and actor_metadata:
                    actor_names = [name for name in actor_metadata if name != "robot"]
                else:
                    actor_names = list(getattr(simulator, "_object_urdf_by_name", {}).keys())
                for name in actor_names:
                    try:
                        actor_state = simulator.get_actor_states([name], env_ids)
                    except Exception:
                        continue
                    if actor_state.numel() == 0:
                        continue
                    actor_states[str(name)] = actor_state[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
            except Exception:
                actor_states = {}
            if actor_states:
                record["actors"] = actor_states
        perception_manager = getattr(self.env, "perception_manager", None)
        if perception_manager is not None and getattr(perception_manager, "enabled", False):
            try:
                env_ids = torch.tensor([0], device=perception_manager.device, dtype=torch.long)
                cam_body_pos, cam_body_quat = perception_manager.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=False,
                    apply_pitch=False,
                )
                record["camera_body_pose_pos"] = (
                    cam_body_pos[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                )
                record["camera_body_pose_quat_xyzw"] = (
                    cam_body_quat[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                )
            except Exception:
                pass
            try:
                cam_pos, cam_quat = perception_manager.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=True,
                    apply_pitch=True,
                )
                record["camera_pose_pos"] = cam_pos[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
                record["camera_pose_quat_xyzw"] = cam_quat[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
            except Exception:
                pass
        record.update(torque_record)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._eval_debug_count = debug_count + 1

    def _pre_eval_env_step(self, actor_state: dict):
        actor_obs_raw = torch.cat([actor_state["obs"][k] for k in self.actor_obs_keys], dim=1)
        actor_obs = actor_obs_raw
        actor_obs = self._normalize_actor_obs(actor_obs, update=False)
        policy_state = {"actor_obs": actor_obs}
        if self.actor_perception_key and self.actor_perception_key in actor_state["obs"]:
            policy_state[self.actor_perception_key] = actor_state["obs"][self.actor_perception_key]
        actions = self.eval_policy(policy_state)
        self._maybe_debug_eval_policy_io(
            step=actor_state.get("step"),
            actor_obs_raw=actor_obs_raw,
            actor_obs=actor_obs,
            policy_state=policy_state,
            actions=actions,
        )
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
