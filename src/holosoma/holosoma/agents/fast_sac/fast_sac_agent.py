from __future__ import annotations

import copy
import itertools
import math
import numbers
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Sequence

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor, CNNCritic, Critic
from holosoma.agents.fast_sac.fast_sac_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import FastSACConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.checkpoint_validation import (
    load_verified_torch_checkpoint,
    validate_checkpoint_iterations,
    validate_finite_tree,
    validate_module_state_compatibility,
)
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.policy_init_preflight import (
    ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
    allow_legacy_unverified_policy_load,
    validate_policy_init_payload_identity,
)
from holosoma.utils.rng_checkpoint import (
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
)
from holosoma.utils.safe_torch_import import (
    F,
    GradScaler,
    TensorboardSummaryWriter,
    TensorDict,
    autocast,
    nn,
    optim,
    torch,
)
from holosoma.utils.training_provenance import validate_training_provenance

torch.set_float32_matmul_precision("high")


class FastSACEnv:
    def __init__(
        self,
        env: BaseTask,
        actor_obs_keys: Sequence[str],
        critic_obs_keys: Sequence[str],
        action_boundary_mode: str,
    ):
        self._env = env
        extras_contract_setter = getattr(env, "set_collection_extras_contract", None)
        if callable(extras_contract_setter):
            extras_contract_setter(dense_episode_stats=True)
        else:
            setattr(env, "_dense_episode_stats_each_step", True)
        self._actor_obs_keys = actor_obs_keys
        self._critic_obs_keys = critic_obs_keys
        self._action_boundary_mode = action_boundary_mode
        self._include_critic_obs = True

        # Initialize the versioned transform from tanh outputs to the raw
        # action vector consumed by the environment action manager.
        self._action_boundaries, self._action_bias = self._compute_action_transform()

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self._env, name)

    def reset(self) -> torch.Tensor:
        obs_dict = self._env.reset_all()
        return torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._include_critic_obs:
            raise RuntimeError("Critic observations are unavailable in FastSAC evaluation-only mode.")
        obs_dict = self._env.reset_all()
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = (
            torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
            if self._include_critic_obs
            else actor_obs
        )
        return actor_obs, critic_obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Actions are now already scaled by the actor, so pass them directly to the environment
        obs_dict, rew_buf, reset_buf, info_dict = self._env.step({"actions": actions})  # type: ignore[attr-defined]
        actor_obs = torch.cat([obs_dict[k] for k in self._actor_obs_keys], dim=1)
        critic_obs = torch.cat([obs_dict[k] for k in self._critic_obs_keys], dim=1)
        if "final_observations" in info_dict:
            # Use true final observations when available
            final_actor_obs = torch.cat([info_dict["final_observations"][k] for k in self._actor_obs_keys], dim=1)
            final_critic_obs = (
                torch.cat([info_dict["final_observations"][k] for k in self._critic_obs_keys], dim=1)
                if self._include_critic_obs
                else final_actor_obs
            )
        else:
            final_actor_obs = actor_obs
            final_critic_obs = critic_obs
        extras = {
            "time_outs": info_dict["time_outs"],
            "observations": {
                "critic": critic_obs,
                "final": {
                    "actor_obs": final_actor_obs,
                    "critic_obs": final_critic_obs,
                },
            },
            "episode": info_dict["episode"],
            "episode_all": info_dict["episode_all"],
            "raw_episode": info_dict.get("raw_episode", {}),
            "raw_episode_all": info_dict.get("raw_episode_all", {}),
            "to_log": info_dict["to_log"],
        }
        return actor_obs, rew_buf, reset_buf, extras

    def _compute_action_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the versioned affine tanh-to-environment action transform."""

        robot_config = self._env.robot_config
        device = self._env.device
        dof_pos_lower_limits = torch.as_tensor(
            robot_config.dof_pos_lower_limit_list,
            dtype=torch.float32,
            device=device,
        )
        dof_pos_upper_limits = torch.as_tensor(
            robot_config.dof_pos_upper_limit_list,
            dtype=torch.float32,
            device=device,
        )
        num_dof = len(robot_config.dof_names)
        if dof_pos_lower_limits.shape != (num_dof,) or dof_pos_upper_limits.shape != (num_dof,):
            raise ValueError(
                "FastSAC joint-limit arrays must match robot DOF order: "
                f"num_dof={num_dof}, lower={tuple(dof_pos_lower_limits.shape)}, "
                f"upper={tuple(dof_pos_upper_limits.shape)}."
            )
        if not bool(torch.isfinite(dof_pos_lower_limits).all().item()) or not bool(
            torch.isfinite(dof_pos_upper_limits).all().item()
        ):
            raise ValueError("FastSAC joint limits must be finite.")
        if bool((dof_pos_upper_limits <= dof_pos_lower_limits).any().item()):
            raise ValueError("FastSAC upper joint limits must be strictly greater than lower limits.")

        default_joint_angles = torch.zeros(num_dof, dtype=torch.float32, device=device)
        for i, joint_name in enumerate(robot_config.dof_names):
            if joint_name in robot_config.init_state.default_joint_angles:
                default_joint_angles[i] = robot_config.init_state.default_joint_angles[joint_name]
        if not bool(torch.isfinite(default_joint_angles).all().item()):
            raise ValueError("FastSAC default joint angles must be finite.")

        if self._action_boundary_mode == "legacy_max_range_scalar_v1":
            scalar_scale = robot_config.control.action_scale
            if isinstance(scalar_scale, bool) or not math.isfinite(float(scalar_scale)) or float(scalar_scale) <= 0:
                raise ValueError(
                    f"Legacy FastSAC control.action_scale must be finite and positive, got {scalar_scale!r}."
                )
            max_range = torch.maximum(
                torch.abs(dof_pos_lower_limits - default_joint_angles),
                torch.abs(dof_pos_upper_limits - default_joint_angles),
            )
            actor_scale = max_range / float(scalar_scale)
            actor_bias = torch.zeros_like(actor_scale)
        elif self._action_boundary_mode == "joint_limit_affine_v2":
            if robot_config.control.control_type != "P":
                raise ValueError(
                    "FastSAC joint_limit_affine_v2 requires position control (control_type='P')."
                )
            action_manager = getattr(self._env, "action_manager", None)
            iter_terms = getattr(action_manager, "iter_terms", None)
            if not callable(iter_terms):
                raise RuntimeError(
                    "FastSAC joint_limit_affine_v2 requires an initialized action manager."
                )
            joint_terms = [
                (name, term)
                for name, term in iter_terms()
                if isinstance(term, JointPositionActionTerm)
            ]
            if len(joint_terms) != 1:
                raise RuntimeError(
                    "FastSAC requires exactly one JointPositionActionTerm for a scientific action transform; "
                    f"found {[name for name, _ in joint_terms]!r}."
                )
            term_name, joint_term = joint_terms[0]
            if int(getattr(action_manager, "total_action_dim", -1)) != num_dof or joint_term.action_dim != num_dof:
                raise ValueError(
                    "FastSAC action-manager dimension must equal robot actions_dim/num_dof: "
                    f"term={term_name!r}, manager_dim={getattr(action_manager, 'total_action_dim', None)!r}, "
                    f"term_dim={joint_term.action_dim}, num_dof={num_dof}."
                )
            effective_scale = joint_term.action_scales.detach().to(device=device, dtype=torch.float32)
            if effective_scale.shape != (num_dof,) or not bool(torch.isfinite(effective_scale).all().item()):
                raise ValueError(
                    "FastSAC JointPositionActionTerm.action_scales must be a finite per-DOF vector."
                )
            if bool((effective_scale <= 0).any().item()):
                raise ValueError("FastSAC effective JointPositionActionTerm action scales must be positive.")
            if bool((default_joint_angles < dof_pos_lower_limits).any().item()) or bool(
                (default_joint_angles > dof_pos_upper_limits).any().item()
            ):
                raise ValueError("FastSAC default joint angles must lie within configured joint limits.")
            half_range = 0.5 * (dof_pos_upper_limits - dof_pos_lower_limits)
            midpoint = 0.5 * (dof_pos_upper_limits + dof_pos_lower_limits)
            actor_scale = half_range / effective_scale
            actor_bias = (midpoint - default_joint_angles) / effective_scale
            if bool(getattr(robot_config.control, "clip_actions", False)):
                clip_limit = float(robot_config.control.action_clip_value)
                if not math.isfinite(clip_limit) or clip_limit <= 0:
                    raise ValueError("FastSAC action clip limit must be finite and positive.")
                endpoint_magnitude = torch.maximum(
                    torch.abs(actor_bias - actor_scale),
                    torch.abs(actor_bias + actor_scale),
                )
                if bool((endpoint_magnitude > clip_limit).any().item()):
                    raise ValueError(
                        "FastSAC joint-limit affine transform exceeds the environment raw-action clip; "
                        "the requested joint-limit mapping would be silently truncated."
                    )
        else:
            raise ValueError(
                "Unsupported FastSAC action_boundary_mode "
                f"{self._action_boundary_mode!r}."
            )

        if not bool(torch.isfinite(actor_scale).all().item()) or not bool(
            torch.isfinite(actor_bias).all().item()
        ):
            raise ValueError("FastSAC actor action scale/bias must be finite.")
        logger.info(
            "Computed FastSAC action transform mode={} scale={} bias={}",
            self._action_boundary_mode,
            actor_scale,
            actor_bias,
        )
        return actor_scale, actor_bias


class FastSACAgent(BaseAlgo):
    """
    FastSAC is an efficient variant of Soft Actor-Critic (SAC) tuned for
    large-scale training with massively parallel simulation.
    See https://arxiv.org/abs/2505.22642 for more details about FastTD3.
    Detailed technical report for FastSAC will be available soon.
    """

    config: FastSACConfig
    env: FastSACEnv  # type: ignore[assignment]
    actor: Actor
    qnet: Critic

    def __init__(
        self, env: BaseTask, config: FastSACConfig, device: str, log_dir: str, multi_gpu_cfg: dict | None = None
    ):
        wrapped_env = FastSACEnv(
            env,
            config.actor_obs_keys,
            config.critic_obs_keys,
            config.action_boundary_mode,
        )

        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=config.logging_interval,
            num_learning_iterations=config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self.training_metrics = TensorAverageMeterDict()

    def setup(self) -> None:
        logger.info("Setting up FastSAC")

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

        args = self.config
        device = self.device
        env = self.env
        evaluation_only = bool(getattr(self, "_evaluation_only", False))
        for field_name in ("num_updates", "policy_frequency"):
            value = getattr(args, field_name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
                raise ValueError(f"FastSAC {field_name} must be a positive integer, got {value!r}.")
        env._include_critic_obs = not evaluation_only

        required_obs_groups = list(args.actor_obs_keys)
        if not evaluation_only:
            required_obs_groups.extend(args.critic_obs_keys)
        algo_obs_dim_dict = self.env.observation_manager.get_obs_dims(required_obs_groups)

        def concatenated_group_dim(group_name: str) -> int:
            if group_name not in algo_obs_dim_dict:
                raise KeyError(f"FastSAC observation group {group_name!r} does not exist.")
            group_cfg = self.env.observation_manager.cfg.groups.get(group_name)
            if group_cfg is None or getattr(group_cfg, "concatenate", None) is not True:
                raise ValueError(
                    f"FastSAC observation group {group_name!r} must concatenate its terms."
                )
            dimension = algo_obs_dim_dict[group_name]
            if type(dimension) is not int or dimension <= 0:
                raise ValueError(
                    f"FastSAC observation group {group_name!r} must have a positive integer dimension, "
                    f"got {dimension!r}."
                )
            # ObservationManager.get_obs_dims() already includes group history.
            return dimension

        actor_obs_keys = self.config.actor_obs_keys
        critic_obs_keys = self.config.critic_obs_keys

        n_act = self.env.robot_config.actions_dim

        # Compute actor observation dimensions and store indices
        actor_obs_dim = 0
        self.actor_obs_indices = {}
        for obs_key in actor_obs_keys:
            obs_size = concatenated_group_dim(obs_key)

            # Store start and end indices for this observation key
            self.actor_obs_indices[obs_key] = {
                "start": actor_obs_dim,
                "end": actor_obs_dim + obs_size,
                "size": obs_size,
            }
            actor_obs_dim += obs_size

        self.actor_obs_dim = actor_obs_dim

        # Compute critic observation dimensions and store indices
        critic_obs_dim = 0
        self.critic_obs_indices = {}
        if not evaluation_only:
            for obs_key in critic_obs_keys:
                obs_size = concatenated_group_dim(obs_key)

                # Store start and end indices for this observation key
                self.critic_obs_indices[obs_key] = {
                    "start": critic_obs_dim,
                    "end": critic_obs_dim + obs_size,
                    "size": obs_size,
                }
                critic_obs_dim += obs_size

        self.obs_normalization = args.obs_normalization
        if args.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()

        # Get action scaling parameters from the environment
        action_scale = env._action_boundaries if args.use_tanh else torch.ones(n_act, device=device)
        action_bias = env._action_bias if args.use_tanh else torch.zeros(n_act, device=device)

        # Handle CNN actor/critic
        if args.use_cnn_encoder:
            # We assume that MLP doesn't take raw encoder observations
            actor_mlp_obs_keys = [k for k in actor_obs_keys if k != args.encoder_obs_key]
            critic_mlp_obs_keys = [k for k in critic_obs_keys if k != args.encoder_obs_key]
        else:
            actor_mlp_obs_keys = list(actor_obs_keys)
            critic_mlp_obs_keys = list(critic_obs_keys)
        actor_cls = CNNActor if args.use_cnn_encoder else Actor

        self.actor = actor_cls(
            obs_indices=self.actor_obs_indices,
            obs_keys=actor_mlp_obs_keys,
            n_act=n_act,
            num_envs=env.num_envs,
            device=device,
            hidden_dim=args.actor_hidden_dim,
            log_std_max=args.log_std_max,
            log_std_min=args.log_std_min,
            use_tanh=args.use_tanh,
            use_layer_norm=args.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )

        self.policy = self.actor.explore
        if evaluation_only:
            self.actor.eval()
            self.obs_normalizer.eval()
            logger.info(
                "FastSAC evaluation-only setup constructed actor/actor-normalizer only; "
                "critic, target, optimizers, scaler, replay buffer, symmetry, and training collectives skipped."
            )
            return

        # Count optimizer updates rather than collection iterations.  This
        # keeps delayed actor updates correct across collection boundaries
        # when num_updates is not a multiple of policy_frequency.
        self._critic_update_step = 0
        self.scaler = GradScaler(enabled=args.amp)
        if args.obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.critic_obs_normalizer = nn.Identity()

        critic_cls = CNNCritic if args.use_cnn_encoder else Critic
        self.qnet = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )

        print(self.actor)
        print(self.qnet)

        self.log_alpha = torch.tensor([math.log(args.alpha_init)], requires_grad=True, device=device)
        self.qnet_target = critic_cls(
            obs_indices=self.critic_obs_indices,
            obs_keys=critic_mlp_obs_keys,
            n_act=n_act,
            num_atoms=args.num_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            hidden_dim=args.critic_hidden_dim,
            device=device,
            use_layer_norm=args.use_layer_norm,
            num_q_networks=args.num_q_networks,
            encoder_obs_key=args.encoder_obs_key,
            encoder_obs_shape=args.encoder_obs_shape,
        )
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=args.critic_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.target_entropy = -n_act * args.target_entropy_ratio
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.alpha_learning_rate, fused=True, betas=(0.9, 0.95))

        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}")

        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=args.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
        )

        if args.use_symmetry:
            # using env._env is not really ideal..
            self.symmetry_utils = SymmetryUtils(env._env)

        # Synchronize model parameters across GPUs for consistent initialization
        if self.is_multi_gpu:
            self._synchronize_model_parameters()

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self):
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast qnet weights from rank 0 to all other ranks
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast log_alpha parameter from rank 0 to all other ranks
        torch.distributed.broadcast(self.log_alpha.data, src=0)

        # Load qnet_target weights from synced qnet
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs to reduce NCCL call count.

        This flattens all existing parameter gradients into a single contiguous
        tensor, performs one all_reduce, averages by world size, and then
        scatters the reduced values back into the original gradient tensors.
        """
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    def _update_main(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = args.gamma ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM)
                    self.log_alpha.grad.data.copy_(self.log_alpha.grad.data / self.gpu_world_size)

            scaler.unscale_(alpha_optimizer)

            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_pol(self, data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            # For logging, this is a bit wasteful though, but could be useful
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                # Compute policy entropy (negative log probability)
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _sample_and_prepare_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * num_updates
        large_data = self.rb.sample(large_batch_size)
        samples_per_update = batch_size * self.env.num_envs

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {"next": {}}

            augmented_large_data["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(actions=large_data["actions"])
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["observations"],
                env=self.env,
                obs_list=self.config.actor_obs_keys,
            )
            augmented_large_data["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )
            augmented_large_data["next"]["critic_observations"] = self.symmetry_utils.augment_observations(
                obs=large_data["next"]["critic_observations"],
                env=self.env,
                obs_list=self.config.critic_obs_keys,
            )

            # Calculate augmentation factor and repeat non-augmented data
            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0])
            augmented_large_data["next"]["rewards"] = large_data["next"]["rewards"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["truncations"] = large_data["next"]["truncations"].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"]["effective_n_steps"].repeat(num_aug)  # type: ignore[index]

            # Override large_data
            large_data = augmented_large_data

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                    "critic_observations": large_data["critic_observations"][start_idx:end_idx],
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"]["critic_observations"][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        raise RuntimeError(
            "FastSAC exact full resume is intentionally disabled: existing checkpoints do not "
            "serialize the replay buffer or complete rank-local RNG/rollout continuation state. "
            "Restoring global_step and optimizers without that state repeats or changes the data "
            "distribution and is not a scientific continuation. Use "
            "--training.policy-init-checkpoint for an explicit actor-only warm start, or start a "
            "fresh run until a versioned full-state FastSAC checkpoint schema is implemented."
        )

    def _runtime_policy_config_dict(self) -> dict[str, Any] | None:
        runtime_config = getattr(self, "_policy_load_runtime_config", None)
        if runtime_config is None or not hasattr(runtime_config, "to_serializable_dict"):
            return None
        serialized = runtime_config.to_serializable_dict()
        return serialized if isinstance(serialized, dict) else None

    @staticmethod
    def _validate_fast_sac_completed_step(checkpoint: dict[str, Any]) -> int:
        completed_step, _next_step = validate_checkpoint_iterations(checkpoint)
        global_step = checkpoint.get("global_step")
        if isinstance(global_step, bool) or not isinstance(global_step, numbers.Integral):
            raise ValueError(
                f"FastSAC checkpoint global_step must be an integer, got {global_step!r}."
            )
        if int(global_step) < 0:
            raise ValueError(f"FastSAC checkpoint global_step must be non-negative, got {global_step!r}.")
        if int(global_step) != completed_step:
            raise ValueError(
                "FastSAC checkpoint step metadata is inconsistent: "
                f"global_step={int(global_step)}, iteration={completed_step}."
            )
        return completed_step

    def _validate_actor_action_transform(self, actor_state: dict[str, Any]) -> None:
        reference_state = self.actor.state_dict()
        for key in ("action_scale", "action_bias"):
            checkpoint_value = actor_state.get(key)
            runtime_value = reference_state.get(key)
            if not isinstance(checkpoint_value, torch.Tensor) or not isinstance(runtime_value, torch.Tensor):
                raise ValueError(f"FastSAC actor state must contain tensor buffer {key!r}.")
            if not torch.allclose(
                checkpoint_value.detach().to(device="cpu"),
                runtime_value.detach().to(device="cpu"),
                rtol=1e-6,
                atol=1e-7,
            ):
                max_error = float(
                    torch.max(
                        torch.abs(
                            checkpoint_value.detach().to(device="cpu")
                            - runtime_value.detach().to(device="cpu")
                        )
                    ).item()
                )
                raise ValueError(
                    "FastSAC checkpoint actor action transform disagrees with the versioned runtime "
                    f"robot/action contract for {key!r}; max_abs_error={max_error:.9g}."
                )

    def _load_actor_only_impl(
        self,
        ckpt_path: str,
        *,
        evaluation: bool,
    ) -> dict[str, Any] | None:
        legacy_unverified = allow_legacy_unverified_policy_load()
        current_config = self._runtime_policy_config_dict()
        current_provenance = None if evaluation else getattr(self, "_training_provenance", None)
        expected_sha256: str | None = None
        if current_provenance is not None and not isinstance(current_provenance, dict):
            raise ValueError(
                "Attached FastSAC policy-init training provenance must be a mapping when present."
            )
        if not evaluation:
            if isinstance(current_provenance, dict):
                current_provenance = validate_training_provenance(
                    current_provenance,
                    require_finalized=True,
                )
                if current_provenance.get("policy_init_enabled") is not True:
                    raise ValueError(
                        "FastSAC.load_policy_init was called while training provenance does not enable "
                        "policy initialization."
                    )
                expected_sha256 = current_provenance.get("policy_init_sha256")
                if not isinstance(expected_sha256, str):
                    raise ValueError(
                        "FastSAC policy-init provenance has no authenticated policy_init_sha256."
                    )
            elif not legacy_unverified:
                raise ValueError(
                    "Scientific FastSAC policy initialization requires finalized current training "
                    "provenance with policy_init_sha256. Set "
                    f"{ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV}=1 only for an explicitly "
                    "non-scientific legacy warm start."
                )
            else:
                logger.warning(
                    "{}=1: allowing FastSAC policy initialization without authenticated provenance.",
                    ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
                )
        if current_config is None and not legacy_unverified:
            raise ValueError(
                "FastSAC actor-only loading requires attached runtime experiment_config metadata so "
                "equal-shaped observation/action tensors cannot hide semantic drift. Set "
                f"{ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV}=1 only for an explicitly non-scientific "
                "legacy policy load."
            )

        checkpoint, actual_sha256 = load_verified_torch_checkpoint(
            ckpt_path,
            expected_sha256=expected_sha256,
            map_location="cpu",
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("FastSAC checkpoint payload must be a mapping.")
        saved_config = checkpoint.get("experiment_config")
        if not isinstance(saved_config, dict):
            raise ValueError("FastSAC checkpoint is missing mapping experiment_config metadata.")
        if current_config is not None:
            validate_policy_init_payload_identity(checkpoint, current_config)
        else:
            validate_policy_init_payload_identity(checkpoint, saved_config)
            logger.warning(
                "{}=1: FastSAC actor load has no live semantic-contract comparison.",
                ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD_ENV,
            )

        actor_state = checkpoint.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise ValueError("FastSAC checkpoint actor_state_dict must be a mapping.")
        validate_finite_tree(actor_state, path="actor_state_dict")
        validate_module_state_compatibility(
            actor_state,
            reference_state=self.actor.state_dict(),
            path="actor_state_dict",
        )
        self._validate_actor_action_transform(actor_state)

        normalizer_state = checkpoint.get("obs_normalizer_state")
        if not isinstance(normalizer_state, dict):
            raise ValueError("FastSAC checkpoint obs_normalizer_state must be a mapping.")
        validate_module_state_compatibility(
            normalizer_state,
            reference_state=self.obs_normalizer.state_dict(),
            path="obs_normalizer_state",
        )

        completed_step: int | None = None
        source_provenance: dict[str, Any] | None = None
        if evaluation:
            completed_step = self._validate_fast_sac_completed_step(checkpoint)
            raw_source_provenance = checkpoint.get("training_provenance")
            if raw_source_provenance is not None:
                source_provenance = validate_training_provenance(
                    raw_source_provenance,
                    require_finalized=True,
                )

        self.actor.load_state_dict(actor_state, strict=True)
        self.obs_normalizer.load_state_dict(normalizer_state, strict=True)
        validate_finite_tree(self.actor.state_dict(), path="live.actor_state_dict")
        validate_finite_tree(
            self.obs_normalizer.state_dict(),
            path="live.obs_normalizer_state",
        )
        self.actor.eval() if evaluation else self.actor.train()
        self.obs_normalizer.eval() if evaluation else self.obs_normalizer.train()

        if evaluation:
            assert completed_step is not None
            self._evaluation_completed_iteration = completed_step
            self._source_checkpoint_sha256 = actual_sha256
            self._source_experiment_config_dict = copy.deepcopy(saved_config)
            self._training_provenance = source_provenance
            logger.info(
                "Loaded FastSAC actor-only evaluation checkpoint step={}; ignored critic, target, "
                "optimizers, scaler, replay buffer, global_step, env state, and critic normalizer; "
                "retained source SHA/config/provenance.",
                completed_step,
            )
        else:
            logger.info(
                "Loaded FastSAC actor-only policy initializer; training counter, critic, target, "
                "optimizers, scaler, replay buffer, environment, and RNG streams remain fresh."
            )
        infos = checkpoint.get("infos")
        return infos if isinstance(infos, dict) else None

    def load_policy_init(self, ckpt_path: str | None) -> dict[str, Any] | None:
        if ckpt_path is None:
            return None
        rng_state = capture_rng_checkpoint_state()
        try:
            return self._load_actor_only_impl(ckpt_path, evaluation=False)
        finally:
            restore_rng_checkpoint_state(
                rng_state,
                path="pre_fast_sac_policy_init_rng_state",
            )

    def load_evaluation(self, ckpt_path: str | None) -> dict[str, Any] | None:
        if ckpt_path is None:
            return None
        if getattr(self, "_evaluation_only", False) is not True:
            raise RuntimeError(
                "FastSAC.load_evaluation requires attach_evaluation_metadata() before setup()."
            )
        rng_state = (
            getattr(self, "_evaluation_rng_boundary_state", None)
            or capture_rng_checkpoint_state()
        )
        try:
            return self._load_actor_only_impl(ckpt_path, evaluation=True)
        finally:
            restore_rng_checkpoint_state(
                rng_state,
                path="pre_fast_sac_evaluation_setup_rng_state",
            )

    def _record_critic_update_and_should_update_policy(self) -> bool:
        frequency = getattr(self.config, "policy_frequency", None)
        if isinstance(frequency, bool) or not isinstance(frequency, numbers.Integral) or int(frequency) <= 0:
            raise ValueError(f"FastSAC policy_frequency must be a positive integer, got {frequency!r}.")
        self._critic_update_step = int(getattr(self, "_critic_update_step", 0)) + 1
        return self._critic_update_step % int(frequency) == 0

    def learn(self) -> None:
        args = self.config
        device = self.device
        if args.compile:
            update_main = torch.compile(self._update_main)
            update_pol = torch.compile(self._update_pol)
            policy = torch.compile(self.policy)
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            update_main = self._update_main
            update_pol = self._update_pol
            policy = self.policy
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward
        qnet = self.qnet
        qnet_target = self.qnet_target
        env = self.env
        rb = self.rb

        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)

        dones = None
        # Initialize metrics that might not be updated every step
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)

        while self.global_step <= args.num_learning_iterations:
            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                with torch.no_grad(), self._maybe_amp():
                    norm_obs = normalize_obs(obs, update=False)
                    actions = policy(obs=norm_obs, dones=dones)

                next_obs, rewards, dones, infos = env.step(actions.float())
                truncations = infos["time_outs"]

                # Update episode stats using logging helper
                self.logging_helper.update_episode_stats(rewards, dones, infos)

                next_critic_obs = infos["observations"]["critic"]

                # Compute 'true' next_obs and next_critic_obs for saving
                true_next_obs = torch.where(
                    truncations[:, None] > 0, infos["observations"]["final"]["actor_obs"], next_obs
                )
                true_next_critic_obs = torch.where(
                    truncations[:, None] > 0,
                    infos["observations"]["final"]["critic_obs"],
                    next_critic_obs,
                )
                transition = TensorDict(
                    {
                        "observations": obs,
                        "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                        "next": {
                            "observations": true_next_obs,
                            "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                            "truncations": truncations.long(),
                            "dones": dones.long(),
                        },
                    },
                    batch_size=(env.num_envs,),
                    device=device,
                )
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic_obs

                obs = next_obs
                critic_obs = next_critic_obs

                rb.extend(transition)

            # NOTE: args.batch_size is the global batch size
            batch_size = max(args.batch_size // env.num_envs // self.gpu_world_size, 1)
            if self.global_step > args.learning_starts:
                with self.logging_helper.record_learn_time():
                    # Use batched sampling: sample once, normalize once, split into updates
                    prepared_batches = self._sample_and_prepare_batches(
                        batch_size, args.num_updates, normalize_obs, normalize_critic_obs
                    )
                    for data in prepared_batches:
                        # Data is already normalized, just run the updates
                        (
                            buffer_rewards,
                            critic_grad_norm,
                            qf_loss,
                            qf_max,
                            qf_min,
                            alpha_loss,
                        ) = update_main(data)
                        if self._record_critic_update_and_should_update_policy():
                            actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)

                        # Accumulate training metrics for smoother logging
                        current_metrics = {
                            "actor_loss": actor_loss,
                            "qf_loss": qf_loss,
                            "qf_max": qf_max,
                            "qf_min": qf_min,
                            "actor_grad_norm": actor_grad_norm,
                            "critic_grad_norm": critic_grad_norm,
                            "buffer_rewards": buffer_rewards,
                            "alpha_loss": alpha_loss,
                            "alpha_value": self.log_alpha.exp().detach().mean(),
                            "policy_entropy": policy_entropy,
                            "action_std": action_std,
                        }
                        self.training_metrics.add(current_metrics)

                        with torch.no_grad():
                            src_ps = [p.data for p in qnet.parameters()]
                            tgt_ps = [p.data for p in qnet_target.parameters()]
                            torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                            torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

                if self.global_step % args.logging_interval == 0:
                    with torch.no_grad():
                        # Use accumulated training metrics for smoother logging (reduces noise)
                        accumulated_metrics = self.training_metrics.mean_and_clear()

                        # Convert tensor values to float for logging
                        loss_dict = {}
                        for key, value in accumulated_metrics.items():
                            if isinstance(value, torch.Tensor):
                                loss_dict[key] = value.item()
                            else:
                                loss_dict[key] = float(value)

                        # Add current env rewards (not part of training loop accumulation)
                        loss_dict["env_rewards"] = rewards.mean().item()

                    # Use logging helper
                    self.logging_helper.post_epoch_logging(it=self.global_step, loss_dict=loss_dict, extra_log_dicts={})
                if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                    if self.is_main_process:
                        logger.info(f"Saving model at global step {self.global_step}")
                        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                        self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

            # Avoid global_step being incremented beyond args.num_learning_iterations, so that the final checkpoint is
            # saved at exactly args.num_learning_iterations. In the `while` condition, we check for self.global_step <=
            # args.num_learning_iterations, so that we have complete logging data at the final step too (assuming
            # `args.num_learning_iterations` is a multiple of `args.logging_interval`).
            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            env_state=env_state or None,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
        )

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.unwrapped_env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.config.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.config.critic_obs_keys], dim=1),
        }

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        # Use the underlying module for inference
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalization:
                normalized_obs = obs_normalizer(obs["actor_obs"], update=False)
            else:
                normalized_obs = obs["actor_obs"]
            # Actions are already scaled by the actor
            return policy(normalized_obs)[0]

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        # Use the underlying module for ONNX export
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs):
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                # Actions are already scaled by the actor
                return self.actor(normalized_obs)[0]

        return ActorWrapper(actor, obs_normalizer if self.obs_normalization else None)

    def extract_actor_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened actor observation tensor.

        Args:
            obs: Flattened actor observation tensor of shape [batch_size, actor_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'actor_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.actor_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in actor observations. "
                f"Available keys: {list(self.actor_obs_indices.keys())}"
            )

        indices = self.actor_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def extract_critic_obs(self, obs: torch.Tensor, obs_key: str) -> torch.Tensor:
        """
        Extract a specific observation component from the flattened critic observation tensor.

        Args:
            obs: Flattened critic observation tensor of shape [batch_size, critic_obs_dim]
            obs_key: The observation key to extract (e.g., 'perception_obs', 'critic_state_obs')

        Returns:
            Extracted observation tensor of shape [batch_size, obs_size]
        """
        if obs_key not in self.critic_obs_indices:
            raise ValueError(
                f"Observation key '{obs_key}' not found in critic observations. "
                f"Available keys: {list(self.critic_obs_indices.keys())}"
            )

        indices = self.critic_obs_indices[obs_key]
        return obs[..., indices["start"] : indices["end"]]

    def get_actor_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about actor observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.actor_obs_indices.copy()

    def get_critic_obs_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about critic observation indices.

        Returns:
            Dictionary with obs_key -> {'start': int, 'end': int, 'size': int}
        """
        return self.critic_obs_indices.copy()

    def export(self, onnx_file_path: str) -> None:
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `FastSACAgent.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self.actor.eval()
        if self.obs_normalization:
            self.obs_normalizer.eval()

        # Create dummy all-zero input for ONNX tracing.
        example_input_list = torch.zeros(1, self.actor_obs_dim, device="cpu")

        export_policy_as_onnx(
            wrapper=self.actor_onnx_wrapper,
            onnx_file_path=onnx_file_path,
            example_obs_dict={"actor_obs": example_input_list},
        )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
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
        completed_iteration = (
            self._evaluation_completed_iteration
            if self._evaluation_completed_iteration is not None
            else self.global_step
        )
        metadata.update(self._checkpoint_metadata(iteration=completed_iteration))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        self.logging_helper.save_to_wandb(onnx_file_path)

        # Restore original training state
        if was_training:
            self.actor.train()
            if self.obs_normalization:
                self.obs_normalizer.train()

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        obs = self.env.reset()

        for _ in itertools.islice(itertools.count(), max_eval_steps):
            if self.obs_normalization:
                normalized_obs = self.obs_normalizer(obs, update=False)
            else:
                normalized_obs = obs
            # Actions are already scaled by the actor
            actions = self.actor(normalized_obs)[0]
            obs, _, _, _ = self.env.step(actions)
