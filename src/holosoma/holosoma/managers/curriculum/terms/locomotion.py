"""Curriculum hooks for locomotion tasks."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from holosoma.managers.curriculum.base import CurriculumTermBase


class AverageEpisodeLengthTracker(CurriculumTermBase):
    """Track moving average of episode length for locomotion tasks."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        base_num_compute_average_epl = float(params.get("num_compute_average_epl", 1000))
        base_denominator = getattr(env, "BASE_NUM_ENVS", env.num_envs)
        self.num_compute_average_epl = max(1, int(base_num_compute_average_epl * env.num_envs / base_denominator))
        self.average_episode_length = torch.tensor(0.0, device=env.device, dtype=torch.float)
        self._suppress_next_update = False

    def setup(self) -> None:
        self.average_episode_length = torch.as_tensor(
            float(self.average_episode_length), device=self.env.device, dtype=torch.float
        )

    def reset(self, env_ids) -> None:
        if env_ids is None:
            return

        if not torch.is_tensor(env_ids):
            env_ids_tensor = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.env.device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        pending = self.env._pending_episode_lengths
        mask_tensor = self.env._pending_episode_update_mask

        update_mask = mask_tensor.index_select(0, env_ids_tensor)
        if not torch.any(update_mask):
            return
        active_ids = env_ids_tensor[update_mask]

        if active_ids.numel() == 0:
            return

        episode_lengths = pending.index_select(0, active_ids).to(dtype=torch.float)
        if episode_lengths.numel() == 0:
            return

        self.update(active_ids, episode_lengths)

        zeros_long = torch.zeros(active_ids.shape, device=pending.device, dtype=pending.dtype)
        pending.index_copy_(0, active_ids, zeros_long)

        zeros_bool = torch.zeros(active_ids.shape, device=mask_tensor.device, dtype=mask_tensor.dtype)
        mask_tensor.index_copy_(0, active_ids, zeros_bool)

    def step(self) -> None:
        return

    def update(self, env_ids: torch.Tensor, episode_lengths: torch.Tensor) -> None:
        if self._suppress_next_update:
            self._suppress_next_update = False
            return

        num = env_ids.numel()
        if num == 0:
            return
        current_average = torch.mean(episode_lengths.to(dtype=torch.float), dtype=torch.float)
        weight = min(num / self.num_compute_average_epl, 1.0)
        self.average_episode_length = self.average_episode_length * (1 - weight) + current_average * weight

    def suppress_next_update(self) -> None:
        self._suppress_next_update = True

    def get_average(self) -> torch.Tensor:
        return self.average_episode_length

    def set_average(self, value: float | torch.Tensor, *, suppress_update: bool = True) -> None:
        self.average_episode_length = torch.as_tensor(float(value), device=self.env.device, dtype=torch.float)
        if suppress_update:
            self._suppress_next_update = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "average_episode_length": self.average_episode_length.detach().to("cpu"),
            "suppress_next_update": self._suppress_next_update,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        avg = state.get("average_episode_length")
        if avg is not None:
            self.average_episode_length = torch.as_tensor(float(avg), device=self.env.device, dtype=torch.float)
        self._suppress_next_update = bool(state.get("suppress_next_update", False))


class WObjectDifficultyCurriculum(CurriculumTermBase):
    """Single-knob curriculum for w-object training.

    A global difficulty scalar ``lambda_value`` in [0, 1] controls:
    - Assistive base support scale (decreases as lambda increases)
    - (Optional) imitation vs. generalization task-mix probability
    - (Optional) reset randomization strength for generalization episodes
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}

        self.enabled = bool(params.get("enabled", True))
        self.lambda_value = float(params.get("initial_lambda", 0.0))
        self.lambda_step_up = float(params.get("lambda_step_up", 0.01))
        self.lambda_step_down = float(params.get("lambda_step_down", 0.01))

        self.early_termination_threshold = float(params.get("early_termination_threshold", 0.30))
        self.similarity_metric_key = str(params.get("similarity_metric_key", "motion/error_body_pos"))
        self.similarity_sigma = float(params.get("similarity_sigma", 0.50))
        self.similarity_threshold = float(params.get("similarity_threshold", 0.60))

        self.imitation_prob_start = float(params.get("imitation_prob_start", 1.0))
        self.imitation_prob_target = float(params.get("imitation_prob_target", 0.5))
        self.enable_task_mixing = bool(params.get("enable_task_mixing", False))

        self.assist_beta_max = float(params.get("assist_beta_max", 1.0))
        self.assist_kp_pos = float(params.get("assist_kp_pos", 4.0))
        self.assist_kd_lin_vel = float(params.get("assist_kd_lin_vel", 2.0))
        self.assist_kd_ang_vel = float(params.get("assist_kd_ang_vel", 1.5))
        self.assist_max_delta_lin = float(params.get("assist_max_delta_lin", 0.20))
        self.assist_max_delta_ang = float(params.get("assist_max_delta_ang", 0.20))

        self.generalization_noise_scale_min = float(params.get("generalization_noise_scale_min", 1.0))
        self.generalization_noise_scale_max = float(params.get("generalization_noise_scale_max", 2.0))
        self.generalization_start_zero_prob_scale_min = float(
            params.get("generalization_start_zero_prob_scale_min", 1.0)
        )
        self.generalization_start_zero_prob_scale_max = float(
            params.get("generalization_start_zero_prob_scale_max", 0.25)
        )

        self._last_early_termination_rate = 1.0
        self._last_motion_end_rate = 0.0
        self._last_similarity = 0.0

    def setup(self) -> None:
        self.lambda_value = float(np.clip(self.lambda_value, 0.0, 1.0))
        self._publish_state()

    def reset(self, env_ids) -> None:
        if not self.enabled:
            self._publish_state()
            return
        if env_ids is None:
            self._publish_state()
            return

        if not torch.is_tensor(env_ids):
            env_ids_tensor = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.env.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            self._publish_state()
            return

        time_out_buf = getattr(self.env, "time_out_buf", None)
        if time_out_buf is None:
            self._publish_state()
            return

        timed_out = time_out_buf.index_select(0, env_ids_tensor).to(dtype=torch.bool)
        motion_end_mask = None
        termination_manager = getattr(self.env, "termination_manager", None)
        if termination_manager is not None and hasattr(termination_manager, "get_last_term_result"):
            motion_end_mask = termination_manager.get_last_term_result("motion_ends")
        if torch.is_tensor(motion_end_mask):
            motion_ended = motion_end_mask.index_select(0, env_ids_tensor).to(dtype=torch.bool)
        else:
            motion_ended = torch.zeros_like(timed_out, dtype=torch.bool)

        # Early termination should exclude true timeout and clean motion-end resets.
        early_termination = ~(timed_out | motion_ended)
        early_termination_rate = float(early_termination.to(dtype=torch.float32).mean().item())
        self._last_motion_end_rate = float(motion_ended.to(dtype=torch.float32).mean().item())
        similarity = self._compute_similarity(env_ids_tensor)

        self._last_early_termination_rate = early_termination_rate
        self._last_similarity = similarity

        if not getattr(self.env, "is_evaluating", False):
            should_increase = (
                early_termination_rate <= self.early_termination_threshold and similarity >= self.similarity_threshold
            )
            if should_increase:
                self.lambda_value = min(1.0, self.lambda_value + self.lambda_step_up)
            else:
                self.lambda_value = max(0.0, self.lambda_value - self.lambda_step_down)

        self._publish_state()

    def step(self) -> None:
        self._update_log_dict()
        if not self.enabled or getattr(self.env, "is_evaluating", False):
            return

        motion_command = None
        if hasattr(self.env, "command_manager"):
            motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            return
        if not hasattr(motion_command, "motion") or not getattr(motion_command.motion, "has_object", False):
            return

        assist_scale = float(getattr(self.env, "_wobj_curriculum_assist_scale", 0.0))
        if assist_scale <= 1e-6:
            return

        simulator = getattr(self.env, "simulator", None)
        if simulator is None or not hasattr(simulator, "robot_root_states"):
            return

        dt = float(getattr(self.env, "dt", 0.0))
        if dt <= 0.0:
            return

        try:
            pos_err = motion_command.ref_pos_w - motion_command.robot_ref_pos_w
            lin_vel_err = motion_command.ref_lin_vel_w - motion_command.robot_ref_lin_vel_w
            ang_vel_err = motion_command.ref_ang_vel_w - motion_command.robot_ref_ang_vel_w

            lin_delta = (
                self.assist_kp_pos * pos_err + self.assist_kd_lin_vel * lin_vel_err
            ) * (assist_scale * dt)
            ang_delta = (self.assist_kd_ang_vel * ang_vel_err) * (assist_scale * dt)

            lin_delta = torch.clamp(lin_delta, -self.assist_max_delta_lin, self.assist_max_delta_lin)
            ang_delta = torch.clamp(ang_delta, -self.assist_max_delta_ang, self.assist_max_delta_ang)

            simulator.robot_root_states[:, 7:10] = simulator.robot_root_states[:, 7:10] + lin_delta
            simulator.robot_root_states[:, 10:13] = simulator.robot_root_states[:, 10:13] + ang_delta
            simulator.set_actor_root_state_tensor_robots(None, simulator.robot_root_states)
        except Exception:
            # Keep curriculum non-fatal: if simulator backend does not support this path, skip assist.
            return

    def _compute_similarity(self, env_ids: torch.Tensor) -> float:
        log_dict = getattr(self.env, "log_dict", None)
        if not isinstance(log_dict, dict):
            return 0.0
        metric = log_dict.get(self.similarity_metric_key)
        if metric is None:
            return 0.0

        if torch.is_tensor(metric):
            metric_tensor = metric.to(device=self.env.device, dtype=torch.float32)
            if metric_tensor.ndim > 0 and metric_tensor.shape[0] == self.env.num_envs:
                metric_tensor = metric_tensor.index_select(0, env_ids)
            mean_error = float(metric_tensor.mean().item())
        else:
            try:
                mean_error = float(metric)
            except Exception:
                return 0.0

        sigma = max(self.similarity_sigma, 1e-6)
        similarity = math.exp(-mean_error / sigma)
        return float(np.clip(similarity, 0.0, 1.0))

    def _publish_state(self) -> None:
        lam = float(np.clip(self.lambda_value, 0.0, 1.0))
        self.lambda_value = lam

        if getattr(self.env, "is_evaluating", False):
            p_imitation = 1.0
            assist_scale = 0.0
            gen_noise_scale = 1.0
            gen_start_zero_scale = 1.0
        else:
            if self.enable_task_mixing:
                p_imitation = (1.0 - lam) * self.imitation_prob_start + lam * self.imitation_prob_target
                gen_noise_scale = self.generalization_noise_scale_min + lam * (
                    self.generalization_noise_scale_max - self.generalization_noise_scale_min
                )
                gen_start_zero_scale = self.generalization_start_zero_prob_scale_min + lam * (
                    self.generalization_start_zero_prob_scale_max - self.generalization_start_zero_prob_scale_min
                )
            else:
                p_imitation = 1.0
                gen_noise_scale = 1.0
                gen_start_zero_scale = 1.0
            assist_scale = (1.0 - lam) * self.assist_beta_max

        self.env._wobj_curriculum_enabled = bool(self.enabled)
        self.env._wobj_curriculum_lambda = float(lam)
        self.env._wobj_curriculum_imitation_prob_start = float(self.imitation_prob_start)
        self.env._wobj_curriculum_imitation_prob_target = float(self.imitation_prob_target)
        self.env._wobj_curriculum_task_mixing_enabled = bool(self.enable_task_mixing)
        self.env._wobj_curriculum_p_imitation = float(np.clip(p_imitation, 0.0, 1.0))
        self.env._wobj_curriculum_assist_scale = float(max(assist_scale, 0.0))
        self.env._wobj_curriculum_generalization_noise_scale = float(max(gen_noise_scale, 1e-6))
        self.env._wobj_curriculum_generalization_start_zero_prob_scale = float(np.clip(gen_start_zero_scale, 0.0, 1.0))

        self._update_log_dict()

    def _update_log_dict(self) -> None:
        if not hasattr(self.env, "log_dict"):
            return
        device = getattr(self.env, "device", "cpu")
        self.env.log_dict["curriculum/wobj_lambda"] = torch.tensor(self.lambda_value, device=device, dtype=torch.float)
        self.env.log_dict["curriculum/wobj_assist_scale"] = torch.tensor(
            float(getattr(self.env, "_wobj_curriculum_assist_scale", 0.0)),
            device=device,
            dtype=torch.float,
        )
        self.env.log_dict["curriculum/wobj_p_imitation"] = torch.tensor(
            float(getattr(self.env, "_wobj_curriculum_p_imitation", 1.0)),
            device=device,
            dtype=torch.float,
        )
        self.env.log_dict["curriculum/wobj_gen_noise_scale"] = torch.tensor(
            float(getattr(self.env, "_wobj_curriculum_generalization_noise_scale", 1.0)),
            device=device,
            dtype=torch.float,
        )
        self.env.log_dict["curriculum/wobj_early_term_rate"] = torch.tensor(
            self._last_early_termination_rate,
            device=device,
            dtype=torch.float,
        )
        self.env.log_dict["curriculum/wobj_motion_end_rate"] = torch.tensor(
            self._last_motion_end_rate,
            device=device,
            dtype=torch.float,
        )
        self.env.log_dict["curriculum/wobj_similarity"] = torch.tensor(
            self._last_similarity,
            device=device,
            dtype=torch.float,
        )


class PenaltyCurriculum(CurriculumTermBase):
    """Stateful penalty curriculum that scales reward term weights based on episode length.

    This curriculum term adaptively scales penalty reward weights during training.
    When episodes are short (robot falls quickly), penalties are reduced to make
    learning easier. As episodes get longer (robot stays up), penalties gradually
    increase to refine behavior.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)

        # Get parameters from config
        params = cfg.params
        self.enabled = params.get("enabled", True)
        self.tag = params.get("tag", "penalty_curriculum")
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))
        self.level_down_threshold = float(params.get("level_down_threshold", 150.0))
        self.level_up_threshold = float(params.get("level_up_threshold", 850.0))
        self.degree = float(params.get("degree", 0.0))

        # State variables (previously stored on env)
        self.current_scale = float(params.get("initial_scale", 1.0))
        self.penalty_reward_names: list[str] = []
        self.original_weights: dict[str, float] = {}

    def setup(self) -> None:
        """Setup penalty curriculum - identify rewards and apply initial scaling."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        # Identify penalty rewards by tag using public reward manager APIs
        for term_name in self.env.reward_manager.active_terms:
            term_cfg = self.env.reward_manager.get_term_cfg(term_name)
            if self.tag in term_cfg.tags:
                self.penalty_reward_names.append(term_name)

        # Store original weights and apply initial scaling
        for name in self.penalty_reward_names:
            if name not in self.env.reward_manager.active_terms:
                continue

            term_cfg = self.env.reward_manager.get_term_cfg(name)
            # Store original weight
            self.original_weights[name] = float(term_cfg.weight)
            # Apply initial scale
            scaled_cfg = replace(term_cfg, weight=term_cfg.weight * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

        # Set flag for logging compatibility
        self.env.use_reward_penalty_curriculum = True
        self.env.reward_penalty_scale = self.current_scale

    def reset(self, env_ids) -> None:
        """Update penalty scale based on average episode length."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        average_length = float(self.env.average_episode_length)

        # Update current scale based on episode length
        if average_length < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif average_length > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree

        # Clamp scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Apply scale to each penalty reward's weight
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            # Set weight = original_weight * current_scale
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

        # Update for logging
        self.env.reward_penalty_scale = self.current_scale

        # Update log_dict for WandB logging
        if hasattr(self.env, "log_dict"):
            self.env.log_dict["penalty_scale"] = torch.tensor(self.current_scale, dtype=torch.float)

    def step(self) -> None:
        """Clamp penalty scale within bounds each step."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        # Clamp current_scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Re-apply clamped scale to weights
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)


# ================================================================================================
# Legacy stateless functions (backward compatibility)
# ================================================================================================


def configure_reward_penalty(
    env,
    *,
    enabled: bool = True,
    tag: str = "penalty_curriculum",
    initial_scale: float = 1.0,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    level_down_threshold: float = 150.0,
    level_up_threshold: float = 750.0,
    degree: float = 0.0,
) -> None:
    """Configure reward-penalty curriculum parameters.

    This modifies the reward term weights directly in the reward manager,
    scaling them by initial_scale and storing the original weights for reference.

    Args:
        enabled: Whether to enable penalty curriculum
        tag: Tag to filter reward terms (e.g., "penalty_curriculum").
        initial_scale: Initial scaling factor
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
        level_down_threshold: Episode length threshold for decreasing penalty scale
        level_up_threshold: Episode length threshold for increasing penalty scale
        degree: Adjustment rate when updating penalty scale
    """
    env.use_reward_penalty_curriculum = bool(enabled)

    # Determine which rewards to apply curriculum to
    # Use tag-based selection
    penalty_names = []
    for term_name in env.reward_manager.active_terms:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if tag in term_cfg.tags:
            penalty_names.append(term_name)

    env._curriculum_penalty_reward_names = penalty_names

    # Store original weights and apply initial scaling
    env._curriculum_penalty_original_weights = {}
    if env.use_reward_penalty_curriculum and hasattr(env, "reward_manager"):
        for name in penalty_names:
            if name not in env.reward_manager.active_terms:
                continue
            term_cfg = env.reward_manager.get_term_cfg(name)
            # Store original weight
            env._curriculum_penalty_original_weights[name] = float(term_cfg.weight)
            # Apply initial scale
            scaled_cfg = replace(term_cfg, weight=term_cfg.weight * initial_scale)
            env.reward_manager.set_term_cfg(name, scaled_cfg)

    env._curriculum_penalty_cfg = {
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "level_down_threshold": float(level_down_threshold),
        "level_up_threshold": float(level_up_threshold),
        "degree": float(degree),
        "current_scale": float(initial_scale),
    }

    # Set reward_penalty_scale for logging compatibility
    env.reward_penalty_scale = float(initial_scale)


def update_reward_penalty(env, env_ids, **_) -> None:
    """Update penalty scale based on average episode length.

    Modifies reward term weights directly in the reward manager.
    """
    if not getattr(env, "use_reward_penalty_curriculum", False):
        return

    cfg = getattr(env, "_curriculum_penalty_cfg", None)
    penalty_names = getattr(env, "_curriculum_penalty_reward_names", [])
    original_weights = getattr(env, "_curriculum_penalty_original_weights", {})
    if not cfg or not penalty_names or not hasattr(env, "reward_manager"):
        return

    average_length = float(env.average_episode_length)
    degree = cfg["degree"]

    # Update current scale based on episode length
    current_scale = cfg["current_scale"]
    if average_length < cfg["level_down_threshold"]:
        current_scale *= 1.0 - degree
    elif average_length > cfg["level_up_threshold"]:
        current_scale *= 1.0 + degree

    # Clamp scale
    current_scale = float(np.clip(current_scale, cfg["min_scale"], cfg["max_scale"]))
    cfg["current_scale"] = current_scale

    # Apply scale to each penalty reward's weight
    for name in penalty_names:
        if name not in original_weights or name not in env.reward_manager.active_terms:
            continue
        term_cfg = env.reward_manager.get_term_cfg(name)
        # Set weight = original_weight * current_scale
        scaled_cfg = replace(term_cfg, weight=original_weights[name] * current_scale)
        env.reward_manager.set_term_cfg(name, scaled_cfg)

    # Update reward_penalty_scale for logging
    env.reward_penalty_scale = current_scale

    # Update log_dict for WandB logging
    if hasattr(env, "log_dict"):
        import torch

        env.log_dict["penalty_scale"] = torch.tensor(env.reward_penalty_scale, dtype=torch.float)


def clamp_reward_penalty(env, **_) -> None:
    """Ensure penalty scale stays within configured bounds each step.

    Re-applies clamping to reward weights in case of any drift.
    """
    if not getattr(env, "use_reward_penalty_curriculum", False):
        return

    cfg = getattr(env, "_curriculum_penalty_cfg", None)
    penalty_names = getattr(env, "_curriculum_penalty_reward_names", [])
    original_weights = getattr(env, "_curriculum_penalty_original_weights", {})
    if not cfg or not penalty_names or not hasattr(env, "reward_manager"):
        return

    # Clamp current_scale
    current_scale = cfg.get("current_scale", 1.0)
    current_scale = float(np.clip(current_scale, cfg["min_scale"], cfg["max_scale"]))
    cfg["current_scale"] = current_scale

    # Re-apply clamped scale to weights
    for name in penalty_names:
        if name not in original_weights or name not in env.reward_manager.active_terms:
            continue
        term_cfg = env.reward_manager.get_term_cfg(name)
        scaled_cfg = replace(term_cfg, weight=original_weights[name] * current_scale)
        env.reward_manager.set_term_cfg(name, scaled_cfg)
