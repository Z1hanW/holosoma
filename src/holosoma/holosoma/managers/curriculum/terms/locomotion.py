"""Curriculum hooks for locomotion tasks."""

from __future__ import annotations

import math
import numbers
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from loguru import logger

from holosoma.managers.curriculum.base import CurriculumTermBase
from holosoma.utils.rotations import quat_conjugate, quat_mul, quat_normalize, quat_to_exp_map


def _finite_penalty_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Penalty curriculum {name} must be a finite real number, got {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Penalty curriculum {name} must be finite, got {value!r}.")
    return parsed


def _validate_penalty_curriculum_parameters(
    *,
    enabled: Any,
    tag: Any,
    initial_scale: Any,
    min_scale: Any,
    max_scale: Any,
    level_down_threshold: Any,
    level_up_threshold: Any,
    degree: Any,
) -> tuple[bool, str, float, float, float, float, float, float]:
    if type(enabled) is not bool:
        raise ValueError(
            f"Penalty curriculum enabled must be an explicit boolean, got {enabled!r}."
        )
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("Penalty curriculum tag must be a non-empty string.")
    initial = _finite_penalty_scalar("initial_scale", initial_scale)
    minimum = _finite_penalty_scalar("min_scale", min_scale)
    maximum = _finite_penalty_scalar("max_scale", max_scale)
    level_down = _finite_penalty_scalar("level_down_threshold", level_down_threshold)
    level_up = _finite_penalty_scalar("level_up_threshold", level_up_threshold)
    parsed_degree = _finite_penalty_scalar("degree", degree)
    if not 0.0 <= minimum <= initial <= maximum:
        raise ValueError(
            "Penalty curriculum scales must satisfy 0 <= min_scale <= initial_scale <= max_scale."
        )
    if not 0.0 <= level_down <= level_up:
        raise ValueError(
            "Penalty curriculum thresholds must satisfy 0 <= level_down_threshold <= level_up_threshold."
        )
    if not 0.0 <= parsed_degree < 1.0:
        raise ValueError("Penalty curriculum degree must satisfy 0 <= degree < 1.")
    return (
        enabled,
        tag,
        initial,
        minimum,
        maximum,
        level_down,
        level_up,
        parsed_degree,
    )


class AverageEpisodeLengthTracker(CurriculumTermBase):
    """Track moving average of episode length for locomotion tasks."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        raw_horizon = params.get("num_compute_average_epl", 1000)
        if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, numbers.Real):
            raise ValueError("num_compute_average_epl must be a finite positive real number.")
        base_num_compute_average_epl = float(raw_horizon)
        if not math.isfinite(base_num_compute_average_epl) or base_num_compute_average_epl <= 0.0:
            raise ValueError("num_compute_average_epl must be a finite positive real number.")
        base_denominator = getattr(env, "BASE_NUM_ENVS", env.num_envs)
        if (
            isinstance(base_denominator, bool)
            or not isinstance(base_denominator, numbers.Real)
            or not math.isfinite(float(base_denominator))
            or float(base_denominator) <= 0.0
        ):
            raise ValueError("BASE_NUM_ENVS must be a finite positive real number.")
        if (
            isinstance(env.num_envs, bool)
            or not isinstance(env.num_envs, numbers.Integral)
            or int(env.num_envs) <= 0
        ):
            raise ValueError("Environment num_envs must be a positive integer.")
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

    def set_next_update_suppressed(self, suppressed: bool) -> None:
        """Set the one-shot reset guard exactly, including clearing it."""

        if not isinstance(suppressed, bool):
            raise TypeError("AverageEpisodeLengthTracker suppression flag must be boolean.")
        self._suppress_next_update = suppressed

    def get_average(self) -> torch.Tensor:
        return self.average_episode_length

    def set_average(self, value: float | torch.Tensor, *, suppress_update: bool = True) -> None:
        self.average_episode_length = torch.as_tensor(float(value), device=self.env.device, dtype=torch.float)
        self.set_next_update_suppressed(suppress_update)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "num_compute_average_epl": self.num_compute_average_epl,
            # ``to('cpu')`` aliases an already-CPU tensor.  Checkpoint state is
            # also used as an in-memory snapshot around canonical resets, so
            # it must own its storage rather than track later live mutation.
            "average_episode_length": self.average_episode_length.detach().to("cpu").clone(),
            "suppress_next_update": self._suppress_next_update,
        }

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("AverageEpisodeLengthTracker state must be a dictionary.")
        version = state.get("version", 0)
        if isinstance(version, bool) or not isinstance(version, int) or version not in (0, 1):
            raise ValueError(f"Unsupported AverageEpisodeLengthTracker state version: {version!r}.")
        if version == 1:
            horizon = state.get("num_compute_average_epl")
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon != self.num_compute_average_epl
            ):
                raise ValueError(
                    "AverageEpisodeLengthTracker averaging horizon differs from the active runtime: "
                    f"checkpoint={horizon!r}, runtime={self.num_compute_average_epl}."
                )
        avg = state.get("average_episode_length")
        if torch.is_tensor(avg):
            if avg.numel() != 1:
                raise ValueError("AverageEpisodeLengthTracker average must be scalar.")
            avg = avg.detach().cpu().item()
        if isinstance(avg, bool) or not isinstance(avg, numbers.Real):
            raise ValueError("AverageEpisodeLengthTracker average must be a real scalar.")
        if not math.isfinite(float(avg)) or float(avg) < 0.0:
            raise ValueError(
                "AverageEpisodeLengthTracker average must be finite and non-negative."
            )
        suppress = state.get("suppress_next_update", False)
        if not isinstance(suppress, bool):
            raise ValueError("AverageEpisodeLengthTracker suppress_next_update must be boolean.")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.validate_state_dict(state)
        avg = state["average_episode_length"]
        if torch.is_tensor(avg):
            avg = avg.detach().cpu().item()
        self.average_episode_length = torch.as_tensor(
            float(avg),
            device=self.env.device,
            dtype=torch.float,
        )
        self._suppress_next_update = state.get("suppress_next_update", False)


class WObjectDifficultyCurriculum(CurriculumTermBase):
    """Single-knob curriculum for w-object training.

    A global difficulty scalar ``lambda_value`` in [0, 1] controls:
    - Object assistive controller scale (decreases as lambda increases).
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

        self.assist_beta_max = float(params.get("assist_beta_max", 1.0))
        self.object_pos_kp = float(params.get("object_pos_kp", 4.0))
        self.object_lin_vel_kd = float(params.get("object_lin_vel_kd", 2.0))
        self.object_rot_kp = float(params.get("object_rot_kp", 3.0))
        self.object_ang_vel_kd = float(params.get("object_ang_vel_kd", 1.5))
        self.object_force_to_velocity = float(params.get("object_force_to_velocity", 1.0))
        self.object_torque_to_ang_velocity = float(params.get("object_torque_to_ang_velocity", 1.0))
        self.object_max_delta_lin = float(params.get("object_max_delta_lin", 0.20))
        self.object_max_delta_ang = float(params.get("object_max_delta_ang", 0.20))
        self.object_max_lin_vel_abs = float(params.get("object_max_lin_vel_abs", 2.5))
        self.object_max_ang_vel_abs = float(params.get("object_max_ang_vel_abs", 6.0))
        self.object_max_pos_err = float(params.get("object_max_pos_err", 0.50))
        self.object_max_rot_err = float(params.get("object_max_rot_err", 1.20))

        self._last_early_termination_rate = 1.0
        self._last_motion_end_rate = 0.0
        self._last_similarity = 0.0
        # In distributed training the curriculum decision is made once at the
        # rollout boundary from global sufficient statistics.  Updating the
        # scalar independently in ``reset()`` makes both the assist trajectory
        # and the resumed objective depend on how terminations happen to be
        # partitioned across ranks.
        self._pending_early_termination_count = 0
        self._pending_motion_end_count = 0
        self._pending_episode_count = 0
        self._pending_similarity_error_sum = 0.0
        self._pending_similarity_count = 0
        self._assist_exception_logged = False

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
        early_count = int(early_termination.sum(dtype=torch.long).item())
        motion_end_count = int(motion_ended.sum(dtype=torch.long).item())
        episode_count = int(env_ids_tensor.numel())
        similarity_error_sum, similarity_count = self._compute_similarity_statistics(env_ids_tensor)

        if getattr(self.env, "is_evaluating", False):
            # Evaluation never mutates training curriculum state.  Local
            # diagnostics remain useful and are not consumed by a later train.
            self._last_early_termination_rate = early_count / episode_count
            self._last_motion_end_rate = motion_end_count / episode_count
            self._last_similarity = self._similarity_from_statistics(
                similarity_error_sum,
                similarity_count,
            )
        else:
            self._pending_early_termination_count += early_count
            self._pending_motion_end_count += motion_end_count
            self._pending_episode_count += episode_count
            self._pending_similarity_error_sum += similarity_error_sum
            self._pending_similarity_count += similarity_count

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
        if simulator is None or not hasattr(simulator, "all_root_states"):
            return

        dt = float(getattr(self.env, "dt", 0.0))
        if dt <= 0.0:
            return

        try:
            object_indices = motion_command._get_active_object_indices()
            shared_object_states = getattr(motion_command, "simulator_object_state_snapshot", None)
            if isinstance(shared_object_states, torch.Tensor):
                object_states = shared_object_states
            else:
                object_states = simulator.all_root_states[object_indices][:, :13]
            if object_states.numel() == 0:
                return

            ref_pos = motion_command.object_pos_w
            ref_quat = motion_command.object_quat_w
            ref_lin_vel = motion_command.object_lin_vel_w

            finite_mask = torch.isfinite(object_states[:, :13]).all(dim=-1)
            finite_mask = finite_mask & torch.isfinite(ref_pos).all(dim=-1)
            finite_mask = finite_mask & torch.isfinite(ref_quat).all(dim=-1)
            finite_mask = finite_mask & torch.isfinite(ref_lin_vel).all(dim=-1)
            if not torch.any(finite_mask):
                return

            pos_err = ref_pos - object_states[:, :3]
            pos_err = torch.clamp(pos_err, -self.object_max_pos_err, self.object_max_pos_err)
            lin_vel_err = ref_lin_vel - object_states[:, 7:10]

            quat_err = quat_mul(ref_quat, quat_conjugate(object_states[:, 3:7], w_last=True), w_last=True)
            rot_err = quat_to_exp_map(quat_normalize(quat_err))
            rot_err = torch.clamp(rot_err, -self.object_max_rot_err, self.object_max_rot_err)
            ang_vel_err = -object_states[:, 10:13]

            force_cmd = self.object_pos_kp * pos_err + self.object_lin_vel_kd * lin_vel_err
            torque_cmd = self.object_rot_kp * rot_err + self.object_ang_vel_kd * ang_vel_err

            lin_delta = force_cmd * (assist_scale * dt * self.object_force_to_velocity)
            ang_delta = torque_cmd * (assist_scale * dt * self.object_torque_to_ang_velocity)

            lin_delta = torch.clamp(lin_delta, -self.object_max_delta_lin, self.object_max_delta_lin)
            ang_delta = torch.clamp(ang_delta, -self.object_max_delta_ang, self.object_max_delta_ang)

            updated_states = object_states.clone()
            updated_states[:, 7:10] = torch.clamp(
                updated_states[:, 7:10] + lin_delta,
                -self.object_max_lin_vel_abs,
                self.object_max_lin_vel_abs,
            )
            updated_states[:, 10:13] = torch.clamp(
                updated_states[:, 10:13] + ang_delta,
                -self.object_max_ang_vel_abs,
                self.object_max_ang_vel_abs,
            )
            updated_states = torch.where(finite_mask.unsqueeze(-1), updated_states, object_states)
            simulator.all_root_states[object_indices, :13] = updated_states
            update_object_snapshot = getattr(
                motion_command,
                "_update_simulator_object_state_snapshot",
                None,
            )
            if callable(update_object_snapshot):
                env_ids = torch.arange(updated_states.shape[0], device=updated_states.device, dtype=torch.long)
                update_object_snapshot(env_ids, updated_states)
            # Do not call simulator.write_state_updates() here.
            # IsaacSim already calls scene.write_data_to_sim() once per physics step.
        except Exception as exc:
            # Keep curriculum non-fatal: if simulator backend does not support this path, skip assist.
            if not self._assist_exception_logged:
                logger.warning("WObjectDifficultyCurriculum assist step skipped due to error: {}", exc)
                self._assist_exception_logged = True
            return

    def _compute_similarity_statistics(self, env_ids: torch.Tensor) -> tuple[float, int]:
        """Return an error sum/count suitable for an exact DDP reduction.

        The configured WBT metrics are one scalar error per environment.  A
        scalar metric is also accepted and interpreted as the mean error for
        every episode in this reset batch.  Other shapes are ambiguous and are
        rejected while the curriculum is enabled rather than silently giving
        ranks different statistical weights.
        """

        log_dict = getattr(self.env, "log_dict", None)
        if not isinstance(log_dict, dict):
            return 0.0, 0
        metric = log_dict.get(self.similarity_metric_key)
        if metric is None:
            return 0.0, 0

        if torch.is_tensor(metric):
            metric_tensor = metric.to(device=self.env.device, dtype=torch.float32)
            if metric_tensor.ndim == 0:
                metric_per_episode = metric_tensor.expand(env_ids.numel())
            elif metric_tensor.shape == (self.env.num_envs,):
                metric_per_episode = metric_tensor.index_select(0, env_ids)
            else:
                raise ValueError(
                    f"WObject similarity metric {self.similarity_metric_key!r} must be scalar or "
                    f"shape ({self.env.num_envs},), got {tuple(metric_tensor.shape)}."
                )
        else:
            try:
                scalar_metric = float(metric)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"WObject similarity metric {self.similarity_metric_key!r} is not numeric."
                ) from exc
            metric_per_episode = torch.full(
                (env_ids.numel(),),
                scalar_metric,
                device=self.env.device,
                dtype=torch.float32,
            )

        if not torch.isfinite(metric_per_episode).all():
            raise ValueError(f"WObject similarity metric {self.similarity_metric_key!r} contains NaN or Inf.")
        if torch.any(metric_per_episode < 0.0):
            raise ValueError(f"WObject similarity metric {self.similarity_metric_key!r} contains negative errors.")
        return float(metric_per_episode.sum(dtype=torch.float64).item()), int(metric_per_episode.numel())

    def _similarity_from_statistics(self, error_sum: float, count: int) -> float:
        if count <= 0:
            return 0.0
        mean_error = error_sum / float(count)

        sigma = max(self.similarity_sigma, 1e-6)
        similarity = math.exp(-mean_error / sigma)
        return float(np.clip(similarity, 0.0, 1.0))

    def _compute_similarity(self, env_ids: torch.Tensor) -> float:
        """Compatibility helper returning similarity for one local batch."""

        error_sum, count = self._compute_similarity_statistics(env_ids)
        return self._similarity_from_statistics(error_sum, count)

    def _apply_sufficient_statistics(
        self,
        *,
        early_termination_count: int,
        motion_end_count: int,
        episode_count: int,
        similarity_error_sum: float,
        similarity_count: int,
    ) -> None:
        """Apply one curriculum decision from one globally weighted batch."""

        if episode_count < 0 or early_termination_count < 0 or motion_end_count < 0:
            raise ValueError("WObject curriculum counts must be non-negative.")
        if early_termination_count > episode_count or motion_end_count > episode_count:
            raise ValueError("WObject curriculum termination counts exceed the episode count.")
        if early_termination_count + motion_end_count > episode_count:
            raise ValueError("WObject early-termination and motion-end counts overlap.")
        if similarity_count < 0 or similarity_count > episode_count:
            raise ValueError("WObject similarity count is inconsistent with the episode count.")
        if similarity_count not in (0, episode_count):
            raise ValueError(
                "WObject similarity statistics cover only part of the episode batch; "
                "refusing to bias the curriculum toward ranks with available metrics."
            )
        if not math.isfinite(similarity_error_sum) or similarity_error_sum < 0.0:
            raise ValueError("WObject similarity error sum must be finite and non-negative.")
        if similarity_count == 0 and similarity_error_sum != 0.0:
            raise ValueError("WObject similarity error sum must be zero when its count is zero.")
        if episode_count == 0:
            if any(
                value != 0
                for value in (
                    early_termination_count,
                    motion_end_count,
                    similarity_count,
                )
            ) or similarity_error_sum != 0.0:
                raise ValueError("WObject empty statistics contain non-zero sufficient statistics.")
            self._publish_state()
            return

        early_termination_rate = early_termination_count / float(episode_count)
        motion_end_rate = motion_end_count / float(episode_count)
        similarity = self._similarity_from_statistics(similarity_error_sum, similarity_count)

        self._last_early_termination_rate = early_termination_rate
        self._last_motion_end_rate = motion_end_rate
        self._last_similarity = similarity
        should_increase = (
            early_termination_rate <= self.early_termination_threshold and similarity >= self.similarity_threshold
        )
        if should_increase:
            self.lambda_value = min(1.0, self.lambda_value + self.lambda_step_up)
        else:
            self.lambda_value = max(0.0, self.lambda_value - self.lambda_step_down)
        self._publish_state()

    def synchronize_state(self, *, device: str, world_size: int, process_group=None) -> None:
        """Reduce pending reset statistics and advance one shared DDP state."""

        if not self.enabled:
            self._publish_state()
            return
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(f"WObject curriculum world_size must be positive, got {world_size!r}.")

        local_validation_error: Exception | None = None
        try:
            self.validate_checkpoint_state(self.get_checkpoint_state())
        except Exception as exc:  # carried to every rank below before raising
            local_validation_error = exc
        if world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("WObject curriculum synchronization requires an initialized process group.")

            validation_ok = torch.tensor(
                0 if local_validation_error is not None else 1,
                device=device,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(validation_ok, op=torch.distributed.ReduceOp.MIN, group=process_group)
            if int(validation_ok.item()) != 1:
                detail = (
                    f" Local validation error: {type(local_validation_error).__name__}: {local_validation_error}"
                    if local_validation_error is not None
                    else " Another rank reported invalid state."
                )
                raise RuntimeError("WObject curriculum state validation failed on at least one rank." + detail)

            lambda_min = torch.tensor(self.lambda_value, device=device, dtype=torch.float64)
            lambda_max = lambda_min.clone()
            torch.distributed.all_reduce(lambda_min, op=torch.distributed.ReduceOp.MIN, group=process_group)
            torch.distributed.all_reduce(lambda_max, op=torch.distributed.ReduceOp.MAX, group=process_group)
            min_value = float(lambda_min.item())
            max_value = float(lambda_max.item())
            if not math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    "WObject curriculum lambda differs across ranks before synchronization: "
                    f"min={min_value}, max={max_value}. Refusing to average path-dependent state."
                )
            self.lambda_value = min_value

            statistics = torch.tensor(
                (
                    self._pending_early_termination_count,
                    self._pending_motion_end_count,
                    self._pending_episode_count,
                    self._pending_similarity_error_sum,
                    self._pending_similarity_count,
                ),
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM, group=process_group)
            reduced = statistics.detach().cpu().tolist()
        else:
            if local_validation_error is not None:
                raise local_validation_error
            reduced = [
                self._pending_early_termination_count,
                self._pending_motion_end_count,
                self._pending_episode_count,
                self._pending_similarity_error_sum,
                self._pending_similarity_count,
            ]

        integral_values = []
        for name, value in zip(
            (
                "early_termination_count",
                "motion_end_count",
                "episode_count",
                "similarity_count",
            ),
            (reduced[0], reduced[1], reduced[2], reduced[4]),
        ):
            if not math.isfinite(float(value)) or not float(value).is_integer():
                raise RuntimeError(f"Reduced WObject {name} is not a finite integer: {value!r}.")
            integral_values.append(int(value))

        self._apply_sufficient_statistics(
            early_termination_count=integral_values[0],
            motion_end_count=integral_values[1],
            episode_count=integral_values[2],
            similarity_error_sum=float(reduced[3]),
            similarity_count=integral_values[3],
        )
        self._pending_early_termination_count = 0
        self._pending_motion_end_count = 0
        self._pending_episode_count = 0
        self._pending_similarity_error_sum = 0.0
        self._pending_similarity_count = 0

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return exact adaptive and pending state for rollout-boundary resume."""

        return {
            "version": 1,
            "semantics": {
                "enabled": self.enabled,
                "lambda_step_up": self.lambda_step_up,
                "lambda_step_down": self.lambda_step_down,
                "early_termination_threshold": self.early_termination_threshold,
                "similarity_metric_key": self.similarity_metric_key,
                "similarity_sigma": self.similarity_sigma,
                "similarity_threshold": self.similarity_threshold,
            },
            "lambda_value": self.lambda_value,
            "last_early_termination_rate": self._last_early_termination_rate,
            "last_motion_end_rate": self._last_motion_end_rate,
            "last_similarity": self._last_similarity,
            "pending_early_termination_count": self._pending_early_termination_count,
            "pending_motion_end_count": self._pending_motion_end_count,
            "pending_episode_count": self._pending_episode_count,
            "pending_similarity_error_sum": self._pending_similarity_error_sum,
            "pending_similarity_count": self._pending_similarity_count,
        }

    @staticmethod
    def _checkpoint_float(state: dict[str, Any], name: str, *, lower: float, upper: float | None) -> float:
        value = state.get(name)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"WObject checkpoint {name} must be scalar.")
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"WObject checkpoint {name} must be numeric.")
        result = float(value)
        if not math.isfinite(result) or result < lower or (upper is not None and result > upper):
            interval = f"[{lower}, {upper}]" if upper is not None else f"[{lower}, +inf)"
            raise ValueError(f"WObject checkpoint {name} must be finite and in {interval}.")
        return result

    @staticmethod
    def _checkpoint_count(state: dict[str, Any], name: str) -> int:
        value = state.get(name)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"WObject checkpoint {name} must be scalar.")
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"WObject checkpoint {name} must be a non-negative integer.")
        return int(value)

    def validate_checkpoint_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("WObject curriculum checkpoint state must be a dictionary.")
        version = state.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError(f"Unsupported WObject curriculum checkpoint version: {version!r}.")
        expected_semantics = self.get_checkpoint_state()["semantics"]
        if state.get("semantics") != expected_semantics:
            raise ValueError(
                "WObject curriculum checkpoint semantics differ from the active configuration: "
                f"checkpoint={state.get('semantics')!r}, expected={expected_semantics!r}."
            )

        self._checkpoint_float(state, "lambda_value", lower=0.0, upper=1.0)
        self._checkpoint_float(state, "last_early_termination_rate", lower=0.0, upper=1.0)
        self._checkpoint_float(state, "last_motion_end_rate", lower=0.0, upper=1.0)
        self._checkpoint_float(state, "last_similarity", lower=0.0, upper=1.0)
        early_count = self._checkpoint_count(state, "pending_early_termination_count")
        motion_end_count = self._checkpoint_count(state, "pending_motion_end_count")
        episode_count = self._checkpoint_count(state, "pending_episode_count")
        error_sum = self._checkpoint_float(
            state,
            "pending_similarity_error_sum",
            lower=0.0,
            upper=None,
        )
        similarity_count = self._checkpoint_count(state, "pending_similarity_count")
        if early_count > episode_count or motion_end_count > episode_count:
            raise ValueError("WObject checkpoint termination counts exceed pending_episode_count.")
        if early_count + motion_end_count > episode_count:
            raise ValueError("WObject checkpoint early/motion-end counts overlap.")
        if similarity_count > episode_count:
            raise ValueError("WObject checkpoint similarity count exceeds pending_episode_count.")
        if similarity_count not in (0, episode_count):
            raise ValueError("WObject checkpoint similarity statistics cover only part of the pending episodes.")
        if similarity_count == 0 and error_sum != 0.0:
            raise ValueError("WObject checkpoint similarity error sum is non-zero with zero count.")

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Atomically restore state after complete validation."""

        self.validate_checkpoint_state(state)
        self.lambda_value = self._checkpoint_float(state, "lambda_value", lower=0.0, upper=1.0)
        self._last_early_termination_rate = self._checkpoint_float(
            state,
            "last_early_termination_rate",
            lower=0.0,
            upper=1.0,
        )
        self._last_motion_end_rate = self._checkpoint_float(
            state,
            "last_motion_end_rate",
            lower=0.0,
            upper=1.0,
        )
        self._last_similarity = self._checkpoint_float(state, "last_similarity", lower=0.0, upper=1.0)
        self._pending_early_termination_count = self._checkpoint_count(
            state,
            "pending_early_termination_count",
        )
        self._pending_motion_end_count = self._checkpoint_count(state, "pending_motion_end_count")
        self._pending_episode_count = self._checkpoint_count(state, "pending_episode_count")
        self._pending_similarity_error_sum = self._checkpoint_float(
            state,
            "pending_similarity_error_sum",
            lower=0.0,
            upper=None,
        )
        self._pending_similarity_count = self._checkpoint_count(state, "pending_similarity_count")
        self._publish_state()

    def _publish_state(self) -> None:
        lam = float(np.clip(self.lambda_value, 0.0, 1.0))
        self.lambda_value = lam

        if getattr(self.env, "is_evaluating", False):
            assist_scale = 0.0
        else:
            assist_scale = (1.0 - lam) * self.assist_beta_max

        self.env._wobj_curriculum_enabled = bool(self.enabled)
        self.env._wobj_curriculum_lambda = float(lam)
        self.env._wobj_curriculum_assist_scale = float(max(assist_scale, 0.0))

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
        params = cfg.params or {}
        (
            self.enabled,
            self.tag,
            self.current_scale,
            self.min_scale,
            self.max_scale,
            self.level_down_threshold,
            self.level_up_threshold,
            self.degree,
        ) = _validate_penalty_curriculum_parameters(
            enabled=params.get("enabled", True),
            tag=params.get("tag", "penalty_curriculum"),
            initial_scale=params.get("initial_scale", 1.0),
            min_scale=params.get("min_scale", 0.0),
            max_scale=params.get("max_scale", 1.0),
            level_down_threshold=params.get("level_down_threshold", 150.0),
            level_up_threshold=params.get("level_up_threshold", 850.0),
            degree=params.get("degree", 0.0),
        )

        # State variables (previously stored on env)
        self.penalty_reward_names: list[str] = []
        self.original_weights: dict[str, float] = {}
        self._setup_complete = False

    def setup(self) -> None:
        """Setup penalty curriculum - identify rewards and apply initial scaling."""
        if not self.enabled:
            return
        if self._setup_complete:
            return
        if not hasattr(self.env, "reward_manager"):
            raise RuntimeError("Enabled penalty curriculum requires a reward manager.")

        # Identify penalty rewards by tag using public reward manager APIs
        penalty_reward_names: list[str] = []
        original_weights: dict[str, float] = {}
        for term_name in self.env.reward_manager.active_terms:
            term_cfg = self.env.reward_manager.get_term_cfg(term_name)
            if self.tag in term_cfg.tags:
                original_weight = _finite_penalty_scalar(
                    f"original reward weight for {term_name!r}",
                    term_cfg.weight,
                )
                derived_weights = (
                    original_weight * self.min_scale,
                    original_weight * self.current_scale,
                    original_weight * self.max_scale,
                )
                if not all(math.isfinite(weight) for weight in derived_weights):
                    raise ValueError(
                        f"Penalty curriculum derived weight range for {term_name!r} is non-finite."
                    )
                penalty_reward_names.append(term_name)
                original_weights[term_name] = original_weight
        if not penalty_reward_names:
            raise ValueError(
                f"Enabled penalty curriculum tag {self.tag!r} matches no active reward terms."
            )

        # Validation above is intentionally complete before any live reward
        # configuration is mutated.
        self.penalty_reward_names = penalty_reward_names
        self.original_weights = original_weights
        self._apply_scale(self.current_scale)
        self._setup_complete = True

        # Set flag for compatibility with algorithm-level curriculum routing.
        self.env.use_reward_penalty_curriculum = True

    def _apply_scale(self, scale: Any) -> None:
        """Atomically validate and publish the authoritative penalty scale."""

        parsed_scale = _finite_penalty_scalar("current_scale", scale)
        if not self.min_scale <= parsed_scale <= self.max_scale:
            raise ValueError(
                "Penalty curriculum current_scale must remain within configured bounds."
            )
        scaled_configs: dict[str, Any] = {}
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            scaled_weight = float(self.original_weights[name]) * parsed_scale
            if not math.isfinite(scaled_weight):
                raise ValueError(
                    f"Penalty curriculum derived weight for {name!r} is non-finite."
                )
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            scaled_configs[name] = replace(term_cfg, weight=scaled_weight)
        for name, scaled_cfg in scaled_configs.items():
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)
        self.current_scale = parsed_scale
        self.env.reward_penalty_scale = parsed_scale
        if hasattr(self.env, "log_dict"):
            self.env.log_dict["penalty_scale"] = torch.tensor(
                parsed_scale,
                dtype=torch.float,
            )

    def reset(self, env_ids) -> None:
        """Update penalty scale based on average episode length."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        # ``BaseTask`` invokes every manager reset hook on every control step,
        # including steps where no environment terminated.  An empty reset is
        # not curriculum evidence and must not advance this global scale.
        if env_ids is None:
            return
        if torch.is_tensor(env_ids):
            if env_ids.numel() == 0:
                return
        else:
            try:
                if len(env_ids) == 0:
                    return
            except TypeError:
                # Scalar identifiers are accepted for backward-compatible
                # direct callers and represent one actual reset.
                pass

        if not self.penalty_reward_names or not self.original_weights:
            return

        average_length = float(self.env.average_episode_length)

        # Update current scale based on episode length
        if average_length < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif average_length > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree

        # Clamp and publish one coherent term/mirror/reward/log state.
        self._apply_scale(float(np.clip(self.current_scale, self.min_scale, self.max_scale)))

    def step(self) -> None:
        """Clamp penalty scale within bounds each step."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        # Clamp current_scale and keep every published representation aligned.
        self._apply_scale(float(np.clip(self.current_scale, self.min_scale, self.max_scale)))

    def _checkpoint_semantics(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "tag": str(self.tag),
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "level_down_threshold": self.level_down_threshold,
            "level_up_threshold": self.level_up_threshold,
            "degree": self.degree,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "semantics": self._checkpoint_semantics(),
            "current_scale": self.current_scale,
        }

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("PenaltyCurriculum state must be a dictionary.")
        version = state.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError(f"Unsupported PenaltyCurriculum state version: {version!r}.")
        if state.get("semantics") != self._checkpoint_semantics():
            raise ValueError(
                "PenaltyCurriculum checkpoint semantics differ from the active runtime."
            )
        scale = state.get("current_scale")
        if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
            raise ValueError("PenaltyCurriculum current_scale must be a real scalar.")
        scale = float(scale)
        if not math.isfinite(scale) or not self.min_scale <= scale <= self.max_scale:
            raise ValueError(
                "PenaltyCurriculum current_scale must be finite and within configured bounds."
            )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.validate_state_dict(state)
        self._apply_scale(state["current_scale"])


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
    (
        enabled,
        tag,
        initial_scale,
        min_scale,
        max_scale,
        level_down_threshold,
        level_up_threshold,
        degree,
    ) = _validate_penalty_curriculum_parameters(
        enabled=enabled,
        tag=tag,
        initial_scale=initial_scale,
        min_scale=min_scale,
        max_scale=max_scale,
        level_down_threshold=level_down_threshold,
        level_up_threshold=level_up_threshold,
        degree=degree,
    )
    if not hasattr(env, "reward_manager"):
        raise RuntimeError("Penalty curriculum configuration requires a reward manager.")

    # Determine which rewards to apply curriculum to
    # Use tag-based selection
    penalty_names: list[str] = []
    original_weights: dict[str, float] = {}
    scaled_configs: dict[str, Any] = {}
    for term_name in env.reward_manager.active_terms:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if tag in term_cfg.tags:
            original_weight = _finite_penalty_scalar(
                f"original reward weight for {term_name!r}",
                term_cfg.weight,
            )
            derived_weights = (
                original_weight * min_scale,
                original_weight * initial_scale,
                original_weight * max_scale,
            )
            if not all(math.isfinite(weight) for weight in derived_weights):
                raise ValueError(
                    f"Penalty curriculum derived weight range for {term_name!r} is non-finite."
                )
            scaled_weight = derived_weights[1]
            penalty_names.append(term_name)
            original_weights[term_name] = original_weight
            scaled_configs[term_name] = replace(term_cfg, weight=scaled_weight)
    if enabled and not penalty_names:
        raise ValueError(
            f"Enabled penalty curriculum tag {tag!r} matches no active reward terms."
        )

    # Publish runtime state only after every configured reward and derived
    # weight has been validated.
    if enabled:
        for name, scaled_cfg in scaled_configs.items():
            env.reward_manager.set_term_cfg(name, scaled_cfg)
    env.use_reward_penalty_curriculum = enabled
    env._curriculum_penalty_reward_names = penalty_names
    env._curriculum_penalty_original_weights = original_weights
    env._curriculum_penalty_cfg = {
        "min_scale": min_scale,
        "max_scale": max_scale,
        "level_down_threshold": level_down_threshold,
        "level_up_threshold": level_up_threshold,
        "degree": degree,
        "current_scale": initial_scale,
    }

    # Set reward_penalty_scale for logging compatibility
    env.reward_penalty_scale = initial_scale
    if hasattr(env, "log_dict"):
        env.log_dict["penalty_scale"] = torch.tensor(initial_scale, dtype=torch.float)


def update_reward_penalty(env, env_ids, **_) -> None:
    """Update penalty scale based on average episode length.

    Modifies reward term weights directly in the reward manager.
    """
    if not getattr(env, "use_reward_penalty_curriculum", False):
        return

    # Match the stateful term: manager reset hooks are also called with an
    # empty selection on ordinary non-terminal control steps.
    if env_ids is None:
        return
    if torch.is_tensor(env_ids):
        if env_ids.numel() == 0:
            return
    else:
        try:
            if len(env_ids) == 0:
                return
        except TypeError:
            pass

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
