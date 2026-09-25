from __future__ import annotations

import time
from typing import Any

import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.curriculum.terms.locomotion import (
    AverageEpisodeLengthTracker,
    WObjectDifficultyCurriculum,
)

# from holosoma.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from holosoma.utils.simulator_config import SimulatorType
from holosoma.utils.step_timing import env_int


class WholeBodyTrackingManager(BaseTask):
    # _reset_buffers_callback marks exactly the rows passed to reset_envs_idx;
    # BaseTask may therefore reuse its already-materialized reset selection.
    supports_direct_post_reset_refresh_ids = True

    def __init__(self, tyro_config, *, device):
        self._motion_metrics_interval = max(1, env_int("HOLOSOMA_MOTION_METRICS_INTERVAL", default=1))
        self._motion_metrics_step = 0
        super().__init__(tyro_config, device=device)
        self._live_motion_metric_keys = self._curriculum_live_motion_metric_keys()
        motion_command = self.command_manager.get_state("motion_command")
        self._validate_live_motion_metric_keys(motion_command, self._live_motion_metric_keys)

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._configure_default_dof_pos()
        self._init_domain_rand_buffers()

    def _configure_default_dof_pos(self):
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

    def _pre_compute_observations_callback(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.base_quat[:] = self.simulator.base_quat[:]
        else:
            self.base_quat[env_ids] = self.simulator.base_quat[env_ids]
        command_manager = getattr(self, "command_manager", None)
        motion_command = command_manager.get_state("motion_command") if command_manager is not None else None
        refresh_object_snapshot = getattr(
            motion_command,
            "refresh_simulator_object_state_snapshot",
            None,
        )
        if callable(refresh_object_snapshot):
            refresh_object_snapshot(env_ids)
        super()._pre_compute_observations_callback(env_ids)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self.need_to_refresh_envs[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # pending_episode_update_mask is only used in curriculum_term::AverageEpisodeLengthTracker.
        self._pending_episode_update_mask[env_ids] = True

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback(env_ids)

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    def _curriculum_live_motion_metric_keys(self) -> frozenset[str]:
        if self.curriculum_manager is None:
            return frozenset()
        metric_keys: set[str] = set()
        for _, term in self.curriculum_manager.iter_terms():
            if not bool(getattr(term, "enabled", False)):
                continue
            if not hasattr(term, "similarity_metric_key"):
                continue
            # WObjectDifficultyCurriculum is currently the sole consumer, but
            # collect every declared key rather than filtering by prefix.  A
            # misspelled prefix must fail at setup instead of silently yielding
            # zero curriculum observations forever.
            metric_key = str(term.similarity_metric_key)
            if not metric_key:
                raise ValueError(
                    "Enabled curriculum similarity_metric_key must be non-empty."
                )
            metric_keys.add(metric_key)
        return frozenset(metric_keys)

    @staticmethod
    def _validate_live_motion_metric_keys(motion_command, metric_keys: frozenset[str]) -> None:
        if not metric_keys:
            return
        supported_keys_getter = getattr(motion_command, "supported_live_metric_keys", None)
        if not callable(supported_keys_getter):
            raise RuntimeError(
                "An enabled curriculum consumes live motion metrics, but motion_command does not "
                "provide the selective live-metric contract."
            )
        supported_keys = frozenset(str(key) for key in supported_keys_getter())
        unsupported = metric_keys - supported_keys
        if unsupported:
            raise ValueError(
                "Enabled curriculum similarity_metric_key is not a supported live tracking error: "
                f"{sorted(unsupported)}. Supported keys are {sorted(supported_keys)}."
            )

    # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------

    def _update_log_dict(self):
        # _update_log_dict happens before reset_envs_idx
        timing = self.step_timing if self.step_timing.enabled else None
        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        if timing is None:
            avg = self._get_average_episode_tracker().get_average()
        else:
            with timing.record("post/log_update/average_episode_length"):
                avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach()
        # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------
        # Add tracking metrics to log_dict
        motion_command = self.command_manager.get_state("motion_command")
        self._motion_metrics_step += 1
        should_update_full_metrics = (
            self._motion_metrics_interval <= 1
            or not getattr(motion_command, "metrics", None)
            or (self._motion_metrics_step - 1) % self._motion_metrics_interval == 0
        )
        if timing is None:
            if should_update_full_metrics:
                motion_command.update_metrics()
            elif self._live_motion_metric_keys:
                motion_command.update_live_metrics(self._live_motion_metric_keys)
        else:
            with timing.record("post/log_update/motion_metrics"):
                if should_update_full_metrics:
                    motion_command.update_metrics()
                elif self._live_motion_metric_keys:
                    motion_command.update_live_metrics(self._live_motion_metric_keys)
        self.log_dict.update(motion_command.metrics)

    def reset_all(self):
        # Clear episode-local buffers without erasing adaptive curriculum restored
        # by algo.load() immediately before PPO.learn() calls its initial reset_all().
        motion_command = self.command_manager.get_state("motion_command")
        curriculum_state = self.get_checkpoint_state()
        motion_command.init_buffers()
        self._load_checkpoint_state(curriculum_state, restore_perception=False)
        observations = super().reset_all()
        # BaseTask.reset_all() takes one zero-action simulator step to build the
        # initial observation/history.  That warm-up step must not inject
        # phantom AS exposure or WObject termination statistics into restored
        # adaptive state.
        self._load_checkpoint_state(curriculum_state, restore_perception=False)
        return observations

    def _compute_final_observations(self):
        """Return the actual next WBT state for timeout value bootstrapping.

        WBT advances its motion command after termination handling.  The base
        implementation therefore observes the post-physics robot state paired
        with the *previous* motion target at a timeout.  Temporarily preview the
        deterministic part of ``MotionCommand.step()`` so the critic sees the
        same target clock as an ordinary next observation, then restore the
        live command before reset handling continues.

        A stochastic freeze, runtime-prepend transition, clip rollover, or an
        observation backed by command state that changes during ``step()``
        cannot be previewed without changing RNG/reset/curriculum semantics.
        Those rare transitions remain terminal but are removed from
        ``time_out_buf`` so PPO cannot bootstrap from a fabricated state.
        """

        timeout_mask = self.time_out_buf.to(dtype=torch.bool)
        if not torch.any(timeout_mask):
            return super()._compute_final_observations()

        motion_command = self.command_manager.get_state("motion_command")
        if motion_command is None or not isinstance(getattr(motion_command, "time_steps", None), torch.Tensor):
            # A WBT configuration without a previewable motion clock must not
            # silently bootstrap from the pre-command observation.
            self._reject_timeout_bootstrap_preview(timeout_mask)
            return super()._compute_final_observations()

        preview_mask = timeout_mask.clone()
        if self._timeout_preview_has_stateful_command_observations():
            preview_mask.zero_()
        if self._timeout_preview_has_enabled_stateful_curriculum():
            preview_mask.zero_()
        preview_mask &= ~self._timeout_preview_due_random_push_mask()

        runtime_prepend_active = getattr(motion_command, "_runtime_default_pose_prepend_active", None)
        if isinstance(runtime_prepend_active, torch.Tensor):
            preview_mask &= ~runtime_prepend_active.to(device=preview_mask.device, dtype=torch.bool)

        time_steps = motion_command.time_steps
        advance_mask = torch.ones_like(preview_mask)
        freeze_getter = getattr(motion_command, "_current_freeze_at_timestep_zero_prob", None)
        freeze_probability = float(freeze_getter()) if callable(freeze_getter) else 0.0
        zero_mask = time_steps == 0
        if freeze_probability >= 1.0:
            advance_mask &= ~zero_mask
        elif freeze_probability > 0.0:
            # Sampling a counterfactual Bernoulli would consume RNG that the
            # subsequent real reset/step path also owns.  Do not guess.
            preview_mask &= ~zero_mask

        next_time_steps = time_steps + advance_mask.to(dtype=time_steps.dtype)
        clip_lengths = getattr(motion_command, "current_clip_lengths", None)
        if not isinstance(clip_lengths, torch.Tensor):
            preview_mask.zero_()
        elif bool(getattr(motion_command, "_disable_clip_end_reset", False)):
            next_time_steps = torch.minimum(next_time_steps, torch.clamp(clip_lengths - 1, min=0))
        else:
            # Normal MotionCommand.step() performs a stochastic reset and also
            # rewrites simulator state at clip rollover.  It is not the same
            # physical next state and cannot be represented by a time increment.
            preview_mask &= next_time_steps < clip_lengths

        rejected_mask = timeout_mask & ~preview_mask
        if torch.any(rejected_mask):
            self._reject_timeout_bootstrap_preview(rejected_mask)
        if not torch.any(preview_mask):
            return super()._compute_final_observations()

        original_time_steps = time_steps[preview_mask].clone()
        future_target_poses = getattr(motion_command, "future_target_poses", None)
        original_future_targets = (
            future_target_poses.clone() if isinstance(future_target_poses, torch.Tensor) else None
        )
        try:
            time_steps[preview_mask] = next_time_steps[preview_mask]
            update_future_targets = getattr(motion_command, "_update_future_target_poses", None)
            if callable(update_future_targets):
                update_future_targets()
            return super()._compute_final_observations()
        finally:
            time_steps[preview_mask] = original_time_steps
            if original_future_targets is not None:
                future_target_poses.copy_(original_future_targets)

    def _reject_timeout_bootstrap_preview(self, rejected_mask: torch.Tensor) -> None:
        """Turn an unpreviewable truncation into a terminal transition."""
        self.time_out_buf[rejected_mask] = False
        if hasattr(self, "log_dict"):
            self.log_dict["termination/timeout_bootstrap_rejected_frac"] = (
                rejected_mask.to(dtype=torch.float32).mean().detach().cpu()
            )

    def _timeout_preview_has_stateful_command_observations(self) -> bool:
        """Detect observation terms whose next value needs a full command step."""
        manager = self.observation_manager
        cfg = getattr(manager, "cfg", None)
        groups = getattr(cfg, "groups", None)
        if not isinstance(groups, dict):
            return False
        active_groups = getattr(manager, "active_group_names", None)
        group_names = groups.keys() if active_groups is None else active_groups
        for group_name in group_names:
            group_cfg = groups[group_name]
            for term_name, term_cfg in group_cfg.terms.items():
                func = term_cfg.func
                if isinstance(func, str):
                    func_name = func
                else:
                    func_name = f"{getattr(func, '__module__', '')}:{getattr(func, '__name__', '')}"
                if term_name == "obj_picked_flag" or func_name.endswith(":obj_picked_flag"):
                    return True
                if "contact_prior_" in term_name or "contact_prior_" in func_name:
                    return True
        return False

    def _timeout_preview_has_enabled_stateful_curriculum(self) -> bool:
        """Conservatively reject previews when curriculum step mutates task state."""
        manager = getattr(self, "curriculum_manager", None)
        iter_terms = getattr(manager, "iter_terms", None)
        if not callable(iter_terms):
            return False
        for term_name, term in iter_terms():
            if term_name == "average_episode_tracker":
                continue
            if bool(getattr(term, "enabled", False)):
                return True
        return False

    def _timeout_preview_due_random_push_mask(self) -> torch.Tensor:
        """Return envs whose counterfactual next state includes a random push."""
        rejected = torch.zeros_like(self.time_out_buf, dtype=torch.bool)
        manager = getattr(self, "randomization_manager", None)
        if manager is None or bool(getattr(self, "is_evaluating", False)):
            return rejected

        cfg = getattr(manager, "cfg", None)
        step_terms = getattr(cfg, "step_terms", None)
        if isinstance(step_terms, dict):
            known_terms = {"push_randomizer_state", "apply_pushes"}
            if set(step_terms) - known_terms:
                # A custom per-step randomizer may alter simulator/observation
                # state after the final observation point.  Do not assume that
                # a motion-clock-only preview represents it.
                return torch.ones_like(rejected)

        get_state = getattr(manager, "get_state", None)
        state = get_state("push_randomizer_state") if callable(get_state) else None
        if state is None or not bool(getattr(state, "enabled", False)):
            return rejected
        counters = getattr(state, "push_robot_counter", None)
        intervals_s = getattr(state, "push_interval_s", None)
        if not isinstance(counters, torch.Tensor) or not isinstance(intervals_s, torch.Tensor):
            return torch.ones_like(rejected)
        interval_steps = torch.clamp((intervals_s / float(self.dt)).to(dtype=torch.long), min=1)
        return (counters.to(dtype=torch.long) + 1) == interval_steps

    @property
    def distributed_loss_weight(self) -> float:
        """Per-rank multiplier that restores global clip weighting under DDP."""
        motion_command = self.command_manager.get_state("motion_command")
        return float(getattr(motion_command, "distributed_loss_weight", 1.0))

    @property
    def curriculum_state_sync_enabled(self) -> bool:
        """Whether WBT has adaptive state that should be synchronized."""
        wobject_curriculum = self._get_wobject_curriculum_term()
        if wobject_curriculum is not None and wobject_curriculum.enabled:
            return True
        motion_command = self.command_manager.get_state("motion_command")
        if getattr(motion_command, "adaptive_timesteps_sampler", None) is not None:
            return True
        return any(
            isinstance(getattr(motion_command, name, None), torch.Tensor)
            for name in ("_clip_success_counts", "_clip_total_counts")
        )

    @property
    def curriculum_state_checkpoint_required(self) -> bool:
        """Whether omitting environment state would alter a resumed objective."""

        return self.curriculum_state_sync_enabled

    def _get_wobject_curriculum_term(self) -> WObjectDifficultyCurriculum | None:
        manager = getattr(self, "curriculum_manager", None)
        iter_terms = getattr(manager, "iter_terms", None)
        if not callable(iter_terms):
            return None
        matches = [term for _, term in iter_terms() if isinstance(term, WObjectDifficultyCurriculum)]
        if len(matches) > 1:
            raise RuntimeError("WBT has multiple WObjectDifficultyCurriculum instances; state ownership is ambiguous.")
        return matches[0] if matches else None

    def get_checkpoint_state(self) -> dict[str, Any]:
        motion_command = self.command_manager.get_state("motion_command")
        wobject_curriculum = self._get_wobject_curriculum_term()
        average_tracker = self._get_average_episode_tracker()
        curriculum_terms = {
            "average_episode_tracker": average_tracker.state_dict(),
        }
        if wobject_curriculum is not None:
            curriculum_terms["w_object_difficulty_curriculum"] = wobject_curriculum.get_checkpoint_state()
        return {
            "version": 4,
            "motion_command": motion_command.get_checkpoint_state(),
            "curriculum_terms": curriculum_terms,
            "perception_managers": self._get_perception_checkpoint_state(),
        }

    def validate_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Validate WBT curriculum state without changing live sampler state."""
        if not state:
            return
        if not isinstance(state, dict):
            raise ValueError("WBT environment checkpoint state must be a dictionary.")
        version = state.get("version", 0)
        if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported WBT environment checkpoint version: {version!r}.")
        motion_command_state = state.get("motion_command")
        if not isinstance(motion_command_state, dict) or not motion_command_state:
            raise ValueError("WBT environment checkpoint is missing motion_command state.")
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.validate_checkpoint_state(motion_command_state)

        wobject_curriculum = self._get_wobject_curriculum_term()
        if version == 1:
            if wobject_curriculum is not None and wobject_curriculum.enabled:
                raise ValueError(
                    "Legacy WBT environment checkpoint version 1 has no WObject curriculum state, "
                    "but WObjectDifficultyCurriculum is enabled; exact resume is impossible."
                )
            self._validate_perception_checkpoint_state(None)
            return

        curriculum_terms = state.get("curriculum_terms")
        if not isinstance(curriculum_terms, dict):
            raise ValueError("WBT environment checkpoint is missing curriculum_terms state.")
        supported_terms = {"w_object_difficulty_curriculum"}
        if version in (3, 4):
            supported_terms.add("average_episode_tracker")
        unexpected_terms = set(curriculum_terms) - supported_terms
        if unexpected_terms:
            raise ValueError(
                f"WBT environment checkpoint has unsupported curriculum terms: {sorted(unexpected_terms)}."
            )
        wobject_state = curriculum_terms.get("w_object_difficulty_curriculum")
        if wobject_curriculum is None:
            if wobject_state is not None:
                raise ValueError(
                    "WBT checkpoint contains WObject curriculum state, but the active environment has no "
                    "WObjectDifficultyCurriculum term."
                )
        elif wobject_state is None:
            if wobject_curriculum.enabled:
                raise ValueError("WBT checkpoint is missing enabled WObject curriculum state.")
        else:
            wobject_curriculum.validate_checkpoint_state(wobject_state)
        if version in (3, 4):
            average_state = curriculum_terms.get("average_episode_tracker")
            if not isinstance(average_state, dict):
                raise ValueError("WBT checkpoint is missing average_episode_tracker state.")
            self._get_average_episode_tracker().validate_state_dict(average_state)
        if version == 4:
            self._validate_perception_checkpoint_state(state.get("perception_managers"))
        else:
            self._validate_perception_checkpoint_state(None)

    def _load_checkpoint_state(
        self,
        state: dict[str, Any] | None,
        *,
        restore_perception: bool,
    ) -> None:
        if not state:
            return
        self.validate_checkpoint_state(state)
        motion_command_state = state["motion_command"]
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.load_checkpoint_state(motion_command_state)
        if state["version"] in (2, 3, 4):
            wobject_curriculum = self._get_wobject_curriculum_term()
            wobject_state = state["curriculum_terms"].get("w_object_difficulty_curriculum")
            if wobject_curriculum is not None and wobject_state is not None:
                wobject_curriculum.load_checkpoint_state(wobject_state)
        if state["version"] in (3, 4):
            self._get_average_episode_tracker().load_state_dict(
                state["curriculum_terms"]["average_episode_tracker"]
            )
        if restore_perception and state["version"] == 4:
            self._load_perception_checkpoint_state(state["perception_managers"])

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        self._load_checkpoint_state(state, restore_perception=True)

    def _restore_checkpoint_state_after_canonical_reset(self, state: dict[str, Any] | None) -> None:
        self._load_checkpoint_state(state, restore_perception=False)

    def synchronize_curriculum_state(
        self,
        *,
        device: str,
        world_size: int,
        process_group=None,
    ) -> None:
        """Synchronize adaptive state without mixing unrelated shard clips."""
        wobject_curriculum = self._get_wobject_curriculum_term()
        if world_size <= 1:
            if wobject_curriculum is not None and wobject_curriculum.enabled:
                wobject_curriculum.synchronize_state(device=device, world_size=1, process_group=process_group)
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("WBT curriculum synchronization requires an initialized process group.")
        motion_command = self.command_manager.get_state("motion_command")
        # Rank-local shards have different clip identities and tensor shapes,
        # so a dense all-reduce would mix unrelated clips.  Some shards do,
        # however, duplicate a global clip to keep every rank non-empty.  Those
        # copies must share one AS estimate or the sampling curriculum changes
        # with the number of ranks used for the experiment.
        if getattr(motion_command, "_rank_local_shard_metadata", None) is not None:
            self._synchronize_rank_local_duplicate_sampler_state(
                motion_command,
                world_size=world_size,
                process_group=process_group,
            )
        else:
            tensors: list[torch.Tensor] = []
            sampler = getattr(motion_command, "adaptive_timesteps_sampler", None)
            if sampler is not None:
                tensors.extend(
                    (
                        sampler.current_bin_failed_count,
                        sampler.bin_failed_count,
                        sampler.current_bin_exposure_count,
                        sampler.bin_exposure_count,
                    )
                )
            for name in ("_clip_success_counts", "_clip_total_counts"):
                value = getattr(motion_command, name, None)
                if isinstance(value, torch.Tensor):
                    tensors.append(value)

            for target in tensors:
                reduced = target.detach().to(device=device)
                torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM, group=process_group)
                reduced /= float(world_size)
                target.copy_(reduced.to(device=target.device, dtype=target.dtype))
            if getattr(motion_command, "clip_weighting_strategy", None) == "success_rate_adaptive":
                motion_command._refresh_adaptive_clip_weights()

        if wobject_curriculum is not None and wobject_curriculum.enabled:
            wobject_curriculum.synchronize_state(
                device=device,
                world_size=world_size,
                process_group=process_group,
            )

    @staticmethod
    def _synchronize_rank_local_duplicate_sampler_state(
        motion_command,
        *,
        world_size: int,
        process_group=None,
    ) -> None:
        """Average AS rows only for global clips present on multiple ranks.

        Rank-local sampler tensors are padded to each shard's longest clip and
        therefore cannot be reduced as dense tensors.  We exchange only the
        valid rows for clips whose shard metadata reports ``cover_count > 1``.
        Unique clips are deliberately omitted and remain untouched.
        """

        sampler = getattr(motion_command, "adaptive_timesteps_sampler", None)
        metadata = getattr(motion_command, "_rank_local_shard_metadata", None)
        if sampler is None or not isinstance(metadata, dict):
            return
        backend = str(torch.distributed.get_backend(process_group)).lower()
        if "nccl" in backend:
            raise RuntimeError(
                "Rank-local adaptive-sampler object synchronization requires a Gloo process group; "
                "refusing to pickle sampler state through NCCL."
            )

        clip_ids = [str(clip_id) for clip_id in getattr(motion_command.motion, "clip_ids", [])]
        cover_counts = metadata.get("clip_cover_counts")
        if not isinstance(cover_counts, dict):
            raise ValueError("Rank-local AS synchronization requires clip_cover_counts metadata.")
        if len(clip_ids) != int(sampler.num_bins_per_clip.numel()):
            raise ValueError(
                "Rank-local AS sampler clip metadata is inconsistent: "
                f"clip_ids={len(clip_ids)}, sampler_rows={int(sampler.num_bins_per_clip.numel())}."
            )

        state_names = (
            "current_bin_failed_count",
            "bin_failed_count",
            "current_bin_exposure_count",
            "bin_exposure_count",
        )
        local_clips: dict[str, dict[str, Any]] = {}
        for local_idx, clip_id in enumerate(clip_ids):
            try:
                expected_cover = int(cover_counts[clip_id])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Rank-local AS metadata has no valid cover count for clip {clip_id!r}."
                ) from exc
            if expected_cover <= 0:
                raise ValueError(f"Rank-local AS cover count must be positive for clip {clip_id!r}.")
            if expected_cover == 1:
                continue

            valid_bins = int(sampler.num_bins_per_clip[local_idx].item())
            states: dict[str, torch.Tensor] = {}
            for state_name in state_names:
                state_tensor = getattr(sampler, state_name, None)
                if not isinstance(state_tensor, torch.Tensor) or state_tensor.ndim != 2:
                    raise ValueError(f"Adaptive sampler state {state_name!r} must be a rank-2 tensor.")
                if local_idx >= state_tensor.shape[0] or valid_bins > state_tensor.shape[1]:
                    raise ValueError(
                        f"Adaptive sampler state {state_name!r} cannot provide {valid_bins} bins "
                        f"for local clip index {local_idx}; shape={tuple(state_tensor.shape)}."
                    )
                states[state_name] = state_tensor[local_idx, :valid_bins].detach().to("cpu").clone()
            local_clips[clip_id] = {
                "cover_count": expected_cover,
                "valid_bins": valid_bins,
                "states": states,
            }

        local_payload = {
            "rank": int(metadata.get("rank", -1)),
            # Exchange every local identity, not just state-bearing duplicate
            # rows.  Otherwise two shards that both incorrectly label the same
            # clip as cover_count=1 silently skip synchronization and bias both
            # AS state and DDP weighting.
            "clip_cover_counts": {
                clip_id: int(cover_counts[clip_id])
                for clip_id in clip_ids
            },
            "clips": local_clips,
        }
        gathered_payloads: list[dict[str, Any] | None] = [None] * int(world_size)
        torch.distributed.all_gather_object(
            gathered_payloads,
            local_payload,
            group=process_group,
        )

        gathered_by_clip: dict[str, list[dict[str, Any]]] = {}
        gathered_coverage: dict[str, list[int]] = {}
        for payload in gathered_payloads:
            if not isinstance(payload, dict):
                raise RuntimeError("Rank-local AS synchronization received an invalid rank payload.")
            payload_cover_counts = payload.get("clip_cover_counts")
            if not isinstance(payload_cover_counts, dict):
                raise RuntimeError("Rank-local AS synchronization payload is missing clip coverage metadata.")
            for clip_id, raw_advertised_cover in payload_cover_counts.items():
                try:
                    advertised_cover = int(raw_advertised_cover)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid synchronized AS cover count for clip {clip_id!r}."
                    ) from exc
                gathered_coverage.setdefault(str(clip_id), []).append(advertised_cover)
            payload_clips = payload.get("clips")
            if not isinstance(payload_clips, dict):
                raise RuntimeError("Rank-local AS synchronization payload is missing its clip mapping.")
            for clip_id, clip_payload in payload_clips.items():
                if not isinstance(clip_payload, dict):
                    raise RuntimeError(f"Invalid synchronized AS payload for clip {clip_id!r}.")
                gathered_by_clip.setdefault(str(clip_id), []).append(clip_payload)

        for local_idx, clip_id in enumerate(clip_ids):
            expected_cover = int(cover_counts[clip_id])
            advertised_covers = gathered_coverage.get(clip_id, [])
            if len(advertised_covers) != expected_cover or any(
                advertised_cover != expected_cover for advertised_cover in advertised_covers
            ):
                raise RuntimeError(
                    "Rank-local AS clip coverage differs from shard metadata: "
                    f"clip={clip_id!r}, gathered={len(advertised_covers)}, "
                    f"advertised={advertised_covers}, expected={expected_cover}."
                )
            if expected_cover == 1:
                continue
            clip_payloads = gathered_by_clip.get(clip_id, [])
            if len(clip_payloads) != expected_cover:
                raise RuntimeError(
                    "Rank-local AS clip coverage differs from shard metadata: "
                    f"clip={clip_id!r}, gathered={len(clip_payloads)}, expected={expected_cover}."
                )

            valid_bins = int(sampler.num_bins_per_clip[local_idx].item())
            for clip_payload in clip_payloads:
                if int(clip_payload.get("cover_count", -1)) != expected_cover:
                    raise RuntimeError(f"Ranks disagree on cover_count for duplicated clip {clip_id!r}.")
                if int(clip_payload.get("valid_bins", -1)) != valid_bins:
                    raise RuntimeError(f"Ranks disagree on valid AS bins for duplicated clip {clip_id!r}.")

            for state_name in state_names:
                rows = []
                for clip_payload in clip_payloads:
                    states = clip_payload.get("states")
                    row = states.get(state_name) if isinstance(states, dict) else None
                    if not isinstance(row, torch.Tensor) or row.shape != (valid_bins,):
                        raise RuntimeError(
                            f"Invalid {state_name!r} row for duplicated clip {clip_id!r}; "
                            f"expected shape {(valid_bins,)}."
                        )
                    rows.append(row.to(dtype=torch.float32))
                averaged = torch.stack(rows, dim=0).mean(dim=0)
                target = getattr(sampler, state_name)
                target[local_idx, :valid_bins].copy_(
                    averaged.to(device=target.device, dtype=target.dtype)
                )

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # TODO(jchen): Now,reset robot/object states is implemented in command/terms/wbt.MotionCommand.reset
        # discuss whether to move to here in the future.
        pass

    ########################################################### Push robots #########################################
    # TODO: This should be moved to the randomization manager.
    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(6, dtype=torch.float32, device=self.device)

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 6:
            raise ValueError("WholeBodyTracking push velocity vector must have exactly 6 components.")

        rand = torch.rand(len(env_ids), 6, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        # Match IsaacLab/BeyondMimic semantics: pushes add velocity on top of current state.
        self.simulator.robot_root_states[env_ids, 7:13] += self.push_robot_vel_buf[env_ids]
        # This writes through immediately.  Do not queue reset-only refresh work:
        # that path also clears contact history and re-runs perception for the rows.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    #########################################################################################################
    ## Debug visualization
    #########################################################################################################

    def _draw_debug_vis_isaacsim(self):
        motion_command = self.command_manager.get_state("motion_command")
        if not hasattr(motion_command, "visualization_markers"):
            return
        if motion_command.visualization_markers is None:
            return
        # torso link
        real_robot_pos_xyz = motion_command.robot_ref_pos_w.clone()
        real_robot_quat_xyzw = motion_command.robot_ref_quat_w.clone()
        real_robot_quat_wxyz = real_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["real_robot"].visualize(real_robot_pos_xyz, real_robot_quat_wxyz)

        motion_robot_pos_xyz = motion_command.ref_pos_w.clone()
        motion_robot_quat_xyzw = motion_command.ref_quat_w.clone()
        motion_robot_quat_wxyz = motion_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["motion_robot"].visualize(motion_robot_pos_xyz, motion_robot_quat_wxyz)

        for body_idx, body_names in enumerate(motion_command.motion_cfg.body_names_to_track):
            motion_robot_body_pos_xyz = motion_command.body_pos_w[0, body_idx].clone()
            motion_command.visualization_markers[f"motion_{body_names}"].visualize(
                motion_robot_body_pos_xyz.unsqueeze(0)
            )

        # object
        if motion_command.motion.has_object:
            real_object_pos_xyz = motion_command.simulator_object_pos_w.clone()
            real_object_quat_xyzw = motion_command.simulator_object_quat_w.clone()
            real_object_quat_wxyz = real_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["real_object"].visualize(real_object_pos_xyz, real_object_quat_wxyz)

            motion_object_pos_xyz = motion_command.object_pos_w.clone()
            motion_object_quat_xyzw = motion_command.object_quat_w.clone()
            motion_object_quat_wxyz = motion_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["motion_object"].visualize(
                motion_object_pos_xyz, motion_object_quat_wxyz
            )

    def _draw_debug_vis_isaacgym(self):
        self.simulator.clear_lines()
        n_bodies = len(self.motion_command.motion_cfg.body_names_to_track)
        for env_id in range(self.num_envs):
            for body_idx in range(n_bodies):
                color = (0.0, 1.0, 0.0)
                self.simulator.draw_sphere(
                    self.motion_command.body_pos_relative_w[env_id, body_idx], 0.03, color, env_id, body_idx
                )

                color = (0.0, 0.0, 1.0)
                self.simulator.draw_sphere(
                    self.motion_command.robot_body_pos_w[env_id, body_idx], 0.03, color, env_id, n_bodies + body_idx
                )

            color = (0.0, 1.0, 0.0)
            self.simulator.draw_sphere(self.motion_command.ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 0)
            color = (0.0, 0.0, 1.0)
            self.simulator.draw_sphere(
                self.motion_command.robot_ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 1
            )

    def _draw_debug_vis(self):
        if self.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._draw_debug_vis_isaacsim()
        elif self.simulator.get_simulator_type() == SimulatorType.ISAACGYM:
            self._draw_debug_vis_isaacgym()

    def step_visualize_motion(self, actions, *, advance_motion: bool = True):
        """Render one kinematic motion state for viewers and replay capture.

        Normal callers advance the command exactly as before.  Formal replay
        uses ``advance_motion=False`` once to materialize the reset state at
        runtime-prepend alpha zero before recording the remaining motion clock.
        Keeping this as an explicit keyword preserves the training/episodic
        ``motion_end_mask`` contract and avoids capture-only clock mutation.
        """

        if hasattr(self, "_viser_live") and getattr(self._viser_live, "enabled", False):
            self._viser_live.apply_pending_controls()
            self._viser_live.wait_if_paused()

        motion_command = self.command_manager.get_state("motion_command")
        dt = 1.0 / float(motion_command.motion.fps)
        if advance_motion:
            motion_command.step()
        self._draw_debug_vis()

        # set root_states_from_motion_command
        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()  # wxyz
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.dof_pos[env_ids] = joint_pos
        self.simulator.dof_vel[env_ids] = joint_vel

        self.simulator.robot_root_states[env_ids, :3] = root_pos
        self.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids)

        if motion_command.motion.has_object:
            # set object root_states from motion command
            object_pos = motion_command.object_pos_w.clone()
            object_ori = motion_command.object_quat_w.clone()
            object_lin_vel = motion_command.object_lin_vel_w.clone()

            object_states = torch.zeros(len(env_ids), 13, device=self.device)
            object_states[:, :3] = object_pos[:]
            object_states[:, 3:7] = object_ori[:]
            object_states[:, 7:10] = object_lin_vel[:]
            object_states[:, 10:13] = torch.zeros_like(object_lin_vel[:])
            if hasattr(motion_command, "_set_simulator_object_states"):
                motion_command._set_simulator_object_states(env_ids, object_states)
            else:
                self.simulator.set_actor_states(["object"], env_ids, object_states)

        sim_type = self.simulator.get_simulator_type()
        if sim_type == SimulatorType.MUJOCO:
            write_state_updates = getattr(self.simulator, "write_state_updates", None)
            if callable(write_state_updates):
                write_state_updates()
            render = getattr(self.simulator, "render", None)
            viewer = getattr(self.simulator, "viewer", None)
            if callable(render) and viewer is not None:
                render(sync_frame_time=False)
        else:
            self.simulator.scene.write_data_to_sim()
            self.simulator.sim.forward()
            self.simulator.sim.render()
        self.simulator.refresh_sim_tensors()
        # Keep replay perception outputs in sync with the kinematic state written above.
        # Without this call, ray debug overlays can update while camera depth maps remain stale.
        self._pre_compute_observations_callback()
        if hasattr(self, "_viser_live"):
            self._viser_live.record_step()
        self._draw_scandots_in_viewer()

        if getattr(self.simulator, "viewer", None) is not None:
            time.sleep(dt)

        return bool(motion_command.motion_end_mask()[0].item())
