from __future__ import annotations

import math
import os
import pathlib
import statistics
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Generator, TypedDict

import wandb
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.safe_torch_import import (
    TensorboardSummaryWriter as SummaryWriter,
    torch,
)

console = Console()

REWARD_LOG_GROUPS: dict[str, tuple[str, ...]] = {
    "Track": (
        "motion_global_ref_position_error_exp",
        "motion_global_ref_orientation_error_exp",
        "motion_relative_body_position_error_exp",
        "motion_relative_body_orientation_error_exp",
        "motion_global_body_lin_vel",
        "motion_global_body_ang_vel",
    ),
    "Object": (
        "object_global_ref_position_error_exp",
        "object_global_ref_orientation_error_exp",
    ),
    "Contact": (
        "offline_wrist_target_guidance",
        "offline_contact_guidance",
    ),
    "Regularize": (
        "action_rate_l2",
        "limits_dof_pos",
        "undesired_contacts",
    ),
}
REWARD_LOG_GROUP_BY_TERM = {
    term_name: group_name for group_name, term_names in REWARD_LOG_GROUPS.items() for term_name in term_names
}
REWARD_LOG_GROUP_ORDER = ("Track", "Object", "Contact", "Regularize", "Rest")
REWARD_LOG_PARAM_KEYS = (
    "sigma",
    "sigma_xy",
    "sigma_yaw",
    "sigma_z",
    "position_sigma",
    "force_threshold",
    "force_sigma",
    "wrist_weight",
    "contact_weight",
)
REWARD_REPORTING_CONTRACT = {
    "legacy_episode": "sum(weight * raw_reward * dt) / max_episode_length_s",
    "legacy_raw_episode": "sum(raw_reward) / max_episode_length_s",
    "episode_rate": "sum(weight * raw_reward * dt) / actual_alive_time_s",
    "raw_episode_mean": "sum(raw_reward) / actual_alive_steps",
    "mean_reward_per_alive_step": "sum(actual_policy_reward) / sum(actual_alive_steps)",
    "termination_condition_fractions": "per-env-step fractions; component conditions may overlap",
}


def get_reward_log_group(term_name: str) -> str:
    """Return the W&B reward panel group for a reward term."""
    return REWARD_LOG_GROUP_BY_TERM.get(term_name, "Rest")


def collect_reward_wandb_metadata(reward_cfg: Any) -> tuple[dict[str, Any], dict[str, float | str]]:
    """Build W&B config/summary metadata for grouped reward terms."""
    if reward_cfg is None or not hasattr(reward_cfg, "terms"):
        return {}, {}

    grouped_spec: dict[str, dict[str, dict[str, Any]]] = {group_name: {} for group_name in REWARD_LOG_GROUP_ORDER}
    summary: dict[str, float | str] = {}

    for term_name, term_cfg in reward_cfg.terms.items():
        weight = float(getattr(term_cfg, "weight", 0.0))
        if weight == 0.0:
            continue

        group_name = get_reward_log_group(str(term_name))
        params = dict(getattr(term_cfg, "params", {}) or {})
        term_spec: dict[str, Any] = {
            "weight": weight,
        }
        summary[f"RewardSpec/{group_name}/{term_name}/weight"] = weight

        for param_key in REWARD_LOG_PARAM_KEYS:
            if param_key not in params:
                continue
            param_value = params[param_key]
            if isinstance(param_value, (int, float, str, bool)):
                term_spec[param_key] = param_value
                if isinstance(param_value, (int, float, bool)):
                    summary[f"RewardSpec/{group_name}/{term_name}/{param_key}"] = float(param_value)
                else:
                    summary[f"RewardSpec/{group_name}/{term_name}/{param_key}"] = str(param_value)

        grouped_spec[group_name][str(term_name)] = term_spec

    grouped_spec = {group_name: terms for group_name, terms in grouped_spec.items() if terms}
    if not grouped_spec:
        return {}, {}
    return {
        "reward_group_spec": grouped_spec,
        "reward_reporting_contract": REWARD_REPORTING_CONTRACT,
    }, summary


class LogDict(TypedDict):
    """Dictionary containing iteration info, timing, and buffers for logging."""

    it: int
    """Current iteration number."""

    loss_dict: dict[str, float]
    """Dictionary of loss values."""


class TrainLogDict(TypedDict):
    """Dictionary containing training metrics."""

    fps: float
    """Frames per second (training speed)."""

    # Additional metrics can be added here


class LoggingHelper:
    _WANDB_METRIC_SUMMARIES = {
        "Eval/fixed_bc_mu_mse": "min",
        "Eval/fixed_bc_final_mu_mse": "last",
        "Eval/fixed_bc_guard_reference_min_mu_mse": "min",
        "Eval/fixed_bc_guard_effective_threshold_mu_mse": "last",
        "Eval/fixed_bc_guard_consecutive_exceedances": "last",
        "Eval/fixed_bc_guard_last_mu_mse": "last",
    }
    _WANDB_HIDDEN_METRIC_EXACT = {
        "Train/num_samples",
        "Train/command_goal_training_iteration",
        "Train/mean_episode_length_motion_total",
        "Train/mean_episode_length_motion_total/time",
        "Train/ppo_dagger_target_coeff",
        "Train/ppo_dagger_coeff",
        "Train/ppo_dagger_bc_weight",
        "Train/teacher_action_mix_ratio",
        "Train/teacher_action_mix_ratio_start",
        "Train/teacher_action_mix_ratio_end",
        "Train/teacher_action_mix_ratio_end_iteration",
        "Loss/teacher_bc_mask_fraction",
        "Eval/fixed_bc_num_samples",
        "Eval/fixed_bc_terminal_observation",
        "Eval/fixed_bc_scheduled_evaluation",
        "Eval/fixed_bc_guard_applied",
    }
    _WANDB_HIDDEN_METRIC_PREFIXES = ("Perf/",)
    _WANDB_HIDDEN_METRIC_SUFFIXES = ("/time",)

    def __init__(
        self,
        writer: SummaryWriter,
        log_dir: str | pathlib.Path,
        num_envs: int,
        num_steps_per_env: int,
        num_learning_iterations: int,
        device: str = "cpu",
        prefix: str = "",
        title: str = "Training Log",
        is_main_process: bool = True,
        num_gpus: int = 1,
    ):
        """Initialize the logging helper.

        Parameters
        ----------
        writer : SummaryWriter
            TensorBoard writer for logging metrics
        log_dir : str
            Directory to store logs
        num_envs : int
            Number of environments to track
        num_steps_per_env : int
            Number of steps per environment between each call to `post_epoch_logging`.
        num_learning_iterations : int
            Number of total learning iterations.
        device : str, optional
            Device to use for tensors, by default "cpu"
        prefix : str, optional
            Prefix to add to all the logging keys.
        title : str, optional
            Title of the logging panel.
        is_main_process : bool, optional
            Whether this is the main process.
        num_gpus : int, optional
            Number of GPUs to use.
        """
        self.writer: SummaryWriter = writer
        self.log_dir: str = str(log_dir)
        self.device: str = device
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.collection_time: float = 0.0
        self.learn_time: float = 0.0
        self.num_envs: int = num_envs
        self.num_steps_per_env: int = num_steps_per_env
        self.num_learning_iterations: int = num_learning_iterations
        self.prefix: str = prefix
        self.title: str = title
        self.is_main_process: bool = is_main_process
        self.num_gpus: int = num_gpus

        # Book keeping
        self.ep_infos: list[dict[str, Any]] = []
        self.raw_ep_infos: list[dict[str, Any]] = []
        self.episode_rate_infos: list[dict[str, Any]] = []
        self.raw_episode_mean_infos: list[dict[str, Any]] = []
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.rewweightbuffer: deque[float] = deque(maxlen=100)
        self.lenweightbuffer: deque[float] = deque(maxlen=100)
        # Completed episodes are tracked separately so distributed logging can
        # transfer each episode exactly once.  Gathering the rolling deques
        # themselves would duplicate old episodes at every iteration.
        self._completed_rewards_since_sync: list[float] = []
        self._completed_lengths_since_sync: list[float] = []
        # A distributed iteration can complete far more than 100 episodes.  If
        # rank payloads are appended to ``rewbuffer`` in rank order, deque
        # truncation keeps only the highest ranks and is badly biased when a
        # rank owns a fixed motion shard.  Store one exact, globally weighted
        # aggregate per synchronization instead.  The window is measured in
        # objective-weighted episode mass and trims the oldest aggregate
        # proportionally when it crosses the legacy 100-episode horizon.
        self._distributed_episode_batches: deque[tuple[float, float, float]] = deque()
        self._distributed_episode_window_size = 100.0
        self.distributed_loss_weight_sum = 1.0
        self.distributed_effective_episode_count = 0.0
        self.cur_reward_sum: torch.Tensor = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length: torch.Tensor = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        self.episode_env_tensors: TensorAverageMeterDict = TensorAverageMeterDict()
        self._wandb_defined_metrics: set[str] = set()

    @contextmanager
    def record_collection_time(self) -> Generator[None, None, None]:
        """Record the time taken for collection."""
        start_time = time.perf_counter()
        yield
        self.collection_time += time.perf_counter() - start_time

    @contextmanager
    def record_learn_time(self) -> Generator[None, None, None]:
        """Record the time taken for learning."""
        start_time = time.perf_counter()
        yield
        self.learn_time += time.perf_counter() - start_time

    def update_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict[str, Any]) -> None:
        """Update episode statistics.

        Parameters
        ----------
        rewards : torch.Tensor
            Rewards from the environment
        dones : torch.Tensor
            Done flags from the environment
        infos : dict[str, Any]
            Additional info from the environment
        """
        episode_info = infos.get("episode", {})
        if episode_info:
            self.ep_infos.append(episode_info)
        # Also process raw episode data if it exists
        raw_episode_info = infos.get("raw_episode", {})
        if raw_episode_info:
            self.raw_ep_infos.append(raw_episode_info)
        episode_rate_info = infos.get("episode_rate", {})
        if episode_rate_info:
            self.episode_rate_infos.append(episode_rate_info)
        raw_episode_mean_info = infos.get("raw_episode_mean", {})
        if raw_episode_mean_info:
            self.raw_episode_mean_infos.append(raw_episode_mean_info)
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        # BaseTask already materializes the transition-local reset IDs while
        # selecting environment resets.  Reuse them to avoid a second CUDA
        # ``nonzero`` synchronization; legacy/fake environments that do not
        # publish the contract retain the historical dones-based fallback.
        reset_env_ids = infos.get("reset_env_ids")
        if isinstance(reset_env_ids, torch.Tensor):
            new_ids = reset_env_ids.to(
                device=self.cur_reward_sum.device,
                dtype=torch.long,
            ).reshape(-1, 1)
        else:
            new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
            completed_rewards = self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
            completed_lengths = self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
            self.rewbuffer.extend(completed_rewards)
            self.lenbuffer.extend(completed_lengths)
            self.rewweightbuffer.extend([1.0] * len(completed_rewards))
            self.lenweightbuffer.extend([1.0] * len(completed_lengths))
            self._completed_rewards_since_sync.extend(completed_rewards)
            self._completed_lengths_since_sync.extend(completed_lengths)
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

        # Update episode environment tensors
        self.episode_env_tensors.add(infos["to_log"])

    @staticmethod
    def _prepare_tensor_dict_summary_entries(
        items: list[dict[str, Any]],
    ) -> list[tuple[str, torch.Tensor, int]]:
        """Build per-value device sums in the historical traversal order."""

        entries: list[tuple[str, torch.Tensor, int]] = []
        for item in items:
            for key, raw_value in item.items():
                value = raw_value.detach() if isinstance(raw_value, torch.Tensor) else torch.as_tensor(raw_value)
                value = value.to(dtype=torch.float64).reshape(-1)
                if value.numel() == 0:
                    continue
                entries.append((str(key), value.sum(), int(value.numel())))
        return entries

    def _prepare_env_summary_entries(self) -> list[tuple[str, torch.Tensor, int]]:
        """Build environment-meter sums in the historical key/step order."""

        entries: list[tuple[str, torch.Tensor, int]] = []
        for key, meter in self.episode_env_tensors.data.items():
            for raw_value in meter.tensors:
                value = raw_value.detach().to(dtype=torch.float64).reshape(-1)
                if value.numel() == 0:
                    continue
                entries.append((str(key), value.sum(), int(value.numel())))
        return entries

    @staticmethod
    def _summarize_prepared_tensor_groups(
        groups: dict[str, list[tuple[str, torch.Tensor, int]]],
    ) -> dict[str, dict[str, tuple[float, int]]]:
        """Bulk-copy scalar sums once per device, then merge on the host.

        Each input entry is already ordered exactly like the former nested
        summary loops.  Device transfers are batched across all metric groups,
        while the final Python additions still happen entry-by-entry in that
        original order.  This preserves the existing floating-point rounding,
        key insertion order, and element-count semantics.
        """

        host_values: dict[str, list[float | None]] = {
            group_name: [None] * len(entries)
            for group_name, entries in groups.items()
        }
        by_device: dict[torch.device, list[tuple[str, int, torch.Tensor]]] = {}
        for group_name, entries in groups.items():
            for entry_index, (_, value_sum, _) in enumerate(entries):
                by_device.setdefault(value_sum.device, []).append(
                    (group_name, entry_index, value_sum)
                )

        for device_entries in by_device.values():
            copied_values = torch.stack(
                [value_sum for _, _, value_sum in device_entries]
            ).cpu().tolist()
            for (group_name, entry_index, _), copied_value in zip(
                device_entries,
                copied_values,
                strict=True,
            ):
                host_values[group_name][entry_index] = float(copied_value)

        summaries: dict[str, dict[str, tuple[float, int]]] = {}
        for group_name, entries in groups.items():
            summary: dict[str, tuple[float, int]] = {}
            for (key, _, value_count), value_sum in zip(
                entries,
                host_values[group_name],
                strict=True,
            ):
                if value_sum is None:
                    raise RuntimeError(
                        f"Metric sum for group {group_name!r}, key {key!r} was not copied to the host."
                    )
                old_sum, old_count = summary.get(key, (0.0, 0))
                summary[key] = (
                    old_sum + value_sum,
                    old_count + value_count,
                )
            summaries[group_name] = summary
        return summaries

    @staticmethod
    def _summarize_tensor_dicts(items: list[dict[str, Any]]) -> dict[str, tuple[float, int]]:
        """Return per-key sums/counts without retaining rank-sized tensors."""

        groups = {
            "summary": LoggingHelper._prepare_tensor_dict_summary_entries(items),
        }
        return LoggingHelper._summarize_prepared_tensor_groups(groups)["summary"]

    def _summarize_env_tensors(self) -> dict[str, tuple[float, int]]:
        groups = {"summary": self._prepare_env_summary_entries()}
        return self._summarize_prepared_tensor_groups(groups)["summary"]

    def _summarize_iteration_tensors(self) -> dict[str, dict[str, tuple[float, int]]]:
        """Summarize all iteration tensor metrics with one copy per device."""

        return self._summarize_prepared_tensor_groups(
            {
                "episode": self._prepare_tensor_dict_summary_entries(self.ep_infos),
                "raw_episode": self._prepare_tensor_dict_summary_entries(self.raw_ep_infos),
                "episode_rate": self._prepare_tensor_dict_summary_entries(self.episode_rate_infos),
                "raw_episode_mean": self._prepare_tensor_dict_summary_entries(self.raw_episode_mean_infos),
                "env": self._prepare_env_summary_entries(),
            }
        )

    @staticmethod
    def _merge_summaries(
        payloads: list[dict[str, Any]],
        field: str,
    ) -> dict[str, float]:
        totals: dict[str, tuple[float, float]] = {}
        for payload in payloads:
            rank_weight = float(payload["loss_weight"])
            if rank_weight <= 0.0:
                continue
            for key, (value_sum, value_count) in payload[field].items():
                old_sum, old_count = totals.get(key, (0.0, 0.0))
                totals[key] = (
                    old_sum + rank_weight * float(value_sum),
                    old_count + rank_weight * int(value_count),
                )
        return {key: value_sum / value_count for key, (value_sum, value_count) in totals.items() if value_count > 0}

    @staticmethod
    def _weighted_buffer_mean(values: deque[float], weights: deque[float]) -> float:
        if values and not weights:
            # Backward-compatible for callers/tests that seed the public
            # rolling value deques directly.
            return statistics.mean(values)
        if len(values) != len(weights):
            raise RuntimeError("Distributed episode value/weight buffers are misaligned.")
        denominator = sum(weights)
        if denominator <= 0.0:
            return statistics.mean(values)
        return sum(value * weight for value, weight in zip(values, weights, strict=True)) / denominator

    def _append_distributed_episode_batch(self, payloads: list[dict[str, Any]]) -> None:
        reward_sum = 0.0
        length_sum = 0.0
        weighted_count = 0.0
        for payload in payloads:
            rank_weight = float(payload["loss_weight"])
            episode_count = int(payload["completed_episode_count"])
            if episode_count < 0:
                raise RuntimeError("Distributed completed-episode count cannot be negative.")
            if episode_count == 0 or rank_weight == 0.0:
                continue
            reward_sum += rank_weight * float(payload["completed_reward_sum"])
            length_sum += rank_weight * float(payload["completed_length_sum"])
            weighted_count += rank_weight * episode_count

        if weighted_count <= 0.0:
            # A no-completion iteration must not evict or dilute the existing
            # rolling window.
            self.distributed_effective_episode_count = sum(batch[2] for batch in self._distributed_episode_batches)
            return
        if not all(math.isfinite(value) for value in (reward_sum, length_sum, weighted_count)):
            raise RuntimeError("Distributed episode aggregates must be finite.")

        self._distributed_episode_batches.append((reward_sum, length_sum, weighted_count))
        total_count = sum(batch[2] for batch in self._distributed_episode_batches)
        window_size = float(self._distributed_episode_window_size)
        while (
            self._distributed_episode_batches and total_count - self._distributed_episode_batches[0][2] >= window_size
        ):
            total_count -= self._distributed_episode_batches.popleft()[2]
        if self._distributed_episode_batches and total_count > window_size:
            excess = total_count - window_size
            old_reward_sum, old_length_sum, old_count = self._distributed_episode_batches[0]
            keep_fraction = (old_count - excess) / old_count
            self._distributed_episode_batches[0] = (
                old_reward_sum * keep_fraction,
                old_length_sum * keep_fraction,
                old_count * keep_fraction,
            )
            total_count = window_size
        self.distributed_effective_episode_count = total_count

    def _distributed_episode_means(self) -> tuple[float, float] | None:
        if not self._distributed_episode_batches:
            return None
        denominator = sum(batch[2] for batch in self._distributed_episode_batches)
        if denominator <= 0.0:
            return None
        reward_mean = sum(batch[0] for batch in self._distributed_episode_batches) / denominator
        length_mean = sum(batch[1] for batch in self._distributed_episode_batches) / denominator
        return reward_mean, length_mean

    def _mean_reward(self) -> float:
        distributed_means = self._distributed_episode_means()
        if distributed_means is not None:
            return distributed_means[0]
        return self._weighted_buffer_mean(self.rewbuffer, self.rewweightbuffer)

    def _mean_episode_length(self) -> float:
        distributed_means = self._distributed_episode_means()
        if distributed_means is not None:
            return distributed_means[1]
        return self._weighted_buffer_mean(self.lenbuffer, self.lenweightbuffer)

    def _mean_reward_per_alive_step(self) -> float | None:
        """Return actual policy reward per control step over completed episodes."""

        if not self._has_reward_statistics() or not self._has_length_statistics():
            return None
        mean_episode_length = self._mean_episode_length()
        if mean_episode_length <= 0.0:
            return None
        return self._mean_reward() / mean_episode_length

    def _has_reward_statistics(self) -> bool:
        return bool(self.rewbuffer) or self._distributed_episode_means() is not None

    def _has_length_statistics(self) -> bool:
        return bool(self.lenbuffer) or self._distributed_episode_means() is not None

    @staticmethod
    def _merge_weighted_losses(payloads: list[dict[str, Any]]) -> dict[str, float]:
        keys = {str(key) for payload in payloads for key in payload["loss_dict"]}
        result: dict[str, float] = {}
        for key in keys:
            entries = [
                (float(payload["loss_dict"][key]), float(payload["loss_weight"]))
                for payload in payloads
                if key in payload["loss_dict"]
            ]
            positive_weight = sum(weight for _, weight in entries if weight > 0.0)
            if positive_weight > 0.0:
                result[key] = sum(value * weight for value, weight in entries if weight > 0.0) / positive_weight
            elif entries:
                result[key] = sum(value for value, _ in entries) / len(entries)
        return result

    def synchronize_distributed_metrics(
        self,
        loss_dict: dict[str, float],
        *,
        loss_weight: float = 1.0,
        process_group=None,
    ) -> dict[str, float]:
        """Aggregate compact training statistics across ranks before rank-zero logging.

        Every rank must call this method in the same order.  Only sums, counts,
        newly completed episode returns/lengths, and scalar losses are moved;
        rollout-sized tensors never leave their owning rank.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            self._completed_rewards_since_sync.clear()
            self._completed_lengths_since_sync.clear()
            return loss_dict

        world_size = torch.distributed.get_world_size(group=process_group)
        if world_size <= 1:
            self._completed_rewards_since_sync.clear()
            self._completed_lengths_since_sync.clear()
            return loss_dict

        backend = str(torch.distributed.get_backend(process_group)).lower()
        if "nccl" in backend:
            raise RuntimeError(
                "Distributed metric object synchronization requires a Gloo process group; "
                "refusing to pickle metrics through NCCL."
            )
        rank = torch.distributed.get_rank(group=process_group)
        if bool(rank == 0) != bool(self.is_main_process):
            raise RuntimeError("Distributed metric process-group rank 0 must be the configured main process.")
        completed_episode_count = len(self._completed_rewards_since_sync)
        if completed_episode_count != len(self._completed_lengths_since_sync):
            raise RuntimeError("Completed reward/length deltas are misaligned before distributed synchronization.")
        tensor_summaries = self._summarize_iteration_tensors()
        payload = {
            "loss_dict": {str(key): float(value) for key, value in loss_dict.items()},
            "loss_weight": float(loss_weight),
            "episode": tensor_summaries["episode"],
            "raw_episode": tensor_summaries["raw_episode"],
            "episode_rate": tensor_summaries["episode_rate"],
            "raw_episode_mean": tensor_summaries["raw_episode_mean"],
            "env": tensor_summaries["env"],
            "completed_reward_sum": float(sum(self._completed_rewards_since_sync)),
            "completed_length_sum": float(sum(self._completed_lengths_since_sync)),
            "completed_episode_count": completed_episode_count,
            "collection_time": float(self.collection_time),
            "learn_time": float(self.learn_time),
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, payload, group=process_group)

        self._completed_rewards_since_sync.clear()
        self._completed_lengths_since_sync.clear()
        if not all(item is not None for item in gathered):
            raise RuntimeError("Distributed metric synchronization returned an incomplete payload list.")
        payloads = [item for item in gathered if item is not None]
        loss_weights = [float(item["loss_weight"]) for item in payloads]
        if not all(math.isfinite(weight) and weight >= 0.0 for weight in loss_weights):
            raise RuntimeError(f"Distributed loss weights must be finite and non-negative, got {loss_weights}.")
        self.distributed_loss_weight_sum = sum(loss_weights)
        if self.distributed_loss_weight_sum <= 0.0:
            raise RuntimeError("At least one rank must have a positive distributed loss weight.")
        if not math.isclose(
            self.distributed_loss_weight_sum,
            float(world_size),
            rel_tol=1.0e-5,
            abs_tol=1.0e-5,
        ):
            raise RuntimeError(
                "Distributed loss weights must sum to world_size because gradient reduction divides by "
                f"world_size: sum={self.distributed_loss_weight_sum}, world_size={world_size}."
            )
        merged_loss = self._merge_weighted_losses(payloads)
        if rank != 0:
            # Non-main ranks never call post_epoch_logging(), so clear their
            # per-iteration data here rather than accumulating it forever.
            self.ep_infos.clear()
            self.raw_ep_infos.clear()
            self.episode_rate_infos.clear()
            self.raw_episode_mean_infos.clear()
            self.episode_env_tensors.clear()
            self.collection_time = 0.0
            self.learn_time = 0.0
            return merged_loss

        episode_means = self._merge_summaries(payloads, "episode")
        raw_episode_means = self._merge_summaries(payloads, "raw_episode")
        episode_rate_means = self._merge_summaries(payloads, "episode_rate")
        raw_episode_mean_means = self._merge_summaries(payloads, "raw_episode_mean")
        env_means = self._merge_summaries(payloads, "env")
        self.ep_infos = (
            [{key: torch.tensor([value], device=self.device) for key, value in episode_means.items()}]
            if episode_means
            else []
        )
        self.raw_ep_infos = (
            [{key: torch.tensor([value], device=self.device) for key, value in raw_episode_means.items()}]
            if raw_episode_means
            else []
        )
        self.episode_rate_infos = (
            [{key: torch.tensor([value], device=self.device) for key, value in episode_rate_means.items()}]
            if episode_rate_means
            else []
        )
        self.raw_episode_mean_infos = (
            [{key: torch.tensor([value], device=self.device) for key, value in raw_episode_mean_means.items()}]
            if raw_episode_mean_means
            else []
        )
        self.episode_env_tensors.clear()
        if env_means:
            self.episode_env_tensors.add(
                {key: torch.tensor([value], device=self.device) for key, value in env_means.items()}
            )

        self._append_distributed_episode_batch(payloads)
        # Use the collection/learning split from the actual slowest rank.  The
        # previous max(collection)+max(learning) could combine two different
        # ranks and overstate iteration time.
        slowest_payload = max(
            payloads,
            key=lambda item: float(item["collection_time"]) + float(item["learn_time"]),
        )
        self.collection_time = float(slowest_payload["collection_time"])
        self.learn_time = float(slowest_payload["learn_time"])
        return merged_loss

    def post_epoch_logging(
        self,
        it: int,
        loss_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Handle post-epoch logging for training metrics.

        This method handles all logging operations after each training epoch, including:
        - Updating total timesteps and time
        - Logging episode information
        - Writing metrics to TensorBoard
        - Creating and displaying console output
        - Clearing episode information after logging

        Parameters
        ----------
        it : int
            Current iteration number
        loss_dict : dict[str, float]
            Dictionary containing loss values
        extra_log_dicts : dict[str, dict[str, float]]
            Dictionary containing extra metrics to log: {section_name: {metric_name: metric_value}}
        width : int, optional
            Width of the console output, by default 80
        pad : int, optional
            Padding for aligned console output, by default 35
        """
        self.tot_timesteps += self.num_steps_per_env * self.num_envs * self.num_gpus
        self.tot_time += self.collection_time + self.learn_time
        iteration_time = self.collection_time + self.learn_time

        # Log episode info
        ep_string, ep_scalars_to_log = self._log_episode_info()

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        fps = int(
            self.num_steps_per_env * self.num_envs * self.num_gpus / (self.collection_time + self.learn_time + 1e-8)
        )

        # Log to tensorboard
        self._logging_to_writer(
            it=it,
            loss_dict=loss_dict,
            extra_log_dicts=extra_log_dicts,
            env_log_dict=env_log_dict,
            fps=fps,
            ep_scalars_to_log=ep_scalars_to_log,
        )

        # Create console output
        log_string = self._create_console_output(
            it=it,
            loss_dict=loss_dict,
            env_log_dict=env_log_dict,
            extra_log_dicts=extra_log_dicts,
            ep_string=ep_string,
            width=width,
            pad=pad,
            iteration_time=iteration_time,
            fps=fps,
        )

        panel = Panel(log_string, title=self.title)
        force_live = os.environ.get("HOLOSOMA_FORCE_RICH_LIVE_LOGGING", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if force_live or console.is_terminal:
            with Live(panel, refresh_per_second=4, console=console):
                pass
        else:
            console.print(panel)

        # Clear episode infos after logging
        self.ep_infos.clear()
        self.raw_ep_infos.clear()
        self.episode_rate_infos.clear()
        self.raw_episode_mean_infos.clear()
        self.learn_time = 0.0
        self.collection_time = 0.0

    def _log_episode_info(self) -> tuple[str, dict[str, float]]:
        """Log episode information and return formatted string.

        Parameters
        ----------
        it : int
            Current iteration number

        Returns
        -------
        str
            Formatted string containing episode statistics
        """
        if not self.is_main_process:
            return "", {}
        ep_string = ""
        scalars_to_log: dict[str, float] = {}

        # Process regular episode info
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if len(infotensor) == 0:
                    continue
                value = torch.mean(infotensor).item()
                scalars_to_log[f"Episode/{key}"] = value
                ep_string += f"""{f"Mean episode {key}:":>35} {value:.4f}\n"""

        # Process raw episode info if it exists
        if self.raw_ep_infos:
            for key in self.raw_ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.raw_ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if len(infotensor) == 0:
                    continue
                value = torch.mean(infotensor).item()
                scalars_to_log[f"RawEpisode/{key}"] = value
                ep_string += f"""{f"Mean raw episode {key}:":>35} {value:.4f}\n"""

        # Actual-duration-normalized metrics are intended for dashboards and
        # comparisons. Keep them out of the already long console report.
        for infos, namespace in (
            (self.episode_rate_infos, "EpisodeRate"),
            (self.raw_episode_mean_infos, "RawEpisodeMean"),
        ):
            if not infos:
                continue
            for key in infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for info in infos:
                    value = info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.Tensor([value])
                    if len(value.shape) == 0:
                        value = value.unsqueeze(0)
                    infotensor = torch.cat((infotensor, value.to(self.device)))
                if len(infotensor) == 0:
                    continue
                scalars_to_log[f"{namespace}/{key}"] = torch.mean(infotensor).item()

        return ep_string, scalars_to_log

    def _logging_to_writer(
        self,
        it: int,
        loss_dict: dict[str, float],
        env_log_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        fps: int,
        ep_scalars_to_log: dict[str, float],
    ) -> None:
        """Log metrics to tensorboard writer.

        Parameters
        ----------
        it : int
            Current iteration number
        loss_dict : dict[str, float]
            Dictionary containing loss metrics
        env_log_dict : dict[str, float]
            Dictionary containing environment metrics
        extra_log_dicts : dict[str, float]
            Dictionary containing extra metrics to log: {section_name: {metric_name: metric_value}}
        fps : int
            Frames per second (training speed).
        ep_scalars_to_log : dict[str, float]
            Dictionary containing episode metrics to log.
        """
        if not self.is_main_process:
            return

        # Log loss metrics
        scalars_to_log: dict[str, float] = {}
        for loss_key, loss_value in loss_dict.items():
            scalars_to_log[f"Loss/{loss_key}"] = loss_value

        scalars_to_log.update(env_log_dict)
        scalars_to_log.update(ep_scalars_to_log)

        # Log extra metrics
        for section_name, section_dict in extra_log_dicts.items():
            for key, value in section_dict.items():
                scalars_to_log[f"{section_name}/{key}"] = value

        # Log performance metrics
        scalars_to_log["Perf/total_fps"] = fps
        scalars_to_log["Perf/collection_time"] = self.collection_time
        scalars_to_log["Perf/learning_time"] = self.learn_time

        # Log reward metrics if available
        if self._has_reward_statistics():
            mean_reward = self._mean_reward()
            scalars_to_log["Train/mean_reward"] = mean_reward
            scalars_to_log["Train/mean_reward/time"] = mean_reward
            scalars_to_log["Reward/mean"] = mean_reward
            mean_reward_per_alive_step = self._mean_reward_per_alive_step()
            if mean_reward_per_alive_step is not None:
                scalars_to_log["Train/mean_reward_per_alive_step"] = mean_reward_per_alive_step
                scalars_to_log["Reward/mean_per_alive_step"] = mean_reward_per_alive_step
        if self._has_length_statistics():
            mean_episode_length = self._mean_episode_length()
            scalars_to_log["Train/mean_episode_length"] = mean_episode_length
            scalars_to_log["Train/mean_episode_length/time"] = mean_episode_length
            scalars_to_log["Episode Length/mean"] = mean_episode_length

        scalars_to_log["Train/num_samples"] = self.tot_timesteps
        self._add_reward_group_aliases(scalars_to_log)

        # Add prefix to all keys
        scalars_to_log = {f"{self.prefix}{k}": v for k, v in scalars_to_log.items()}

        for k, v in scalars_to_log.items():
            self.writer.add_scalar(k, v, global_step=it)
        if wandb.run is not None:
            self._configure_wandb_metrics(scalars_to_log)
            wandb.log(dict(scalars_to_log, global_step=it), step=it)

    def _add_reward_group_aliases(self, scalars_to_log: dict[str, float]) -> None:
        """Add W&B-friendly reward panels from existing Episode/rew_* metrics."""
        for source_prefix, alias_prefix, total_name in (
            ("Episode/rew_", "Reward", "total_episode_terms"),
            ("EpisodeRate/rew_", "RewardRate", "total_episode_terms"),
        ):
            group_totals: dict[str, float] = {}
            reward_total = 0.0
            found_reward_terms = False

            for metric_name, value in list(scalars_to_log.items()):
                if not metric_name.startswith(source_prefix):
                    continue
                term_name = metric_name[len(source_prefix) :]
                group_name = get_reward_log_group(term_name)
                scalar_value = float(value)
                scalars_to_log[f"{alias_prefix}/{group_name}/{term_name}"] = scalar_value
                group_totals[group_name] = group_totals.get(group_name, 0.0) + scalar_value
                reward_total += scalar_value
                found_reward_terms = True

            if not found_reward_terms:
                continue

            for group_name in REWARD_LOG_GROUP_ORDER:
                if group_name in group_totals:
                    scalars_to_log[f"{alias_prefix}/{group_name}"] = group_totals[group_name]
            scalars_to_log[f"{alias_prefix}/{total_name}"] = reward_total

    def _strip_prefix(self, metric_name: str) -> str:
        if self.prefix and metric_name.startswith(self.prefix):
            return metric_name[len(self.prefix) :]
        return metric_name

    def _should_hide_wandb_metric(self, metric_name: str) -> bool:
        metric_name = self._strip_prefix(metric_name)
        if metric_name in self._WANDB_HIDDEN_METRIC_EXACT:
            return True
        if any(metric_name.startswith(prefix) for prefix in self._WANDB_HIDDEN_METRIC_PREFIXES):
            return True
        if any(metric_name.endswith(suffix) for suffix in self._WANDB_HIDDEN_METRIC_SUFFIXES):
            return True
        return False

    def _configure_wandb_metrics(self, scalars_to_log: dict[str, float]) -> None:
        if wandb.run is None:
            return
        if "global_step" not in self._wandb_defined_metrics:
            wandb.define_metric("global_step", hidden=True, summary="none")
            self._wandb_defined_metrics.add("global_step")
        for metric_name in scalars_to_log:
            if metric_name in self._wandb_defined_metrics:
                continue
            if self._should_hide_wandb_metric(metric_name):
                wandb.define_metric(metric_name, hidden=True, summary="last")
            elif self._strip_prefix(metric_name) in self._WANDB_METRIC_SUMMARIES:
                wandb.define_metric(
                    metric_name,
                    summary=self._WANDB_METRIC_SUMMARIES[
                        self._strip_prefix(metric_name)
                    ],
                )
            self._wandb_defined_metrics.add(metric_name)

    def _create_console_output(
        self,
        it: int,
        loss_dict: dict[str, float],
        env_log_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        ep_string: str,
        width: int,
        pad: int,
        iteration_time: float,
        fps: int,
    ) -> str:
        """Create formatted console output string.

        Parameters
        ----------
        it : int
            Current iteration number
        loss_dict : dict[str, float]
            Dictionary containing loss metrics
        env_log_dict : dict[str, float]
            Dictionary containing environment metrics
        extra_log_dicts : dict[str, dict[str, float]]
            Dictionary containing extra metrics to log: {section_name: {metric_name: metric_value}}
        ep_string : str
            Formatted string containing episode statistics
        width : int
            Width of the console output
        pad : int
            Padding for aligned console output
        iteration_time : float
            Time taken for the current iteration
        fps : int
            Frames per second (training speed).

        Returns
        -------
        str
            Formatted string for console output
        """
        if not self.is_main_process:
            return ""
        header = f" \033[1m Learning iteration {it}/{self.num_learning_iterations} \033[0m "

        # Base log string with computation info
        log_string = (
            f"""{header.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s """
            f"""(Collection: {self.collection_time:.3f}s, Learning {self.learn_time:.3f}s)\n"""
        )

        # Add training metrics if available
        if self._has_reward_statistics():
            mean_reward = self._mean_reward()
            log_string += f"""{"Mean reward:":>{pad}} {mean_reward:.2f}\n"""
        if self._has_length_statistics():
            mean_episode_length = self._mean_episode_length()
            log_string += f"""{"Mean episode length:":>{pad}} {mean_episode_length:.2f}\n"""

        # Add loss metrics
        for key, value in loss_dict.items():
            formatted_value = f"{value:.3e}" if key.endswith("learning_rate") else f"{value:.4f}"
            log_string += f"{f'{key}:':>{pad}} {formatted_value}\n"

        # Add environment metrics
        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string

        # Add extra metrics
        for section_name, section_dict in extra_log_dicts.items():
            for key, value in section_dict.items():
                log_string += f"{f'{section_name}/{key}:':>{pad}} {value:.4f}\n"

        # Add episode info
        log_string += ep_string

        eta = self.tot_time / (it + 1) * (self.num_learning_iterations - it)

        # Add timing info
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {eta:.1f}s\n"""
        )
        log_string += f"Logging Directory: {self.log_dir}"

        return log_string

    def save_checkpoint_artifact(
        self,
        state_dict: dict[str, Any],
        path: str,
        *,
        upload: bool = True,
    ) -> None:
        # Resolve the parent (rather than the final component) so an existing
        # symlink at the checkpoint name is atomically replaced instead of
        # followed.  pathlib/commonpath semantics also avoid the old string
        # prefix bug where e.g. ``/logs-other`` passed a ``/logs`` check.
        log_root = pathlib.Path(self.log_dir).expanduser().resolve()
        requested_path = pathlib.Path(path).expanduser()
        if not requested_path.is_absolute():
            requested_path = pathlib.Path.cwd() / requested_path
        target_parent = requested_path.parent.resolve()
        target_path = target_parent / requested_path.name
        try:
            target_path.relative_to(log_root)
        except ValueError as exc:
            raise ValueError(f"Path {path} is not in the logging directory {self.log_dir}") from exc

        target_parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving checkpoint atomically to {}", target_path)
        temp_fd, temp_path_raw = tempfile.mkstemp(
            dir=target_parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
        )
        os.close(temp_fd)
        temp_path = pathlib.Path(temp_path_raw)
        try:
            # torch.save() closes the file it opens, but explicitly fsync the
            # completed archive before publication.  os.replace() is atomic
            # because the temporary file lives in the target directory.
            torch.save(state_dict, temp_path)
            if temp_path.stat().st_size <= 0:
                raise RuntimeError(f"Checkpoint serialization produced an empty file: {temp_path}")
            serialized_fd = os.open(temp_path, os.O_RDONLY)
            try:
                os.fsync(serialized_fd)
            finally:
                os.close(serialized_fd)

            os.replace(temp_path, target_path)
            directory_fd = os.open(target_parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            # If serialization or publication fails, leave any previously
            # published checkpoint untouched and remove only our private temp.
            temp_path.unlink(missing_ok=True)

        if upload:
            self.save_to_wandb(str(target_path))

    def save_to_wandb(self, file_path: str) -> None:
        """Saves file to wandb if run is initialized."""
        if wandb.run is None:
            return
        if os.environ.get("HOLOSOMA_SKIP_WANDB_FILE_UPLOAD", "").lower() in ("1", "true", "yes", "on"):
            logger.info("Skipping wandb file upload for {} due to HOLOSOMA_SKIP_WANDB_FILE_UPLOAD.", file_path)
            return
        if (
            os.environ.get("HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD", "").lower() in ("1", "true", "yes", "on")
            and pathlib.Path(file_path).name.startswith("model_")
            and pathlib.Path(file_path).suffix == ".pt"
        ):
            logger.info(
                "Skipping wandb checkpoint upload for {} due to HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD.",
                file_path,
            )
            return
        wandb.save(file_path, base_path=self.log_dir)
