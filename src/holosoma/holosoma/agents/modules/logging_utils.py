from __future__ import annotations

import os
import pathlib
import statistics
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Generator, TypedDict

import torch
import wandb
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from torch.utils.tensorboard import SummaryWriter

from holosoma.utils.average_meters import TensorAverageMeterDict

console = Console()

REWARD_LOG_GROUPS: dict[str, tuple[str, ...]] = {
    "Track": (
        "teacher_rollout_global_ref_position_error_exp",
        "teacher_rollout_global_ref_orientation_error_exp",
        "teacher_rollout_relative_body_position_error_exp",
        "teacher_rollout_relative_body_orientation_error_exp",
        "teacher_rollout_global_body_lin_vel",
        "teacher_rollout_global_body_ang_vel",
    ),
    "Object": (
        "teacher_rollout_object_global_ref_position_error_exp",
        "teacher_rollout_object_global_ref_orientation_error_exp",
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
    return {"reward_group_spec": grouped_spec}, summary


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
    }
    _WANDB_HIDDEN_METRIC_PREFIXES = (
        "Perf/",
    )
    _WANDB_HIDDEN_METRIC_SUFFIXES = (
        "/time",
    )

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
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
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
        if not self.is_main_process:
            return
        self.ep_infos.append(infos["episode"])
        # Also process raw episode data if it exists
        if "raw_episode" in infos:
            self.raw_ep_infos.append(infos["raw_episode"])
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

        # Update episode environment tensors
        self.episode_env_tensors.add(infos["to_log"])

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

        # Use rich Live to update console
        with Live(Panel(log_string, title=self.title), refresh_per_second=4, console=console):
            pass

        # Clear episode infos after logging
        self.ep_infos.clear()
        self.raw_ep_infos.clear()
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
        if len(self.rewbuffer) > 0:
            mean_reward = statistics.mean(self.rewbuffer)
            scalars_to_log["Train/mean_reward"] = mean_reward
            scalars_to_log["Train/mean_reward/time"] = mean_reward
            scalars_to_log["Reward/mean"] = mean_reward
        if len(self.lenbuffer) > 0:
            mean_episode_length = statistics.mean(self.lenbuffer)
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
        group_totals: dict[str, float] = {}
        reward_total = 0.0
        found_reward_terms = False

        for metric_name, value in list(scalars_to_log.items()):
            if not metric_name.startswith("Episode/rew_"):
                continue
            term_name = metric_name[len("Episode/rew_") :]
            group_name = get_reward_log_group(term_name)
            scalar_value = float(value)
            scalars_to_log[f"Reward/{group_name}/{term_name}"] = scalar_value
            group_totals[group_name] = group_totals.get(group_name, 0.0) + scalar_value
            reward_total += scalar_value
            found_reward_terms = True

        if not found_reward_terms:
            return

        for group_name in REWARD_LOG_GROUP_ORDER:
            if group_name in group_totals:
                scalars_to_log[f"Reward/{group_name}"] = group_totals[group_name]
        scalars_to_log["Reward/total_episode_terms"] = reward_total

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
            elif metric_name == f"{self.prefix}Eval/fixed_bc_mu_mse":
                wandb.define_metric(metric_name, summary="min")
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
        if len(self.rewbuffer) > 0:
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
        if len(self.lenbuffer) > 0:
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

        # Add loss metrics
        for key, value in loss_dict.items():
            log_string += f"{f'{key}:':>{pad}} {value:.4f}\n"

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

    def save_checkpoint_artifact(self, state_dict: dict[str, Any], path: str) -> None:
        if not path.startswith(self.log_dir):
            raise ValueError(f"Path {path} is not in the logging directory {self.log_dir}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving checkpoint to {path}")
        torch.save(state_dict, path)
        self.save_to_wandb(path)

    def save_to_wandb(self, file_path: str) -> None:
        """Saves file to wandb if run is initialized."""
        if wandb.run is None:
            return
        wandb.save(file_path, base_path=self.log_dir)
