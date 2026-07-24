"""Utility functions for computing experiment directory paths."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.experiment import TrainingConfig
    from holosoma.config_types.logger import LoggerConfig


EXPERIMENT_TIMESTAMP_ENV = "HOLOSOMA_EXPERIMENT_TIMESTAMP"
EXPERIMENT_DIR_ENV = "HOLOSOMA_EXPERIMENT_DIR"
EXPERIMENT_TASK_ENV = "HOLOSOMA_EXPERIMENT_TASK_NAME"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{8}_[0-9]{6}$")


def get_timestamp() -> str:
    """Get current timestamp in experiment format."""
    return datetime.now(tz=timezone.utc).strftime(_TIMESTAMP_FORMAT)


def validate_experiment_timestamp(timestamp: str) -> str:
    """Validate the canonical UTC experiment timestamp representation."""

    if not isinstance(timestamp, str) or not _TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValueError(
            "Experiment timestamp must use canonical UTC format YYYYMMDD_HHMMSS, "
            f"got {timestamp!r}."
        )
    try:
        datetime.strptime(timestamp, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"Experiment timestamp is not a valid UTC date/time: {timestamp!r}.") from exc
    return timestamp


def _normalized_experiment_dir(path: str | Path) -> Path:
    raw_path = str(path)
    if not raw_path.strip():
        raise ValueError("Experiment directory must not be empty.")
    return Path(raw_path).expanduser().resolve(strict=False)


def get_experiment_dir(
    logger_config: LoggerConfig,
    training_config: TrainingConfig,
    timestamp: str | None = None,
    task_name: str = "locomotion",
) -> Path:
    """Compute experiment directory from logger and training config.

    Parameters
    ----------
    logger_config : LoggerConfig
        Logger configuration (WandbLoggerConfig or DisabledLoggerConfig)
    training_config : TrainingConfig
        Training configuration with project/name
    timestamp : str | None
        Timestamp string. If None, generates a new one.
    task_name : str
        Task name for the experiment (e.g., "locomotion", "manipulation")

    Returns
    -------
    Path
        Experiment directory path

    Examples
    --------
    >>> exp_dir = get_experiment_dir(logger_cfg, training_cfg, "20250115_143022", "locomotion")
    >>> # Result: logs/my_project/20250115_143022-my_run-locomotion
    """
    if timestamp is None:
        timestamp = get_timestamp()
    timestamp = validate_experiment_timestamp(timestamp)

    base_dir = Path(logger_config.base_dir)

    # Fallback chain: training config → logger config → default
    project = training_config.project or getattr(logger_config, "project", None) or "default_project"
    name = training_config.name or getattr(logger_config, "name", None) or "run"

    # Build structured path if we have any project/name info
    if project or name:
        group = getattr(logger_config, "group", None)
        exp_name = f"{timestamp}-{name}-{group or task_name}"
        return base_dir / project / exp_name

    # Fallback to simple structure
    return base_dir / "runs" / timestamp


def set_experiment_dir_override(
    logger_config: LoggerConfig,
    training_config: TrainingConfig,
    *,
    timestamp: str,
    experiment_dir: str | Path,
    task_name: str,
) -> Path:
    """Install one validated process-wide experiment identity.

    ``BaseTask`` is constructed after the training entry point has selected its
    log directory.  Publishing both components explicitly prevents it (and any
    simulator-side video paths) from silently generating a later timestamp.
    """

    timestamp = validate_experiment_timestamp(timestamp)
    expected_dir = _normalized_experiment_dir(
        get_experiment_dir(logger_config, training_config, timestamp, task_name=task_name)
    )
    resolved_dir = _normalized_experiment_dir(experiment_dir)
    if resolved_dir != expected_dir:
        raise ValueError(
            "Experiment directory does not match its timestamp/config identity: "
            f"expected={expected_dir}, received={resolved_dir}."
        )
    os.environ[EXPERIMENT_TIMESTAMP_ENV] = timestamp
    os.environ[EXPERIMENT_DIR_ENV] = str(resolved_dir)
    os.environ[EXPERIMENT_TASK_ENV] = task_name
    return resolved_dir


def get_process_experiment_dir(
    logger_config: LoggerConfig,
    training_config: TrainingConfig,
    *,
    task_name: str,
    require_override: bool = False,
    use_override_task_name: bool = False,
) -> Path:
    """Resolve the shared experiment directory or fail on a partial/stale override."""

    timestamp_override = os.environ.get(EXPERIMENT_TIMESTAMP_ENV)
    dir_override = os.environ.get(EXPERIMENT_DIR_ENV)
    task_override = os.environ.get(EXPERIMENT_TASK_ENV)
    override_values = (timestamp_override, dir_override, task_override)
    present_override_count = sum(value is not None for value in override_values)
    if present_override_count not in (0, len(override_values)):
        raise RuntimeError(
            f"{EXPERIMENT_TIMESTAMP_ENV}, {EXPERIMENT_DIR_ENV}, and {EXPERIMENT_TASK_ENV} "
            "must be set together; "
            "a partial experiment identity could split one run across directories."
        )
    if timestamp_override is None:
        if require_override:
            raise RuntimeError(
                "A synchronized experiment identity is required, but neither "
                f"{EXPERIMENT_TIMESTAMP_ENV}, {EXPERIMENT_DIR_ENV}, nor {EXPERIMENT_TASK_ENV} is set."
            )
        timestamp = get_timestamp()
        return get_experiment_dir(logger_config, training_config, timestamp, task_name=task_name)

    assert dir_override is not None
    assert task_override is not None
    if not task_override.strip():
        raise RuntimeError(f"{EXPERIMENT_TASK_ENV} must not be empty.")
    if not use_override_task_name and task_override != task_name:
        raise RuntimeError(
            "Process experiment task override is inconsistent with the caller: "
            f"caller={task_name!r}, override={task_override!r}."
        )
    effective_task_name = task_override if use_override_task_name else task_name
    timestamp = validate_experiment_timestamp(timestamp_override)
    expected_dir = _normalized_experiment_dir(
        get_experiment_dir(logger_config, training_config, timestamp, task_name=effective_task_name)
    )
    resolved_dir = _normalized_experiment_dir(dir_override)
    if resolved_dir != expected_dir:
        raise RuntimeError(
            "Process experiment-directory override is inconsistent with the active config: "
            f"expected={expected_dir}, override={resolved_dir}."
        )
    return resolved_dir


def get_output_dir(experiment_dir: Path) -> Path:
    """Get output directory from experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory path

    Returns
    -------
    Path
        Output directory path (experiment_dir/output)
    """
    return experiment_dir / "output"


def get_video_dir(experiment_dir: Path) -> Path:
    """Get video directory from experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory path

    Returns
    -------
    Path
        Video directory path (experiment_dir/renderings_training)
    """
    return experiment_dir / "renderings_training"


def get_eval_log_dir(
    logger_config: LoggerConfig, training_config: TrainingConfig, eval_timestamp: str | None = None
) -> Path:
    """Compute evaluation log directory from logger and training config.

    Parameters
    ----------
    logger_config : LoggerConfig
        Logger configuration
    training_config : TrainingConfig
        Training configuration with project name
    eval_timestamp : str | None
        Evaluation timestamp. If None, generates a new one.

    Returns
    -------
    Path
        Evaluation log directory path
    """
    if eval_timestamp is None:
        eval_timestamp = get_timestamp()

    base_dir = Path(logger_config.base_dir).parent / "logs_eval"

    # Use training config for project, with fallback to logger config
    project: str | None = training_config.project
    if not project and hasattr(logger_config, "project"):
        project = logger_config.project

    if project:
        return base_dir / project / eval_timestamp
    return base_dir / eval_timestamp
