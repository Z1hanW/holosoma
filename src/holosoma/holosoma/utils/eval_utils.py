from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

import yaml
from loguru import logger
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from tqdm import tqdm

# CONFIG_NAME is "holosoma_config.yaml" - the primary configuration file for Holosoma
# This file contains all settings for training and evaluation of models
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.logging import LoguruLoggingBridge
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type

_WANDB_PREFIX = "wandb://"
_WANDB_REFERENCE_FORMAT = f"{_WANDB_PREFIX}<entity>/<project>/<run_id>/[<artifact_name>]"


def _parse_wandb_reference(reference: str) -> tuple[str, str | None]:
    """Split a wandb:// URI into run path and optional artifact/checkpoint name."""

    if not reference.startswith(_WANDB_PREFIX):
        raise ValueError(f"Invalid wandb URI: {reference}. Expected format {_WANDB_REFERENCE_FORMAT}")
    remainder = reference[len(_WANDB_PREFIX) :]
    parts = remainder.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid wandb URI: {reference}. Expected format {_WANDB_REFERENCE_FORMAT}")
    entity, project = parts[0], parts[1]
    run_id_index = 2
    if len(parts) > 3 and parts[2] == "runs":
        run_id_index = 3
    if run_id_index >= len(parts):
        raise ValueError(f"Invalid wandb URI: {reference}. Expected format {_WANDB_REFERENCE_FORMAT}")
    identity_parts = [parts[0], parts[1], parts[run_id_index]]
    if any(part in {"", ".", ".."} for part in identity_parts):
        raise ValueError(f"Invalid wandb URI: {reference}. Expected format {_WANDB_REFERENCE_FORMAT}")
    run_id = parts[run_id_index]
    artifact_start = run_id_index + 1
    wandb_run_path = f"{entity}/{project}/{run_id}"
    artifact_parts = parts[artifact_start:]
    if any(part in {"", ".", ".."} for part in artifact_parts):
        raise ValueError(f"Invalid wandb URI: {reference}. Expected format {_WANDB_REFERENCE_FORMAT}")
    artifact_path = "/".join(artifact_parts) or None
    return wandb_run_path, artifact_path


def _validated_relative_wandb_name(file_name: str) -> Path:
    """Validate that a W&B run-file name is a safe, unambiguous relative path."""

    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"Invalid W&B file name: {file_name!r}")
    parts = file_name.split("/")
    path = Path(file_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid W&B file name: {file_name!r}")
    return path


def _download_wandb_file_exact(run, file_name: str, root: str | Path) -> Path:
    """Download one exact W&B run file and verify the SDK's returned path.

    W&B's public API returns an open file handle whose ``name`` is the path it
    actually wrote.  Passing an absolute root and checking both the remote name
    and returned path prevents a stale or differently named local file from
    being accepted as the requested checkpoint.
    """

    relative_name = _validated_relative_wandb_name(file_name)
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    expected_path = (root_path / relative_name).resolve()
    try:
        expected_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"W&B file path escapes download root: {file_name!r}") from exc

    remote_file = run.file(file_name)
    if remote_file is None:
        raise FileNotFoundError(f"W&B run does not contain requested file: {file_name}")
    remote_name = getattr(remote_file, "name", None)
    if remote_name != file_name:
        raise RuntimeError(
            f"W&B returned a different run file than requested: requested={file_name!r}, returned={remote_name!r}"
        )

    downloaded = remote_file.download(root=str(root_path), replace=True)
    try:
        downloaded_name = getattr(downloaded, "name", None)
        if not isinstance(downloaded_name, (str, os.PathLike)):
            raise RuntimeError(f"W&B download returned no usable file path for {file_name!r}")
        downloaded_path = Path(downloaded_name).expanduser()
        if not downloaded_path.is_absolute():
            downloaded_path = root_path / downloaded_path
        downloaded_path = downloaded_path.resolve()
    finally:
        close = getattr(downloaded, "close", None)
        if callable(close):
            close()

    if downloaded_path != expected_path:
        raise RuntimeError(
            "W&B downloaded a different path than requested: "
            f"requested={expected_path}, returned={downloaded_path}"
        )
    if not downloaded_path.is_file():
        raise FileNotFoundError(f"W&B reported a download that does not exist as a file: {downloaded_path}")
    return downloaded_path


def init_eval_logging() -> None:
    logger.remove()

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    py_log_level = os.environ.get("PY_LOG_LEVEL", console_log_level).upper()
    level = logging._nameToLevel.get(py_log_level, logging.INFO)
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
    logging.getLogger().addHandler(LoguruLoggingBridge())
    if level > logging.DEBUG:
        for name in ("websockets", "websockets.server", "trimesh"):
            logging.getLogger(name).setLevel(level)


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint: str | None = None
    """Path to a local checkpoint file, or W&B URI in the format `wandb://<entity>/<project>/<run_id>[/<checkpoint_name>]`."""


def load_saved_experiment_config(checkpoint_cfg: CheckpointConfig) -> tuple[ExperimentConfig, str | None]:
    """Load checkpoint configuration from either W&B run or local checkpoint.

    Returns
    -------
    (ExperimentConfig, str | None)
        Loaded experiment config and the originating wandb run path, if available.
    """

    checkpoint = checkpoint_cfg.checkpoint

    if checkpoint is None:
        raise ValueError("No checkpoint provided")

    checkpoint_str = str(checkpoint)
    if not checkpoint_str.startswith(_WANDB_PREFIX):
        checkpoint_path = Path(checkpoint_str).expanduser()
        config, stored_wandb_path = _load_config_from_checkpoint(checkpoint_path)
        if stored_wandb_path:
            logger.info(f"Checkpoint originated from W&B run: {stored_wandb_path}")
        logger.info(f"Loaded experiment config from checkpoint: {checkpoint_path}")
        return config, stored_wandb_path

    wandb_run_path, artifact_path = _parse_wandb_reference(checkpoint_str)

    # Lazy import W&B to avoid conflicts with site-packages Python and Isaac
    # for the entirely local checkpoint path above.
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)
    if artifact_path:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = _download_wandb_file_exact(run, artifact_path, temp_dir)
                config, stored_wandb_path = _load_config_from_checkpoint(checkpoint_path)
                effective_wandb_path = stored_wandb_path or wandb_run_path
                logger.info(f"Loaded experiment config from W&B checkpoint payload: {checkpoint_str}")
                return config, effective_wandb_path
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load experiment config from exact W&B checkpoint payload: {checkpoint_str}"
            ) from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = _download_wandb_file_exact(run, CONFIG_NAME, temp_dir)
        with config_path.open(encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
    if not isinstance(config_data, Mapping):
        raise ValueError(f"W&B {CONFIG_NAME} must contain a mapping: {wandb_run_path}")
    return ExperimentConfig(**dict(config_data)), wandb_run_path


def _load_config_from_checkpoint(checkpoint_path: Path) -> tuple[ExperimentConfig, str | None]:
    """Attempt to load the serialized ExperimentConfig from a checkpoint file."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist as a file: {checkpoint_path}")
    checkpoint_contents, _ = load_verified_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    if not isinstance(checkpoint_contents, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    config_data = checkpoint_contents.get("experiment_config")
    if not isinstance(config_data, Mapping):
        raise ValueError(f"Checkpoint is missing a mapping experiment_config: {checkpoint_path}")
    # Work on a regular mutable copy. Some older serialized scene configs used
    # null for list fields that now have strict list types.
    config_data = dict(config_data)
    algo_wrapper = config_data.get("algo")
    if isinstance(algo_wrapper, dict):
        algo_config = algo_wrapper.get("config")
        if isinstance(algo_config, dict) and "actor_obs_keys" in algo_config:
            # FastSAC checkpoints written before the action transform was
            # versioned used the scalar/max-range implementation. Preserve
            # that exact policy function for evaluation instead of silently
            # applying the corrected affine mapping default.
            algo_config.setdefault(
                "action_boundary_mode",
                "legacy_max_range_scalar_v1",
            )
    simulator_cfg = config_data.get("simulator")
    if isinstance(simulator_cfg, dict):
        sim_cfg = simulator_cfg.get("config")
        if isinstance(sim_cfg, dict):
            scene_cfg = sim_cfg.get("scene")
            if isinstance(scene_cfg, dict):
                if scene_cfg.get("scene_files") is None:
                    scene_cfg["scene_files"] = []
                if scene_cfg.get("rigid_objects") is None:
                    scene_cfg["rigid_objects"] = []
    stored_wandb_path = checkpoint_contents.get("wandb_run_path")
    if stored_wandb_path is not None and not isinstance(stored_wandb_path, str):
        raise ValueError(f"Checkpoint wandb_run_path must be a string or null: {checkpoint_path}")
    return ExperimentConfig(**config_data), stored_wandb_path


class CheckpointMetadata(TypedDict):
    file_name: str
    """Name of the checkpoint file."""

    global_step: int
    """Global step of the checkpoint."""

    train_runtime: float | None
    """Number of seconds that have elapsed since the start of training."""

    num_samples: int | None
    """Number of training samples that have been collected up to the checkpoint."""


def get_all_checkpoint_metadata(override_config: DictConfig) -> list[CheckpointMetadata]:
    """Get all checkpoint names and their global steps from either W&B run or local directory.

    Parameters
    ----------
    override_config : DictConfig
        Configuration object containing:
        - wandb_run_path: str | None
            Path to the W&B run (e.g., 'username/project/run_id'). If None, checkpoint_dir must be provided.
        - checkpoint_dir: str | None
            Path to local directory containing checkpoints. If None, wandb_run_path must be provided.
        - checkpoint_names: list[str] | None
            List of checkpoint names to evaluate. If None, all checkpoints will be evaluated.

    Returns
    -------
    list[CheckpointMetadata]
        List of checkpoint metadata.

    Raises
    ------
    ValueError
        If neither wandb_run_path nor checkpoint_dir is provided.
    """
    import wandb

    def extract_global_step(filename: str) -> int | None:
        """Extract global step from checkpoint filename."""
        match = re.match(r"model_(\d+)\.pt", filename)
        if match:
            return int(match.group(1))
        return None

    checkpoint_metadata: list[CheckpointMetadata]
    if override_config.get("wandb_run_path", None) is not None:
        api = wandb.Api()
        run = api.run(override_config.wandb_run_path)
        # Get all files in the run
        files = run.files()
        # Filter for checkpoint files (assuming they end with .pt)
        checkpoint_names = [f.name for f in files if f.name.endswith(".pt") and extract_global_step(f.name) is not None]
        runtimes: dict[int, float] = {}
        num_samples: dict[int, int] = {}
        logger.info("Scanning W&B history to extract runtime data...")
        for hist in tqdm(run.scan_history(keys=["_runtime", "global_step", "Train/num_samples"])):
            hist_global_step = hist["global_step"]
            hist_runtime = hist["_runtime"]
            hist_num_samples = hist["Train/num_samples"]
            if hist_global_step is not None and hist_runtime is not None:
                runtimes[hist_global_step] = min(runtimes.get(hist_global_step, float("inf")), hist_runtime)
            if hist_global_step is not None and hist_num_samples is not None:
                num_samples[hist_global_step] = hist_num_samples
        checkpoint_metadata = []
        for checkpoint_name in checkpoint_names:
            checkpoint_global_step = extract_global_step(checkpoint_name)
            assert checkpoint_global_step is not None
            if checkpoint_global_step not in runtimes:
                logger.warning(
                    f"Checkpoint {checkpoint_name} and the corresponding global step {checkpoint_global_step} "
                    "has no _runtime data in W&B. Setting train_runtime to 0."
                )
            if checkpoint_global_step not in num_samples:
                logger.warning(
                    f"Checkpoint {checkpoint_name} and the corresponding global step {checkpoint_global_step} "
                    "has no Train/num_samples data in W&B. Setting num_samples to 0."
                )
            checkpoint_metadata.append(
                {
                    "file_name": checkpoint_name,
                    "global_step": checkpoint_global_step,
                    "train_runtime": runtimes.get(checkpoint_global_step, 0.0),
                    "num_samples": num_samples.get(checkpoint_global_step, 0),
                }
            )
    elif override_config.get("checkpoint_dir", None) is not None:
        checkpoint_dir = Path(override_config.checkpoint_dir)
        # Get all checkpoint files in the directory
        checkpoint_names = [f.name for f in checkpoint_dir.glob("*.pt") if extract_global_step(f.name) is not None]
        checkpoint_metadata = [
            {
                "file_name": checkpoint_name,
                "global_step": cast("int", extract_global_step(checkpoint_name)),
                "train_runtime": None,
                "num_samples": None,
            }
            for checkpoint_name in checkpoint_names
        ]
    else:
        raise ValueError("No checkpoint directory or wandb run path provided")

    if override_config.get("checkpoint_names", None) is not None:
        checkpoint_metadata = [
            metadata for metadata in checkpoint_metadata if metadata["file_name"] in override_config.checkpoint_names
        ]

    return sorted(checkpoint_metadata, key=lambda x: x["global_step"])


def load_checkpoint(checkpoint: str, log_dir: str) -> Path:
    """Download checkpoint from W&B or use local checkpoint.

    Parameters
    ----------
    checkpoint : str
        W&B checkpoint URI or path to local checkpoint file.
    log_dir : str
        Directory to save downloaded checkpoint.

    Returns
    -------
    Path
        Path to the downloaded or local checkpoint file.
    """

    wandb_run_path: str | None = None
    if checkpoint.startswith(_WANDB_PREFIX):
        wandb_run_path_from_uri, wandb_checkpoint_name = _parse_wandb_reference(checkpoint)
        if wandb_checkpoint_name is None:
            raise ValueError(
                f"Invalid wandb checkpoint path: {checkpoint}. "
                f"Expected format: {_WANDB_PREFIX}<entity>/<project>/<run_id>/<checkpoint_name>"
            )
        wandb_run_path = wandb_run_path_from_uri
        checkpoint = wandb_checkpoint_name

    if wandb_run_path is not None:
        import wandb

        api = wandb.Api()
        run = api.run(wandb_run_path)
        # Stage under the destination filesystem and publish atomically. A
        # failed/retried network download can therefore never be mistaken for
        # the exact requested checkpoint on a later run.
        log_dir_path = Path(log_dir).expanduser().resolve()
        log_dir_path.mkdir(parents=True, exist_ok=True)
        relative_checkpoint = _validated_relative_wandb_name(checkpoint)
        checkpoint_path = (log_dir_path / relative_checkpoint).resolve()
        try:
            checkpoint_path.relative_to(log_dir_path)
        except ValueError as exc:
            raise ValueError(f"Checkpoint destination escapes log directory: {checkpoint!r}") from exc
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".wandb-download-", dir=log_dir_path) as temp_dir:
            downloaded_path = _download_wandb_file_exact(run, checkpoint, temp_dir)
            os.replace(downloaded_path, checkpoint_path)
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Downloaded checkpoint is missing or empty: {checkpoint_path}")
        logger.info(f"Finished downloading checkpoint {checkpoint} to {checkpoint_path} from W&B run {wandb_run_path}")
    else:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Local checkpoint does not exist as a file: {checkpoint_path}")
    return checkpoint_path


def init_sim_imports(tyro_config: ExperimentConfig):
    """Initialize simulator imports - DEPRECATED.

    This function is deprecated in favor of the more focused functions in sim_utils.py.
    Use setup_simulation_environment() for new code.

    Parameters
    ----------
    tyro_config : ExperimentConfig
        Configuration containing simulator settings.

    Returns
    -------
    Any | None
        Simulation app instance for IsaacSim, None for other simulators.
    """
    from holosoma.utils.sim_utils import setup_isaaclab_launcher, setup_simulator_imports

    # Use the new focused functions
    setup_simulator_imports(tyro_config)

    simulator_type = get_simulator_type()
    if simulator_type == SimulatorType.ISAACSIM:
        return setup_isaaclab_launcher(tyro_config)

    # For other simulators, no app is needed
    return None
