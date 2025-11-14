from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TypedDict, cast

import yaml
from loguru import logger
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from tqdm import tqdm

# CONFIG_NAME is "hv_config.yaml" - the primary configuration file for HumanoidVerse
# This file contains all settings for training and evaluation of models
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.logging import HydraLoggerBridge
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type

_WANDB_PREFIX = "wandb://"


def init_eval_logging() -> None:
    logger.remove()

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())


@dataclass(frozen=True)
class CheckpointConfig:
    wandb_run_path: str | None = None
    """Path to the W&B run (e.g., 'username/project/run_id'). If None, checkpoint must be provided."""

    checkpoint: str | None = None
    """Path to local checkpoint file, or W&B checkpoint path in the format of `wandb://<entity>/<project>/<run_id>/<checkpoint_name>`."""


def load_saved_experiment_config(checkpoint_cfg: CheckpointConfig) -> ExperimentConfig | None:
    """Load checkpoint configuration from either W&B run or local checkpoint.

    Raises
    ------
    ValueError
        If neither wandb_run_path nor checkpoint is provided.
    """

    # lazy import wandb to avoid conflicts with site-packages python and Isaac
    import wandb

    checkpoint = checkpoint_cfg.checkpoint
    wandb_run_path = checkpoint_cfg.wandb_run_path

    if checkpoint is not None:
        if checkpoint.startswith(_WANDB_PREFIX):
            wandb_entity, wandb_project, wandb_run_id, checkpoint = checkpoint[len(_WANDB_PREFIX) :].split("/", 3)
            wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
            checkpoint = None

    if wandb_run_path is not None:
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # Get the config file (hv_config.yaml) which contains all model and environment settings
        config_file = run.file(CONFIG_NAME)  # Get the config file by CONFIG_NAME (hv_config.yaml)
        with tempfile.TemporaryDirectory() as temp_dir, config_file.download(root=temp_dir) as file:
            config = ExperimentConfig(**yaml.safe_load(file))
    elif checkpoint is not None:
        config_path = Path(checkpoint).parent / CONFIG_NAME
        if config_path.exists():
            with open(config_path) as file:
                config = ExperimentConfig(**yaml.safe_load(file))
        else:
            config = None
    else:
        raise ValueError("No checkpoint or wandb run path provided")

    return config


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


def load_checkpoint(
    wandb_run_path: str | None,
    checkpoint: str,
    log_dir: str,
) -> Path:
    """Download checkpoint from W&B or use local checkpoint.

    Parameters
    ----------
    wandb_run_path : str | None
        Path to the W&B run (e.g., 'username/project/run_id'). If None, checkpoint must be provided.
    checkpoint : str
        Name of checkpoint file in W&B run or path to local checkpoint file.
    log_dir : str
        Directory to save downloaded checkpoint.

    Returns
    -------
    Path
        Path to the downloaded or local checkpoint file.
    """

    import wandb

    if checkpoint.startswith(_WANDB_PREFIX):
        try:
            wandb_entity, wandb_project, wandb_run_id, checkpoint = checkpoint[len(_WANDB_PREFIX) :].split("/", 3)
        except ValueError:
            raise ValueError(
                f"Invalid wandb checkpoint path: {checkpoint}. "
                f"Expected format: {_WANDB_PREFIX}<entity>/<project>/<run_id>/<checkpoint_name>"
            )
        wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"

    if wandb_run_path is not None:
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # Create log dir
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        # Download checkpoint to log_dir
        checkpoint_file = run.file(checkpoint)  # Get the specific checkpoint file
        checkpoint_file.download(root=log_dir)
        logger.info(f"Finished downloading checkpoint {checkpoint} to {log_dir} from W&B run {wandb_run_path}")
        checkpoint_path = log_dir_path / checkpoint
    else:
        checkpoint_path = Path(checkpoint)
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
