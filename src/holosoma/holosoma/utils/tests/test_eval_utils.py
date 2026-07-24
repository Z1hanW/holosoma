from __future__ import annotations

import dataclasses
import pickle
import subprocess
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
)
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    get_all_checkpoint_metadata,
    load_checkpoint,
    load_saved_experiment_config,
)


class _UnsafeCheckpointConfigPayload:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return subprocess.call, (["touch", self.marker],)


@pytest.fixture
def mock_wandb_run() -> mock.MagicMock:
    """Create a mock wandb run object."""
    return mock.MagicMock()


@pytest.fixture
def mock_wandb_api(mock_wandb_run: mock.MagicMock) -> mock.MagicMock:
    """Create a mock wandb API object."""
    mock_api = mock.MagicMock()
    mock_api.run.return_value = mock_wandb_run
    return mock_api


def test_get_all_checkpoint_metadata_from_wandb(mock_wandb_api: mock.MagicMock) -> None:
    """Test getting checkpoint metadata from wandb run."""
    # Create mock files with proper name attribute
    mock_files = []
    for name in ["model_100.pt", "model_2.pt", "config.yaml", "model_10.pt", "invalid.pt"]:
        mock_file = mock.MagicMock()
        mock_file.name = name
        mock_files.append(mock_file)

    # Set up the mock to return the files
    mock_wandb_api.run.return_value.files.return_value = mock_files

    # Mock the scan_history to return runtime data
    mock_history = [
        {"global_step": 2, "_runtime": 100.0, "Train/num_samples": 1000},
        {"global_step": 10, "_runtime": 200.0, "Train/num_samples": 5000},
        {"global_step": 100, "_runtime": 300.0, "Train/num_samples": 50000},
    ]
    mock_wandb_api.run.return_value.scan_history.return_value = mock_history

    # Create override config
    override_config = OmegaConf.create(
        {
            "wandb_run_path": "test_user/test_project/test_run",
            "checkpoint_dir": None,
        }
    )

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_metadata = get_all_checkpoint_metadata(override_config)

    # Verify the checkpoint metadata is in order
    expected_metadata = [
        {"file_name": "model_2.pt", "global_step": 2, "train_runtime": 100.0, "num_samples": 1000},
        {"file_name": "model_10.pt", "global_step": 10, "train_runtime": 200.0, "num_samples": 5000},
        {"file_name": "model_100.pt", "global_step": 100, "train_runtime": 300.0, "num_samples": 50000},
    ]
    assert checkpoint_metadata == expected_metadata
    mock_wandb_api.run.assert_called_once_with("test_user/test_project/test_run")


def test_get_all_checkpoint_metadata_from_local(tmp_path: Path) -> None:
    """Test getting checkpoint metadata from local directory."""
    # Create checkpoint files
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_100.pt").touch()
    (checkpoint_dir / "model_2.pt").touch()
    (checkpoint_dir / "config.yaml").touch()
    (checkpoint_dir / "model_10.pt").touch()
    (checkpoint_dir / "invalid.pt").touch()

    # Create override config
    override_config = OmegaConf.create(
        {
            "wandb_run_path": None,
            "checkpoint_dir": str(checkpoint_dir),
        }
    )

    checkpoint_metadata = get_all_checkpoint_metadata(override_config)

    # Verify the checkpoint metadata is in order
    expected_metadata = [
        {"file_name": "model_2.pt", "global_step": 2, "train_runtime": None, "num_samples": None},
        {"file_name": "model_10.pt", "global_step": 10, "train_runtime": None, "num_samples": None},
        {"file_name": "model_100.pt", "global_step": 100, "train_runtime": None, "num_samples": None},
    ]
    assert checkpoint_metadata == expected_metadata


def test_get_all_checkpoint_metadata_no_inputs() -> None:
    """Test that get_all_checkpoint_metadata raises ValueError when no inputs are provided."""
    override_config = OmegaConf.create(
        {
            "wandb_run_path": None,
            "checkpoint_dir": None,
        }
    )

    with pytest.raises(ValueError, match="No checkpoint directory or wandb run path provided"):
        get_all_checkpoint_metadata(override_config)


def _create_yaml_config(tmp_path, content=None):
    config_path = tmp_path / CONFIG_NAME
    config_content = (
        content
        or """
    base_field: original_value
    nested:
        field1: original_nested_value
        field2: unchanged_value
    override_field:
        base_field: override_value
        nested:
            field1: overridden_nested_value
    """
    )
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def _mock_wandb_file_download(mock_run: mock.MagicMock, source_path: Path, remote_name: str) -> mock.MagicMock:
    """Model the real W&B File.download contract with a file on disk."""

    mock_file = mock.MagicMock()
    mock_file.name = remote_name

    def download(*, root: str, replace: bool = False):
        destination = Path(root) / remote_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not replace:
            raise ValueError("destination already exists")
        destination.write_bytes(source_path.read_bytes())
        # wandb.File.download returns an open TextIOWrapper whose name is the
        # actual path written under root.
        return destination.open()

    mock_file.download.side_effect = download
    mock_run.file.return_value = mock_file
    return mock_file


def _mock_wandb_config_download(mock_wandb_api: mock.MagicMock, config_path: Path) -> mock.MagicMock:
    return _mock_wandb_file_download(mock_wandb_api.run.return_value, config_path, CONFIG_NAME)


def _create_checkpoint_payload(tmp_path: Path, name: str = "model_100.pt") -> tuple[Path, ExperimentConfig]:
    checkpoint_path = tmp_path / "source" / name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    torch.save(
        {
            "actor_model_state_dict": {},
            "experiment_config": cfg.to_serializable_dict(),
            "wandb_run_path": "test_entity/test_project/test_run_id",
        },
        checkpoint_path,
    )
    return checkpoint_path, cfg


def test_load_saved_experiment_config_from_wandb(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    config_path = _create_yaml_config(tmp_path)
    _mock_wandb_config_download(mock_wandb_api, config_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_user/test_project/test_run",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_user/test_project/test_run"
    mock_wandb_api.run.assert_called_once_with("test_user/test_project/test_run")
    mock_wandb_api.run.return_value.file.assert_called_once_with(CONFIG_NAME)


def test_load_saved_experiment_config_with_wandb_prefix(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    checkpoint_path, expected_cfg = _create_checkpoint_payload(tmp_path)
    mock_file = _mock_wandb_file_download(mock_wandb_api.run.return_value, checkpoint_path, "model_100.pt")
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg == expected_cfg
    assert run_path == "test_entity/test_project/test_run_id"
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_wandb_api.run.return_value.file.assert_called_once_with("model_100.pt")
    assert mock_file.download.call_args.kwargs["replace"] is True
    assert Path(mock_file.download.call_args.kwargs["root"]).is_absolute()


def test_load_saved_experiment_config_with_wandb_runs_segment(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    checkpoint_path, expected_cfg = _create_checkpoint_payload(tmp_path)
    mock_file = _mock_wandb_file_download(mock_wandb_api.run.return_value, checkpoint_path, "model_100.pt")
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/runs/test_run_id/model_100.pt",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg == expected_cfg
    assert run_path == "test_entity/test_project/test_run_id"
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_wandb_api.run.return_value.file.assert_called_once_with("model_100.pt")
    assert mock_file.download.call_args.kwargs["replace"] is True


def test_load_saved_experiment_config_with_wandb_run_only(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    """Ensure wandb:// URIs without explicit checkpoint names can load configs."""
    config_path = _create_yaml_config(tmp_path)
    _mock_wandb_config_download(mock_wandb_api, config_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/test_run_id",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_entity/test_project/test_run_id"
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_wandb_api.run.return_value.file.assert_called_once_with(CONFIG_NAME)


def test_save_config_preserves_mapping_order(tmp_path: Path) -> None:
    cfg = dataclasses.replace(
        ExperimentConfig(),
        observation=g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    )
    expected = cfg.to_serializable_dict()
    config_path = tmp_path / CONFIG_NAME

    cfg.save_config(str(config_path))

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_terms = expected["observation"]["groups"]["actor_obs_proprio_with_actions_no_linvel"]["terms"]
    loaded_terms = loaded["observation"]["groups"]["actor_obs_proprio_with_actions_no_linvel"]["terms"]
    assert list(loaded_terms) == list(expected_terms)
    assert list(loaded_terms) == ["base_ang_vel", "dof_pos", "dof_vel", "actions"]


def test_load_saved_experiment_config_from_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    cfg = ExperimentConfig()
    torch.save(
        {
            "actor_model_state_dict": {},
            "experiment_config": cfg.to_serializable_dict(),
            "wandb_run_path": "entity/project/run",
        },
        checkpoint_path,
    )
    checkpoint_cfg = CheckpointConfig(
        checkpoint=str(checkpoint_path),
    )
    loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg == cfg
    assert run_path == "entity/project/run"


def test_load_saved_experiment_config_never_executes_pickle_globals(tmp_path: Path) -> None:
    marker = tmp_path / "pickle_executed"
    checkpoint_path = tmp_path / "unsafe.pt"
    torch.save(
        {
            "experiment_config": ExperimentConfig().to_serializable_dict(),
            "unused": _UnsafeCheckpointConfigPayload(str(marker)),
        },
        checkpoint_path,
    )

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint_path)))

    assert not marker.exists()


def test_load_saved_experiment_config_no_inputs() -> None:
    """Test that load_saved_experiment_config raises ValueError when no inputs are provided."""
    checkpoint_cfg = CheckpointConfig(
        checkpoint=None,
    )

    with pytest.raises(ValueError, match="No checkpoint provided"):
        load_saved_experiment_config(checkpoint_cfg)


def test_load_checkpoint(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    """Test downloading checkpoints from W&B and using local checkpoints.

    Parameters
    ----------
    mock_wandb_api : mock.MagicMock
        Mock wandb API object
    tmp_path : Path
        Temporary directory for test files
    """
    # Test W&B download
    mock_run = mock.MagicMock()
    mock_wandb_api.run.return_value = mock_run
    source_checkpoint, _ = _create_checkpoint_payload(tmp_path)
    mock_file = _mock_wandb_file_download(mock_run, source_checkpoint, "model_100.pt")
    download_dir = tmp_path / "downloads"

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_user/test_project/test_run/model_100.pt",
            log_dir=str(download_dir),
        )

    mock_wandb_api.run.assert_called_once_with("test_user/test_project/test_run")
    mock_run.file.assert_called_once_with("model_100.pt")
    assert mock_file.download.call_args.kwargs["replace"] is True
    assert Path(mock_file.download.call_args.kwargs["root"]).parent == download_dir
    assert checkpoint_path == download_dir / "model_100.pt"
    assert checkpoint_path.read_bytes() == source_checkpoint.read_bytes()

    # Test local checkpoint
    local_checkpoint = tmp_path / "local_model.pt"
    local_checkpoint.touch()  # Create empty file
    checkpoint_path = load_checkpoint(
        checkpoint=str(local_checkpoint),
        log_dir=str(tmp_path),
    )
    assert checkpoint_path == local_checkpoint


def test_load_checkpoint_with_wandb_prefix(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    """Test loading checkpoint with wandb:// prefix.

    Parameters
    ----------
    mock_wandb_api : mock.MagicMock
        Mock wandb API object
    tmp_path : Path
        Temporary directory for test files
    """
    # Test wandb:// prefix parsing
    mock_run = mock.MagicMock()
    mock_wandb_api.run.return_value = mock_run
    source_checkpoint, _ = _create_checkpoint_payload(tmp_path)
    mock_file = _mock_wandb_file_download(mock_run, source_checkpoint, "model_100.pt")
    download_dir = tmp_path / "downloads"

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
            log_dir=str(download_dir),
        )

    # Verify the wandb path was correctly parsed and used
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_run.file.assert_called_once_with("model_100.pt")
    assert mock_file.download.call_args.kwargs["replace"] is True
    assert checkpoint_path == download_dir / "model_100.pt"
    assert checkpoint_path.read_bytes() == source_checkpoint.read_bytes()


def test_load_checkpoint_with_wandb_runs_segment(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    mock_run = mock.MagicMock()
    mock_wandb_api.run.return_value = mock_run
    source_checkpoint, _ = _create_checkpoint_payload(tmp_path)
    mock_file = _mock_wandb_file_download(mock_run, source_checkpoint, "model_100.pt")
    download_dir = tmp_path / "downloads"

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_entity/test_project/runs/test_run_id/model_100.pt",
            log_dir=str(download_dir),
        )

    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_run.file.assert_called_once_with("model_100.pt")
    assert mock_file.download.call_args.kwargs["replace"] is True
    assert checkpoint_path == download_dir / "model_100.pt"
    assert checkpoint_path.read_bytes() == source_checkpoint.read_bytes()


def test_load_checkpoint_with_wandb_prefix_missing_checkpoint_name(tmp_path: Path) -> None:
    """Ensure wandb:// URIs for checkpoints include the artifact name."""
    with pytest.raises(
        ValueError,
        match="Expected format: wandb://<entity>/<project>/<run_id>/<checkpoint_name>",
    ):
        load_checkpoint(
            checkpoint="wandb://test_entity/test_project/test_run_id",
            log_dir=str(tmp_path),
        )


def test_named_wandb_checkpoint_config_is_fail_closed(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    """A named checkpoint must not silently fall back to mutable run config."""

    corrupt_checkpoint = tmp_path / "corrupt.pt"
    corrupt_checkpoint.write_bytes(b"not a torch checkpoint")
    mock_file = _mock_wandb_file_download(
        mock_wandb_api.run.return_value,
        corrupt_checkpoint,
        "model_100.pt",
    )
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
    )

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        with pytest.raises(RuntimeError, match="exact W&B checkpoint payload"):
            load_saved_experiment_config(checkpoint_cfg)

    mock_wandb_api.run.return_value.file.assert_called_once_with("model_100.pt")
    mock_file.download.assert_called_once()


def test_load_checkpoint_fails_when_wandb_reports_missing_file(
    mock_wandb_api: mock.MagicMock, tmp_path: Path
) -> None:
    mock_run = mock_wandb_api.run.return_value
    mock_file = mock.MagicMock()
    mock_file.name = "model_100.pt"

    def missing_download(*, root: str, replace: bool = False):
        assert replace is True
        handle = mock.MagicMock()
        handle.name = str(Path(root) / "model_100.pt")
        return handle

    mock_file.download.side_effect = missing_download
    mock_run.file.return_value = mock_file

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        with pytest.raises(FileNotFoundError, match="does not exist as a file"):
            load_checkpoint(
                checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
                log_dir=str(tmp_path / "downloads"),
            )


def test_load_checkpoint_fails_for_missing_local_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Local checkpoint does not exist"):
        load_checkpoint(str(tmp_path / "missing.pt"), str(tmp_path / "downloads"))


@pytest.mark.parametrize(
    "checkpoint",
    [
        "wandb://test_entity/test_project/test_run_id/../model_100.pt",
        "wandb://test_entity/test_project//model_100.pt",
        "wandb://test_entity/test_project/test_run_id/",
    ],
)
def test_load_checkpoint_rejects_ambiguous_wandb_paths(checkpoint: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid wandb URI"):
        load_checkpoint(checkpoint, str(tmp_path))
