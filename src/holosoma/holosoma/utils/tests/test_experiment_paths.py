from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from holosoma import train_agent
from holosoma.utils.experiment_paths import (
    EXPERIMENT_DIR_ENV,
    EXPERIMENT_TASK_ENV,
    EXPERIMENT_TIMESTAMP_ENV,
    get_experiment_dir,
    get_process_experiment_dir,
    set_experiment_dir_override,
    validate_experiment_timestamp,
)


def _configs(tmp_path: Path):
    logger_config = SimpleNamespace(
        base_dir=str(tmp_path / "logs"),
        project=None,
        name=None,
        group=None,
    )
    training_config = SimpleNamespace(project="project", name="scientific_run")
    return logger_config, training_config


def _clear_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EXPERIMENT_TIMESTAMP_ENV, raising=False)
    monkeypatch.delenv(EXPERIMENT_DIR_ENV, raising=False)
    monkeypatch.delenv(EXPERIMENT_TASK_ENV, raising=False)


class _FakeDistributed:
    def __init__(self, *, rank: int, world_size: int, backend: str, received_identity=None):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.received_identity = received_identity
        self.broadcast_device = None
        self.broadcast_value = None
        self.broadcast_group = None
        self.created_group = None
        self.destroyed_group = None

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def get_backend(self):
        return self.backend

    def new_group(self, *, backend, timeout):
        self.created_group = SimpleNamespace(backend=backend, timeout=timeout)
        return self.created_group

    def destroy_process_group(self, group):
        self.destroyed_group = group

    def broadcast_object_list(self, values, *, src, device, group=None):
        assert src == 0
        self.broadcast_device = device
        self.broadcast_group = group
        if self.rank == src:
            self.broadcast_value = values[0]
        else:
            values[0] = self.received_identity


def test_validate_experiment_timestamp_rejects_noncanonical_or_impossible_values():
    assert validate_experiment_timestamp("20260711_202501") == "20260711_202501"
    for invalid in ("2026-07-11_20:25:01", "20260711_2025", "20261311_202501", ""):
        with pytest.raises(ValueError):
            validate_experiment_timestamp(invalid)


def test_process_override_round_trips_exact_directory(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    timestamp = "20260711_202501"
    expected = get_experiment_dir(
        logger_config,
        training_config,
        timestamp,
        task_name="locomotion",
    ).resolve()

    installed = set_experiment_dir_override(
        logger_config,
        training_config,
        timestamp=timestamp,
        experiment_dir=expected,
        task_name="locomotion",
    )

    assert installed == expected
    assert Path(os.environ[EXPERIMENT_DIR_ENV]) == expected
    assert os.environ[EXPERIMENT_TIMESTAMP_ENV] == timestamp
    assert os.environ[EXPERIMENT_TASK_ENV] == "locomotion"
    assert (
        get_process_experiment_dir(
            logger_config,
            training_config,
            task_name="locomotion",
            require_override=True,
        )
        == expected
    )


@pytest.mark.parametrize(
    "present_key",
    [EXPERIMENT_TIMESTAMP_ENV, EXPERIMENT_DIR_ENV, EXPERIMENT_TASK_ENV],
)
def test_process_override_rejects_partial_identity(tmp_path, monkeypatch, present_key):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    monkeypatch.setenv(
        present_key,
        {
            EXPERIMENT_TIMESTAMP_ENV: "20260711_202501",
            EXPERIMENT_DIR_ENV: str(tmp_path / "logs"),
            EXPERIMENT_TASK_ENV: "locomotion",
        }[present_key],
    )

    with pytest.raises(RuntimeError, match="must be set together"):
        get_process_experiment_dir(
            logger_config,
            training_config,
            task_name="locomotion",
        )


def test_process_override_rejects_stale_directory(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    monkeypatch.setenv(EXPERIMENT_TIMESTAMP_ENV, "20260711_202501")
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(tmp_path / "wrong"))
    monkeypatch.setenv(EXPERIMENT_TASK_ENV, "locomotion")

    with pytest.raises(RuntimeError, match="inconsistent with the active config"):
        get_process_experiment_dir(
            logger_config,
            training_config,
            task_name="locomotion",
        )


def test_single_process_training_installs_shared_identity(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    monkeypatch.setattr(train_agent, "get_timestamp", lambda: "20260711_202501")

    timestamp, experiment_dir = train_agent._synchronize_experiment_identity(
        dist_module=None,
        distributed_conf=None,
        device="cpu",
        logger_config=logger_config,
        training_config=training_config,
        task_name="locomotion",
    )

    assert timestamp == "20260711_202501"
    assert Path(os.environ[EXPERIMENT_DIR_ENV]) == experiment_dir
    assert os.environ[EXPERIMENT_TIMESTAMP_ENV] == timestamp


def test_base_task_style_resolution_reuses_training_task_override(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    timestamp = "20260711_202501"
    expected = get_experiment_dir(
        logger_config,
        training_config,
        timestamp,
        task_name="locomotion",
    ).resolve()
    set_experiment_dir_override(
        logger_config,
        training_config,
        timestamp=timestamp,
        experiment_dir=expected,
        task_name="locomotion",
    )

    # WholeBodyTrackingManager's fallback class-derived task name differs from
    # train_agent's canonical training task, but BaseTask must reuse the run.
    assert (
        get_process_experiment_dir(
            logger_config,
            training_config,
            task_name="wholebodytrackingmanager",
            use_override_task_name=True,
        )
        == expected
    )


def test_nccl_nonzero_rank_reuses_rank_zero_identity(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    timestamp = "20260711_202501"
    expected_dir = get_experiment_dir(
        logger_config,
        training_config,
        timestamp,
        task_name="locomotion",
    ).resolve()
    fake_dist = _FakeDistributed(
        rank=3,
        world_size=8,
        backend="nccl",
        received_identity=(timestamp, str(expected_dir)),
    )
    monkeypatch.setattr(
        train_agent,
        "get_timestamp",
        lambda: pytest.fail("Only rank zero may generate the experiment timestamp"),
    )

    received_timestamp, received_dir = train_agent._synchronize_experiment_identity(
        dist_module=fake_dist,
        distributed_conf={"global_rank": 3, "local_rank": 3, "world_size": 8},
        device="cuda:3",
        logger_config=logger_config,
        training_config=training_config,
        task_name="locomotion",
    )

    assert received_timestamp == timestamp
    assert received_dir == expected_dir
    assert fake_dist.broadcast_device.type == "cuda"
    assert fake_dist.broadcast_device.index == 3


def test_gloo_rank_zero_broadcasts_identity_on_cpu(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    monkeypatch.setattr(train_agent, "get_timestamp", lambda: "20260711_202501")
    fake_dist = _FakeDistributed(rank=0, world_size=2, backend="gloo")

    timestamp, experiment_dir = train_agent._synchronize_experiment_identity(
        dist_module=fake_dist,
        distributed_conf={"global_rank": 0, "local_rank": 0, "world_size": 2},
        device="cuda:0",
        logger_config=logger_config,
        training_config=training_config,
        task_name="locomotion",
    )

    assert fake_dist.broadcast_device.type == "cpu"
    assert fake_dist.broadcast_value == (timestamp, str(experiment_dir))


def test_nccl_launcher_gloo_control_contract_uses_temporary_cpu_group(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    timestamp = "20260711_202501"
    expected_dir = get_experiment_dir(
        logger_config,
        training_config,
        timestamp,
        task_name="locomotion",
    ).resolve()
    fake_dist = _FakeDistributed(
        rank=1,
        world_size=8,
        backend="nccl",
        received_identity=(timestamp, str(expected_dir)),
    )
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
    monkeypatch.setenv("TORCH_DIST_TIMEOUT_SEC", "123")

    received_timestamp, received_dir = train_agent._synchronize_experiment_identity(
        dist_module=fake_dist,
        distributed_conf={"global_rank": 1, "local_rank": 1, "world_size": 8},
        device="cuda:1",
        logger_config=logger_config,
        training_config=training_config,
        task_name="locomotion",
    )

    assert received_timestamp == timestamp
    assert received_dir == expected_dir
    assert fake_dist.created_group.backend == "gloo"
    assert fake_dist.broadcast_group is fake_dist.created_group
    assert fake_dist.broadcast_device.type == "cpu"
    assert fake_dist.destroyed_group is fake_dist.created_group


def test_distributed_identity_rejects_rank_zero_path_inconsistent_with_local_config(tmp_path, monkeypatch):
    _clear_identity(monkeypatch)
    logger_config, training_config = _configs(tmp_path)
    fake_dist = _FakeDistributed(
        rank=1,
        world_size=2,
        backend="gloo",
        received_identity=("20260711_202501", str(tmp_path / "rank-zero-wrong-path")),
    )

    with pytest.raises(ValueError, match="does not match its timestamp/config identity"):
        train_agent._synchronize_experiment_identity(
            dist_module=fake_dist,
            distributed_conf={"global_rank": 1, "local_rank": 1, "world_size": 2},
            device="cuda:1",
            logger_config=logger_config,
            training_config=training_config,
            task_name="locomotion",
        )
