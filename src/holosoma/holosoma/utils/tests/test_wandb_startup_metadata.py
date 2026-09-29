import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import holosoma.train_agent as train_agent
from holosoma.train_agent import (
    _bool_env,
    _collect_training_provenance_wandb_metadata,
    _finish_wandb_run,
    _publish_wandb_startup_metadata,
    _rank_training_seed,
    _resolve_wandb_startup_outcome,
    _run_rank_zero_wandb_startup,
    _strict_active_wandb_run_path,
    _synchronize_wandb_startup_outcome,
    _validate_prestarted_runtime_provenance,
    _validate_required_wandb_logger_mode,
    _wandb_init_failure_is_fatal,
)


def _wandb_stub() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(update=MagicMock()),
        run=SimpleNamespace(summary={}),
        log=MagicMock(),
    )


def test_startup_metadata_does_not_commit_a_training_history_step() -> None:
    wandb = _wandb_stub()

    _publish_wandb_startup_metadata(
        wandb,
        config_metadata={"provenance/source_sha256": "abc123"},
    )

    wandb.config.update.assert_called_once_with(
        {"provenance/source_sha256": "abc123"},
        allow_val_change=True,
    )
    assert wandb.run.summary == {"provenance/source_sha256": "abc123"}
    wandb.log.assert_not_called()

    # The first training metrics row remains valid at the scientific iteration
    # index 0 because startup did not move W&B's history cursor to step 1.
    wandb.log({"Loss/value": 0.25, "global_step": 0}, step=0)
    wandb.log.assert_called_once_with({"Loss/value": 0.25, "global_step": 0}, step=0)


def test_startup_metadata_can_keep_detailed_config_and_scalar_summary_separate() -> None:
    wandb = _wandb_stub()
    config_metadata = {"reward_group_spec": {"Track": {"term": {"weight": 1.0}}}}
    summary_metadata = {"RewardSpec/Track/term/weight": 1.0}

    _publish_wandb_startup_metadata(
        wandb,
        config_metadata=config_metadata,
        summary_metadata=summary_metadata,
    )

    wandb.config.update.assert_called_once_with(config_metadata, allow_val_change=True)
    assert wandb.run.summary == summary_metadata
    wandb.log.assert_not_called()


def test_wandb_provenance_metadata_retains_full_runtime_asset_manifest() -> None:
    manifest = {"version": 2, "robot": {"urdf": {"sha256": "a" * 64}}}
    with patch(
        "holosoma.train_agent.training_provenance_from_env",
        return_value={"runtime_asset_manifest": manifest},
    ):
        metadata = _collect_training_provenance_wandb_metadata()

    assert metadata["provenance/runtime_asset_manifest"] == manifest


def test_wandb_provenance_metadata_retains_semantic_environment_mapping() -> None:
    semantic_environment = {
        "HOLOSOMA_DEFM_FORWARD_BATCH_SIZE": "0",
        "HOLOSOMA_DISABLE_AUTO_RESET": None,
    }
    with patch(
        "holosoma.train_agent.training_provenance_from_env",
        return_value={
            "environment": {
                "execution_runtime": {
                    "semantic_environment": semantic_environment,
                }
            }
        },
    ):
        metadata = _collect_training_provenance_wandb_metadata()

    assert metadata["provenance/environment"]["execution_runtime"][
        "semantic_environment"
    ] == semantic_environment


def test_wandb_must_resume_init_failure_is_fatal() -> None:
    assert _wandb_init_failure_is_fatal("must") is True
    assert _wandb_init_failure_is_fatal(" MUST ") is True
    assert _wandb_init_failure_is_fatal("allow") is False
    assert _wandb_init_failure_is_fatal(None) is False


@pytest.mark.parametrize("exit_code", [0, 1, 17])
def test_wandb_shutdown_preserves_exit_code_without_global_teardown(exit_code: int) -> None:
    wandb = SimpleNamespace(finish=MagicMock(), teardown=MagicMock())

    _finish_wandb_run(wandb, exit_code=exit_code)

    wandb.finish.assert_called_once_with(exit_code=exit_code)
    wandb.teardown.assert_not_called()


@pytest.mark.parametrize("exit_code", [True, -1, 1.0, "1"])
def test_wandb_shutdown_rejects_ambiguous_exit_code(exit_code: object) -> None:
    wandb = SimpleNamespace(finish=MagicMock())

    with pytest.raises(ValueError, match="non-negative integer"):
        _finish_wandb_run(wandb, exit_code=exit_code)  # type: ignore[arg-type]

    wandb.finish.assert_not_called()


def _startup_wandb_stub(*, run: object | None = None, init_error: Exception | None = None) -> SimpleNamespace:
    wandb = SimpleNamespace(run=None, finish=MagicMock())

    def init(**_kwargs: object) -> None:
        if init_error is not None:
            raise init_error
        wandb.run = run

    def finish(*, exit_code: int | None = None) -> None:
        del exit_code
        wandb.run = None

    wandb.init = MagicMock(side_effect=init)
    wandb.finish = MagicMock(side_effect=finish)
    return wandb


def _active_run() -> SimpleNamespace:
    return SimpleNamespace(entity="entity-a", project="project-a", id="run123")


def test_require_wandb_env_uses_strict_boolean_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLOSOMA_REQUIRE_WANDB_RUN", "yes")
    assert _bool_env("HOLOSOMA_REQUIRE_WANDB_RUN") is True
    monkeypatch.setenv("HOLOSOMA_REQUIRE_WANDB_RUN", "invalid")
    with pytest.raises(ValueError, match="must be a boolean"):
        _bool_env("HOLOSOMA_REQUIRE_WANDB_RUN")


@pytest.mark.parametrize(
    ("wandb_enabled", "wandb_mode", "message"),
    [
        (False, None, "logger.type='wandb'"),
        (True, "offline", "logger.mode='online'"),
    ],
)
def test_required_wandb_rejects_disabled_or_offline_logger(
    wandb_enabled: bool,
    wandb_mode: str | None,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validate_required_wandb_logger_mode(
            require_run=True,
            wandb_enabled=wandb_enabled,
            wandb_mode=wandb_mode,
        )


def test_required_fresh_wandb_init_exception_becomes_fatal_outcome() -> None:
    wandb = _startup_wandb_stub(init_error=RuntimeError("network down"))
    outcome = _run_rank_zero_wandb_startup(
        wandb,
        wandb_enabled=True,
        wandb_mode="online",
        require_run=True,
        wandb_kwargs={"project": "project-a"},
        publish_startup=MagicMock(),
    )

    assert outcome["ok"] is False
    assert outcome["error_type"] == "RuntimeError"
    with pytest.raises(RuntimeError, match="all ranks are aborting"):
        _resolve_wandb_startup_outcome(outcome, require_run=True, resume_mode=None)


def test_required_fresh_wandb_none_run_is_failure() -> None:
    wandb = _startup_wandb_stub(run=None)
    publish = MagicMock()

    outcome = _run_rank_zero_wandb_startup(
        wandb,
        wandb_enabled=True,
        wandb_mode="online",
        require_run=True,
        wandb_kwargs={},
        publish_startup=publish,
    )

    assert outcome["ok"] is False
    assert "without creating an active run" in str(outcome["error_message"])
    publish.assert_not_called()


def test_required_wandb_metadata_failure_closes_partial_run() -> None:
    wandb = _startup_wandb_stub(run=_active_run())

    def fail_metadata() -> None:
        raise OSError("metadata save failed")

    outcome = _run_rank_zero_wandb_startup(
        wandb,
        wandb_enabled=True,
        wandb_mode="online",
        require_run=True,
        wandb_kwargs={},
        publish_startup=fail_metadata,
    )

    assert outcome["ok"] is False
    assert outcome["run_path"] is None
    wandb.finish.assert_called_once_with(exit_code=1)
    assert wandb.run is None


def test_successful_wandb_startup_publishes_strict_path() -> None:
    wandb = _startup_wandb_stub(run=_active_run())
    publish = MagicMock()

    outcome = _run_rank_zero_wandb_startup(
        wandb,
        wandb_enabled=True,
        wandb_mode="online",
        require_run=True,
        wandb_kwargs={
            "mode": "online",
            "entity": "entity-a",
            "project": "project-a",
            "id": "run123",
        },
        publish_startup=publish,
    )

    assert outcome == {
        "ok": True,
        "run_path": "entity-a/project-a/run123",
        "error_type": None,
        "error_message": None,
        "force_fatal": False,
    }
    publish.assert_called_once_with()
    assert _resolve_wandb_startup_outcome(outcome, require_run=True, resume_mode=None) == outcome["run_path"]


@pytest.mark.parametrize("bad_id", [None, "", "bad/id", "bad\nrun", 123])
def test_wandb_run_path_rejects_ambiguous_identity(bad_id: object) -> None:
    wandb = SimpleNamespace(run=SimpleNamespace(entity="entity-a", project="project-a", id=bad_id))
    with pytest.raises(RuntimeError, match="invalid URL-path identity"):
        _strict_active_wandb_run_path(wandb)


@pytest.mark.parametrize(
    ("expected_kwargs", "mismatch_field"),
    [
        ({"expected_entity": "other-entity"}, "entity"),
        ({"expected_project": "other-project"}, "project"),
        ({"expected_run_id": "other-run"}, "id"),
    ],
)
def test_wandb_run_path_rejects_valid_but_wrong_requested_identity(
    expected_kwargs: dict[str, str],
    mismatch_field: str,
) -> None:
    wandb = SimpleNamespace(run=_active_run())
    with pytest.raises(RuntimeError, match=rf"{mismatch_field} requested=.* active="):
        _strict_active_wandb_run_path(wandb, **expected_kwargs)


def test_wandb_identity_mismatch_closes_partial_run() -> None:
    wandb = _startup_wandb_stub(run=_active_run())
    outcome = _run_rank_zero_wandb_startup(
        wandb,
        wandb_enabled=True,
        wandb_mode="online",
        require_run=True,
        wandb_kwargs={"entity": "entity-a", "project": "wrong-project", "id": "run123"},
        publish_startup=MagicMock(),
    )

    assert outcome["ok"] is False
    assert "does not match the requested launch identity" in str(outcome["error_message"])
    wandb.finish.assert_called_once_with(exit_code=1)


def test_optional_fresh_wandb_failure_preserves_synchronized_fallback() -> None:
    wandb = _startup_wandb_stub(init_error=RuntimeError("optional failure"))
    outcome = _run_rank_zero_wandb_startup(
        wandb,
        wandb_enabled=True,
        wandb_mode="online",
        require_run=False,
        wandb_kwargs={},
        publish_startup=MagicMock(),
    )
    assert _resolve_wandb_startup_outcome(outcome, require_run=False, resume_mode=None) is None


def test_must_resume_failure_is_fatal_after_outcome() -> None:
    wandb = _startup_wandb_stub(init_error=RuntimeError("resume unavailable"))
    outcome = _run_rank_zero_wandb_startup(
        wandb,
        wandb_enabled=True,
        wandb_mode="online",
        require_run=False,
        wandb_kwargs={},
        publish_startup=MagicMock(),
    )
    with pytest.raises(RuntimeError, match="all ranks are aborting"):
        _resolve_wandb_startup_outcome(outcome, require_run=False, resume_mode="must")


class _FakeGlooDist:
    ReduceOp = SimpleNamespace(MAX="max")

    def __init__(self, *, rank: int, broadcast_value: dict[str, object] | None = None) -> None:
        self.rank = rank
        self.broadcast_value = broadcast_value
        self.events: list[str] = []

    def is_initialized(self) -> bool:
        return True

    def get_rank(self) -> int:
        return self.rank

    def get_backend(self) -> str:
        return "gloo"

    def broadcast_object_list(self, payload: list[object], **_kwargs: object) -> None:
        self.events.append("broadcast")
        if self.rank != 0:
            payload[0] = self.broadcast_value

    def all_reduce(self, _tensor: object, **_kwargs: object) -> None:
        self.events.append("all_reduce")


class _FakeNcclDist(_FakeGlooDist):
    def __init__(self, *, rank: int) -> None:
        super().__init__(rank=rank)
        self.broadcast_kwargs: dict[str, object] = {}
        self.all_reduce_kwargs: dict[str, object] = {}
        self.control_group = object()

    def get_backend(self) -> str:
        return "nccl"

    def new_group(self, **_kwargs: object) -> object:
        self.events.append("new_group")
        return self.control_group

    def broadcast_object_list(self, payload: list[object], **kwargs: object) -> None:
        self.events.append("broadcast")
        self.broadcast_kwargs = kwargs

    def all_reduce(self, _tensor: object, **kwargs: object) -> None:
        self.events.append("all_reduce")
        self.all_reduce_kwargs = kwargs

    def destroy_process_group(self, group: object) -> None:
        assert group is self.control_group
        self.events.append("destroy_group")


def test_nonzero_rank_receives_same_success_path() -> None:
    success = {
        "ok": True,
        "run_path": "entity-a/project-a/run123",
        "error_type": None,
        "error_message": None,
        "force_fatal": False,
    }
    dist = _FakeGlooDist(
        rank=1,
        broadcast_value={
            "outcome": success,
            "require_run": True,
            "resume_must": False,
        },
    )
    shared = _synchronize_wandb_startup_outcome(
        dist_module=dist,
        distributed_conf={"global_rank": 1, "local_rank": 1, "world_size": 2},
        device="cpu",
        rank_zero_outcome=None,
        local_require_run=True,
        local_resume_must=False,
    )
    assert _resolve_wandb_startup_outcome(shared, require_run=True, resume_mode=None) == success["run_path"]
    assert dist.events == ["broadcast", "all_reduce"]


def test_required_failure_collective_happens_before_raise() -> None:
    failure = {
        "ok": False,
        "run_path": None,
        "error_type": "RuntimeError",
        "error_message": "rank zero failed",
        "force_fatal": False,
    }
    dist = _FakeGlooDist(rank=0)
    with pytest.raises(RuntimeError, match="all ranks are aborting"):
        shared = _synchronize_wandb_startup_outcome(
            dist_module=dist,
            distributed_conf={"global_rank": 0, "local_rank": 0, "world_size": 2},
            device="cpu",
            rank_zero_outcome=failure,
            local_require_run=True,
            local_resume_must=False,
        )
        _resolve_wandb_startup_outcome(shared, require_run=True, resume_mode=None)
    assert dist.events == ["broadcast", "all_reduce"]


@pytest.mark.parametrize("use_gloo_control", [False, True])
def test_nccl_wandb_policy_collective_uses_rank_device_or_same_gloo_group(
    monkeypatch: pytest.MonkeyPatch,
    use_gloo_control: bool,
) -> None:
    import torch

    success = {
        "ok": True,
        "run_path": "entity-a/project-a/run123",
        "error_type": None,
        "error_message": None,
        "force_fatal": False,
    }
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1" if use_gloo_control else "0")
    dist = _FakeNcclDist(rank=0)
    mismatch_tensor = SimpleNamespace(item=lambda: 0)
    with patch("torch.tensor", return_value=mismatch_tensor) as make_tensor:
        shared = _synchronize_wandb_startup_outcome(
            dist_module=dist,
            distributed_conf={"global_rank": 0, "local_rank": 3, "world_size": 8},
            device="cuda:3",
            rank_zero_outcome=success,
            local_require_run=True,
            local_resume_must=False,
        )

    assert shared == success
    expected_device = torch.device("cpu" if use_gloo_control else "cuda:3")
    assert dist.broadcast_kwargs["device"] == expected_device
    assert make_tensor.call_args.kwargs["device"] == expected_device
    if use_gloo_control:
        assert dist.events == ["new_group", "broadcast", "all_reduce", "destroy_group"]
        assert dist.broadcast_kwargs["group"] is dist.control_group
        assert dist.all_reduce_kwargs["group"] is dist.control_group
    else:
        assert dist.events == ["broadcast", "all_reduce"]
        assert "group" not in dist.broadcast_kwargs
        assert "group" not in dist.all_reduce_kwargs


def _gloo_wandb_outcome_worker(
    rank: int,
    rendezvous_path: str,
    result_root: str,
    fail: bool,
    mismatch_field: str | None,
) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=2,
    )
    try:
        rank_zero_outcome = None
        if rank == 0:
            rank_zero_outcome = (
                {
                    "ok": False,
                    "run_path": None,
                    "error_type": "RuntimeError",
                    "error_message": "gloo startup failure",
                    "force_fatal": False,
                }
                if fail
                else {
                    "ok": True,
                    "run_path": "entity-a/project-a/run123",
                    "error_type": None,
                    "error_message": None,
                    "force_fatal": False,
                }
            )
        local_require_run = not (rank == 1 and mismatch_field == "require_run")
        local_resume_must = rank == 1 and mismatch_field == "resume_must"
        try:
            shared = _synchronize_wandb_startup_outcome(
                dist_module=dist,
                distributed_conf={"global_rank": rank, "local_rank": rank, "world_size": 2},
                device="cpu",
                rank_zero_outcome=rank_zero_outcome,
                local_require_run=local_require_run,
                local_resume_must=local_resume_must,
            )
        except RuntimeError as exc:
            if "policy differs across ranks" not in str(exc):
                raise
            Path(result_root, f"rank-{rank}.txt").write_text("policy-mismatch", encoding="utf-8")
            return
        try:
            resolved = _resolve_wandb_startup_outcome(
                shared,
                require_run=True,
                resume_mode=None,
            )
        except RuntimeError:
            resolved = "fatal"
        Path(result_root, f"rank-{rank}.txt").write_text(str(resolved), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail", [False, True])
def test_real_two_rank_gloo_wandb_outcome(tmp_path: Path, fail: bool) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = tmp_path / ("failure-rendezvous" if fail else "success-rendezvous")
    mp.spawn(
        _gloo_wandb_outcome_worker,
        args=(str(rendezvous_path), str(tmp_path), fail, None),
        nprocs=2,
        join=True,
    )
    expected = "fatal" if fail else "entity-a/project-a/run123"
    assert [
        (tmp_path / f"rank-{rank}.txt").read_text(encoding="utf-8")
        for rank in range(2)
    ] == [expected, expected]


@pytest.mark.parametrize("mismatch_field", ["require_run", "resume_must"])
def test_real_two_rank_gloo_rejects_divergent_wandb_policy(
    tmp_path: Path,
    mismatch_field: str,
) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = tmp_path / f"{mismatch_field}-rendezvous"
    mp.spawn(
        _gloo_wandb_outcome_worker,
        args=(str(rendezvous_path), str(tmp_path), False, mismatch_field),
        nprocs=2,
        join=True,
    )
    assert [
        (tmp_path / f"rank-{rank}.txt").read_text(encoding="utf-8")
        for rank in range(2)
    ] == ["policy-mismatch", "policy-mismatch"]


def test_fresh_training_rng_is_seeded_after_logger_sync_and_before_environment() -> None:
    """Rank-0-only logger setup must not perturb the seeded simulator stream."""

    source = inspect.getsource(train_agent.train)
    wandb_startup = source.index("_run_rank_zero_wandb_startup(")
    outcome_collective = source.index("_synchronize_wandb_startup_outcome(", wandb_startup)
    outcome_resolution = source.index("_resolve_wandb_startup_outcome(", outcome_collective)
    logger_barrier = source.index("_distributed_barrier(dist, distributed_conf)", outcome_resolution)
    seed_call = source.index("seeding(seed, torch_deterministic=", logger_barrier)
    environment_construction = source.index("env = get_class(env_target)", seed_call)

    assert wandb_startup < outcome_collective < outcome_resolution < logger_barrier
    assert logger_barrier < seed_call < environment_construction


def test_rank_training_seed_covers_world_without_collisions() -> None:
    seeds = [
        _rank_training_seed(0, world_size=104, global_rank=rank)
        for rank in range(104)
    ]
    assert seeds == list(range(104))


def test_rank_training_seed_accepts_numpy_upper_boundary() -> None:
    base_seed = 2**32 - 104
    assert _rank_training_seed(base_seed, world_size=104, global_rank=103) == 2**32 - 1


@pytest.mark.parametrize("base_seed", [-1, 2**32 - 103])
def test_rank_training_seed_rejects_world_that_escapes_numpy_range(base_seed: int) -> None:
    with pytest.raises(ValueError, match="NumPy"):
        _rank_training_seed(base_seed, world_size=104, global_rank=0)


def _runtime_provenance(*, hash_seed: str = "0", cublas: str = ":4096:8") -> dict:
    return {
        "environment": {
            "execution_runtime": {
                "PYTHONHASHSEED": hash_seed,
                "CUBLAS_WORKSPACE_CONFIG": cublas,
            }
        }
    }


def test_prestarted_runtime_matches_scientific_provenance(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "000")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _validate_prestarted_runtime_provenance(_runtime_provenance())


@pytest.mark.parametrize(
    ("hash_seed", "cublas", "expected"),
    [
        (None, ":4096:8", "PYTHONHASHSEED"),
        ("0", ":16:8", "does not match"),
    ],
)
def test_prestarted_runtime_rejects_unrecorded_or_drifted_settings(
    monkeypatch,
    hash_seed,
    cublas,
    expected,
) -> None:
    if hash_seed is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", hash_seed)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", cublas)
    with pytest.raises(RuntimeError, match=expected):
        _validate_prestarted_runtime_provenance(_runtime_provenance())
