from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

from holosoma.utils.checkpoint_validation import (
    fixed_bc_dataset_sha256,
    fixed_bc_global_dataset_sha256,
    terminal_fixed_bc_eval_state_sha256,
)
from holosoma.utils.runtime_asset_manifest import (
    embedded_runtime_asset_manifest_sha256,
)
from holosoma.utils.training_provenance import disabled_checkpoint_sha256
from scripts.validate_terminal_policy_init import (
    validate_controller_terminal_policy_init,
)


_SOURCE_SNAPSHOT_ID = "src-" + "5" * 64


def _payload() -> dict:
    target = 8
    completed = 7
    budget = 2
    rank_state = {
        "allocation_version": 1,
        "allocation_scheme": "rank_quotient_remainder",
        "global_sample_budget": budget,
        "world_size": 1,
        "rank": 0,
        "local_target": budget,
        "ready": True,
        "size": budget,
        "actor_obs_raw": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "teacher_actions": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        "actor_perception": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    local_digest = fixed_bc_dataset_sha256(
        rank_state,
        expected_rows=budget,
        required_tensor_keys={"actor_obs_raw", "teacher_actions", "actor_perception"},
        context="controller fixture",
    )
    global_digest = fixed_bc_global_dataset_sha256(
        {"0": local_digest},
        global_sample_budget=budget,
        world_size=1,
    )
    terminal_state = {
        "version": 1,
        "terminal_observation": True,
        "completed_iteration": completed,
        "next_iteration": target,
        "run_target_iteration": target,
        "scheduled_evaluation": False,
        "guard_enabled": False,
        "guard_applied": False,
        "fixed_bc_eval_log_interval": 2,
        "fixed_bc_eval_num_samples": budget,
        "world_size": 1,
        "fixed_bc_global_dataset_sha256": global_digest,
        "fixed_bc_guard_config_sha256": "a" * 64,
        "fixed_bc_guard_state_sha256": None,
        "fixed_bc_guard_threshold_mu_mse": None,
        "fixed_bc_terminal_within_threshold": None,
        "fixed_bc_mu_mse": 0.03,
        "fixed_bc_num_samples": budget,
        "fixed_bc_weighted_num_samples": float(budget),
        "fixed_bc_expected_weighted_num_samples": float(budget),
        "fixed_bc_rank_strata": 1,
    }
    return {
        "iter": completed,
        "iteration": completed,
        "next_iter": target,
        "experiment_config": {
            "algo": {
                "config": {
                    "num_learning_iterations": target,
                    "module_dict": {
                        "actor": {
                            "layer_config": {
                                "perception_input_name": "perception_obs"
                            }
                        }
                    },
                }
            },
            "robot": {"actions_dim": 2},
        },
        "actor_model_state_dict": {"weight": torch.ones(2, 2)},
        "fixed_bc_eval_by_rank": {"0": rank_state},
        "terminal_fixed_bc_eval": terminal_state,
        "terminal_fixed_bc_eval_sha256": terminal_fixed_bc_eval_state_sha256(
            terminal_state
        ),
        "wandb_run_path": "entity/project/replacement-canary",
    }


def _fresh_finalized_provenance(
    *,
    source_snapshot_id: str | None = None,
) -> dict:
    runtime_manifest = {"version": 2, "fixture": "terminal-policy-init-controller"}
    provenance = {
        "version": 2,
        "training_regime": "distillation",
        "teacher_enabled": True,
        "teacher_sha256": "1" * 64,
        "policy_init_enabled": False,
        "policy_init_sha256": disabled_checkpoint_sha256("policy_init"),
        "training_resume_enabled": False,
        "training_resume_sha256": disabled_checkpoint_sha256("training_resume"),
        "motion_shard_manifest_sha256": "2" * 64,
        "contact_sidecar_manifest_sha256": "3" * 64,
        "source_bundle_sha256": "4" * 64,
        "runtime_asset_manifest_phase": "final",
        "runtime_asset_manifest": runtime_manifest,
        "runtime_asset_manifest_sha256": embedded_runtime_asset_manifest_sha256(
            runtime_manifest
        ),
    }
    if source_snapshot_id is not None:
        provenance["source_snapshot_id"] = source_snapshot_id
        provenance["source_manifest_sha256"] = source_snapshot_id.removeprefix("src-")
    return provenance


def _publish_private(tmp_path: Path, payload: dict) -> tuple[Path, Path, str]:
    cache_root = tmp_path / "private-cache"
    object_dir = cache_root / "object"
    object_dir.mkdir(parents=True)
    os.chmod(cache_root, 0o700)
    os.chmod(object_dir, 0o700)
    staged = object_dir / "staged.pt"
    torch.save(payload, staged)
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    checkpoint = object_dir / f"{digest}.pt"
    staged.rename(checkpoint)
    os.chmod(checkpoint, 0o400)
    os.chmod(object_dir, 0o500)
    return cache_root, checkpoint, digest


def _validate(cache_root: Path, checkpoint: Path, digest: str, **kwargs):
    return validate_controller_terminal_policy_init(
        checkpoint,
        cache_root=cache_root,
        expected_sha256=digest,
        required_target=8,
        **kwargs,
    )


def test_controller_accepts_private_exact_terminal_source(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())

    terminal = _validate(
        cache_root,
        checkpoint,
        digest,
        expected_world_size=1,
        expected_wandb_run_path="entity/project/replacement-canary",
    )

    assert terminal["completed_iteration"] == 7
    assert terminal["next_iteration"] == 8


def test_controller_rejects_nonprivate_file_mode(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())
    os.chmod(checkpoint, 0o444)

    with pytest.raises(ValueError, match="private file metadata contract"):
        _validate(cache_root, checkpoint, digest)


def test_controller_rejects_hardlinked_checkpoint(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())
    os.chmod(checkpoint.parent, 0o700)
    os.link(checkpoint, checkpoint.parent / "alias.pt")
    os.chmod(checkpoint.parent, 0o500)

    with pytest.raises(ValueError, match="private file metadata contract"):
        _validate(cache_root, checkpoint, digest)


def test_controller_rejects_symlink_checkpoint(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())
    os.chmod(checkpoint.parent, 0o700)
    real_checkpoint = checkpoint.with_name("real.pt")
    checkpoint.rename(real_checkpoint)
    checkpoint.symlink_to(real_checkpoint.name)
    os.chmod(checkpoint.parent, 0o500)

    with pytest.raises(OSError, match="no-follow regular file"):
        _validate(cache_root, checkpoint, digest)


def test_controller_rejects_writable_object_directory(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())
    os.chmod(checkpoint.parent, 0o700)

    with pytest.raises(ValueError, match="mode 0500"):
        _validate(cache_root, checkpoint, digest)


def test_controller_rejects_digest_basename_mismatch(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())
    os.chmod(checkpoint.parent, 0o700)
    wrong_name = checkpoint.with_name("wrong.pt")
    checkpoint.rename(wrong_name)
    os.chmod(checkpoint.parent, 0o500)

    with pytest.raises(ValueError, match="basename"):
        _validate(cache_root, wrong_name, digest)


def test_controller_rejects_internally_mutated_frozen_dataset(tmp_path):
    payload = _payload()
    payload["fixed_bc_eval_by_rank"]["0"]["actor_obs_raw"][0, 0] += 1.0
    cache_root, checkpoint, digest = _publish_private(tmp_path, payload)

    with pytest.raises(ValueError, match="does not authenticate.*frozen dataset"):
        _validate(cache_root, checkpoint, digest)


def test_controller_rejects_wrong_topology_and_wandb_identity(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())

    with pytest.raises(ValueError, match="world size"):
        _validate(cache_root, checkpoint, digest, expected_world_size=2)
    with pytest.raises(ValueError, match="W&B identity"):
        _validate(
            cache_root,
            checkpoint,
            digest,
            expected_wandb_run_path="entity/project/not-the-canary",
        )


def test_controller_fresh_source_gate_rejects_missing_provenance(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())

    with pytest.raises(ValueError, match="finalized training provenance"):
        _validate(cache_root, checkpoint, digest, require_fresh_source=True)


def test_controller_accepts_fresh_finalized_source_provenance(tmp_path):
    payload = _payload()
    payload["training_provenance"] = _fresh_finalized_provenance()
    cache_root, checkpoint, digest = _publish_private(tmp_path, payload)

    terminal = _validate(
        cache_root,
        checkpoint,
        digest,
        require_fresh_source=True,
    )

    assert terminal["run_target_iteration"] == 8


def test_controller_accepts_exact_fresh_source_snapshot_identity(tmp_path):
    payload = _payload()
    payload["training_provenance"] = _fresh_finalized_provenance(
        source_snapshot_id=_SOURCE_SNAPSHOT_ID,
    )
    cache_root, checkpoint, digest = _publish_private(tmp_path, payload)

    terminal = _validate(
        cache_root,
        checkpoint,
        digest,
        require_fresh_source=True,
        expected_source_snapshot_id=_SOURCE_SNAPSHOT_ID,
    )

    assert terminal["run_target_iteration"] == 8


@pytest.mark.parametrize(
    "recorded_source_snapshot_id",
    [None, "src-" + "6" * 64],
)
def test_controller_rejects_missing_or_wrong_expected_source_snapshot_identity(
    tmp_path,
    recorded_source_snapshot_id,
):
    payload = _payload()
    payload["training_provenance"] = _fresh_finalized_provenance(
        source_snapshot_id=recorded_source_snapshot_id,
    )
    cache_root, checkpoint, digest = _publish_private(tmp_path, payload)

    with pytest.raises(ValueError, match="source snapshot identity differs"):
        _validate(
            cache_root,
            checkpoint,
            digest,
            require_fresh_source=True,
            expected_source_snapshot_id=_SOURCE_SNAPSHOT_ID,
        )


def test_controller_expected_source_snapshot_requires_fresh_gate(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())

    with pytest.raises(ValueError, match="requires fresh-source provenance"):
        _validate(
            cache_root,
            checkpoint,
            digest,
            expected_source_snapshot_id=_SOURCE_SNAPSHOT_ID,
        )


def test_controller_rejects_malformed_expected_source_snapshot_identity(tmp_path):
    cache_root, checkpoint, digest = _publish_private(tmp_path, _payload())

    with pytest.raises(ValueError, match="src-<64 lowercase hexadecimal"):
        _validate(
            cache_root,
            checkpoint,
            digest,
            require_fresh_source=True,
            expected_source_snapshot_id="src-not-a-digest",
        )
