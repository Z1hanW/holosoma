from __future__ import annotations

import hashlib
import os
from unittest.mock import patch

import pytest
import torch

from holosoma.utils.checkpoint_validation import (
    CheckpointFileSecurityContract,
    canonical_student_policy_type,
    load_verified_torch_checkpoint,
    validate_student_actor_contract,
)


@pytest.mark.parametrize(
    ("saved_type", "expected"),
    [
        ("MLP", "mlp"),
        ("modules.MLP", "mlp"),
        ("MLPPerceptionEncoder", "mlp"),
        ("holosoma.agents.modules.MLPPerceptionEncoder", "mlp"),
        ("FlowMLP", "flow"),
        ("FlowMLPPerceptionEncoder", "flow"),
        ("holosoma.agents.modules.FlowMLPPerceptionEncoder", "flow"),
    ],
)
def test_canonical_student_policy_type_accepts_persisted_runtime_types(
    saved_type: str,
    expected: str,
) -> None:
    assert canonical_student_policy_type(saved_type) == expected


@pytest.mark.parametrize(
    "saved_type",
    [None, "", "   ", "NotAFlowPolicy", "FlowTransformer", "MLPExtra"],
)
def test_canonical_student_policy_type_rejects_missing_or_lookalike_types(
    saved_type: object,
) -> None:
    with pytest.raises(ValueError, match="Unsupported actor type"):
        canonical_student_policy_type(saved_type)


def _actor_contract(*, actor_type: str = "MLPPerceptionEncoder") -> dict:
    return {
        "type": actor_type,
        "input_dim": ["root", "proprio"],
        "layer_config": {
            "hidden_dims": [128, 64],
            "flow_integration_steps": 4,
            "flow_train_noise_std": 1.0,
            "flow_time_epsilon": 1e-4,
            "flow_inference_noise_std": 0.0,
        },
    }


def test_validate_student_actor_contract_canonicalizes_flow_runtime_type() -> None:
    actor = _actor_contract(actor_type="FlowMLPPerceptionEncoder")
    actor["layer_config"]["flow_integration_steps"] = 8
    actor["layer_config"]["flow_time_epsilon"] = 0.01

    contract = validate_student_actor_contract(actor)

    assert contract == {
        "policy_type": "flow",
        "hidden_dims": (128, 64),
        "actor_inputs": ("root", "proprio"),
        "flow_steps": 8,
        "flow_train_noise": 1.0,
        "flow_epsilon": 0.01,
        "flow_inference_noise": 0.0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda actor: actor["layer_config"].update(hidden_dims=[True, 64]), "hidden dims"),
        (lambda actor: actor.update(input_dim=["root", "root"]), "input groups"),
        (
            lambda actor: actor["layer_config"].update(flow_integration_steps=True),
            "flow_integration_steps",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_integration_steps=4097),
            "flow_integration_steps",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_train_noise_std=float("nan")),
            "flow_train_noise_std",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_train_noise_std=1.0e19),
            "flow_train_noise_std",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_train_noise_std=10**400),
            "flow_train_noise_std",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_time_epsilon=0.5),
            "flow_time_epsilon",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_inference_noise_std=float("inf")),
            "flow_inference_noise_std",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_inference_noise_std=1.0e19),
            "flow_inference_noise_std",
        ),
        (
            lambda actor: actor["layer_config"].update(flow_integration_steps=8),
            "Non-default persisted flow settings",
        ),
    ],
)
def test_validate_student_actor_contract_rejects_malformed_persisted_schema(
    mutation,
    message: str,
) -> None:
    actor = _actor_contract()
    mutation(actor)

    with pytest.raises(ValueError, match=message):
        validate_student_actor_contract(actor)


def _private_contract() -> CheckpointFileSecurityContract:
    return CheckpointFileSecurityContract(
        owner_uid=os.geteuid(),
        mode=0o400,
        link_count=1,
        minimum_size=1,
        bind_pathname=True,
    )


def test_verified_loader_accepts_private_bound_checkpoint(tmp_path):
    checkpoint = tmp_path / "private.pt"
    torch.save({"value": torch.ones(1)}, checkpoint)
    os.chmod(checkpoint, 0o400)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    payload, actual_digest = load_verified_torch_checkpoint(
        checkpoint,
        expected_sha256=digest,
        file_security=_private_contract(),
    )

    assert actual_digest == digest
    assert torch.equal(payload["value"], torch.ones(1))


def test_verified_loader_private_contract_rejects_wrong_mode(tmp_path):
    checkpoint = tmp_path / "public.pt"
    torch.save({"value": torch.ones(1)}, checkpoint)
    os.chmod(checkpoint, 0o444)

    with pytest.raises(ValueError, match="private file metadata contract"):
        load_verified_torch_checkpoint(
            checkpoint,
            file_security=_private_contract(),
        )


def test_verified_loader_private_contract_rejects_hardlink(tmp_path):
    checkpoint = tmp_path / "linked.pt"
    alias = tmp_path / "alias.pt"
    torch.save({"value": torch.ones(1)}, checkpoint)
    os.chmod(checkpoint, 0o400)
    os.link(checkpoint, alias)

    with pytest.raises(ValueError, match="private file metadata contract"):
        load_verified_torch_checkpoint(
            checkpoint,
            file_security=_private_contract(),
        )


def test_verified_loader_private_contract_rejects_symlink(tmp_path):
    checkpoint = tmp_path / "real.pt"
    alias = tmp_path / "alias.pt"
    torch.save({"value": torch.ones(1)}, checkpoint)
    os.chmod(checkpoint, 0o400)
    alias.symlink_to(checkpoint)

    with pytest.raises(OSError, match="no-follow regular file"):
        load_verified_torch_checkpoint(
            alias,
            file_security=_private_contract(),
        )


def test_verified_loader_private_contract_rejects_path_replacement(tmp_path):
    checkpoint = tmp_path / "replaced.pt"
    torch.save({"value": torch.ones(1)}, checkpoint)
    os.chmod(checkpoint, 0o400)
    original_bytes = checkpoint.read_bytes()
    digest = hashlib.sha256(original_bytes).hexdigest()

    def replace_path(_stream, **_kwargs):
        checkpoint.unlink()
        checkpoint.write_bytes(original_bytes)
        os.chmod(checkpoint, 0o400)
        return {"value": torch.ones(1)}

    with (
        patch(
            "holosoma.utils.checkpoint_validation.torch.load",
            side_effect=replace_path,
        ),
        pytest.raises(RuntimeError, match="changed while it was being safely deserialized|pathname changed"),
    ):
        load_verified_torch_checkpoint(
            checkpoint,
            expected_sha256=digest,
            file_security=_private_contract(),
        )
