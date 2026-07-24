from __future__ import annotations

import hashlib
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint
from holosoma.utils.rng_checkpoint import (
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
    validate_rng_checkpoint_state,
)


def _assert_same_state(left, right) -> None:
    assert left["python_random_state"] == right["python_random_state"]
    left_np = left["numpy_random_state"]
    right_np = right["numpy_random_state"]
    assert left_np[0] == right_np[0]
    assert torch.equal(left_np[1], right_np[1])
    assert left_np[2:] == right_np[2:]
    assert torch.equal(left["torch_cpu_rng_state"], right["torch_cpu_rng_state"])
    assert left["torch_cuda_visible_device_count"] == right["torch_cuda_visible_device_count"]
    assert left["torch_cuda_current_device"] == right["torch_cuda_current_device"]
    if left["torch_cuda_rng_state"] is None:
        assert right["torch_cuda_rng_state"] is None
    else:
        assert torch.equal(left["torch_cuda_rng_state"], right["torch_cuda_rng_state"])


def test_capture_restore_continues_python_numpy_and_torch_streams() -> None:
    original = capture_rng_checkpoint_state()
    try:
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        target = capture_rng_checkpoint_state()
        expected = (
            [random.random() for _ in range(4)],
            np.random.random(4),
            torch.rand(4),
        )

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restore_rng_checkpoint_state(target)
        actual = (
            [random.random() for _ in range(4)],
            np.random.random(4),
            torch.rand(4),
        )

        assert actual[0] == expected[0]
        assert np.array_equal(actual[1], expected[1])
        assert torch.equal(actual[2], expected[2])
    finally:
        restore_rng_checkpoint_state(original)


def test_validation_clones_mutable_rng_tensors_and_numpy_keys() -> None:
    source = capture_rng_checkpoint_state()
    validated = validate_rng_checkpoint_state(source)
    source["numpy_random_state"][1][0] ^= 1
    source["torch_cpu_rng_state"][0] ^= 1

    assert not torch.equal(
        source["numpy_random_state"][1],
        validated["numpy_random_state"][1],
    )
    assert not torch.equal(source["torch_cpu_rng_state"], validated["torch_cpu_rng_state"])


def test_invalid_rng_payload_is_rejected_without_global_mutation() -> None:
    before = capture_rng_checkpoint_state()
    corrupt = capture_rng_checkpoint_state()
    corrupt["numpy_random_state"] = (
        "MT19937",
        np.zeros(3, dtype=np.uint32),
        0,
        0,
        0.0,
    )

    with pytest.raises(ValueError, match=r"shape \(624,\)"):
        restore_rng_checkpoint_state(corrupt)

    _assert_same_state(before, capture_rng_checkpoint_state())


def test_rng_payload_survives_torch_checkpoint_round_trip(tmp_path) -> None:
    path = tmp_path / "rng.pt"
    expected = capture_rng_checkpoint_state()
    torch.save({"rng": expected}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)["rng"]
    actual = validate_rng_checkpoint_state(loaded)
    _assert_same_state(expected, actual)


def test_verified_weights_only_loader_authenticates_rng_checkpoint(tmp_path) -> None:
    path = tmp_path / "rng-verified.pt"
    expected_state = capture_rng_checkpoint_state()
    torch.save({"rng": expected_state}, path)
    expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded, actual_digest = load_verified_torch_checkpoint(
        path,
        expected_sha256=expected_digest,
    )

    assert actual_digest == expected_digest
    _assert_same_state(expected_state, validate_rng_checkpoint_state(loaded["rng"]))


def test_verified_loader_rejects_in_place_mutation_during_deserialization(tmp_path) -> None:
    path = tmp_path / "mutating.pt"
    torch.save({"value": torch.ones(1)}, path)
    expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def mutate_checkpoint(_stream, **_kwargs):
        with path.open("r+b") as writer:
            writer.seek(0)
            writer.write(b"mutated!")
            writer.flush()
        return {"value": torch.ones(1)}

    with (
        patch(
            "holosoma.utils.checkpoint_validation.torch.load",
            side_effect=mutate_checkpoint,
        ),
        pytest.raises(RuntimeError, match="changed while it was being safely deserialized"),
    ):
        load_verified_torch_checkpoint(path, expected_sha256=expected_digest)


def test_legacy_numpy_array_rng_keys_normalize_to_weights_only_safe_tensor() -> None:
    state = capture_rng_checkpoint_state()
    state["numpy_random_state"] = (
        state["numpy_random_state"][0],
        state["numpy_random_state"][1].numpy().astype(np.uint32, copy=True),
        *state["numpy_random_state"][2:],
    )

    normalized = validate_rng_checkpoint_state(state)

    assert isinstance(normalized["numpy_random_state"][1], torch.Tensor)
    assert normalized["numpy_random_state"][1].dtype == torch.int64


def test_rng_validation_rejects_cuda_visibility_drift() -> None:
    state = capture_rng_checkpoint_state()
    with pytest.raises(ValueError, match="CUDA visibility differs"):
        validate_rng_checkpoint_state(
            state,
            expected_cuda_device_count=state["torch_cuda_visible_device_count"] + 1,
            validate_cuda_generators=False,
        )


@pytest.mark.parametrize(
    ("visible_device_count", "current_device"),
    [(8, 3), (1, 0)],
)
def test_rng_metadata_supports_conventional_and_rank_visible_cuda_topologies(
    visible_device_count,
    current_device,
) -> None:
    state = capture_rng_checkpoint_state()
    state["torch_cuda_visible_device_count"] = visible_device_count
    state["torch_cuda_current_device"] = current_device
    state["torch_cuda_rng_state"] = torch.ones(16, dtype=torch.uint8)

    normalized = validate_rng_checkpoint_state(
        state,
        expected_cuda_device_count=visible_device_count,
        expected_cuda_device_index=current_device,
        # Structural preflight deliberately does not initialize a CUDA
        # generator; PPO.load performs the actual device-generator check.
        validate_cuda_generators=False,
    )

    assert normalized["torch_cuda_visible_device_count"] == visible_device_count
    assert normalized["torch_cuda_current_device"] == current_device
