from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from holosoma.agents.modules.data_utils import RolloutStorage


def _filled_storage(*, num_envs: int, num_steps: int) -> RolloutStorage:
    storage = RolloutStorage(num_envs, num_steps)
    storage.register("index", shape=(1,), dtype=torch.long)
    values = torch.arange(num_envs * num_steps).view(num_steps, num_envs, 1)
    for step in range(num_steps):
        storage.add(index=values[step])
    return storage


def test_add_rejects_unknown_or_missing_transition_fields_without_partial_write():
    storage = RolloutStorage(num_envs=2, num_transitions_per_env=1)
    storage.register("obs", shape=(1,))
    storage.register("actions", shape=(1,))

    with pytest.raises(KeyError, match="unregistered rollout fields"):
        storage.add(
            obs=torch.ones(2, 1),
            actions=torch.ones(2, 1),
            typo=torch.ones(2, 1),
        )
    assert storage.step == 0
    assert torch.equal(storage["obs"], torch.zeros_like(storage["obs"]))
    assert torch.equal(storage["actions"], torch.zeros_like(storage["actions"]))

    with pytest.raises(KeyError, match="missing required rollout fields"):
        storage.add(obs=torch.ones(2, 1))
    assert storage.step == 0
    assert torch.equal(storage["obs"], torch.zeros_like(storage["obs"]))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.ones(2), "has shape"),
        (torch.ones(2, 1, dtype=torch.float64), "has dtype"),
    ],
)
def test_add_rejects_implicit_shape_or_dtype_conversion(value, message):
    storage = RolloutStorage(num_envs=2, num_transitions_per_env=1)
    storage.register("obs", shape=(1,), dtype=torch.float32)

    with pytest.raises(ValueError, match=message):
        storage.add(obs=value)

    assert storage.step == 0
    assert torch.equal(storage["obs"], torch.zeros_like(storage["obs"]))


def test_derived_fields_require_complete_post_rollout_write():
    storage = RolloutStorage(num_envs=2, num_transitions_per_env=2)
    storage.register("obs", shape=(1,))
    storage.register("returns", shape=(1,), required_on_add=False)
    assert storage.required_on_add_keys == frozenset({"obs"})
    assert storage.derived_keys == frozenset({"returns"})

    storage.add(obs=torch.ones(2, 1))
    storage.add(obs=torch.full((2, 1), 2.0))
    with pytest.raises(RuntimeError, match="Derived fields were not populated|derived fields were not populated|Rollout-derived fields"):
        list(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    with pytest.raises(RuntimeError, match="Rollout-derived fields"):
        list(storage.sequence_mini_batch_generator(num_mini_batches=1, num_epochs=1))

    expected_returns = torch.full((2, 2, 1), 3.0)
    storage["returns"] = expected_returns
    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    assert torch.equal(batch["returns"], expected_returns.flatten(0, 1))


def test_derived_fields_cannot_be_written_transition_by_transition():
    storage = RolloutStorage(num_envs=1, num_transitions_per_env=1)
    storage.register("obs", shape=(1,))
    storage.register("advantages", shape=(1,), required_on_add=False)

    with pytest.raises(ValueError, match="Derived rollout fields"):
        storage.add(obs=torch.ones(1, 1), advantages=torch.ones(1, 1))
    assert storage.step == 0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.ones(2, 1), "has shape"),
        (torch.ones(1, 2, 1, dtype=torch.float64), "has dtype"),
    ],
)
def test_complete_buffer_write_rejects_implicit_shape_or_dtype_conversion(value, message):
    storage = RolloutStorage(num_envs=2, num_transitions_per_env=1)
    storage.register("obs", shape=(1,), dtype=torch.float32)
    storage.register("returns", shape=(1,), dtype=torch.float32, required_on_add=False)
    storage.add(obs=torch.ones(2, 1))

    with pytest.raises(ValueError, match=message):
        storage["returns"] = value

    assert torch.equal(storage["returns"], torch.zeros_like(storage["returns"]))


def test_derived_field_cannot_be_marked_ready_before_collection_is_complete():
    storage = RolloutStorage(num_envs=1, num_transitions_per_env=2)
    storage.register("obs", shape=(1,))
    storage.register("returns", shape=(1,), required_on_add=False)
    storage.add(obs=torch.tensor([[1.0]]))

    with pytest.raises(RuntimeError, match="only be written after collection is complete"):
        storage["returns"] = torch.ones(2, 1, 1)

    storage.add(obs=torch.tensor([[2.0]]))
    with pytest.raises(RuntimeError, match="Rollout-derived fields"):
        list(storage.mini_batch_generator(1, 1))


def test_register_rejects_schema_mutation_after_collection_starts():
    storage = RolloutStorage(num_envs=1, num_transitions_per_env=2)
    storage.register("obs", shape=(1,))
    storage.add(obs=torch.tensor([[1.0]]))

    with pytest.raises(RuntimeError, match="before transition collection starts"):
        storage.register("late", shape=(1,))

    storage.clear()
    with pytest.raises(RuntimeError, match="schema_frozen=True"):
        storage.register("still_late", shape=(1,))


@pytest.mark.parametrize("sequence", [False, True])
def test_minibatch_generators_reject_partially_filled_storage(sequence):
    storage = RolloutStorage(num_envs=2, num_transitions_per_env=2)
    storage.register("obs", shape=(1,))
    storage.add(obs=torch.ones(2, 1))

    generator = (
        storage.sequence_mini_batch_generator(1, 1)
        if sequence
        else storage.mini_batch_generator(1, 1)
    )
    with pytest.raises(RuntimeError, match="completely filled"):
        list(generator)


def test_minibatch_generator_rejects_empty_or_unknown_requested_keys():
    storage = _filled_storage(num_envs=2, num_steps=1)

    with pytest.raises(ValueError, match="At least one rollout field"):
        list(storage.mini_batch_generator(1, 1, keys=set()))
    with pytest.raises(KeyError, match="unregistered rollout fields"):
        list(storage.mini_batch_generator(1, 1, keys={"index", "misspelled"}))


def test_clear_invalidates_and_zeroes_derived_fields_before_next_rollout():
    storage = RolloutStorage(num_envs=1, num_transitions_per_env=1)
    storage.register("obs", shape=(1,))
    storage.register("returns", shape=(1,), required_on_add=False)
    storage.add(obs=torch.tensor([[1.0]]))
    storage["returns"] = torch.tensor([[[7.0]]])
    list(storage.mini_batch_generator(1, 1))

    storage.clear()
    assert storage.step == 0
    assert torch.equal(storage["returns"], torch.zeros_like(storage["returns"]))
    storage.add(obs=torch.tensor([[2.0]]))
    with pytest.raises(RuntimeError, match="Rollout-derived fields"):
        list(storage.mini_batch_generator(1, 1))


def test_minibatch_generator_rejects_tail_dropping_configuration():
    storage = _filled_storage(num_envs=2, num_steps=5)

    with pytest.raises(ValueError, match="no samples are dropped"):
        list(storage.mini_batch_generator(num_mini_batches=3, num_epochs=1))


def test_minibatch_generator_reshuffles_each_epoch(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_CONTIGUOUS_MINIBATCHES", raising=False)
    storage = _filled_storage(num_envs=2, num_steps=2)
    permutations = [torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])]

    with patch(
        "holosoma.agents.modules.data_utils.torch.randperm",
        side_effect=permutations,
    ) as randperm:
        batches = list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=2))

    assert randperm.call_count == 2
    epoch_one = torch.cat([batch["index"].view(-1) for batch in batches[:2]])
    epoch_two = torch.cat([batch["index"].view(-1) for batch in batches[2:]])
    assert torch.equal(epoch_one, permutations[0])
    assert torch.equal(epoch_two, permutations[1])


def test_sequence_minibatch_generator_reshuffles_envs_each_epoch():
    storage = _filled_storage(num_envs=4, num_steps=1)
    permutations = [torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])]

    with patch(
        "holosoma.agents.modules.data_utils.torch.randperm",
        side_effect=permutations,
    ) as randperm:
        batches = list(storage.sequence_mini_batch_generator(num_mini_batches=2, num_epochs=2))

    assert randperm.call_count == 2
    epoch_one = torch.cat([batch["index"].view(-1) for batch in batches[:2]])
    epoch_two = torch.cat([batch["index"].view(-1) for batch in batches[2:]])
    assert torch.equal(epoch_one, permutations[0])
    assert torch.equal(epoch_two, permutations[1])
