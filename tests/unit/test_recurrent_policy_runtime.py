from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from holosoma_inference.policies.base import BasePolicy


def _metadata() -> dict:
    contract = {
        "version": 1,
        "kind": "lstm",
        "num_layers": 1,
        "hidden_dim": 4,
        "dtype": "float32",
        "state_input_names": ["hidden_state", "cell_state"],
        "state_output_names": ["hidden_state_out", "cell_state_out"],
        "state_shape": [1, "batch", 4],
        "state_batch_axis": 1,
        "step_semantics": "state_before_observation_to_state_after_observation",
        "reset_semantics": "zero_after_done_before_next_observation",
        "deployment_reset_events": [
            "episode_reset",
            "policy_start",
            "policy_stop",
            "policy_switch",
        ],
    }
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "recurrent_policy_contract": contract,
        "recurrent_policy_contract_sha256": hashlib.sha256(payload).hexdigest(),
    }


class _Session:
    def __init__(self, *, invalid_state: bool = False):
        self.invalid_state = invalid_state

    def run(self, names, feed):
        assert names == ["action", "hidden_state_out", "cell_state_out"]
        hidden = feed["hidden_state"]
        cell = feed["cell_state"]
        action = np.full((1, 2), hidden.mean() + cell.mean(), dtype=np.float32)
        hidden_out = hidden + np.float32(1.0)
        cell_out = cell + np.float32(2.0)
        if self.invalid_state:
            hidden_out = np.full_like(hidden_out, np.nan)
        return [action, hidden_out, cell_out]


def _policy(session: _Session) -> BasePolicy:
    policy = object.__new__(BasePolicy)
    policy.onnx_policy_session = session
    policy.onnx_input_names = ["actor_obs", "hidden_state", "cell_state"]
    policy.onnx_output_names = ["action", "hidden_state_out", "cell_state_out"]
    policy.obs_dict = {"actor_obs": []}
    policy.observation_clip = 100.0
    policy._obs_input_name = "actor_obs"
    policy._perception_obs_input_name = None
    policy._onnx_metadata = {}
    policy._configure_policy_recurrent_state(_metadata())
    return policy


def test_runtime_advances_and_resets_explicit_lstm_state() -> None:
    policy = _policy(_Session())
    feed = {"actor_obs": np.zeros((1, 3), dtype=np.float32)}

    first = policy._run_policy_onnx(feed, ["action"])["action"]
    second = policy._run_policy_onnx(feed, ["action"])["action"]
    assert np.array_equal(first, np.zeros((1, 2), dtype=np.float32))
    assert np.array_equal(second, np.full((1, 2), 3.0, dtype=np.float32))
    assert np.array_equal(policy._policy_recurrent_state["hidden_state"], np.full((1, 1, 4), 2.0))
    assert np.array_equal(policy._policy_recurrent_state["cell_state"], np.full((1, 1, 4), 4.0))

    policy._reset_policy_recurrent_state()
    assert not np.any(policy._policy_recurrent_state["hidden_state"])
    assert not np.any(policy._policy_recurrent_state["cell_state"])


def test_runtime_rejects_invalid_state_without_partial_commit() -> None:
    policy = _policy(_Session(invalid_state=True))
    before = {name: value.copy() for name, value in policy._policy_recurrent_state.items()}
    with pytest.raises(FloatingPointError, match="non-finite"):
        policy._run_policy_onnx(
            {"actor_obs": np.zeros((1, 3), dtype=np.float32)},
            ["action"],
        )
    for name, value in before.items():
        assert np.array_equal(policy._policy_recurrent_state[name], value)
