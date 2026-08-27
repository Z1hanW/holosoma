"""Minimal unit test for ONNX export functionality."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from onnx import TensorProto, helper
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    validate_exported_policy_onnx,
)


class ActorWrapper(nn.Module):
    """Wrapper matching PPO's actor_onnx_wrapper pattern."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": actor_obs})


class ActorWithPerceptionWrapper(nn.Module):
    def forward(self, actor_obs: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        return actor_obs[:, :1] + depth_features[:, :1]


class ValidationCopyTrackingWrapper(nn.Module):
    validation_calls: list[tuple[int, str, bool]] = []

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.cached_nonleaf = self.linear(torch.ones(1, 2))

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        type(self).validation_calls.append(
            (id(self), actor_obs.device.type, self.training)
        )
        return self.linear(actor_obs)


class ExplicitLSTMWrapper(nn.Module):
    onnx_input_names = ["actor_obs", "hidden_state", "cell_state"]
    onnx_output_names = ["action", "hidden_state_out", "cell_state_out"]
    onnx_dynamic_axes = {
        "actor_obs": {0: "batch"},
        "hidden_state": {1: "batch"},
        "cell_state": {1: "batch"},
        "action": {0: "batch"},
        "hidden_state_out": {1: "batch"},
        "cell_state_out": {1: "batch"},
    }

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, actor_obs, hidden_state, cell_state):
        output, (hidden_out, cell_out) = self.lstm(
            actor_obs.unsqueeze(0),
            (hidden_state, cell_state),
        )
        return self.head(output.squeeze(0)), hidden_out, cell_out


def test_export_policy_as_onnx():
    """Test ONNX export, load, and dimension verification."""
    OBS_DIM, ACT_DIM = 10, 5

    # Minimal config for PPOActor
    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[ACT_DIM],
        layer_config=LayerConfig(
            hidden_dims=[64],
            activation="ReLU",
            dropout_prob=0.0,
        ),
        min_noise_std=None,
        min_mean_noise_std=None,
    )

    # Create PPOActor
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": OBS_DIM},
        module_config=module_config,
        num_actions=ACT_DIM,
        init_noise_std=0.1,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    wrapper = ActorWrapper(actor)
    wrapper.eval()

    # Export to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "test_policy.onnx")
        example_obs = torch.zeros(1, OBS_DIM)

        export_policy_as_onnx(
            wrapper=wrapper,
            onnx_file_path=onnx_path,
            example_obs_dict={"actor_obs": example_obs},
        )

        # Load and verify
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Check input/output dims
        assert len(model.graph.input) == 1
        assert len(model.graph.output) == 1

        input_shape = model.graph.input[0].type.tensor_type.shape
        output_shape = model.graph.output[0].type.tensor_type.shape

        assert input_shape.dim[1].dim_value == OBS_DIM
        assert output_shape.dim[1].dim_value == ACT_DIM
        assert input_shape.dim[0].dim_param == "batch"
        assert output_shape.dim[0].dim_param == "batch"

        session = onnxruntime.InferenceSession(onnx_path)
        for batch_size in (1, 3):
            actor_obs = torch.randn(batch_size, OBS_DIM)
            with torch.no_grad():
                expected = wrapper(actor_obs).numpy()
            actual = session.run(["action"], {"actor_obs": actor_obs.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_export_policy_as_onnx_preserves_custom_perception_input_name(tmp_path):
    onnx_path = tmp_path / "custom_perception.onnx"

    export_policy_as_onnx(
        wrapper=ActorWithPerceptionWrapper(),
        onnx_file_path=str(onnx_path),
        example_obs_dict={
            "actor_obs": torch.zeros(1, 3),
            "depth_features": torch.zeros(1, 5),
        },
        perception_input_name="depth_features",
    )

    model = onnx.load(onnx_path)
    assert [value.name for value in model.graph.input] == ["actor_obs", "depth_features"]
    assert all(value.type.tensor_type.shape.dim[0].dim_param == "batch" for value in model.graph.input)
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"

    session = onnxruntime.InferenceSession(str(onnx_path))
    for batch_size in (1, 3):
        actor_obs = torch.randn(batch_size, 3)
        depth_features = torch.randn(batch_size, 5)
        with torch.no_grad():
            expected = ActorWithPerceptionWrapper()(actor_obs, depth_features).numpy()
        actual = session.run(
            ["action"],
            {
                "actor_obs": actor_obs.numpy(),
                "depth_features": depth_features.numpy(),
            },
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_validate_exported_policy_onnx_checks_runtime_and_numerical_parity(tmp_path):
    onnx_path = tmp_path / "validated.onnx"
    wrapper = ActorWithPerceptionWrapper().eval()
    examples = {
        "actor_obs": torch.zeros(1, 3),
        "depth_features": torch.zeros(1, 5),
    }
    export_policy_as_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
        perception_input_name="depth_features",
    )
    attach_onnx_metadata(str(onnx_path), {"iteration": 4})

    report = validate_exported_policy_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
        perception_input_name="depth_features",
    )

    assert report["checker"] == "onnx.checker.check_model"
    assert report["runtime"] == "onnxruntime_cpu"
    assert report["pytorch_vs_ort"] is True
    assert report["input_names"] == ["actor_obs", "depth_features"]
    assert report["output_names"] == ["action"]
    assert report["probe_rows"] == 6
    assert report["rtol"] == 1.0e-3
    assert report["atol"] == 1.0e-5


def test_export_and_validate_explicit_lstm_state_contract(tmp_path):
    wrapper = ExplicitLSTMWrapper().eval()
    examples = {
        "actor_obs": torch.zeros(1, 4),
        "hidden_state": torch.zeros(1, 1, 6),
        "cell_state": torch.zeros(1, 1, 6),
    }
    onnx_path = tmp_path / "explicit_lstm.onnx"
    export_policy_as_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
    )
    report = validate_exported_policy_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
        rtol=1.0e-4,
    )

    assert report["version"] == 2
    assert report["input_names"] == ExplicitLSTMWrapper.onnx_input_names
    assert report["output_names"] == ExplicitLSTMWrapper.onnx_output_names
    session = onnxruntime.InferenceSession(str(onnx_path))
    assert session.get_inputs()[1].shape == [1, "batch", 6]
    assert session.get_outputs()[1].shape == [1, "batch", 6]


def test_validate_exported_policy_onnx_accepts_bounded_float32_backend_drift(tmp_path):
    onnx_path = tmp_path / "bounded-drift.onnx"
    examples = {"actor_obs": torch.zeros(1, 2)}
    exported = nn.Linear(2, 1, bias=False).eval()
    live = nn.Linear(2, 1, bias=False).eval()
    with torch.no_grad():
        exported.weight.fill_(1.0)
        live.weight.fill_(1.0005)
    export_policy_as_onnx(
        wrapper=exported,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
    )

    report = validate_exported_policy_onnx(
        wrapper=live,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
    )

    assert report["pytorch_vs_ort"] is True
    assert 1.0e-4 < report["max_rel_error"] < 1.0e-3


def test_validate_exported_policy_onnx_rejects_different_live_actor(tmp_path):
    onnx_path = tmp_path / "mismatch.onnx"
    examples = {"actor_obs": torch.zeros(1, 2)}
    exported = nn.Linear(2, 1, bias=False).eval()
    live = nn.Linear(2, 1, bias=False).eval()
    with torch.no_grad():
        exported.weight.fill_(1.0)
        live.weight.fill_(2.0)
    export_policy_as_onnx(
        wrapper=exported,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
    )

    with pytest.raises(RuntimeError, match="failed PyTorch-vs-ORT action parity"):
        validate_exported_policy_onnx(
            wrapper=live,
            onnx_file_path=str(onnx_path),
            example_obs_dict=examples,
        )


def test_validate_exported_policy_onnx_uses_detached_eval_cpu_copy(tmp_path):
    onnx_path = tmp_path / "cpu-copy.onnx"
    wrapper = ValidationCopyTrackingWrapper().eval()
    examples = {"actor_obs": torch.zeros(1, 2)}
    export_policy_as_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
    )
    wrapper.train()
    cached_nonleaf = wrapper.cached_nonleaf
    ValidationCopyTrackingWrapper.validation_calls.clear()

    validate_exported_policy_onnx(
        wrapper=wrapper,
        onnx_file_path=str(onnx_path),
        example_obs_dict=examples,
    )

    assert wrapper.training is True
    assert wrapper.cached_nonleaf is cached_nonleaf
    assert ValidationCopyTrackingWrapper.validation_calls
    assert all(
        module_id != id(wrapper) and device == "cpu" and training is False
        for module_id, device, training in ValidationCopyTrackingWrapper.validation_calls
    )


def _write_identity_onnx(path: Path, metadata_entries: list[tuple[str, object]]) -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 1])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["actor_obs"], ["action"])
    model = helper.make_model(helper.make_graph([node], "metadata", [actor_obs], [action]))
    for key, value in metadata_entries:
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)
    onnx.save(model, path)


def test_attach_onnx_metadata_replaces_keys_without_creating_duplicates(tmp_path):
    onnx_path = tmp_path / "metadata.onnx"
    _write_identity_onnx(onnx_path, [("slot", 1), ("preserved", {"value": 3})])

    attach_onnx_metadata(str(onnx_path), {"slot": 2, "new": [4, 5]})

    entries = [(prop.key, json.loads(prop.value)) for prop in onnx.load(onnx_path).metadata_props]
    assert entries == [("preserved", {"value": 3}), ("slot", 2), ("new", [4, 5])]


def test_attach_onnx_metadata_rejects_nonfinite_json(tmp_path):
    onnx_path = tmp_path / "metadata.onnx"
    _write_identity_onnx(onnx_path, [("preserved", 1)])
    original = onnx_path.read_bytes()

    with pytest.raises(ValueError, match="not finite JSON data"):
        attach_onnx_metadata(str(onnx_path), {"iteration": float("nan")})

    assert onnx_path.read_bytes() == original


@pytest.mark.parametrize(
    ("existing_key", "existing_value", "message"),
    [
        ("", 1, "empty metadata key"),
        ("legacy", float("nan"), "not strict finite JSON"),
    ],
)
def test_attach_onnx_metadata_rejects_invalid_preexisting_metadata_without_rewriting(
    existing_key,
    existing_value,
    message,
    tmp_path,
):
    onnx_path = tmp_path / "invalid-existing-metadata.onnx"
    _write_identity_onnx(onnx_path, [(existing_key, existing_value)])
    original = onnx_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        attach_onnx_metadata(str(onnx_path), {"new": 2})

    assert onnx_path.read_bytes() == original


def test_ppo_export_uses_pure_policy_even_with_motion_command(tmp_path):
    """PPO export should not add motion replay tensors or a time_step input."""

    ppo = object.__new__(PPO)
    ppo.actor = nn.Linear(3, 2)
    ppo.device = "cpu"
    ppo.current_learning_iteration = 0
    ppo.actor_perception_key = ""
    ppo._get_zero_input = mock.MagicMock(return_value=torch.zeros(1, 3))
    ppo._get_zero_perception_input = mock.MagicMock(return_value=None)
    ppo._eval_mode = mock.MagicMock()
    ppo._train_mode = mock.MagicMock()
    ppo._checkpoint_metadata = mock.MagicMock(return_value={})
    ppo.logging_helper = SimpleNamespace(save_to_wandb=mock.MagicMock())
    motion_command = SimpleNamespace(
        get_motion_transition_contract=mock.MagicMock(
            return_value={
                "version": 1,
                "control_dt_s": 0.02,
                "source_semantics": "single_clip_static",
                "prepend": {"implementation": "none", "applied": False, "steps": 0},
                "append": {"implementation": "none", "applied": False, "steps": 0},
            }
        )
    )
    ppo.env = SimpleNamespace(
        command_manager=SimpleNamespace(get_state=mock.MagicMock(return_value=motion_command)),
        robot_config=SimpleNamespace(dof_names=[], control=SimpleNamespace(stiffness={}, damping={})),
    )

    with (
        mock.patch.object(PPO, "actor_onnx_wrapper", new_callable=mock.PropertyMock) as actor_onnx_wrapper,
        mock.patch("holosoma.agents.ppo.ppo.export_policy_as_onnx") as export_policy,
        mock.patch("holosoma.agents.ppo.ppo.attach_onnx_metadata"),
        mock.patch(
            "holosoma.agents.ppo.ppo.validate_exported_policy_onnx",
            return_value={
                "version": 1,
                "checker": "onnx.checker.check_model",
                "runtime": "onnxruntime_cpu",
                "pytorch_vs_ort": True,
                "input_names": ["actor_obs"],
                "output_names": ["action"],
                "probe_rows": 6,
                "rtol": 1.0e-4,
                "atol": 1.0e-5,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
            },
        ),
        mock.patch("holosoma.agents.ppo.ppo.get_control_gains_from_config", return_value=([], [])),
        mock.patch("holosoma.agents.ppo.ppo.get_command_ranges_from_env", return_value=None),
        mock.patch("holosoma.agents.ppo.ppo.get_urdf_text_from_robot_config", return_value=("", "")),
    ):
        actor_onnx_wrapper.return_value = nn.Linear(3, 2)
        ppo.export(str(tmp_path / "policy.onnx"))

    export_policy.assert_called_once()
    assert export_policy.call_args.kwargs["example_obs_dict"]["actor_obs"].shape == (1, 3)
    assert export_policy.call_args.kwargs["perception_input_name"] is None


if __name__ == "__main__":
    test_export_policy_as_onnx()
