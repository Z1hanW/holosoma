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
from holosoma.utils.inference_helpers import attach_onnx_metadata, export_policy_as_onnx


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
