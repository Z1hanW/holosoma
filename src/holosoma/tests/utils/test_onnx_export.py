"""Minimal unit test for ONNX export functionality."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import onnx
import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.inference_helpers import export_policy_as_onnx


class ActorWrapper(nn.Module):
    """Wrapper matching PPO's actor_onnx_wrapper pattern."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": actor_obs})


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
    ppo.env = SimpleNamespace(
        command_manager=SimpleNamespace(get_state=mock.MagicMock(return_value=object())),
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


if __name__ == "__main__":
    test_export_policy_as_onnx()
