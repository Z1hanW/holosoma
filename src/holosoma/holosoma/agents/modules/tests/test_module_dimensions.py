"""Tests for module dimension calculations with history length.

This test suite verifies that:
1. BaseModule correctly computes input dimensions when obs_dim_dict includes history
2. PPO's _get_obs_dim correctly computes dimensions
3. Network input dimensions match what is stored in replay buffers/storage
"""

import hashlib
import os
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from holosoma.agents.modules import modules as module_impl
from holosoma.agents.modules.modules import (
    AttentionLinearEncoder,
    BaseModule,
    ConditionalFlowMLP,
    DeFMEfficientNetB2Encoder,
    DeFMRegNetY800MFEncoder,
    DeFMViTS14Encoder,
    FarTrackingDepthSmallEncoder,
    FarTrackingDepthSpatialSoftmaxEncoder,
    SpatialSoftmax2d,
)
from holosoma.agents.modules.ppo_modules import PPOActor, PPOActorEncoder, PPOCriticEncoder
from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.config_values import perception as perception_presets
from holosoma.config_values.wbt.g1 import experiment as g1_experiments
from holosoma.config_values.wbt.g1 import observation as g1_observations
from holosoma.perception.config_utils import apply_perception_overrides


@pytest.fixture
def simple_module_config():
    """Create a simple MLP module configuration."""
    layer_config = LayerConfig(
        hidden_dims=[256, 128],
        activation="ELU",
    )

    return ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[10],  # e.g., number of actions
        layer_config=layer_config,
    )


def test_ppo_actor_optional_max_noise_std_bounds_distribution():
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[2],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
        min_noise_std=0.05,
        max_noise_std=0.35,
    )
    actor = PPOActorEncoder(
        obs_dim_dict={"actor_obs": 4},
        module_config_dict=config,
        num_actions=2,
        init_noise_std=0.25,
        history_length={"actor_obs": 1},
    )
    with torch.no_grad():
        actor.std.copy_(torch.tensor([0.20, 0.80]))

    actor.update_distribution(torch.zeros(3, 4))

    assert torch.allclose(actor.action_std[0], torch.tensor([0.20, 0.35]))


def test_ppo_actor_does_not_mutate_shared_symbolic_action_dimension_config():
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=["robot_action_dim"],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
    )

    actor_two = PPOActor(
        obs_dim_dict={"actor_obs": 4},
        module_config_dict=config,
        num_actions=2,
        init_noise_std=0.25,
        history_length={"actor_obs": 1},
    )
    actor_three = PPOActor(
        obs_dim_dict={"actor_obs": 4},
        module_config_dict=config,
        num_actions=3,
        init_noise_std=0.25,
        history_length={"actor_obs": 1},
    )

    assert config.output_dim == ["robot_action_dim"]
    assert actor_two.act_inference({"actor_obs": torch.zeros(1, 4)}).shape == (1, 2)
    assert actor_three.act_inference({"actor_obs": torch.zeros(1, 4)}).shape == (1, 3)


def test_ppo_actor_without_max_noise_std_keeps_legacy_distribution_scale():
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[2],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
        min_noise_std=0.05,
    )
    actor = PPOActorEncoder(
        obs_dim_dict={"actor_obs": 4},
        module_config_dict=config,
        num_actions=2,
        init_noise_std=0.25,
        history_length={"actor_obs": 1},
    )
    with torch.no_grad():
        actor.std.copy_(torch.tensor([0.20, 0.80]))

    actor.update_distribution(torch.zeros(3, 4))

    assert torch.allclose(actor.action_std[0], torch.tensor([0.20, 0.80]))


def test_ppo_actor_std_projection_never_clamps_non_finite_values_to_finite():
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[3],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
        min_noise_std=0.05,
        max_noise_std=0.35,
    )
    actor = PPOActorEncoder(
        obs_dim_dict={"actor_obs": 4},
        module_config_dict=config,
        num_actions=3,
        init_noise_std=0.25,
        history_length={"actor_obs": 1},
    )
    with torch.no_grad():
        actor.std.copy_(torch.tensor([float("nan"), float("inf"), float("-inf")]))

    projected = actor._safe_std()
    sanitized_scale = actor._sanitize_scale(actor.std)
    actor.update_distribution(torch.zeros(1, 4))

    assert torch.isnan(projected[0]) and torch.isposinf(projected[1]) and torch.isneginf(projected[2])
    assert torch.isnan(sanitized_scale[0])
    assert torch.isposinf(sanitized_scale[1])
    assert torch.isneginf(sanitized_scale[2])
    assert torch.isnan(actor.action_std[0, 0])
    assert torch.isposinf(actor.action_std[0, 1])
    assert torch.isneginf(actor.action_std[0, 2])


def test_ppo_actor_distribution_never_sanitizes_non_finite_mean():
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[2],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
        min_noise_std=0.05,
    )
    actor = PPOActorEncoder(
        obs_dim_dict={"actor_obs": 4},
        module_config_dict=config,
        num_actions=2,
        init_noise_std=0.25,
        history_length={"actor_obs": 1},
    )
    final_linear = [module for module in actor.modules() if isinstance(module, nn.Linear)][-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.copy_(torch.tensor([float("nan"), float("inf")]))

    actor.update_distribution(torch.zeros(1, 4))
    actor._sanitize_distribution()

    assert torch.isnan(actor.action_mean[0, 0])
    assert torch.isposinf(actor.action_mean[0, 1])


def test_ppo_actor_max_noise_std_remains_hard_cap_after_mean_floor_rescale():
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[2],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
        min_mean_noise_std=0.8,
        max_noise_std=0.8,
    )
    actor = PPOActorEncoder(
        obs_dim_dict={"actor_obs": 4},
        module_config_dict=config,
        num_actions=2,
        init_noise_std=0.25,
        history_length={"actor_obs": 1},
    )
    with torch.no_grad():
        actor.std.copy_(torch.tensor([1e-6, 0.8]))

    actor.update_distribution(torch.zeros(3, 4))

    assert torch.all(actor.action_std <= 0.8)
    assert actor.action_std[0].mean().item() >= 0.8 - 1e-6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_noise_std", 0.0),
        ("min_noise_std", float("nan")),
        ("min_mean_noise_std", -0.1),
        ("min_mean_noise_std", float("inf")),
        ("max_noise_std", 0.0),
        ("max_noise_std", float("nan")),
    ],
)
def test_ppo_actor_rejects_invalid_noise_constraint(field, value):
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[2],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
        **{field: value},
    )

    with pytest.raises(ValueError, match=field):
        PPOActorEncoder(
            obs_dim_dict={"actor_obs": 4},
            module_config_dict=config,
            num_actions=2,
            init_noise_std=0.25,
            history_length={"actor_obs": 1},
        )


@pytest.mark.parametrize("init_noise_std", [0.0, -0.1, float("nan"), float("inf")])
def test_ppo_actor_rejects_invalid_initial_noise_std(init_noise_std):
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[2],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
    )

    with pytest.raises(ValueError, match="init_noise_std"):
        PPOActorEncoder(
            obs_dim_dict={"actor_obs": 4},
            module_config_dict=config,
            num_actions=2,
            init_noise_std=init_noise_std,
            history_length={"actor_obs": 1},
        )


def test_ppo_actor_rejects_ambiguous_component_and_mean_noise_floors():
    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[2],
        layer_config=LayerConfig(hidden_dims=[8], activation="ELU"),
        min_noise_std=0.05,
        min_mean_noise_std=0.1,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        PPOActorEncoder(
            obs_dim_dict={"actor_obs": 4},
            module_config_dict=config,
            num_actions=2,
            init_noise_std=0.25,
            history_length={"actor_obs": 1},
        )


def test_base_module_input_dim_with_history(simple_module_config):
    """Test that BaseModule doesn't multiply by history when obs_dim_dict already includes it."""
    # Simulate obs_dim_dict from observation_manager.get_obs_dims()
    # which already includes history (e.g., 100 single-frame * 4 history = 400)
    obs_dim_dict = {
        "actor_obs": 400,  # Already includes history
    }

    history_length = {
        "actor_obs": 4,
    }

    # Create BaseModule
    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length,
    )

    # The input dimension should be 400, NOT 400 * 4 = 1600
    assert module.input_dim == 400, (
        f"Input dimension should be 400 (already includes history), but got {module.input_dim}"
    )


def test_base_module_input_dim_multiple_keys():
    """Test input dimension calculation with multiple observation keys."""
    # Create config with multiple input keys
    layer_config = LayerConfig(
        hidden_dims=[256, 128],
        activation="ELU",
    )

    config = ModuleConfig(
        type="MLP",
        input_dim=["actor_state_obs", "perception_obs"],
        output_dim=[10],
        layer_config=layer_config,
    )

    obs_dim_dict = {
        "actor_state_obs": 200,  # 50 * 4 history
        "perception_obs": 800,  # 200 * 4 history
    }

    history_length = {
        "actor_state_obs": 4,
        "perception_obs": 4,
    }

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        history_length=history_length,
    )

    # Should sum the dimensions without multiplying by history again
    assert module.input_dim == 1000  # 200 + 800


# Skipping numeric input tests as ModuleConfig.input_dim only accepts List[str]
# and numeric inputs don't seem to be used in practice


def test_base_module_input_slices():
    """Test that input slices are correctly computed."""
    layer_config = LayerConfig(
        hidden_dims=[256, 128],
        activation="ELU",
    )

    config = ModuleConfig(
        type="MLP",
        input_dim=["obs_a", "obs_b"],
        output_dim=[10],
        layer_config=layer_config,
    )

    obs_dim_dict = {
        "obs_a": 100,
        "obs_b": 200,
    }

    history_length = {
        "obs_a": 1,
        "obs_b": 1,
    }

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        history_length=history_length,
    )

    # Check slices
    assert module.input_indices_dict["obs_a"] == slice(0, 100)
    assert module.input_indices_dict["obs_b"] == slice(100, 300)


def test_base_module_network_creation(simple_module_config):
    """Test that the network is created with correct input/output dimensions."""
    obs_dim_dict = {"actor_obs": 400}
    history_length = {"actor_obs": 4}

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length,
    )

    # Check that the network was created
    assert hasattr(module, "module")

    # Verify the first layer has correct input dimension
    first_layer = module.module[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 400

    # Verify the last layer has correct output dimension
    last_layer = module.module[-1]
    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 10


def test_base_module_forward_pass(simple_module_config):
    """Test that forward pass works with correct dimensions."""
    obs_dim_dict = {"actor_obs": 400}
    history_length = {"actor_obs": 4}

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length,
    )

    # Create input tensor
    batch_size = 16
    input_tensor = torch.randn(batch_size, 400)

    # Forward pass
    output = module.module(input_tensor)

    # Check output shape
    assert output.shape == (batch_size, 10)


def test_flow_mlp_module_forward_and_flow_loss():
    config = ModuleConfig(
        type="FlowMLP",
        input_dim=["actor_obs"],
        output_dim=[6],
        layer_config=LayerConfig(
            hidden_dims=[32, 32],
            activation="ELU",
            flow_integration_steps=2,
        ),
    )
    module = BaseModule(
        obs_dim_dict={"actor_obs": 12},
        module_config_dict=config,
        history_length={"actor_obs": 1},
    )

    assert isinstance(module.module, ConditionalFlowMLP)
    obs = torch.randn(5, 12)
    target = torch.randn(5, 6)

    actions = module(obs)
    losses = module.flow_matching_loss(obs, target)

    assert actions.shape == (5, 6)
    assert losses.shape == (5,)
    assert torch.isfinite(losses).all()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("flow_integration_steps", True, "flow_integration_steps"),
        ("flow_integration_steps", 3.7, "flow_integration_steps"),
        ("flow_integration_steps", "4", "flow_integration_steps"),
        ("flow_integration_steps", 0, "flow_integration_steps"),
        ("flow_integration_steps", 4097, "flow_integration_steps"),
        ("flow_train_noise_std", True, "flow_train_noise_std"),
        ("flow_train_noise_std", "1.0", "flow_train_noise_std"),
        ("flow_train_noise_std", float("nan"), "flow_train_noise_std"),
        ("flow_train_noise_std", float("inf"), "flow_train_noise_std"),
        ("flow_train_noise_std", 10**400, "flow_train_noise_std"),
        ("flow_train_noise_std", 1.0e19, "flow_train_noise_std"),
        ("flow_train_noise_std", -0.1, "flow_train_noise_std"),
        ("flow_time_epsilon", False, "flow_time_epsilon"),
        ("flow_time_epsilon", "0.1", "flow_time_epsilon"),
        ("flow_time_epsilon", float("nan"), "flow_time_epsilon"),
        ("flow_time_epsilon", float("inf"), "flow_time_epsilon"),
        ("flow_time_epsilon", -0.1, "flow_time_epsilon"),
        ("flow_time_epsilon", 0.5, "flow_time_epsilon"),
        ("flow_inference_noise_std", True, "flow_inference_noise_std"),
        ("flow_inference_noise_std", "0.0", "flow_inference_noise_std"),
        ("flow_inference_noise_std", float("nan"), "flow_inference_noise_std"),
        ("flow_inference_noise_std", float("inf"), "flow_inference_noise_std"),
        ("flow_inference_noise_std", 1.0e19, "flow_inference_noise_std"),
        ("flow_inference_noise_std", -0.1, "flow_inference_noise_std"),
    ],
)
def test_flow_mlp_rejects_invalid_numerical_configuration(field, value, message):
    raw_config = vars(LayerConfig(hidden_dims=[16, 16], activation="ELU")).copy()
    raw_config[field] = value
    layer_config = types.SimpleNamespace(**raw_config)

    with pytest.raises(ValueError, match=message):
        ConditionalFlowMLP(condition_dim=4, action_dim=2, layer_config=layer_config)


def test_flow_mlp_perception_encoder_actor_forward_and_loss():
    config = ModuleConfig(
        type="FlowMLPPerceptionEncoder",
        input_dim=["actor_obs"],
        output_dim=[4],
        layer_config=LayerConfig(
            hidden_dims=[32, 32],
            activation="ELU",
            module_input_name=("actor_obs",),
            perception_input_name="perception_obs",
            perception_output_dim=8,
            perception_encoder_type="gated_linear",
            flow_integration_steps=2,
        ),
    )
    actor = PPOActorEncoder(
        obs_dim_dict={"actor_obs": 10, "perception_obs": 16},
        module_config_dict=config,
        num_actions=4,
        init_noise_std=0.1,
        history_length={"actor_obs": 1},
    )
    policy_state = {
        "actor_obs": torch.randn(3, 10),
        "perception_obs": torch.randn(3, 16),
    }
    target = torch.randn(3, 4)

    actions = actor.act_inference(policy_state)
    losses = actor.flow_matching_loss(policy_state, target)

    assert actions.shape == (3, 4)
    assert losses.shape == (3,)
    assert torch.isfinite(losses).all()


@pytest.mark.parametrize(
    ("history_length", "single_frame_dim"),
    [
        (1, 100),
        (2, 100),
        (4, 100),
        (8, 100),
        (1, 50),
        (4, 200),
    ],
)
def test_various_history_configurations(simple_module_config, history_length, single_frame_dim):
    """Test module creation with various history length and dimension combinations."""
    obs_dim_with_history = single_frame_dim * history_length

    obs_dim_dict = {"actor_obs": obs_dim_with_history}
    history_length_dict = {"actor_obs": history_length}

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=simple_module_config,
        history_length=history_length_dict,
    )

    # Input dimension should match obs_dim_with_history
    assert module.input_dim == obs_dim_with_history

    # Test forward pass
    batch_size = 8
    input_tensor = torch.randn(batch_size, obs_dim_with_history)
    output = module.module(input_tensor)
    assert output.shape == (batch_size, 10)


def test_consistency_with_storage_dimensions():
    """Test that module dimensions match what PPO storage would expect.

    This simulates the scenario where:
    1. observation_manager.get_obs_dims() returns dims with history
    2. PPO._get_obs_dim uses these dims to register storage
    3. BaseModule uses the same dims to create the network

    All three should agree on the dimension!
    """
    # Simulate observation_manager.get_obs_dims() output
    # (which includes history: single_frame_dim * history_length)
    single_frame_dim = 100
    history_length = 4
    obs_dim_from_manager = single_frame_dim * history_length  # 400

    # This is what PPO would use for storage
    obs_dim_dict = {"actor_obs": obs_dim_from_manager}
    history_length_dict = {"actor_obs": history_length}

    # Simulate PPO._get_obs_dim (after our fix)
    storage_dim = obs_dim_dict["actor_obs"]  # Should NOT multiply by history again

    # Create module with same dimensions
    layer_config = LayerConfig(hidden_dims=[256], activation="ELU")
    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[10],
        layer_config=layer_config,
    )

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=module_config,
        history_length=history_length_dict,
    )

    # All three should agree
    assert obs_dim_from_manager == 400
    assert storage_dim == 400
    assert module.input_dim == 400

    # And forward pass should work with this dimension
    batch_size = 16
    observation = torch.randn(batch_size, storage_dim)
    output = module.module(observation)
    assert output.shape == (batch_size, 10)


def test_terrain_transformer_module_builds():
    """Terrain transformer should build with proprio, depth, and target-pose inputs."""
    layer_config = LayerConfig(
        hidden_dims=[64, 32],
        activation="ELU",
        encoder_hidden_dims=[64],
        encoder_activation="ReLU",
        encoder_input_name="motion_future_target_poses",
        encoder_obs_token_name="actor_obs_proprio",
        encoder_num_steps=10,
        transformer_latent_dim=32,
        transformer_num_layers=2,
        transformer_num_heads=2,
        transformer_ff_dim=64,
        transformer_dropout=0.0,
        transformer_pooling="first",
        perception_input_name="perception_obs",
    )

    config = ModuleConfig(
        type="TerrainTransformerObsTokenEncoder",
        input_dim=["actor_obs_proprio", "motion_future_target_poses"],
        output_dim=[12],
        layer_config=layer_config,
    )

    obs_dim_dict = {
        "actor_obs_proprio": 530,
        "motion_future_target_poses": 1000,
        "perception_obs": 512,
    }
    history_length = {
        "actor_obs_proprio": 10,
        "motion_future_target_poses": 1,
        "perception_obs": 1,
    }

    module = BaseModule(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        history_length=history_length,
    )

    assert module.input_dim == 1530
    assert hasattr(module, "encoder")
    assert module.module[0].in_features == 32


def test_terrain_transformer_actor_forward():
    """Terrain transformer actor should consume depth as a token and emit actions."""
    layer_config = LayerConfig(
        hidden_dims=[64, 32],
        activation="ELU",
        encoder_hidden_dims=[64],
        encoder_activation="ReLU",
        encoder_input_name="motion_future_target_poses",
        encoder_obs_token_name="actor_obs_proprio",
        encoder_num_steps=10,
        transformer_latent_dim=32,
        transformer_num_layers=2,
        transformer_num_heads=2,
        transformer_ff_dim=64,
        transformer_dropout=0.0,
        transformer_pooling="first",
        perception_input_name="perception_obs",
    )

    config = ModuleConfig(
        type="TerrainTransformerObsTokenEncoder",
        input_dim=["actor_obs_proprio", "motion_future_target_poses"],
        output_dim=[6],
        layer_config=layer_config,
        min_noise_std=0.05,
    )

    obs_dim_dict = {
        "actor_obs_proprio": 530,
        "motion_future_target_poses": 1000,
        "perception_obs": 512,
    }
    history_length = {
        "actor_obs_proprio": 10,
        "motion_future_target_poses": 1,
        "perception_obs": 1,
    }

    actor = PPOActorEncoder(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        num_actions=6,
        init_noise_std=1.0,
        history_length=history_length,
    )

    batch_size = 4
    actor_obs = torch.randn(batch_size, obs_dim_dict["actor_obs_proprio"] + obs_dim_dict["motion_future_target_poses"])
    policy_state = {
        "actor_obs": actor_obs,
        "perception_obs": torch.randn(batch_size, obs_dim_dict["perception_obs"]),
    }

    actions = actor.act_inference(policy_state)
    assert actions.shape == (batch_size, 6)


def _terrain_transformer_time_gru_config(*, output_dim: int) -> ModuleConfig:
    return ModuleConfig(
        type="TerrainTransformerObsTokenEncoder",
        input_dim=["proprio", "target"],
        output_dim=[output_dim],
        layer_config=LayerConfig(
            hidden_dims=[16],
            activation="ELU",
            encoder_hidden_dims=[16],
            encoder_activation="ReLU",
            encoder_input_name="target",
            encoder_obs_token_name="proprio",
            encoder_num_steps=1,
            transformer_latent_dim=8,
            transformer_num_layers=1,
            transformer_num_heads=1,
            transformer_ff_dim=16,
            transformer_dropout=0.0,
            transformer_pooling="first",
            perception_input_name="depth",
            perception_encoder_type="time_gru",
            perception_output_dim=8,
        ),
    )


def test_terrain_transformer_time_gru_actor_sequence_extra_bypasses_live_gru():
    obs_dims = {"proprio": 4, "target": 2, "depth": 3}
    history = {key: 1 for key in obs_dims}
    actor = PPOActorEncoder(
        obs_dim_dict=obs_dims,
        module_config_dict=_terrain_transformer_time_gru_config(output_dim=2),
        num_actions=2,
        init_noise_std=1.0,
        history_length=history,
    )

    # The ordinary collection path owns and advances the live recurrent state.
    actor.update_distribution_from_policy_state(
        {
            "actor_obs": torch.randn(2, 6),
            "depth": torch.randn(2, 3),
        }
    )
    rollout_hidden = actor.perception_time_gru.hidden.detach().clone()

    # Sequence PPO already encoded [T, B] through forward_sequence. The
    # flattened external embedding must take strict priority over any raw side
    # input and must not advance the collection hidden state.
    actor.update_distribution_from_policy_state(
        {
            "actor_obs": torch.randn(4, 6),
            "extra_actor_input": torch.randn(4, 8),
            "depth": torch.randn(1, 99),
        }
    )

    assert actor.action_mean.shape == (4, 2)
    assert torch.equal(actor.perception_time_gru.hidden, rollout_hidden)


def test_terrain_transformer_time_gru_critic_sequence_extra_bypasses_live_gru():
    obs_dims = {"proprio": 4, "target": 2, "depth": 3}
    history = {key: 1 for key in obs_dims}
    critic = PPOCriticEncoder(
        obs_dim_dict=obs_dims,
        module_config_dict=_terrain_transformer_time_gru_config(output_dim=1),
        history_length=history,
    )

    collection_values = critic.evaluate(
        {
            "critic_obs": torch.randn(2, 6),
            "depth": torch.randn(2, 3),
        }
    )
    assert collection_values.shape == (2, 1)
    rollout_hidden = critic.perception_time_gru.hidden.detach().clone()

    sequence_values = critic.evaluate(
        {
            "critic_obs": torch.randn(4, 6),
            "extra_critic_input": torch.randn(4, 8),
            "depth": torch.randn(1, 99),
        }
    )

    assert sequence_values.shape == (4, 1)
    assert torch.equal(critic.perception_time_gru.hidden, rollout_hidden)


def test_far_tracking_perception_encoder_concatenates_into_actor_input():
    """Far-tracking depth encoder should produce a 32d latent concatenated into actor input."""
    layer_config = LayerConfig(
        hidden_dims=[64, 32],
        activation="ELU",
        module_input_name=("actor_obs",),
        perception_input_name="perception_obs",
        perception_output_dim=32,
        perception_encoder_type="far_tracking_cnn_small",
        perception_input_height=58,
        perception_input_width=87,
        extra_input_to_hidden=False,
    )

    config = ModuleConfig(
        type="MLPPerceptionEncoder",
        input_dim=["actor_obs"],
        output_dim=[6],
        layer_config=layer_config,
        min_noise_std=0.05,
    )

    obs_dim_dict = {
        "actor_obs": 467,
        "perception_obs": 58 * 87,
    }
    history_length = {
        "actor_obs": 1,
        "perception_obs": 1,
    }

    actor = PPOActorEncoder(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        num_actions=6,
        init_noise_std=1.0,
        history_length=history_length,
    )

    assert isinstance(actor.actor_module.perception_encoder, FarTrackingDepthSmallEncoder)
    assert actor.actor_module.module[0].in_features == 467 + 32

    batch_size = 4
    actor_obs = torch.randn(batch_size, obs_dim_dict["actor_obs"])
    policy_state = {
        "actor_obs": actor_obs,
        "perception_obs": torch.randn(batch_size, obs_dim_dict["perception_obs"]),
    }
    input_actor, extra_input = actor._get_input(actor_obs, policy_state)
    assert input_actor.shape == (batch_size, 467 + 32)
    assert extra_input is None


def test_spatial_softmax_preserves_horizontal_location():
    spatial_softmax = SpatialSoftmax2d(height=8, width=11)
    features_left = torch.full((1, 1, 8, 11), -20.0)
    features_right = features_left.clone()
    features_left[0, 0, 4, 1] = 20.0
    features_right[0, 0, 4, 9] = 20.0

    left_xy = spatial_softmax(features_left)
    right_xy = spatial_softmax(features_right)

    assert left_xy.shape == (1, 2)
    assert left_xy[0, 0] < -0.7
    assert right_xy[0, 0] > 0.7
    assert torch.allclose(left_xy[0, 1], right_xy[0, 1], atol=1.0e-6)


def test_far_tracking_spatial_softmax_encoder_keeps_32d_actor_interface():
    layer_config = LayerConfig(
        hidden_dims=[64, 32],
        activation="ELU",
        module_input_name=("actor_obs",),
        perception_input_name="perception_obs",
        perception_output_dim=32,
        perception_encoder_type="far_tracking_cnn_spatial_softmax",
        perception_input_height=58,
        perception_input_width=87,
        extra_input_to_hidden=False,
    )
    config = ModuleConfig(
        type="MLPPerceptionEncoder",
        input_dim=["actor_obs"],
        output_dim=[29],
        layer_config=layer_config,
        min_noise_std=0.05,
    )
    actor = PPOActorEncoder(
        obs_dim_dict={"actor_obs": 94, "perception_obs": 58 * 87},
        module_config_dict=config,
        num_actions=29,
        init_noise_std=1.0,
        history_length={"actor_obs": 1, "perception_obs": 1},
    )

    encoder = actor.actor_module.perception_encoder
    assert isinstance(encoder, FarTrackingDepthSpatialSoftmaxEncoder)
    assert encoder.feature_height == 8
    assert encoder.feature_width == 11
    assert encoder.projection.in_features == 128
    assert encoder.projection.out_features == 32
    assert actor.actor_module.module[0].in_features == 126

    depth = torch.randn(3, 58 * 87, requires_grad=True)
    latent = encoder(depth)
    assert latent.shape == (3, 32)
    assert torch.isfinite(latent).all()
    latent.square().mean().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()


def test_apply_perception_overrides_keeps_critic_plain_for_far_tracking_preset():
    """Far-tracking-aligned depth preset should inject depth into actor only, not critic."""
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd,
        perception=perception_presets.camera_depth_d435i,
    )

    updated = apply_perception_overrides(config)

    actor_cfg = updated.algo.config.module_dict.actor
    critic_cfg = updated.algo.config.module_dict.critic

    assert "perception_obs" in updated.observation.groups
    assert actor_cfg.type == "MLPPerceptionEncoder"
    assert actor_cfg.layer_config.perception_input_name == "perception_obs"
    assert actor_cfg.layer_config.perception_encoder_type == "far_tracking_cnn_small"
    assert actor_cfg.layer_config.perception_input_height == 58
    assert actor_cfg.layer_config.perception_input_width == 87
    assert actor_cfg.layer_config.perception_pretrained is False
    assert actor_cfg.layer_config.perception_pretrained_path is None
    assert actor_cfg.layer_config.perception_pretrained_sha256 is None
    assert actor_cfg.layer_config.perception_freeze_backbone is False
    assert actor_cfg.layer_config.extra_input_to_hidden is False
    assert critic_cfg.type == "MLP"
    assert critic_cfg.layer_config.perception_input_name == ""


def test_defm_vit_s14_encoder_forward_with_mock_runtime():
    """DeFM ViT-S/14 encoder should preprocess depth and project backbone features."""

    class DummyDeFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 384

        def forward(self, x):
            return torch.ones((x.shape[0], self.embed_dim), device=x.device, dtype=x.dtype)

    def fake_create_defm_model(model_name, pretrained=False, pretrained_path=None):
        assert model_name == "defm_vit_s14"
        assert pretrained is False
        assert pretrained_path is None
        return DummyDeFM()

    def fake_preprocess_depth_batch(input_batch, target_size=None, patch_size=None, device="cpu", **kwargs):
        assert tuple(input_batch.shape) == (3, 58, 87)
        assert target_size == 224
        assert patch_size == 14
        return torch.ones((input_batch.shape[0], 3, 224, 224), device=device, dtype=torch.float32)

    encoder = DeFMViTS14Encoder(
        input_height=58,
        input_width=87,
        output_dim=128,
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=True,
        target_size=224,
        patch_size=14,
    )

    with patch(
        "holosoma.agents.modules.modules._load_defm_runtime",
        return_value=(fake_create_defm_model, fake_preprocess_depth_batch),
    ):
        x = torch.randn(3, 58 * 87)
        y = encoder(x)

    assert y.shape == (3, 128)


def test_defm_preprocessing_rejects_nonexportable_antialiased_downsampling():
    with pytest.raises(ValueError, match="does not support preprocessing downsampling"):
        DeFMViTS14Encoder(
            input_height=480,
            input_width=848,
            output_dim=384,
            pretrained=False,
            target_size=224,
            patch_size=14,
        )


def test_defm_onnx_safe_upsampling_matches_pinned_upstream_preprocessing():
    try:
        _factory, upstream_preprocess = module_impl._load_defm_runtime()
    except RuntimeError as exc:
        pytest.skip(f"DeFM submodule runtime is unavailable: {exc}")
    assert os.environ["XFORMERS_DISABLED"] == "1"

    encoder = DeFMViTS14Encoder(
        input_height=58,
        input_width=87,
        output_dim=384,
        pretrained=False,
        target_size=224,
        patch_size=14,
    )
    depth = torch.linspace(0.0, 5.0, steps=2 * 58 * 87, dtype=torch.float32).view(2, 58, 87)

    actual = encoder._preprocess_depth_batch_onnx_safe(depth, device=torch.device("cpu"))
    expected = upstream_preprocess(
        depth,
        target_size=224,
        patch_size=14,
        device="cpu",
    )

    assert torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_defm_materialization_covers_optimizer_and_fresh_strict_state_round_trip():
    """A trainable DeFM backbone must exist in both optimizer and checkpoint schema."""

    class DummyTrainableDeFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(384))
            self.register_buffer("feature_scale", torch.rand(1))

        def forward(self, x):
            return x.new_ones((x.shape[0], 384)) * self.weight

    def fake_create_defm_model(model_name, pretrained=False, pretrained_path=None):
        assert model_name == "defm_vit_s14"
        return DummyTrainableDeFM()

    def build_encoder():
        return DeFMViTS14Encoder(
            input_height=58,
            input_width=87,
            output_dim=32,
            pretrained=False,
            pretrained_path=None,
            freeze_backbone=False,
            target_size=224,
            patch_size=14,
        )

    source = build_encoder()
    fresh = build_encoder()
    with patch(
        "holosoma.agents.modules.modules._load_defm_runtime",
        return_value=(fake_create_defm_model, None),
    ):
        source.materialize_for_setup("cpu")
        optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
        assert source.backbone is not None
        backbone_weight = source.backbone.weight
        optimizer_parameter_ids = {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }
        assert id(backbone_weight) in optimizer_parameter_ids

        source_state = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
        }
        assert "backbone.weight" in source_state
        assert "backbone.feature_scale" in source_state

        fresh.materialize_for_setup("cpu")
        incompatible = fresh.load_state_dict(source_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for key, expected in source_state.items():
        assert torch.equal(fresh.state_dict()[key], expected)


def test_defm_pretrained_checkpoint_requires_digest_and_exact_state_schema(tmp_path):
    class DummyDeFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(384, dtype=torch.float32))

    checkpoint_path = tmp_path / "defm.pth"
    expected_state = DummyDeFM().state_dict()
    torch.save({"model": expected_state}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    def fake_create_defm_model(model_name, pretrained=False, pretrained_path=None):
        assert model_name == "defm_vit_s14"
        assert pretrained is False
        assert pretrained_path is None
        model = DummyDeFM()
        model.weight.data.zero_()
        return model

    encoder = DeFMViTS14Encoder(
        input_height=58,
        input_width=87,
        output_dim=384,
        pretrained=True,
        pretrained_path=str(checkpoint_path),
        pretrained_sha256=digest,
        freeze_backbone=True,
    )
    with patch(
        "holosoma.agents.modules.modules._load_defm_runtime",
        return_value=(fake_create_defm_model, None),
    ):
        encoder.materialize_for_setup("cpu")

    assert encoder.backbone is not None
    assert torch.equal(encoder.backbone.state_dict()["weight"], expected_state["weight"])

    incompatible_path = tmp_path / "incompatible.pth"
    torch.save({"model": {"unexpected": torch.ones(1)}}, incompatible_path)
    incompatible_digest = hashlib.sha256(incompatible_path.read_bytes()).hexdigest()
    incompatible = DeFMViTS14Encoder(
        input_height=58,
        input_width=87,
        output_dim=384,
        pretrained=True,
        pretrained_path=str(incompatible_path),
        pretrained_sha256=incompatible_digest,
        freeze_backbone=True,
    )
    with (
        patch(
            "holosoma.agents.modules.modules._load_defm_runtime",
            return_value=(fake_create_defm_model, None),
        ),
        pytest.raises(ValueError, match="schema does not exactly match"),
    ):
        incompatible.materialize_for_setup("cpu")


def test_frozen_defm_batch_norm_stays_in_eval_mode_after_parent_train(
    monkeypatch: pytest.MonkeyPatch,
):
    """Frozen DeFM running statistics must not mutate across PPO mode changes."""

    class DummyBatchNormDeFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_norm = nn.BatchNorm1d(384)

        def forward(self, x):
            return self.batch_norm(x.new_ones((x.shape[0], 384)))

    def fake_create_defm_model(model_name, pretrained=False, pretrained_path=None):
        assert model_name == "defm_vit_s14"
        return DummyBatchNormDeFM()

    encoder = DeFMViTS14Encoder(
        input_height=58,
        input_width=87,
        output_dim=384,
        pretrained=False,
        freeze_backbone=True,
        target_size=224,
        patch_size=14,
    )
    with patch(
        "holosoma.agents.modules.modules._load_defm_runtime",
        return_value=(fake_create_defm_model, None),
    ):
        encoder.materialize_for_setup("cpu")

    assert encoder.backbone is not None
    encoder.train(True)

    assert encoder.training is True
    assert encoder.backbone.training is False
    assert encoder.backbone.batch_norm.training is False
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())

    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", raising=False)
    monkeypatch.delenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", raising=False)
    ppo = object.__new__(PPO)
    ppo.use_symmetry = False
    ppo.use_time_gru = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.actor = nn.Sequential(encoder)
    ppo.dagger_enabled = False

    ppo._validate_training_objective_configuration()


def test_resolve_defm_repo_root_finds_repo_submodule_without_env(monkeypatch):
    """Local DeFM checkout may live under the repository's submodules directory."""
    submodule_root = None
    for parent in Path(__file__).resolve().parents:
        root_candidate = parent / "defm"
        if (root_candidate / "defm" / "model_factory.py").is_file():
            pytest.skip(f"root-level DeFM checkout shadows submodule checkout: {root_candidate}")
        submodule_candidate = parent / "submodules" / "defm"
        if (submodule_candidate / "defm" / "model_factory.py").is_file():
            submodule_root = submodule_candidate
            break

    if submodule_root is None:
        pytest.skip("DeFM submodule checkout is not initialized.")

    monkeypatch.delenv("HOLOSOMA_DEFM_ROOT", raising=False)
    module_impl._resolve_defm_repo_root.cache_clear()
    try:
        assert module_impl._resolve_defm_repo_root() == submodule_root
    finally:
        module_impl._resolve_defm_repo_root.cache_clear()


def test_defm_runtime_rejects_preimported_xformers_graph(monkeypatch):
    try:
        source_root = module_impl._resolve_defm_repo_root()
    except FileNotFoundError as exc:
        pytest.skip(f"DeFM submodule runtime is unavailable: {exc}")
    fake_attention = types.ModuleType("defm.layers.attention")
    fake_attention.__file__ = str(source_root / "defm" / "layers" / "attention.py")
    fake_attention.XFORMERS_AVAILABLE = True
    monkeypatch.setitem(sys.modules, "defm.layers.attention", fake_attention)
    monkeypatch.setenv("XFORMERS_DISABLED", "ambient-value")
    module_impl._load_defm_runtime.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="imported with xFormers enabled"):
            module_impl._load_defm_runtime()
        assert os.environ["XFORMERS_DISABLED"] == "1"
    finally:
        module_impl._load_defm_runtime.cache_clear()


def test_defm_vit_s14_encoder_chunks_frozen_backbone_forward(monkeypatch):
    """Frozen ViT backbone should avoid one large PPO minibatch forward."""
    chunk_shapes = []

    class DummyDeFM(nn.Module):
        def forward(self, x):
            chunk_shapes.append(tuple(x.shape))
            return torch.ones((x.shape[0], 384), device=x.device, dtype=x.dtype)

    def fake_create_defm_model(model_name, pretrained=False, pretrained_path=None):
        assert model_name == "defm_vit_s14"
        return DummyDeFM()

    monkeypatch.setenv("HOLOSOMA_DEFM_FORWARD_BATCH_SIZE", "2")
    encoder = DeFMViTS14Encoder(
        input_height=58,
        input_width=87,
        output_dim=384,
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=True,
        target_size=224,
        patch_size=14,
    )

    with patch(
        "holosoma.agents.modules.modules._load_defm_runtime",
        return_value=(fake_create_defm_model, None),
    ):
        y = encoder(torch.randn(5, 58 * 87))

    assert y.shape == (5, 384)
    assert chunk_shapes == [(2, 3, 224, 224), (2, 3, 224, 224), (1, 3, 224, 224)]


def test_resolve_defm_forward_batch_size_chunks_frozen_cnn(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DEFM_FORWARD_BATCH_SIZE", raising=False)

    assert module_impl._resolve_defm_forward_batch_size("defm_efficientnet_b2", freeze_backbone=True) == 512


def test_defm_regnet_y_800mf_encoder_forward_with_mock_runtime():
    """DeFM RegNetY-800MF encoder should project the global backbone feature."""

    class DummyDeFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 784

        def forward(self, x):
            raise AssertionError("RegNet encoder should use forward_no_bifpn().")

        def forward_no_bifpn(self, x):
            return {"global_backbone": torch.ones((x.shape[0], self.embed_dim), device=x.device, dtype=x.dtype)}

    def fake_create_defm_model(model_name, pretrained=False, pretrained_path=None):
        assert model_name == "defm_regnet_y_800mf"
        assert pretrained is False
        assert pretrained_path is None
        return DummyDeFM()

    def fake_preprocess_depth_batch(input_batch, target_size=None, patch_size=None, device="cpu", **kwargs):
        assert tuple(input_batch.shape) == (3, 58, 87)
        assert target_size == 224
        assert patch_size is None
        return torch.ones((input_batch.shape[0], 3, 224, 224), device=device, dtype=torch.float32)

    encoder = DeFMRegNetY800MFEncoder(
        input_height=58,
        input_width=87,
        output_dim=128,
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=True,
        target_size=224,
        patch_size=None,
    )

    with patch(
        "holosoma.agents.modules.modules._load_defm_runtime",
        return_value=(fake_create_defm_model, fake_preprocess_depth_batch),
    ):
        x = torch.randn(3, 58 * 87)
        y = encoder(x)

    assert y.shape == (3, 128)


def test_defm_efficientnet_b2_encoder_forward_with_mock_runtime():
    """DeFM EfficientNet-B2 encoder should project the global backbone feature."""

    class DummyDeFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 208

        def forward(self, x):
            raise AssertionError("EfficientNet-B2 encoder should use forward_no_bifpn().")

        def forward_no_bifpn(self, x):
            return {"global_backbone": torch.ones((x.shape[0], self.embed_dim), device=x.device, dtype=x.dtype)}

    def fake_create_defm_model(model_name, pretrained=False, pretrained_path=None):
        assert model_name == "defm_efficientnet_b2"
        assert pretrained is False
        assert pretrained_path is None
        return DummyDeFM()

    encoder = DeFMEfficientNetB2Encoder(
        input_height=58,
        input_width=87,
        output_dim=64,
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=True,
        target_size=224,
        patch_size=None,
    )

    with patch(
        "holosoma.agents.modules.modules._load_defm_runtime",
        return_value=(fake_create_defm_model, None),
    ):
        x = torch.randn(3, 58 * 87)
        y = encoder(x)

    assert y.shape == (3, 64)


def test_attention_linear_encoder_has_live_signal_at_init():
    encoder = AttentionLinearEncoder(input_dim=17 * 17, output_dim=64)
    x = torch.randn(4, 17 * 17)
    y = encoder(x)

    assert y.shape == (4, 64)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_apply_perception_overrides_sets_defm_actor_only_path():
    """DeFM preset should inject actor-only perception with concat fusion."""
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd,
        perception=perception_presets.camera_depth_d435i_defm_vit_s14,
    )

    updated = apply_perception_overrides(config)

    actor_cfg = updated.algo.config.module_dict.actor
    critic_cfg = updated.algo.config.module_dict.critic

    assert actor_cfg.type == "MLPPerceptionEncoder"
    assert actor_cfg.layer_config.perception_encoder_type == "defm_vit_s14"
    assert actor_cfg.layer_config.perception_output_dim == 384
    assert actor_cfg.layer_config.perception_input_height == 58
    assert actor_cfg.layer_config.perception_input_width == 87
    assert actor_cfg.layer_config.perception_pretrained is True
    assert actor_cfg.layer_config.perception_pretrained_path.endswith("defm_vit_s14.pth")
    assert actor_cfg.layer_config.perception_pretrained_sha256 == "37a6e95befea3a16732a743b2ebec854fd5eaed912ebaf9fbffc63a2306f1e90"
    assert actor_cfg.layer_config.perception_freeze_backbone is True
    assert actor_cfg.layer_config.perception_target_size == 224
    assert actor_cfg.layer_config.perception_patch_size == 14
    assert actor_cfg.layer_config.extra_input_to_hidden is False
    assert critic_cfg.type == "MLP"
    assert critic_cfg.layer_config.perception_input_name == ""
    assert config.perception.camera_warp_normalize is False


def test_apply_perception_overrides_sets_defm_regnet_actor_only_path():
    """DeFM RegNet preset should inject actor-only perception with concat fusion."""
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd,
        perception=perception_presets.camera_depth_d435i_defm_regnet_y_800mf,
    )

    updated = apply_perception_overrides(config)

    actor_cfg = updated.algo.config.module_dict.actor
    critic_cfg = updated.algo.config.module_dict.critic

    assert actor_cfg.type == "MLPPerceptionEncoder"
    assert actor_cfg.layer_config.perception_encoder_type == "defm_regnet_y_800mf"
    assert actor_cfg.layer_config.perception_output_dim == 784
    assert actor_cfg.layer_config.perception_input_height == 58
    assert actor_cfg.layer_config.perception_input_width == 87
    assert actor_cfg.layer_config.perception_pretrained is True
    assert actor_cfg.layer_config.perception_pretrained_path.endswith("defm_regnet_y_800mf.pth")
    assert actor_cfg.layer_config.perception_pretrained_sha256 == "6a78e6cce176e691cfbc1c8991815c5c90e98b369ecc153ddf48a6cc8641f14d"
    assert actor_cfg.layer_config.perception_freeze_backbone is True
    assert actor_cfg.layer_config.perception_target_size == 224
    assert actor_cfg.layer_config.perception_patch_size is None
    assert actor_cfg.layer_config.extra_input_to_hidden is False
    assert critic_cfg.type == "MLP"
    assert critic_cfg.layer_config.perception_input_name == ""
    assert config.perception.camera_warp_normalize is False


def test_apply_perception_overrides_sets_defm_efficientnet_actor_only_path():
    """DeFM EfficientNet preset should inject actor-only perception with concat fusion."""
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd,
        perception=perception_presets.camera_depth_d435i_defm_efficientnet_b2,
    )

    updated = apply_perception_overrides(config)

    actor_cfg = updated.algo.config.module_dict.actor
    critic_cfg = updated.algo.config.module_dict.critic

    assert actor_cfg.type == "MLPPerceptionEncoder"
    assert actor_cfg.layer_config.perception_encoder_type == "defm_efficientnet_b2"
    assert actor_cfg.layer_config.perception_output_dim == 208
    assert actor_cfg.layer_config.perception_input_height == 58
    assert actor_cfg.layer_config.perception_input_width == 87
    assert actor_cfg.layer_config.perception_pretrained is True
    assert actor_cfg.layer_config.perception_pretrained_path.endswith("defm_efficientnet_b2.pth")
    assert actor_cfg.layer_config.perception_pretrained_sha256 == "565404bdb073a3e81d5af3f8d6f76200384ba511bc81b51324298c6b630a4b58"
    assert actor_cfg.layer_config.perception_freeze_backbone is True
    assert actor_cfg.layer_config.perception_target_size == 224
    assert actor_cfg.layer_config.perception_patch_size is None
    assert actor_cfg.layer_config.extra_input_to_hidden is False
    assert critic_cfg.type == "MLP"
    assert critic_cfg.layer_config.perception_input_name == ""
    assert config.perception.camera_warp_normalize is False


@pytest.mark.parametrize(
    "preset",
    [
        perception_presets.camera_depth_d435i_defm_vit_s14,
        perception_presets.camera_depth_d435i_defm_regnet_y_800mf,
        perception_presets.camera_depth_d435i_defm_efficientnet_b2,
    ],
)
def test_apply_perception_overrides_rejects_normalized_defm_depth(preset):
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd,
        perception=replace(preset, camera_warp_normalize=True),
    )

    with pytest.raises(ValueError, match="requires metric depth in meters"):
        apply_perception_overrides(config)


def test_apply_perception_overrides_adds_heightmap_to_critic_only_path():
    """Student depth + critic heightmap should keep actor/critic perception paths separate."""
    distill_cfg = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd.algo.config.distill,
        critic_perception_preset="heightmap",
        critic_perception_obs_key="critic_perception_obs",
    )
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd,
        perception=perception_presets.camera_depth_d435i,
        algo=replace(
            g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd.algo,
            config=replace(
                g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd.algo.config,
                distill=distill_cfg,
            ),
        ),
    )

    updated = apply_perception_overrides(config)

    actor_cfg = updated.algo.config.module_dict.actor
    critic_cfg = updated.algo.config.module_dict.critic

    assert "perception_obs" in updated.observation.groups
    assert "critic_perception_obs" in updated.observation.groups
    assert actor_cfg.type == "MLPPerceptionEncoder"
    assert actor_cfg.layer_config.perception_input_name == "perception_obs"
    assert actor_cfg.layer_config.perception_encoder_type == "far_tracking_cnn_small"
    assert critic_cfg.type == "MLPPerceptionEncoder"
    assert critic_cfg.input_dim == ["critic_obs", "critic_proprio_history", "critic_actions"]
    assert critic_cfg.layer_config.module_input_name == ("critic_obs", "critic_proprio_history", "critic_actions")
    assert critic_cfg.layer_config.perception_input_name == "critic_perception_obs"
    assert critic_cfg.layer_config.perception_encoder_type == "attention"
    assert critic_cfg.layer_config.perception_input_height == 17
    assert critic_cfg.layer_config.perception_input_width == 17
    assert critic_cfg.layer_config.extra_input_to_hidden is True


def test_distill_critic_obs_keeps_single_frame_state_and_action():
    obs_cfg = g1_observations.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    critic_group = obs_cfg.groups["critic_obs"]
    proprio_history_group = obs_cfg.groups["critic_proprio_history"]
    critic_actions_group = obs_cfg.groups["critic_actions"]

    assert critic_group.history_length == 1
    assert proprio_history_group.history_length == 1
    assert set(proprio_history_group.terms) == {"base_lin_vel", "base_ang_vel", "dof_pos", "dof_vel"}
    assert critic_actions_group.history_length == 1
    assert set(critic_actions_group.terms) == {"actions"}

    arm_link_regions = {
        "left_elbow",
        "right_elbow",
        "left_wrist_roll",
        "right_wrist_roll",
        "left_wrist_pitch",
        "right_wrist_pitch",
    }
    removed_terms = {
        "contact_prior_confidence",
        "left_wrist_contact_prior_occupancy",
        "right_wrist_contact_prior_occupancy",
        *(f"{region}_contact_prior_occupancy" for region in arm_link_regions),
        "torso_contact_prior_occupancy",
        "left_wrist_contact_prior_force",
        "right_wrist_contact_prior_force",
        *(f"{region}_contact_prior_force" for region in arm_link_regions),
        "torso_contact_prior_force",
        "left_wrist_contact_prior_pos_obj",
        "right_wrist_contact_prior_pos_obj",
        *(f"{region}_contact_prior_pos_obj" for region in arm_link_regions),
        "torso_contact_prior_pos_obj",
        "left_wrist_object_contact_force",
        "right_wrist_object_contact_force",
        *(f"{region}_object_contact_force" for region in arm_link_regions),
        "torso_object_contact_force",
        "feet_object_contact_force",
        "ankle_object_contact_force",
        "left_wrist_object_contact_flag",
        "right_wrist_object_contact_flag",
        *(f"{region}_object_contact_flag" for region in arm_link_regions),
        "torso_object_contact_flag",
        "feet_object_contact_flag",
        "ankle_object_contact_flag",
        "feet_support_contact_force",
        "feet_support_contact_flag",
    }
    assert removed_terms.isdisjoint(critic_group.terms)

    kept_terms = {
        "motion_command",
        "obj_pos_b",
        "obj_ori_b",
        "obj_target_pos_b",
        "obj_target_ori_b",
        "obj_size",
        "obj_lin_vel_b",
        "obj_ang_vel_b",
    }
    assert kept_terms.issubset(critic_group.terms)


def test_sparse_root_distill_observation_uses_root_relative_terms():
    obs_cfg = g1_observations.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    root_group = obs_cfg.groups["actor_obs_root"]
    torso_alias_group = obs_cfg.groups["actor_obs_torso"]

    assert root_group.history_length == 1
    assert torso_alias_group.history_length == 1
    assert set(root_group.terms) == {"sparse_target_root_trajectory_command"}
    assert set(torso_alias_group.terms) == {"sparse_target_root_trajectory_command"}


def test_sparse_root_distill_observation_exposes_no_linvel_proprio_variant():
    obs_cfg = g1_observations.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    proprio_group = obs_cfg.groups["actor_obs_proprio_no_linvel"]

    assert proprio_group.history_length == 1
    assert set(proprio_group.terms) == {"base_ang_vel", "dof_pos", "dof_vel"}


def test_distill_experiment_critic_inputs_include_proprio_history():
    critic_cfg = g1_experiments.g1_29dof_wbt_w_object_distill_sparse_root_cmd.algo.config.module_dict.critic

    assert critic_cfg.input_dim == ["critic_obs", "critic_proprio_history", "critic_actions"]
    assert critic_cfg.layer_config.module_input_name == ("critic_obs", "critic_proprio_history", "critic_actions")


def test_critic_heightmap_encoder_accepts_separate_critic_perception_obs():
    """Critic should accept privileged obs plus a separate heightmap perception key."""
    layer_config = LayerConfig(
        hidden_dims=[64, 32],
        activation="ELU",
        module_input_name=("critic_obs",),
        perception_input_name="critic_perception_obs",
        perception_output_dim=128,
        perception_encoder_type="attention",
        perception_input_height=17,
        perception_input_width=17,
        extra_input_to_hidden=True,
    )

    config = ModuleConfig(
        type="MLPPerceptionEncoder",
        input_dim=["critic_obs"],
        output_dim=[1],
        layer_config=layer_config,
    )

    obs_dim_dict = {
        "critic_obs": 731,
        "critic_perception_obs": 17 * 17,
    }
    history_length = {
        "critic_obs": 1,
        "critic_perception_obs": 1,
    }

    critic = PPOCriticEncoder(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=config,
        history_length=history_length,
    )

    batch_size = 4
    critic_obs = torch.randn(batch_size, obs_dim_dict["critic_obs"])
    policy_state = {
        "critic_obs": critic_obs,
        "critic_perception_obs": torch.randn(batch_size, obs_dim_dict["critic_perception_obs"]),
    }

    values = critic.evaluate(policy_state)
    assert values.shape == (batch_size, 1)


def test_full_policy_lstm_step_sequence_and_done_reset_are_equivalent():
    config = ModuleConfig(
        type="LSTM",
        input_dim=["actor_obs"],
        output_dim=[3],
        layer_config=LayerConfig(
            hidden_dims=[16, 8],
            activation="ELU",
            lstm_hidden_dim=12,
            lstm_num_layers=2,
        ),
    )
    actor = PPOActor(
        obs_dim_dict={"actor_obs": 5},
        module_config_dict=config,
        num_actions=3,
        init_noise_std=0.5,
        history_length={"actor_obs": 1},
    )
    torch.manual_seed(7)
    observations = torch.randn(6, 4, 5)
    dones = torch.zeros(6, 4, 1, dtype=torch.bool)
    dones[1, 0] = True
    dones[2, 2] = True
    dones[4, 1] = True

    initial_hidden = torch.zeros(2, 4, 12)
    initial_cell = torch.zeros_like(initial_hidden)
    sequence_means, sequence_hidden, sequence_cell = (
        actor.actor_module.forward_recurrent_sequence(
            observations,
            dones=dones,
            initial_hidden=initial_hidden,
            initial_cell=initial_cell,
        )
    )

    actor.reset(None)
    step_means = []
    for step in range(observations.shape[0]):
        step_means.append(actor.act_inference({"actor_obs": observations[step]}))
        actor.reset(dones[step])

    assert torch.allclose(torch.stack(step_means), sequence_means, atol=1.0e-7, rtol=1.0e-6)
    assert torch.allclose(actor.actor_module.module.hidden_state, sequence_hidden, atol=1.0e-7, rtol=1.0e-6)
    assert torch.allclose(actor.actor_module.module.cell_state, sequence_cell, atol=1.0e-7, rtol=1.0e-6)
    stored_hidden, stored_cell = actor.recurrent_state_before_step(observations[-1])
    assert stored_hidden.shape == (4, 2, 12)
    assert stored_cell.shape == (4, 2, 12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
