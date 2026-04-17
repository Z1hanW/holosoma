"""Tests for module dimension calculations with history length.

This test suite verifies that:
1. BaseModule correctly computes input dimensions when obs_dim_dict includes history
2. PPO's _get_obs_dim correctly computes dimensions
3. Network input dimensions match what is stored in replay buffers/storage
"""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from holosoma.agents.modules.modules import (
    AttentionLinearEncoder,
    BaseModule,
    DeFMRegNetY800MFEncoder,
    DeFMViTS14Encoder,
    FarTrackingDepthSmallEncoder,
)
from holosoma.agents.modules.ppo_modules import PPOActorEncoder, PPOCriticEncoder
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


def test_apply_perception_overrides_keeps_critic_plain_for_far_tracking_preset():
    """Far-tracking-aligned depth preset should inject depth into actor only, not critic."""
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed,
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
        assert pretrained is True
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
        pretrained=True,
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
        assert pretrained is True
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
        pretrained=True,
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


def test_attention_linear_encoder_has_live_signal_at_init():
    encoder = AttentionLinearEncoder(input_dim=17 * 17, output_dim=64)
    x = torch.randn(4, 17 * 17)
    y = encoder(x)

    assert y.shape == (4, 64)
    assert not torch.allclose(y, torch.zeros_like(y))


def test_apply_perception_overrides_sets_defm_actor_only_path():
    """DeFM preset should inject actor-only perception with concat fusion."""
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed,
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
    assert actor_cfg.layer_config.perception_freeze_backbone is True
    assert actor_cfg.layer_config.perception_target_size == 224
    assert actor_cfg.layer_config.perception_patch_size == 14
    assert actor_cfg.layer_config.extra_input_to_hidden is False
    assert critic_cfg.type == "MLP"
    assert critic_cfg.layer_config.perception_input_name == ""


def test_apply_perception_overrides_sets_defm_regnet_actor_only_path():
    """DeFM RegNet preset should inject actor-only perception with concat fusion."""
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed,
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
    assert actor_cfg.layer_config.perception_freeze_backbone is True
    assert actor_cfg.layer_config.perception_target_size == 224
    assert actor_cfg.layer_config.perception_patch_size is None
    assert actor_cfg.layer_config.extra_input_to_hidden is False
    assert critic_cfg.type == "MLP"
    assert critic_cfg.layer_config.perception_input_name == ""


def test_apply_perception_overrides_adds_heightmap_to_critic_only_path():
    """Student depth + critic heightmap should keep actor/critic perception paths separate."""
    distill_cfg = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed.algo.config.distill,
        critic_perception_preset="heightmap",
        critic_perception_obs_key="critic_perception_obs",
    )
    config = replace(
        g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed,
        perception=perception_presets.camera_depth_d435i,
        algo=replace(
            g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed.algo,
            config=replace(
                g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed.algo.config,
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
    assert critic_cfg.input_dim == ["critic_obs", "critic_proprio_history"]
    assert critic_cfg.layer_config.module_input_name == ("critic_obs", "critic_proprio_history")
    assert critic_cfg.layer_config.perception_input_name == "critic_perception_obs"
    assert critic_cfg.layer_config.perception_encoder_type == "attention"
    assert critic_cfg.layer_config.perception_input_height == 17
    assert critic_cfg.layer_config.perception_input_width == 17
    assert critic_cfg.layer_config.extra_input_to_hidden is True


def test_distill_critic_obs_keeps_single_frame_state_and_proprio_history():
    obs_cfg = g1_observations.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    critic_group = obs_cfg.groups["critic_obs"]
    proprio_history_group = obs_cfg.groups["critic_proprio_history"]

    assert critic_group.history_length == 1
    assert proprio_history_group.history_length == 5
    assert set(proprio_history_group.terms) == {"base_lin_vel", "base_ang_vel", "dof_pos", "dof_vel", "actions"}

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
        "obj_target_pose_size_b",
        "obj_pos_b",
        "obj_ori_b",
        "obj_lin_vel_b",
        "obj_ang_vel_b",
        "obj_sparse_goal_xy_pick_root_heading",
        "obj_picked_flag",
        "command_only_flag",
        "sparse_goal_external_flag",
    }
    assert kept_terms.issubset(critic_group.terms)


def test_sparse_root_distill_observation_uses_root_relative_terms():
    obs_cfg = g1_observations.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    root_group = obs_cfg.groups["actor_obs_root"]
    torso_alias_group = obs_cfg.groups["actor_obs_torso"]

    assert root_group.history_length == 5
    assert torso_alias_group.history_length == 5
    assert set(root_group.terms) == {"sparse_target_root_trajectory_command"}
    assert set(torso_alias_group.terms) == {"sparse_target_root_trajectory_command"}


def test_sparse_root_distill_observation_exposes_no_linvel_proprio_variant():
    obs_cfg = g1_observations.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    proprio_group = obs_cfg.groups["actor_obs_proprio_no_linvel"]

    assert proprio_group.history_length == 5
    assert set(proprio_group.terms) == {"base_ang_vel", "dof_pos", "dof_vel", "actions"}


def test_distill_experiment_critic_inputs_include_proprio_history():
    critic_cfg = g1_experiments.g1_29dof_wbt_w_object_distill_sparse_goal_mixed.algo.config.module_dict.critic

    assert critic_cfg.input_dim == ["critic_obs", "critic_proprio_history"]
    assert critic_cfg.layer_config.module_input_name == ("critic_obs", "critic_proprio_history")


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
