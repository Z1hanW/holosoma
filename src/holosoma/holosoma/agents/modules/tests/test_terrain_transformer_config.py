from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_terrain_mlp_future5,
    g1_29dof_wbt_terrain_transformer,
    g1_29dof_wbt_terrain_transformer_future5,
)


def test_terrain_transformer_uses_explicit_command_and_proprio_obs():
    obs_groups = g1_29dof_wbt_terrain_transformer.observation.groups

    assert set(obs_groups["actor_obs_self"].terms.keys()) == {
        "actions_history",
        "motion_command",
        "motion_ref_pos_b",
        "motion_ref_ori_b",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
    }
    assert obs_groups["actor_obs_self"].history_length == 1

    assert list(obs_groups["actor_obs_target"].terms.keys()) == ["target_joints", "target_root_roll", "target_root_pitch"]
    assert obs_groups["actor_obs_target"].history_length == 1

    actor_cfg = g1_29dof_wbt_terrain_transformer.algo.config.module_dict.actor
    assert actor_cfg.type == "TerrainTransformerObsTokenEncoder"
    assert list(actor_cfg.input_dim) == ["actor_obs_self", "actor_obs_target"]

    layer_cfg = actor_cfg.layer_config
    assert layer_cfg.encoder_obs_token_name == "actor_obs_self"
    assert layer_cfg.encoder_input_name == "actor_obs_target"
    assert tuple(layer_cfg.module_input_name) == ()
    assert layer_cfg.encoder_num_steps == 1

    critic_cfg = g1_29dof_wbt_terrain_transformer.algo.config.module_dict.critic
    assert critic_cfg.type == "MLP"
    assert list(critic_cfg.input_dim) == ["critic_obs"]


def test_terrain_transformer_future5_uses_future_target_tokens():
    obs_groups = g1_29dof_wbt_terrain_transformer_future5.observation.groups

    assert list(obs_groups["actor_obs_target"].terms.keys()) == ["motion_future_target_poses"]
    assert obs_groups["actor_obs_target"].history_length == 1

    motion_cfg = (
        g1_29dof_wbt_terrain_transformer_future5.command.setup_terms["motion_command"]
        .params["motion_config"]
    )
    assert motion_cfg.num_future_steps == 5
    assert motion_cfg.target_pose_type == "max-coords-future-rel-with-time"

    actor_cfg = g1_29dof_wbt_terrain_transformer_future5.algo.config.module_dict.actor
    assert actor_cfg.type == "TerrainTransformerObsTokenEncoder"
    assert list(actor_cfg.input_dim) == ["actor_obs_self", "actor_obs_target"]

    layer_cfg = actor_cfg.layer_config
    assert layer_cfg.encoder_input_name == "actor_obs_target"
    assert layer_cfg.encoder_obs_token_name == "actor_obs_self"
    assert layer_cfg.encoder_num_steps == 5


def test_terrain_mlp_future5_matches_future_target_inputs_without_transformer():
    obs_groups = g1_29dof_wbt_terrain_mlp_future5.observation.groups

    assert list(obs_groups["actor_obs_target"].terms.keys()) == ["motion_future_target_poses"]

    motion_cfg = g1_29dof_wbt_terrain_mlp_future5.command.setup_terms["motion_command"].params["motion_config"]
    assert motion_cfg.num_future_steps == 5

    actor_cfg = g1_29dof_wbt_terrain_mlp_future5.algo.config.module_dict.actor
    assert actor_cfg.type == "MLPEncoder"
    assert list(actor_cfg.input_dim) == ["actor_obs_self", "actor_obs_target"]

    layer_cfg = actor_cfg.layer_config
    assert layer_cfg.module_input_name == ("actor_obs_self",)
    assert layer_cfg.encoder_input_name == "actor_obs_target"
    assert layer_cfg.encoder_hidden_dims == [512, 256]
    assert layer_cfg.encoder_output_dim == 256
