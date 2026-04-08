from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_terrain_transformer


def test_terrain_transformer_uses_explicit_command_and_proprio_obs():
    obs_groups = g1_29dof_wbt_terrain_transformer.observation.groups

    assert list(obs_groups["actor_obs_track"].terms.keys()) == ["motion_command", "motion_ref_ori_b"]
    assert obs_groups["actor_obs_track"].history_length == 1

    assert list(obs_groups["actor_obs_proprio"].terms.keys()) == ["base_ang_vel", "dof_pos", "dof_vel"]
    assert obs_groups["actor_obs_proprio"].history_length == 1

    assert list(obs_groups["actor_obs_actions"].terms.keys()) == ["actions"]
    assert obs_groups["actor_obs_actions"].history_length == 5

    actor_cfg = g1_29dof_wbt_terrain_transformer.algo.config.module_dict.actor
    assert actor_cfg.type == "TerrainTransformerObsTokenEncoder"
    assert list(actor_cfg.input_dim) == ["actor_obs_proprio", "actor_obs_track", "actor_obs_actions"]

    layer_cfg = actor_cfg.layer_config
    assert layer_cfg.encoder_obs_token_name == "actor_obs_proprio"
    assert layer_cfg.encoder_input_name == "actor_obs_track"
    assert tuple(layer_cfg.module_input_name) == ("actor_obs_actions",)

    critic_cfg = g1_29dof_wbt_terrain_transformer.algo.config.module_dict.critic
    assert critic_cfg.type == "MLP"
    assert list(critic_cfg.input_dim) == ["critic_obs"]
