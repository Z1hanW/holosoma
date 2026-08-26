from dataclasses import replace

from holosoma.config_types.algo import PPOModuleDictConfig
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    perception,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="boxer",
        name="g1_29dof_wbt_manager",
        num_envs=8192,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_epochs=5,
            num_learning_iterations=40000,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            use_symmetry=False,
            normalize_actor_obs=False,
            normalize_critic_obs=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.16, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.45, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.30, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.30, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.02, "inf"],
        },
    ),
)

g1_29dof_wbt_motion_tracking = replace(
    g1_29dof_wbt,
    training=replace(
        g1_29dof_wbt.training,
        name="g1_29dof_wbt_motion_tracking_manager",
    ),
    observation=observation.g1_29dof_wbt_observation_motion_tracking,
    command=command.g1_29dof_wbt_command_motion_tracking,
)

_motion_tracking_actor_inputs = ["actor_obs", "motion_future_target_poses"]
_motion_tracking_critic_inputs = ["critic_obs", "motion_future_target_poses"]

_motion_tracking_mlp_layer = replace(
    algo.ppo.config.module_dict.actor.layer_config,
    module_input_name=("actor_obs",),
    encoder_input_name="motion_future_target_poses",
    encoder_hidden_dims=[512, 256],
    encoder_output_dim=256,
)

_motion_tracking_critic_mlp_layer = replace(
    algo.ppo.config.module_dict.critic.layer_config,
    module_input_name=("critic_obs",),
    encoder_input_name="motion_future_target_poses",
    encoder_hidden_dims=[512, 256],
    encoder_output_dim=256,
)

_motion_tracking_mlp_module_dict = PPOModuleDictConfig(
    actor=replace(
        algo.ppo.config.module_dict.actor,
        type="MLPEncoder",
        input_dim=_motion_tracking_actor_inputs,
        layer_config=_motion_tracking_mlp_layer,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="MLPEncoder",
        input_dim=_motion_tracking_critic_inputs,
        layer_config=_motion_tracking_critic_mlp_layer,
    ),
)

_motion_tracking_transformer_layer = replace(
    algo.ppo.config.module_dict.actor.layer_config,
    module_input_name=(),
    encoder_input_name="motion_future_target_poses",
    encoder_obs_token_name="actor_obs",
    encoder_num_steps=10,
    encoder_hidden_dims=[512, 256],
    encoder_activation="ReLU",
    transformer_latent_dim=256,
    transformer_num_layers=2,
    transformer_num_heads=2,
    transformer_ff_dim=512,
    transformer_dropout=0.0,
    transformer_pooling="first",
    hidden_dims=[1024, 512],
)

_motion_tracking_transformer_critic_layer = replace(
    algo.ppo.config.module_dict.critic.layer_config,
    module_input_name=("critic_obs",),
    encoder_input_name="motion_future_target_poses",
    encoder_num_steps=10,
    encoder_activation="ReLU",
    transformer_latent_dim=256,
    transformer_num_layers=2,
    transformer_num_heads=2,
    transformer_ff_dim=512,
    transformer_dropout=0.0,
    transformer_pooling="mean",
    hidden_dims=[1024, 512],
)

_motion_tracking_transformer_module_dict = PPOModuleDictConfig(
    actor=replace(
        algo.ppo.config.module_dict.actor,
        type="TransformerObsTokenEncoder",
        input_dim=_motion_tracking_actor_inputs,
        layer_config=_motion_tracking_transformer_layer,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="MLP",
        input_dim=_motion_tracking_critic_inputs,
    ),
)

g1_29dof_wbt_motion_tracking_mlp_encoder = replace(
    g1_29dof_wbt_motion_tracking,
    observation=observation.g1_29dof_wbt_observation_motion_tracking_split,
    algo=replace(
        algo.ppo,
        config=replace(algo.ppo.config, module_dict=_motion_tracking_mlp_module_dict),
    ),
)

g1_29dof_wbt_motion_tracking_transformer = replace(
    g1_29dof_wbt_motion_tracking,
    observation=observation.g1_29dof_wbt_observation_motion_tracking_split,
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            module_dict=_motion_tracking_transformer_module_dict,
            normalize_actor_obs=False,
            normalize_critic_obs=False,
        ),
    ),
)

_videomimic_actor_inputs = ["actor_obs", "actor_obs_target"]
_videomimic_critic_inputs = ["critic_obs", "critic_obs_target"]
_terrain_transformer_actor_inputs = ["actor_obs_self", "actor_obs_target"]
_terrain_transformer_critic_inputs = ["critic_obs"]

_videomimic_mlp_module_dict = PPOModuleDictConfig(
    actor=replace(
        algo.ppo.config.module_dict.actor,
        type="MLP",
        input_dim=_videomimic_actor_inputs,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="MLP",
        input_dim=_videomimic_critic_inputs,
    ),
)

_videomimic_transformer_layer = replace(
    algo.ppo.config.module_dict.actor.layer_config,
    module_input_name=("actor_obs_target",),
    encoder_input_name="actor_obs",
    encoder_num_steps=1,
    encoder_activation="ReLU",
    transformer_latent_dim=256,
    transformer_num_layers=2,
    transformer_num_heads=2,
    transformer_ff_dim=512,
    transformer_dropout=0.0,
    transformer_pooling="first",
    hidden_dims=[1024, 512],
)

_videomimic_transformer_module_dict = PPOModuleDictConfig(
    actor=replace(
        algo.ppo.config.module_dict.actor,
        type="TransformerEncoder",
        input_dim=_videomimic_actor_inputs,
        layer_config=_videomimic_transformer_layer,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="MLP",
        input_dim=_videomimic_critic_inputs,
    ),
)

_terrain_transformer_layer = replace(
    algo.ppo.config.module_dict.actor.layer_config,
    module_input_name=(),
    encoder_input_name="actor_obs_target",
    encoder_obs_token_name="actor_obs_self",
    encoder_num_steps=1,
    encoder_hidden_dims=[512, 256],
    encoder_activation="ReLU",
    transformer_latent_dim=256,
    transformer_num_layers=2,
    transformer_num_heads=2,
    transformer_ff_dim=512,
    transformer_dropout=0.0,
    transformer_pooling="first",
    hidden_dims=[1024, 512],
)

_terrain_transformer_module_dict = PPOModuleDictConfig(
    actor=replace(
        algo.ppo.config.module_dict.actor,
        type="TerrainTransformerObsTokenEncoder",
        input_dim=_terrain_transformer_actor_inputs,
        layer_config=_terrain_transformer_layer,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="MLP",
        input_dim=_terrain_transformer_critic_inputs,
    ),
)

g1_29dof_wbt_videomimic_mlp = replace(
    g1_29dof_wbt,
    training=replace(
        g1_29dof_wbt.training,
        name="g1_29dof_wbt_videomimic_mlp",
    ),
    observation=observation.g1_29dof_wbt_observation_videomimic,
    algo=replace(
        algo.ppo,
        config=replace(algo.ppo.config, module_dict=_videomimic_mlp_module_dict),
    ),
)

g1_29dof_wbt_videomimic_transformer = replace(
    g1_29dof_wbt,
    training=replace(
        g1_29dof_wbt.training,
        name="g1_29dof_wbt_videomimic_transformer",
    ),
    observation=observation.g1_29dof_wbt_observation_videomimic,
    algo=replace(
        algo.ppo,
        config=replace(algo.ppo.config, module_dict=_videomimic_transformer_module_dict),
    ),
)

g1_29dof_wbt_terrain_transformer = replace(
    g1_29dof_wbt,
    training=replace(
        g1_29dof_wbt.training,
        name="g1_29dof_wbt_terrain_transformer",
    ),
    observation=observation.g1_29dof_wbt_observation_terrain_transformer,
    algo=replace(
        g1_29dof_wbt.algo,
        config=replace(
            g1_29dof_wbt.algo.config,
            module_dict=_terrain_transformer_module_dict,
            save_interval=1000,
            normalize_actor_obs=False,
            normalize_critic_obs=False,
            use_symmetry=False,
        ),
    ),
)

g1_29dof_wbt_videomimic_mlp_w_gru = replace(
    g1_29dof_wbt_videomimic_mlp,
    training=replace(
        g1_29dof_wbt_videomimic_mlp.training,
        name="w_gru",
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="boxer",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=8192,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    curriculum=curriculum.g1_29dof_wbt_curriculum_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
            enabled=True,
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_w_object_extend = replace(
    g1_29dof_wbt_w_object,
    training=replace(
        g1_29dof_wbt_w_object.training,
        name="g1_29dof_wbt_w_object_extend",
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_extend,
)

g1_29dof_wbt_w_object_generalist = replace(
    g1_29dof_wbt_w_object,
    training=replace(
        g1_29dof_wbt_w_object.training,
        name="g1_29dof_wbt_w_object_generalist",
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_generalist,
    command=command.g1_29dof_wbt_command_w_object_generalist,
    termination=termination.g1_29dof_wbt_termination_generalist,
)

g1_29dof_wbt_w_object_generalist_teacher_linvel = replace(
    g1_29dof_wbt_w_object_generalist,
    training=replace(
        g1_29dof_wbt_w_object_generalist.training,
        name="g1_29dof_wbt_w_object_generalist_teacher_linvel",
    ),
    observation=observation.g1_29dof_wbt_observation_w_object_teacher_linvel,
)

g1_29dof_wbt_w_object_generalist_legacy_obs = replace(
    g1_29dof_wbt_w_object_generalist,
    training=replace(
        g1_29dof_wbt_w_object_generalist.training,
        name="g1_29dof_wbt_w_object_generalist_legacy_obs",
    ),
    observation=observation.g1_29dof_wbt_observation_w_object_legacy,
)

_w_object_distill_sparse_root_cmd_actor_inputs = ["actor_obs_root", "actor_obs_proprio", "actor_obs_actions", "actor_obs_box"]
_w_object_distill_sparse_root_cmd_critic_inputs = ["critic_obs", "critic_proprio_history", "critic_actions"]

_w_object_distill_sparse_root_cmd_critic_layer = replace(
    g1_29dof_wbt_w_object_generalist.algo.config.module_dict.critic.layer_config,
    module_input_name=tuple(_w_object_distill_sparse_root_cmd_critic_inputs),
)

_w_object_distill_sparse_root_cmd_module_dict = PPOModuleDictConfig(
    actor=replace(
        g1_29dof_wbt_w_object_generalist.algo.config.module_dict.actor,
        type="MLP",
        input_dim=_w_object_distill_sparse_root_cmd_actor_inputs,
    ),
    critic=replace(
        g1_29dof_wbt_w_object_generalist.algo.config.module_dict.critic,
        type="MLP",
        input_dim=_w_object_distill_sparse_root_cmd_critic_inputs,
        layer_config=_w_object_distill_sparse_root_cmd_critic_layer,
    ),
)

_terrain_distill_sparse_root_cmd_actor_inputs = ["actor_obs_root", "actor_obs_proprio", "actor_obs_actions"]
_terrain_distill_sparse_root_cmd_critic_inputs = ["critic_obs"]

_terrain_distill_sparse_root_cmd_module_dict = PPOModuleDictConfig(
    actor=replace(
        g1_29dof_wbt_videomimic_mlp.algo.config.module_dict.actor,
        type="MLP",
        input_dim=_terrain_distill_sparse_root_cmd_actor_inputs,
    ),
    critic=replace(
        g1_29dof_wbt_videomimic_mlp.algo.config.module_dict.critic,
        type="MLP",
        input_dim=_terrain_distill_sparse_root_cmd_critic_inputs,
    ),
)

g1_29dof_wbt_w_object_distill_sparse_root_cmd = replace(
    g1_29dof_wbt_w_object_generalist,
    training=replace(
        g1_29dof_wbt_w_object_generalist.training,
        name="g1_29dof_wbt_w_object_distill_sparse_root_cmd",
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object_with_action_delay,
    observation=observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    termination=termination.g1_29dof_wbt_termination_distill,
    algo=replace(
        g1_29dof_wbt_w_object_generalist.algo,
        config=replace(
            g1_29dof_wbt_w_object_generalist.algo.config,
            module_dict=_w_object_distill_sparse_root_cmd_module_dict,
        ),
    ),
)

g1_29dof_wbt_w_object_hybrid_stage2 = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.training,
        name="g1_29dof_wbt_w_object_hybrid_stage2",
    ),
    command=command.g1_29dof_wbt_command_w_object_hybrid_stage2,
    observation=observation.g1_29dof_wbt_observation_w_object_hybrid_stage2,
    reward=reward.g1_29dof_wbt_reward_w_object_hybrid_stage2,
    termination=termination.g1_29dof_wbt_termination_hybrid_stage2,
)

g1_29dof_wbt_w_object_hybrid_velocity = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.training,
        name="g1_29dof_wbt_w_object_hybrid_velocity",
    ),
    command=command.g1_29dof_wbt_command_w_object_hybrid_velocity,
    observation=observation.g1_29dof_wbt_observation_w_object_hybrid_velocity,
    reward=reward.g1_29dof_wbt_reward_w_object_hybrid_velocity,
    termination=termination.g1_29dof_wbt_termination_hybrid_velocity,
)


def _policy_command_module_dict(command_group: str) -> PPOModuleDictConfig:
    return replace(
        _w_object_distill_sparse_root_cmd_module_dict,
        actor=replace(
            _w_object_distill_sparse_root_cmd_module_dict.actor,
            input_dim=[
                command_group,
                "actor_obs_drop_button",
                "actor_obs_proprio_with_actions_no_linvel",
            ],
        ),
    )


_pure_rl_critic317_inputs = [
    "critic_obs",
    "critic_actions",
    "actor_obs_root_contact_aware",
    "actor_obs_drop_button",
]

_pure_rl_critic317_base_module_dict = _policy_command_module_dict(
    "actor_obs_root_contact_aware"
)
_pure_rl_critic317_module_dict = replace(
    _pure_rl_critic317_base_module_dict,
    critic=replace(
        _pure_rl_critic317_base_module_dict.critic,
        input_dim=_pure_rl_critic317_inputs,
        layer_config=replace(
            _pure_rl_critic317_base_module_dict.critic.layer_config,
            module_input_name=tuple(_pure_rl_critic317_inputs),
        ),
    ),
)

_hmi_depth_critic317_inputs = [
    "critic_obs",
    "critic_actions",
    "actor_obs_hmi_goal_command",
    "actor_obs_drop_button",
]
_hmi_depth_base_module_dict = _policy_command_module_dict(
    "actor_obs_hmi_goal_command"
)
_hmi_depth_module_dict = replace(
    _hmi_depth_base_module_dict,
    critic=replace(
        _hmi_depth_base_module_dict.critic,
        input_dim=_hmi_depth_critic317_inputs,
        layer_config=replace(
            _hmi_depth_base_module_dict.critic.layer_config,
            module_input_name=tuple(_hmi_depth_critic317_inputs),
        ),
    ),
)


g1_29dof_wbt_w_object_policy_world_velocity = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.training,
        name="g1_29dof_wbt_w_object_policy_world_velocity",
    ),
    observation=observation.g1_29dof_wbt_observation_w_object_policy_world_velocity,
    reward=reward.g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact,
    termination=termination.g1_29dof_wbt_termination_generalist,
    algo=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.algo,
        config=replace(
            g1_29dof_wbt_w_object_distill_sparse_root_cmd.algo.config,
            module_dict=_policy_command_module_dict("actor_obs_world_velocity_command"),
        ),
    ),
)

g1_29dof_wbt_w_object_policy_world_root_error = replace(
    g1_29dof_wbt_w_object_policy_world_velocity,
    training=replace(
        g1_29dof_wbt_w_object_policy_world_velocity.training,
        name="g1_29dof_wbt_w_object_policy_world_root_error",
    ),
    observation=observation.g1_29dof_wbt_observation_w_object_policy_world_root_error,
    algo=replace(
        g1_29dof_wbt_w_object_policy_world_velocity.algo,
        config=replace(
            g1_29dof_wbt_w_object_policy_world_velocity.algo.config,
            module_dict=_policy_command_module_dict("actor_obs_world_root_error_command"),
        ),
    ),
)

g1_29dof_wbt_w_object_hybrid_world_velocity = replace(
    g1_29dof_wbt_w_object_hybrid_velocity,
    training=replace(
        g1_29dof_wbt_w_object_hybrid_velocity.training,
        name="g1_29dof_wbt_w_object_hybrid_world_velocity",
    ),
    command=command.g1_29dof_wbt_command_w_object_hybrid_world_velocity,
    observation=observation.g1_29dof_wbt_observation_w_object_hybrid_world_velocity,
    algo=replace(
        g1_29dof_wbt_w_object_hybrid_velocity.algo,
        config=replace(
            g1_29dof_wbt_w_object_hybrid_velocity.algo.config,
            module_dict=_policy_command_module_dict(
                "actor_obs_hybrid_world_velocity_command"
            ),
        ),
    ),
)

g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.training,
        name="g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift",
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object_pure_rl,
    command=command.g1_29dof_wbt_command_w_object_pure_rl_policy_command_after_lift,
    observation=observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    reward=reward.g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact,
    termination=termination.g1_29dof_wbt_termination_generalist,
)

# 317D critic = critic_obs(284) + previous action(29) + the exact same-step
# actor root command(3) + drop button(1).  This intentionally omits the
# redundant one-frame critic_proprio_history(64) used by older 381D runs.
g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317 = replace(
    g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift,
    training=replace(
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift.training,
        name="g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317",
    ),
    observation=(
        observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_critic317
    ),
    algo=replace(
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift.algo,
        config=replace(
            g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift.algo.config,
            module_dict=_pure_rl_critic317_module_dict,
        ),
    ),
)

g1_29dof_wbt_w_object_hmi_depth_stage1 = replace(
    g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317,
    training=replace(
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.training,
        name="g1_29dof_wbt_w_object_hmi_depth_stage1",
        export_onnx=True,
    ),
    perception=perception.camera_depth_d435i,
    robot=replace(
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.robot,
        object=replace(
            g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.robot.object,
            object_urdf_path="data_demo/_clip_object_urdf_map.json",
        ),
    ),
    simulator=replace(
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.simulator,
        config=replace(
            g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.simulator.config,
            sim=replace(
                g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.simulator.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    command=command.g1_29dof_wbt_command_w_object_hmi_depth_stage1,
    observation=observation.g1_29dof_wbt_observation_w_object_hmi_depth,
    reward=reward.g1_29dof_wbt_reward_w_object_hmi,
    termination=termination.g1_29dof_wbt_termination_hmi,
    algo=replace(
        g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.algo,
        config=replace(
            g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317.algo.config,
            module_dict=_hmi_depth_module_dict,
            num_learning_iterations=15000,
        ),
    ),
)

g1_29dof_wbt_w_object_hmi_depth_stage2 = replace(
    g1_29dof_wbt_w_object_hmi_depth_stage1,
    training=replace(
        g1_29dof_wbt_w_object_hmi_depth_stage1.training,
        name="g1_29dof_wbt_w_object_hmi_depth_stage2",
        export_onnx=True,
    ),
    command=command.g1_29dof_wbt_command_w_object_hmi_depth_stage2,
    algo=replace(
        g1_29dof_wbt_w_object_hmi_depth_stage1.algo,
        config=replace(
            g1_29dof_wbt_w_object_hmi_depth_stage1.algo.config,
            num_learning_iterations=20000,
        ),
    ),
)

g1_29dof_wbt_w_object_distill_sparse_root_cmd_teacher_linvel = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.training,
        name="g1_29dof_wbt_w_object_distill_sparse_root_cmd_teacher_linvel",
    ),
    observation=(
        observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_teacher_linvel
    ),
)

g1_29dof_wbt_terrain_distill_sparse_root_cmd = replace(
    g1_29dof_wbt_videomimic_mlp,
    training=replace(
        g1_29dof_wbt_videomimic_mlp.training,
        name="g1_29dof_wbt_terrain_distill_sparse_root_cmd",
    ),
    randomization=randomization.g1_29dof_wbt_randomization_with_action_delay,
    observation=observation.g1_29dof_wbt_observation_terrain_distill_sparse_root_cmd,
    termination=termination.g1_29dof_wbt_termination_distill,
    algo=replace(
        g1_29dof_wbt_videomimic_mlp.algo,
        config=replace(
            g1_29dof_wbt_videomimic_mlp.algo.config,
            module_dict=_terrain_distill_sparse_root_cmd_module_dict,
        ),
    ),
)

# Legacy variant that keeps clip_phase in actor_obs_root.
# Use only for backward compatibility / old checkpoint reproduction.
g1_29dof_wbt_w_object_distill_sparse_root_cmd_legacy = replace(
    g1_29dof_wbt_w_object_generalist,
    training=replace(
        g1_29dof_wbt_w_object_generalist.training,
        name="g1_29dof_wbt_w_object_distill_sparse_root_cmd_legacy",
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object_with_action_delay,
    observation=observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_legacy,
    termination=termination.g1_29dof_wbt_termination_distill,
    algo=replace(
        g1_29dof_wbt_w_object_generalist.algo,
        config=replace(
            g1_29dof_wbt_w_object_generalist.algo.config,
            module_dict=_w_object_distill_sparse_root_cmd_module_dict,
        ),
    ),
)

g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_contact = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.training,
        name="g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_contact",
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_r2s_contact_guidance,
)

g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd.training,
        name="g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref",
    ),
    reward=reward.g1_29dof_wbt_reward_w_object_r2s_rollout_reference_guidance,
)

_shoo7sr1_debug_observation_groups = dict(
    observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups
)
_shoo7sr1_debug_observation_groups["actor_obs"] = _shoo7sr1_debug_observation_groups["actor_obs_legacy"]
_shoo7sr1_debug_observation_groups["critic_obs"] = replace(
    _shoo7sr1_debug_observation_groups["critic_obs"],
    terms=observation.critic_obs_w_object_command_distill_legacy_target_terms,
)
_shoo7sr1_debug_observation_groups.pop("actor_obs_root_contact_aware", None)
_shoo7sr1_debug_observation_groups.pop("actor_obs_torso_contact_aware", None)
_shoo7sr1_debug_observation = replace(
    observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups=_shoo7sr1_debug_observation_groups,
)
_shoo7sr1_contact_aware_debug_observation_groups = dict(
    observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd.groups
)
_shoo7sr1_contact_aware_debug_observation_groups["actor_obs"] = _shoo7sr1_contact_aware_debug_observation_groups[
    "actor_obs_legacy"
]
_shoo7sr1_contact_aware_debug_observation_groups["critic_obs"] = replace(
    _shoo7sr1_contact_aware_debug_observation_groups["critic_obs"],
    terms=observation.critic_obs_w_object_command_distill_legacy_target_terms,
)
_shoo7sr1_contact_aware_debug_observation = replace(
    observation.g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    groups=_shoo7sr1_contact_aware_debug_observation_groups,
)

g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref_shoo7sr1_debug = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref.training,
        name="g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref_shoo7sr1_debug",
    ),
    observation=_shoo7sr1_debug_observation,
)

g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref_shoo7sr1_contact_aware_debug = replace(
    g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref,
    training=replace(
        g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref.training,
        name="g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref_shoo7sr1_contact_aware_debug",
    ),
    observation=_shoo7sr1_contact_aware_debug_observation,
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
            enabled=True,
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

__all__ = [
    "g1_29dof_wbt",
    "g1_29dof_wbt_motion_tracking",
    "g1_29dof_wbt_motion_tracking_mlp_encoder",
    "g1_29dof_wbt_motion_tracking_transformer",
    "g1_29dof_wbt_terrain_transformer",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_w_object",
    "g1_29dof_wbt_w_object_extend",
    "g1_29dof_wbt_w_object_generalist",
    "g1_29dof_wbt_w_object_generalist_teacher_linvel",
    "g1_29dof_wbt_w_object_generalist_legacy_obs",
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd",
    "g1_29dof_wbt_w_object_hybrid_velocity",
    "g1_29dof_wbt_w_object_hybrid_world_velocity",
    "g1_29dof_wbt_w_object_policy_world_velocity",
    "g1_29dof_wbt_w_object_policy_world_root_error",
    "g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift",
    "g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317",
    "g1_29dof_wbt_w_object_hmi_depth_stage1",
    "g1_29dof_wbt_w_object_hmi_depth_stage2",
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_teacher_linvel",
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_contact",
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref",
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref_shoo7sr1_debug",
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_r2s_rollout_ref_shoo7sr1_contact_aware_debug",
    "g1_29dof_wbt_terrain_distill_sparse_root_cmd",
    "g1_29dof_wbt_w_object_distill_sparse_root_cmd_legacy",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""
