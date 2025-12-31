from dataclasses import replace

from holosoma.config_types.algo import PPOModuleDictConfig
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=8192,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=40000,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
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
        control=replace(robot.g1_29dof.control, action_scale=1.0),
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
    module_input_name=("actor_obs",),
    encoder_input_name="motion_future_target_poses",
    encoder_num_steps=5,
    transformer_latent_dim=256,
    transformer_num_layers=2,
    transformer_num_heads=2,
    transformer_ff_dim=512,
    transformer_dropout=0.0,
    transformer_pooling="mean",
    hidden_dims=[1024, 512],
)

_motion_tracking_transformer_critic_layer = replace(
    algo.ppo.config.module_dict.critic.layer_config,
    module_input_name=("critic_obs",),
    encoder_input_name="motion_future_target_poses",
    encoder_num_steps=5,
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
        type="TransformerEncoder",
        input_dim=_motion_tracking_actor_inputs,
        layer_config=_motion_tracking_transformer_layer,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="TransformerEncoder",
        input_dim=_motion_tracking_critic_inputs,
        layer_config=_motion_tracking_transformer_critic_layer,
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
        config=replace(algo.ppo.config, module_dict=_motion_tracking_transformer_module_dict),
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
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
        control=replace(robot.g1_29dof.control, action_scale=1.0),
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
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
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

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
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
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_w_object",
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
