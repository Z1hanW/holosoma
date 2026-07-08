from dataclasses import replace

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
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=1000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
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
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

g1_29dof_wbt_height_scan = replace(
    g1_29dof_wbt,
    training=replace(g1_29dof_wbt.training, name="g1_29dof_wbt_height_scan_manager"),
    observation=observation.g1_29dof_wbt_observation_with_height_scan,
    simulator=replace(
        g1_29dof_wbt.simulator,
        config=replace(
            g1_29dof_wbt.simulator.config,
            height_scanner=replace(g1_29dof_wbt.simulator.config.height_scanner, enabled=True),
        ),
    ),
)

_tokenhsi_actor_inputs = ["actor_obs", "actor_height_scan"]
_tokenhsi_critic_inputs = ["critic_obs", "critic_height_scan"]
_tokenhsi_object_actor_inputs = ["actor_obs", "actor_height_scan", "actor_object_obs"]
_tokenhsi_object_critic_inputs = ["critic_obs", "critic_height_scan", "critic_object_obs"]

_tokenhsi_actor_layer = replace(
    algo.ppo.config.module_dict.actor.layer_config,
    module_input_name=(),
    encoder_obs_token_name="actor_obs",
    perception_input_name="actor_height_scan",
    encoder_input_name="",
    encoder_hidden_dims=[512, 256],
    encoder_output_dim=256,
    encoder_activation="ReLU",
    transformer_latent_dim=256,
    transformer_num_layers=2,
    transformer_num_heads=4,
    transformer_ff_dim=512,
    transformer_dropout=0.0,
    transformer_pooling="first",
    hidden_dims=[1024, 512],
)

_tokenhsi_critic_layer = replace(
    algo.ppo.config.module_dict.critic.layer_config,
    module_input_name=(),
    encoder_obs_token_name="critic_obs",
    perception_input_name="critic_height_scan",
    encoder_input_name="",
    encoder_hidden_dims=[512, 256],
    encoder_output_dim=256,
    encoder_activation="ReLU",
    transformer_latent_dim=256,
    transformer_num_layers=2,
    transformer_num_heads=4,
    transformer_ff_dim=512,
    transformer_dropout=0.0,
    transformer_pooling="first",
    hidden_dims=[1024, 512],
)

_tokenhsi_module_dict = replace(
    algo.ppo.config.module_dict,
    actor=replace(
        algo.ppo.config.module_dict.actor,
        type="TokenHSI",
        input_dim=_tokenhsi_actor_inputs,
        layer_config=_tokenhsi_actor_layer,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="TokenHSI",
        input_dim=_tokenhsi_critic_inputs,
        layer_config=_tokenhsi_critic_layer,
    ),
)

_tokenhsi_object_actor_layer = replace(
    _tokenhsi_actor_layer,
    encoder_input_name="actor_object_obs",
    encoder_num_steps=1,
    encoder_obs_dim=None,
)

_tokenhsi_object_critic_layer = replace(
    _tokenhsi_critic_layer,
    encoder_input_name="critic_object_obs",
    encoder_num_steps=1,
    encoder_obs_dim=None,
)

_tokenhsi_object_module_dict = replace(
    algo.ppo.config.module_dict,
    actor=replace(
        algo.ppo.config.module_dict.actor,
        type="TokenHSI",
        input_dim=_tokenhsi_object_actor_inputs,
        layer_config=_tokenhsi_object_actor_layer,
    ),
    critic=replace(
        algo.ppo.config.module_dict.critic,
        type="TokenHSI",
        input_dim=_tokenhsi_object_critic_inputs,
        layer_config=_tokenhsi_object_critic_layer,
    ),
)

g1_29dof_wbt_height_scan_tokenhsi = replace(
    g1_29dof_wbt_height_scan,
    training=replace(g1_29dof_wbt_height_scan.training, name="g1_29dof_wbt_height_scan_tokenhsi_manager"),
    observation=observation.g1_29dof_wbt_observation_height_scan_tokenhsi,
    algo=replace(
        g1_29dof_wbt_height_scan.algo,
        config=replace(
            g1_29dof_wbt_height_scan.algo.config,
            module_dict=_tokenhsi_module_dict,
            save_interval=1000,
            use_symmetry=False,
        ),
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=4096,
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

g1_29dof_wbt_w_object_height_scan = replace(
    g1_29dof_wbt_w_object,
    training=replace(g1_29dof_wbt_w_object.training, name="g1_29dof_wbt_w_object_height_scan_manager"),
    observation=observation.g1_29dof_wbt_observation_w_object_with_height_scan,
    simulator=replace(
        g1_29dof_wbt_w_object.simulator,
        config=replace(
            g1_29dof_wbt_w_object.simulator.config,
            height_scanner=replace(g1_29dof_wbt_w_object.simulator.config.height_scanner, enabled=True),
        ),
    ),
)

g1_29dof_wbt_w_object_height_scan_tokenhsi = replace(
    g1_29dof_wbt_w_object_height_scan,
    training=replace(
        g1_29dof_wbt_w_object_height_scan.training,
        name="g1_29dof_wbt_w_object_height_scan_tokenhsi_manager",
    ),
    observation=observation.g1_29dof_wbt_observation_w_object_height_scan_tokenhsi,
    algo=replace(
        g1_29dof_wbt_w_object_height_scan.algo,
        config=replace(
            g1_29dof_wbt_w_object_height_scan.algo.config,
            module_dict=_tokenhsi_object_module_dict,
            save_interval=1000,
            use_symmetry=False,
        ),
    ),
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            **g1_29dof_wbt.nightly.metrics,
            "Episode/rew_object_global_ref_position_error_exp": [0.4, "inf"],
            "Episode/rew_object_global_ref_orientation_error_exp": [0.4, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object_height_scan_tokenhsi_next_target = replace(
    g1_29dof_wbt_w_object_height_scan_tokenhsi,
    training=replace(
        g1_29dof_wbt_w_object_height_scan_tokenhsi.training,
        name="g1_29dof_wbt_w_object_height_scan_tokenhsi_next_target_manager",
    ),
    observation=observation.g1_29dof_wbt_observation_w_object_height_scan_tokenhsi_next_target,
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
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_height_scan",
    "g1_29dof_wbt_height_scan_tokenhsi",
    "g1_29dof_wbt_w_object",
    "g1_29dof_wbt_w_object_height_scan",
    "g1_29dof_wbt_w_object_height_scan_tokenhsi",
    "g1_29dof_wbt_w_object_height_scan_tokenhsi_next_target",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 2b: Robot+Object TokenHSI expert:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object-height-scan-tokenhsi

Example 2c: Robot+Object TokenHSI expert with next-frame object target:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-height-scan \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""
