CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-motion-tracking-transformer \
  --training.num_envs=8192 \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.module_dict.actor.type=TransformerObsTokenEncoder \
  --algo.config.module_dict.critic.type=MLP \
  --algo.config.module_dict.actor.layer_config.encoder_num_steps=10 \
  --algo.config.module_dict.actor.layer_config.encoder_obs_token_name=actor_obs \
  --algo.config.module_dict.actor.layer_config.encoder_activation=ReLU \
  --algo.config.module_dict.actor.layer_config.transformer_pooling=first \
  --algo.config.module_dict.actor.min_noise_std=0.10 \
  --algo.config.save_interval=100 \
  \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  --terrain.terrain-term.num_rows=1 \
  --terrain.terrain-term.num_cols=1 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler=False \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.num_future_steps=10 \
  --command.setup_terms.motion_command.params.motion_config.target_pose_type=max-coords-future-rel-with-time \
  \
  --reward.terms.motion_relative_body_position_error_exp.params.sigma=1.0 \
  --reward.terms.motion_relative_body_position_error_exp.weight=0.2 \
  --reward.terms.motion_relative_body_orientation_error_exp.params.sigma=1.2 \
  --reward.terms.motion_relative_body_orientation_error_exp.weight=0.2 \
  --reward.terms.motion_global_ref_position_error_exp.params.sigma=0.6 \
  --reward.terms.motion_global_ref_position_error_exp.weight=0.5 \
  --reward.terms.motion_global_ref_orientation_error_exp.params.sigma=0.8 \
  --reward.terms.motion_global_ref_orientation_error_exp.weight=0.5 \
  --reward.terms.motion_global_body_lin_vel.params.sigma=2.0 \
  --reward.terms.motion_global_body_lin_vel.weight=0.5 \
  --reward.terms.motion_global_body_ang_vel.params.sigma=4.0 \
  --reward.terms.motion_global_body_ang_vel.weight=0.5 \
  --reward.terms.action_rate_l2.weight=-0.02 \
  --reward.terms.undesired_contacts.weight=0.0 \
  --reward.terms.limits_dof_pos.weight=-10.0 \
  \
  --termination.terms.bad_tracking.params.bad_ref_pos_threshold=1.0 \
  --termination.terms.bad_tracking.params.bad_ref_ori_threshold=1.2 \
  --termination.terms.bad_tracking.params.bad_motion_body_pos_threshold=0.6 \
  \
  logger:wandb \
  --logger.video.interval=1000 \
  --simulator.config.scene.env_spacing=0.0
