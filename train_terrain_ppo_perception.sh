PERCEPTION_MODE=${PERCEPTION_MODE:-none}

CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-motion-tracking-transformer \
  perception:${PERCEPTION_MODE} \
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
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler=True \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.num_future_steps=10 \
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale=0.77 \
  --command.setup_terms.motion_command.params.motion_config.target_pose_type=max-coords-future-rel-with-time \
  logger:wandb \
  --logger.video.interval=1000 \
  --simulator.config.scene.env_spacing=0.0
