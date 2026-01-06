CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  --training.num_envs=30720 \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.save_interval=100 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file /ABS/PATH/holosoma_lafan_npz/motion_bank.h5 \
  --command.setup_terms.motion_command.params.motion_config.clip_weighting_strategy=uniform_clip \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  logger:wandb \
  --logger.video.interval=1000

# Optional adaptive clip weighting:
#  --command.setup_terms.motion_command.params.motion_config.clip_weighting_strategy=success_rate_adaptive \
#  --command.setup_terms.motion_command.params.motion_config.min_weight_factor=0.33 \
#  --command.setup_terms.motion_command.params.motion_config.max_weight_factor=3.0 \
