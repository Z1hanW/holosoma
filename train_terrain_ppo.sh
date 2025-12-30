python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  --training.num_envs=8192 \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.num_mini_batches=8 \
  --algo.config.num_learning_epochs=4 \
  --algo.config.init_noise_std=0.8 \
  --algo.config.module_dict.actor.min_noise_std=0.10 \
  --algo.config.save_interval=100 \
  \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler=True \
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob=0.10 \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0.5 \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0.5 \
  \
  --randomization.setup_terms.push_randomizer_state.params.enabled=False \
  \
  logger:wandb \
  --logger.video.interval=1000

## num_envs: int = 4096
#python src/holosoma/holosoma/replay.py     \
# exp:g1-29dof-wbt \
#    --training.headless=False \
#    --training.num_envs=1     \
#    --terrain:terrain-load-obj \
#     --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj    
# --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz
