python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=../far_data/ckp/model_14000.pt \
  --training.num_envs=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file /home/ANT.AMAZON.COM/zzzihanw/FAR/far_data/multi-motion/robot_only/lafan/aiming1_subject1_original_mj_fps50.npz \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale=0.01 \