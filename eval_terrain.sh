python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=./model_01200.pt \
  --training.num_envs=1 \
  terrain:terrain-load-obj \
  --terrain.terrain-term.spawn.randomize_tiles=False \
  --terrain.terrain-term.obj-file-path stairs.obj \
  --command.setup_terms.motion_command.params.motion_config.motion_file far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 