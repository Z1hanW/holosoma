python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=./model_00500.pt \
  --training.num_envs=1 \
  terrain:terrain-load-obj \
  --terrain.terrain-term.spawn.randomize_tiles=False \
  --terrain.terrain-term.obj-file-path stairs.obj \
  --command.setup_terms.motion_command.params.motion_config.motion_file far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
