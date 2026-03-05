python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=/home/ANT.AMAZON.COM/zzzihanw/Downloads/prefill_pose/prefill_pose/model_06600.pt \
  perception:none \
  --training.num_envs=1 \
  terrain:terrain-load-obj \
  --terrain.terrain-term.spawn.randomize_tiles=False \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=True \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=1 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=True \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=1 \
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale=0.01 \
