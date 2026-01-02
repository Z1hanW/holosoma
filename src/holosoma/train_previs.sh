#python src/holosoma/holosoma/replay.py     \
# exp:g1-29dof-wbt \
#    --training.headless=False \
#    --training.num_envs=1     \
#    --terrain:terrain-load-obj \
#     --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj    
# --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz

python src/holosoma/holosoma/replay.py \
  exp:g1-29dof-wbt \
  --training.headless=False \
  --training.num_envs=1 \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  --terrain.terrain-term.num_rows=1 \
  --terrain.terrain-term.num_cols=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --simulator.config.scene.env_spacing=0
