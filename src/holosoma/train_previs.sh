#python src/holosoma/holosoma/replay.py     \
# exp:g1-29dof-wbt \
#    --training.headless=False \
#    --training.num_envs=1     \
#    --terrain:terrain-load-obj \
#     --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj    
# --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz
# holosoma_lafan_npz/motion_bank.h5
python src/holosoma/holosoma/replay.py \
   exp:g1-29dof-wbt-videomimic-mlp \
  --training.headless=False \
  --training.num_envs=4 \
  --terrain.terrain-term.num_rows=2 \
  --terrain.terrain-term.num_cols=2 \
  --command.setup_terms.motion_command.params.motion_config.motion_file data/converted_res/holosoma_lafan_npz/motion_bank.h5 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 
