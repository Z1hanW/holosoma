#python src/holosoma/holosoma/replay.py     \
# exp:g1-29dof-wbt \
#    --training.headless=False \
#    --training.num_envs=1     \
#    --terrain:terrain-load-obj \
#     --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj    
# --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz

# ----------------------------------------------------------------------
# Multi-env aligned preview (update TILE_STRIDE to match OBJ XY span)
# ----------------------------------------------------------------------
# TILE_STRIDE=2.0
# python src/holosoma/holosoma/replay.py \
#   exp:g1-29dof-wbt \
#   --training.headless=False \
#   --training.num_envs=1024 \
#   terrain:terrain-load-obj \
#   --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
#   --terrain.terrain-term.num_rows=32 \
#   --terrain.terrain-term.num_cols=32 \
#   --terrain.terrain-term.obj_tile_spacing_scale=1.0 \
#   --terrain.spawn.randomize_tiles=False \
#   --terrain.spawn.xy_offset_range=0 \
#   --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale=0 \
#   --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
#   --simulator.config.scene.env_spacing=${TILE_STRIDE}
