source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1 \
    terrain:terrain-load-obj \
    --robot.object.object-urdf-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.urdf \
    --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/stairs_mj_w_obj_omnirt.npz



python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1 \
    terrain:terrain-load-obj \
    --terrain.terrain-term.obj-file-path src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj \
    --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz


python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1 \
    terrain:terrain-load-obj \
    --terrain.terrain-term.obj-file-path src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj \
    --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz

