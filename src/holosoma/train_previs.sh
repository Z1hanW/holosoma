source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1 \
    terrain:terrain-load-obj \
    --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
    --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/stairs_mj_w_obj.npz

python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1


python data_conversion/convert_data_format_mj.py \
    --input_file demo_results/g1/climbing/mocap_climb/far_robot_original.npz \
    --output_fps 50 \
    --output_name converted_res/object_interaction/stairs_mj_w_obj.npz \
    --data_format smplx \
    --object_name "stairs" \
    --has_dynamic_object \
    --once