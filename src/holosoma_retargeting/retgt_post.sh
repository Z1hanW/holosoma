python data_conversion/convert_data_format_mj.py \
    --input_file demo_results/g1/climbing/mocap_climb/far_robot_original.npz \
    --output_fps 50 \
    --output_name converted_res/object_interaction/far_robot_mj.npz \
    --data_format smplx \
    --object_name ground \
    --once \

# Visualize the converted motion with viser (robot + optional scene obj).
SCENE_OBJ_PATH="/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj"  # TODO: set to /path/to/scene.obj if you want to show the scene mesh.

python viser_player.py \
    --qpos_npz converted_res/object_interaction/far_robot_mj.npz \
    --robot_urdf models/g1/g1_29dof.urdf \
    --object_obj "$SCENE_OBJ_PATH" \
