python data_conversion/convert_data_format_mj.py \
    --input_file demo_results/g1/climbing/mocap_climb/far_robot_original.npz \
    --output_fps 50 \
    --output_name converted_res/object_interaction/far_robot_mj.npz \
    --data_format smplx \
    --object_name ground \
    --once \

# Visualize the converted motion with viser (robot only).
python viser_player.py \
    --qpos_npz converted_res/object_interaction/far_robot_mj.npz \
    --robot_urdf models/g1/g1_29dof.urdf \
    --assume_object_in_qpos False

