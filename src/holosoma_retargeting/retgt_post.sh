python data_conversion/convert_data_format_mj.py \
    --input_file demo_results/g1/climbing/mocap_climb/far_robot_original.npz \
    --output_fps 50 \
    --output_name converted_res/object_interaction/stairs_mj_w_obj_omnirt.npz \
    --data_format smplx \
    --object_name "stairs" \
    --use_omniretarget_data --once \

