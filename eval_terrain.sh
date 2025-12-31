python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/home/ANT.AMAZON.COM/zzzihanw/FAR/holosoma/logs/WholeBodyTracking/20251231_025321-g1_29dof_wbt_motion_tracking_manager-locomotion/model_00800.pt \
        terrain:terrain-load-obj \
    --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
    --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz
