python examples/robot_retarget.py \
    --data_path demo_data/far_robot \
    --task-type climbing \
    --task-name far_robot \
    --robot-config.robot-urdf-file models/g1/g1_29dof.urdf \
    --data_format smplx \
    --retargeter.debug \
    --retargeter.visualize \
    --retargeter.algorithm interaction_mesh_foot \
    --retargeter.foot-tracking-weight 1000
