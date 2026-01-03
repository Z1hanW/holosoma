python examples/robot_retarget.py \
    --data_path demo_data/far_robot \
    --task-type climbing \
    --task-name far_robot \
    --task-config.object_name stairs \
    --task-config.object-dir /home/ubuntu/FAR/holosoma/src/holosoma_retargeting/demo_data/far_robot/far_robot \
    --robot-config.robot-urdf-file models/g1/g1_29dof.urdf \
    --data_format smplx \
    --retargeter.debug \
    --retargeter.visualize \
    --retargeter.foot-tracking-weight 1000


    # interaction_mesh_foot

# python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
#    --object_urdf demo_data/far_robot/far_robot/stairs.urdf \
#    --qpos_npz demo_results/g1/climbing/mocap_climb/far_robot_original.npz