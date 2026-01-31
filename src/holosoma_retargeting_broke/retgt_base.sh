python examples/robot_retarget.py \
    --data_path demo_data/far_robot \
    --task-type robot_only \
    --task-name far_robot \
    --task-config.ground-size=0 \
    --retargeter.foot-tracking-weight=0 \
    --retargeter.w-nominal-tracking-init=0 \
    --data_format smplx --retargeter.debug --retargeter.visualize


