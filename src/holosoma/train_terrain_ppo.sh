source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file="/home/ANT.AMAZON.COM/zzzihanw/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/stairs_mj_w_obj_omnirt.npz"

source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --training.headless=False \
    --training.num_envs=1 \
    terrain:terrain-load-obj \
    --terrain.terrain-term.obj-file-path src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj \
    --command.setup_terms.motion_command.params.motion_config.motion_file="/home/ANT.AMAZON.COM/zzzihanw/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/stairs_mj_w_obj_omnirt.npz"
