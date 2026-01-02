DATE="0921"
SCENE="MPH1Library"
METHOD="ours"

tmux set mouse on

# Construct the command with variables replaced
CUDA_VISIBLE_DEVICES=0 python motion_tracking/train_agent.py exp=mimic backbone=isaacgym_smpl_pulse_shape \
+opt=[mimic/target_pose_transformer_with_target_time_final,mimic/global_tracking,mimic/early_termination_tracking_err,mimic/dynamic_sampling] \
dynamic_weight_max=10000 motion_file=motion_data/${DATE}/${SCENE}_${METHOD}.npy ngpu=1 eval_metrics_every=500000 \
training_max_steps=100000000000 name=1217_${DATE}_${SCENE}_${METHOD} num_future_steps=10 critic_units=[1024,512] init_start_prob=0.1 \
num_mini_epochs=1 fix_pd_offsets=False obs_relative_to_surface=True num_envs=1024 batch_size=4096 provide_future_states=True \
headless=True ref_height_adjust=0. scene_file=${DATE}/${SCENE}/${METHOD}/${METHOD} max_epochs=10000
