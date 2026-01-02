DATE="g1"
SCENE="wbt"
METHOD="holosoma"

tmux set mouse on

CUDA_VISIBLE_DEVICES=0 python motion_tracking/train_agent.py exp=mimic_g1 backbone=isaacgym_g1 \
+opt=[mimic/target_pose_transformer_with_target_time_final_g1,mimic/global_tracking,mimic/early_termination_tracking_err,mimic/dynamic_sampling] \
motion_file=motion_data/g1_29dof/sub3_largebox_003_mj.npz \
ngpu=1 eval_metrics_every=500000 training_max_steps=100000000000 name=1217_${DATE}_${SCENE}_${METHOD} \
num_future_steps=5 critic_units=[1024,512] init_start_prob=0.1 num_mini_epochs=1 \
num_envs=1024 batch_size=4096 provide_future_states=True headless=True ref_height_adjust=0.0 \
obs_relative_to_surface=True max_epochs=10000
