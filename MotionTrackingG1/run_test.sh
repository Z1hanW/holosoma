DATE="0921"
SCENE="MPH1Library"
METHOD="ours"
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia HYDRA_FULL_ERROR=1 python motion_tracking/eval_agent.py critic_units='[1024,512]' \
exp=mimic backbone=isaacgym_smpl_pulse_shape +opt=[mimic/target_pose_transformer_with_target_time_final,mimic/global_tracking] \
motion_file=motion_data/${DATE}/${SCENE}_${METHOD}.npy early_reward_term=null headless=True num_future_steps=10 \
checkpoint=results/${DATE}/${DATE}_${SCENE}_${METHOD}/lightning_logs/version_0/last.ckpt max_episode_length=1000 \
num_envs=1 init_start_prob=1 reset_track_steps_min=1000000 reset_track_steps_max=1000001 provide_future_states=True enable_height_termination=False \
fix_pd_offsets=False obs_relative_to_surface=True ref_height_adjust=0.0 scene_file=${DATE}/${SCENE}/${METHOD}/${METHOD} #num_games=1 #save_state_path="abcd" #
