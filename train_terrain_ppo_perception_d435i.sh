#!/usr/bin/env bash
set -euo pipefail

DEPTH_IMPL=${DEPTH_IMPL:-raycast}
if [[ "${DEPTH_IMPL}" == "raycast" ]]; then
  IMAGE_WIDTH=${IMAGE_WIDTH:-106}
  IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
  CAMERA_HFOV=${CAMERA_HFOV:-89.5}
  CAMERA_VFOV=${CAMERA_VFOV:-58.6}
  CAMERA_NEAR=${CAMERA_NEAR:-0.3}
  CAMERA_FAR=${CAMERA_FAR:-3.0}
  CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}

  CAMERA_JITTER_X=${CAMERA_JITTER_X:-0.025}
  CAMERA_JITTER_Y=${CAMERA_JITTER_Y:-0.025}
  CAMERA_JITTER_Z=${CAMERA_JITTER_Z:-0.025}
  CAMERA_JITTER_ROLL=${CAMERA_JITTER_ROLL:-2.5}
  CAMERA_JITTER_PITCH=${CAMERA_JITTER_PITCH:-3.0}
  CAMERA_JITTER_YAW=${CAMERA_JITTER_YAW:-2.5}
  CAMERA_NOISE_STD=${CAMERA_NOISE_STD:-0.05}
  CAMERA_NOISE_DROP=${CAMERA_NOISE_DROP:-0.025}

  CAMERA_MESH_ALLOWLIST=${CAMERA_MESH_ALLOWLIST:-'["pelvis","left_hip_pitch_link","left_hip_roll_link","left_hip_yaw_link","left_knee_link","left_ankle_pitch_link","left_ankle_roll_link","right_hip_pitch_link","right_hip_roll_link","right_hip_yaw_link","right_knee_link","right_ankle_pitch_link","right_ankle_roll_link","waist_yaw_link","waist_roll_link","left_shoulder_pitch_link","left_shoulder_roll_link","left_shoulder_yaw_link","left_elbow_link","left_wrist_roll_link","left_wrist_pitch_link","left_wrist_yaw_link","right_shoulder_pitch_link","right_shoulder_roll_link","right_shoulder_yaw_link","right_elbow_link","right_wrist_roll_link","right_wrist_pitch_link","right_wrist_yaw_link"]'}
else
  IMAGE_WIDTH=${IMAGE_WIDTH:-1280}
  IMAGE_HEIGHT=${IMAGE_HEIGHT:-720}
fi
case "${DEPTH_IMPL}" in
  rendered)
    PERCEPTION_PRESET="camera_depth_d435i_rendered"
    ;;
  depth_sensor)
    PERCEPTION_PRESET="camera_depth_d435i_depth_sensor"
    ;;
  raycast)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  *)
    echo "Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use rendered|depth_sensor|raycast." >&2
    exit 1
    ;;
esac

PERCEPTION_OVERRIDES=(
  --perception.camera_width="$IMAGE_WIDTH"
  --perception.camera_height="$IMAGE_HEIGHT"
)

RANDOMIZATION_OVERRIDES=()
if [[ "${DEPTH_IMPL}" == "raycast" ]]; then
  PERCEPTION_OVERRIDES+=(
    --perception.camera_hfov_deg="$CAMERA_HFOV"
    --perception.camera_vfov_deg="$CAMERA_VFOV"
    --perception.camera_near="$CAMERA_NEAR"
    --perception.camera_far="$CAMERA_FAR"
    --perception.max_distance="$CAMERA_MAX_DISTANCE"
    --perception.camera_include_robot_mesh=True
  )
  RANDOMIZATION_OVERRIDES+=(
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled=True
    --randomization.setup_terms.setup_camera_raycast_randomization.params.mesh_allowlist="${CAMERA_MESH_ALLOWLIST}"
    --randomization.reset_terms.randomize_camera_raycast.params.enabled=True
    --randomization.reset_terms.randomize_camera_raycast.params.translation_range='{"x":[-'"${CAMERA_JITTER_X}"', '"${CAMERA_JITTER_X}"'], "y":[-'"${CAMERA_JITTER_Y}"', '"${CAMERA_JITTER_Y}"'], "z":[-'"${CAMERA_JITTER_Z}"', '"${CAMERA_JITTER_Z}"']}'
    --randomization.reset_terms.randomize_camera_raycast.params.rotation_range_deg='{"roll":[-'"${CAMERA_JITTER_ROLL}"', '"${CAMERA_JITTER_ROLL}"'], "pitch":[-'"${CAMERA_JITTER_PITCH}"', '"${CAMERA_JITTER_PITCH}"'], "yaw":[-'"${CAMERA_JITTER_YAW}"', '"${CAMERA_JITTER_YAW}"']}'
    --randomization.reset_terms.randomize_camera_raycast.params.noise_std_mult_range='[0.0, '"${CAMERA_NOISE_STD}"']'
    --randomization.reset_terms.randomize_camera_raycast.params.noise_drop_prob_range='[0.0, '"${CAMERA_NOISE_DROP}"']'
  )
fi

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-motion-tracking-transformer \
  "perception:${PERCEPTION_PRESET}" \
  --training.num_envs=128 \
  "${PERCEPTION_OVERRIDES[@]}" \
  "${RANDOMIZATION_OVERRIDES[@]}" \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.module_dict.actor.type=TransformerObsTokenEncoder \
  --algo.config.module_dict.critic.type=MLP \
  --algo.config.module_dict.actor.layer_config.encoder_num_steps=10 \
  --algo.config.module_dict.actor.layer_config.encoder_obs_token_name=actor_obs \
  --algo.config.module_dict.actor.layer_config.encoder_activation=ReLU \
  --algo.config.module_dict.actor.layer_config.transformer_pooling=first \
  --algo.config.module_dict.actor.min_noise_std=0.10 \
  --algo.config.save_interval=100 \
  \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  --terrain.terrain-term.num_rows=1 \
  --terrain.terrain-term.num_cols=1 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler=False \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.num_future_steps=10 \
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale=0.77 \
  --command.setup_terms.motion_command.params.motion_config.target_pose_type=max-coords-future-rel-with-time \
  \
  logger:wandb \
  --logger.video.interval=1000 \
  --simulator.config.scene.env_spacing=0.0
