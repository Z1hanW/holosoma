#!/usr/bin/env bash
set -euo pipefail

# Distill a VideoMimic teacher into a policy conditioned on root direction (torso_xy_rel/yaw_rel)
# and proprioception only (no target joints/root roll/pitch). If TEACHER_MODE=perception, the
# student also receives perception_obs.

TEACHER_MODE="blind" # blind|perception
TEACHER_CHECKPOINT=""

MOTION_DIR="src/holosoma_retargeting/converted_res/robot_only/lafan"

PERCEPTION_PRESET="camera_depth_d435i"

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Set TEACHER_CHECKPOINT to the trained teacher policy checkpoint." >&2
  exit 1
fi

TEACHER_OBS_KEYS="actor_obs,actor_obs_target"
EXTRA_ARGS=()
if [[ "${TEACHER_MODE}" == "perception" ]]; then
  EXTRA_ARGS+=("perception:${PERCEPTION_PRESET}")
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS},perception_obs"
elif [[ "${TEACHER_MODE}" != "blind" ]]; then
  echo "Unknown TEACHER_MODE=${TEACHER_MODE}. Use blind|perception." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-distill-mlp \
  "${EXTRA_ARGS[@]}" \
  --observation_overrides.disable_actor_target_inputs=True \
  --algo.config.distill.mode=dagger \
  --algo.config.distill.policy_to_clone="${TEACHER_CHECKPOINT}" \
  --algo.config.distill.bc_loss_coef=1.0 \
  --algo.config.distill.clip_teacher_actions=True \
  --algo.config.distill.clip_actions_threshold=8.0 \
  --algo.config.distill.teacher_obs_keys="${TEACHER_OBS_KEYS}" \
  \
  --training.num_envs=30720 \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.save_interval=1000 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}" \
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=False \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  logger:wandb \
  --logger.video.interval=1000 \
  --logger.name="g1_videomimic_distill_rootpos"
