#!/usr/bin/env bash
set -euo pipefail

TEACHER_CHECKPOINT="${1:-${TEACHER_CHECKPOINT:-}}"
if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Usage: $0 <teacher_checkpoint.pt>" >&2
  exit 1
fi

MOTION_DIR="${MOTION_DIR:-src/holosoma_retargeting/converted_res/robot_only/lafan}"
PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i_rendered}"

NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
NUM_ENVS="${NUM_ENVS:-30720}"
ACTOR_LR="${ACTOR_LR:-7e-5}"
CRITIC_LR="${CRITIC_LR:-7e-5}"
LOGGER="${LOGGER:-logger:wandb}"
RUN_NAME="${RUN_NAME:-g1_videomimic_distill_lafan_perception}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  "perception:${PERCEPTION_PRESET}" \
  --observation_overrides.disable_actor_target_inputs=True \
  --observation_overrides.disable_critic_target=True \
  --algo.config.distill.mode=dagger \
  --algo.config.distill.policy_to_clone="${TEACHER_CHECKPOINT}" \
  --algo.config.distill.bc_loss_coef=1.0 \
  --algo.config.distill.clip_teacher_actions=True \
  --algo.config.distill.clip_actions_threshold=8.0 \
  --algo.config.distill.teacher_obs_keys="actor_obs,actor_obs_target,perception_obs" \
  \
  --training.num_envs="${NUM_ENVS}" \
  --algo.config.actor_learning_rate="${ACTOR_LR}" \
  --algo.config.critic_learning_rate="${CRITIC_LR}" \
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
  "${LOGGER}" \
  --logger.video.interval=1000 \
  --logger.name="${RUN_NAME}"
