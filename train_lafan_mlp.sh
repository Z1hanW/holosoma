#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
NPROC=${NPROC:-$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")}
PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${NPROC}" --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  perception:none \
  --training.num_envs="${NUM_ENVS}" \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.save_interval=1000 \
  --algo.config.num_learning_iterations=1000000 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/robot_only/lafan \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  logger:wandb \
  --logger.video.enabled=False \
  --logger.headless_recording=False \
  --logger.video.upload_to_wandb=False \
  --logger.name="g1_videomimic_newdata"
