#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

OBJ_DIR=${OBJ_DIR:-multi-terrain/test}
MOTION_DIR=${MOTION_DIR:-multi-motion/test}
NUM_ROWS=${NUM_ROWS:-4}
NUM_COLS=${NUM_COLS:-4}
CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
NPROC=${NPROC:-$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")}
PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${NPROC}" --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  perception:none \
  terrain:terrain-load-obj \
  --training.headless=False \
  --training.num_envs="${NUM_ENVS}" \
  --simulator.config.scene.env_spacing=0.0 \
  --terrain.terrain-term.obj-file-path "${OBJ_DIR}" \
  --terrain.terrain-term.num-rows "${NUM_ROWS}" \
  --terrain.terrain-term.num-cols "${NUM_COLS}" \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.save_interval=1000 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}" \
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=True \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  logger:wandb \
  --logger.video.enabled=False \
  --logger.headless_recording=False \
  --logger.video.upload_to_wandb=False \
  --logger.name="g1_videomimic_multiclip_terrain"
