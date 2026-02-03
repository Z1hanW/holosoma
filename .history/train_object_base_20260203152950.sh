#!/usr/bin/env bash
set -euo pipefail

# Base whole-body tracking training with a dynamic object (e.g., large box).
# Override via env vars if needed.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
EXP=${EXP:-g1-29dof-wbt-w-object}
MOTION_FILE=${MOTION_FILE:-src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}
NUM_ENVS=${NUM_ENVS:-12288}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

EXTRA_ARGS=("$@")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent.py \
  "exp:${EXP}" \
  --training.num_envs="${NUM_ENVS}" \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}" \
  logger:wandb \
  --logger.video.interval=2000 \
  "${EXTRA_ARGS[@]}"
