#!/usr/bin/env bash
set -euo pipefail

# Base whole-body tracking training with a dynamic object (large box).

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
EXP=${EXP:-g1-29dof-wbt-w-object}
WANDB_PROJECT=${WANDB_PROJECT:-boxer}
MOTION_FILE=${MOTION_FILE:-src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}
NUM_ENVS=${NUM_ENVS:-12288}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}

EXTRA_ARGS=("$@")

echo "[INFO] Starting training with live Viser on port ${VISER_PORT}"
echo "[INFO] Open: http://localhost:${VISER_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent.py \
  "exp:${EXP}" \
  --training.project="${WANDB_PROJECT}" \
  --training.num_envs="${NUM_ENVS}" \
  --training.enable_viser=True \
  --training.viser_port="${VISER_PORT}" \
  --training.viser_env_id="${VISER_ENV_ID}" \
  --training.viser_update_hz="${VISER_UPDATE_HZ}" \
  --training.viser_sync_to_sim="${VISER_SYNC_TO_SIM}" \
  --training.viser_force_dt="${VISER_FORCE_DT}" \
  --training.viser_recenter="${VISER_RECENTER}" \
  --training.viser_show_scandots="${VISER_SHOW_SCANDOTS}" \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}" \
  --algo.config.save_interval=500 \
  logger:wandb \
  --logger.video.interval=2000 \
  "${EXTRA_ARGS[@]}"
