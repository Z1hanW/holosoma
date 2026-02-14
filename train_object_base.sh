#!/usr/bin/env bash
set -euo pipefail

# Base whole-body tracking training with a dynamic object (large box).
# Workflow:
#   1) Inspect kinematic motion in IsaacSim ("both" markers: real + motion).
#   2) Close the visualizer window.
#   3) Start training with live Viser updates.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
EXP=${EXP:-g1-29dof-wbt-w-object}
MOTION_FILE=${MOTION_FILE:-src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}
NUM_ENVS=${NUM_ENVS:-12288}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

INSPECT_KINEMATIC_FIRST=${INSPECT_KINEMATIC_FIRST:-1}
INSPECT_KINEMATIC_MODE=${INSPECT_KINEMATIC_MODE:-both}
INSPECT_NUM_ENVS=${INSPECT_NUM_ENVS:-1}
INSPECT_HEADLESS=${INSPECT_HEADLESS:-False}
PREVIEW_CUDA_VISIBLE_DEVICES=${PREVIEW_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES%%,*}}
PREVIEW_CUDA_VISIBLE_DEVICES=${PREVIEW_CUDA_VISIBLE_DEVICES:-0}

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}

INSPECT_ENABLE_VISER=${INSPECT_ENABLE_VISER:-1}
INSPECT_VISER_PORT=${INSPECT_VISER_PORT:-$((RANDOM % 8976 + 1024))}
INSPECT_VISER_ENV_ID=${INSPECT_VISER_ENV_ID:-0}
INSPECT_VISER_UPDATE_HZ=${INSPECT_VISER_UPDATE_HZ:-30}
INSPECT_VISER_SYNC_TO_SIM=${INSPECT_VISER_SYNC_TO_SIM:-True}
INSPECT_VISER_FORCE_DT=${INSPECT_VISER_FORCE_DT:-True}
INSPECT_VISER_RECENTER=${INSPECT_VISER_RECENTER:-True}
INSPECT_VISER_SHOW_SCANDOTS=${INSPECT_VISER_SHOW_SCANDOTS:-False}

EXTRA_ARGS=("$@")

if [[ "${INSPECT_KINEMATIC_FIRST}" != "0" ]]; then
  case "${INSPECT_KINEMATIC_MODE}" in
    both)
      ;;
    *)
      echo "[WARN] Unsupported INSPECT_KINEMATIC_MODE=${INSPECT_KINEMATIC_MODE}; falling back to both."
      INSPECT_KINEMATIC_MODE="both"
      ;;
  esac

  echo "[INFO] Inspect kinematic motion mode: ${INSPECT_KINEMATIC_MODE}"
  if [[ "${INSPECT_ENABLE_VISER}" != "0" ]]; then
    echo "[INFO] Inspect Viser: http://localhost:${INSPECT_VISER_PORT}"
  fi
  echo "[INFO] Launching replay viewer. Close it to start training."
  replay_cmd=(
    python src/holosoma/holosoma/replay.py
    "exp:${EXP}"
    "${EXTRA_ARGS[@]}"
    --training.headless="${INSPECT_HEADLESS}"
    --training.num_envs="${INSPECT_NUM_ENVS}"
    --simulator.config.debug_viz=True
    --simulator.config.scene.env_spacing=0.0
    --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}"
  )
  if [[ "${INSPECT_ENABLE_VISER}" != "0" ]]; then
    replay_cmd+=(
      --training.enable_viser=True
      --training.viser_port="${INSPECT_VISER_PORT}"
      --training.viser_env_id="${INSPECT_VISER_ENV_ID}"
      --training.viser_update_hz="${INSPECT_VISER_UPDATE_HZ}"
      --training.viser_sync_to_sim="${INSPECT_VISER_SYNC_TO_SIM}"
      --training.viser_force_dt="${INSPECT_VISER_FORCE_DT}"
      --training.viser_recenter="${INSPECT_VISER_RECENTER}"
      --training.viser_show_scandots="${INSPECT_VISER_SHOW_SCANDOTS}"
    )
  fi
  CUDA_VISIBLE_DEVICES="${PREVIEW_CUDA_VISIBLE_DEVICES}" "${replay_cmd[@]}"
fi

echo "[INFO] Starting training with live Viser on port ${VISER_PORT}"
echo "[INFO] Open: http://localhost:${VISER_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent.py \
  "exp:${EXP}" \
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
