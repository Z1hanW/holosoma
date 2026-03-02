#!/usr/bin/env bash
set -euo pipefail

# Generalist whole-body tracking training with dynamic object from a motion bank directory.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SIM_ENV_BIN=/home/ubuntu/miniconda3/envs/sim/bin
if ! command -v torchrun >/dev/null 2>&1 && [[ -x "${SIM_ENV_BIN}/torchrun" ]]; then
  export PATH="${SIM_ENV_BIN}:${PATH}"
fi

DEFAULT_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}
EXP=${EXP:-g1-29dof-wbt-w-object-generalist}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_aug_mix_ml"}
OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-""}
NUM_ENVS=${NUM_ENVS:-24576}
NPROC=${NPROC:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

AUTO_PREP_MIXED_BANK=${AUTO_PREP_MIXED_BANK:-0}
MIXED_CLEAN_OUT=${MIXED_CLEAN_OUT:-1}
MIXED_LINK_MODE=${MIXED_LINK_MODE:-symlink}
MIXED_BEHAVE_FILTER=${MIXED_BEHAVE_FILTER:-boxmedium,boxlarge}
MIXED_OMOMO_DIR=${MIXED_OMOMO_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
MIXED_BEHAVE_DIR=${MIXED_BEHAVE_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_obj"}
MIXED_BEHAVE_MAP_FILE=${MIXED_BEHAVE_MAP_FILE:-"${MIXED_BEHAVE_DIR}/_clip_object_urdf_map.json"}

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
ENABLE_VISER=${ENABLE_VISER:-0}

EXTRA_ARGS=("$@")

if [[ "${AUTO_PREP_MIXED_BANK}" != "0" ]]; then
  echo "[INFO] Preparing mixed motion bank into: ${MOTION_DIR}"
  OMOMO_DIR="${MIXED_OMOMO_DIR}" \
  BEHAVE_DIR="${MIXED_BEHAVE_DIR}" \
  OUT_DIR="${MOTION_DIR}" \
  BEHAVE_FILTER="${MIXED_BEHAVE_FILTER}" \
  LINK_MODE="${MIXED_LINK_MODE}" \
  CLEAN_OUT="${MIXED_CLEAN_OUT}" \
  BEHAVE_MAP_FILE="${MIXED_BEHAVE_MAP_FILE}" \
  PREFIX_DATASET=1 \
  bash "${SCRIPT_DIR}/prepare_mixed_object_bank.sh"
fi

if [[ -z "${OBJECT_SPEC_PATH}" ]]; then
  default_map="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ -f "${default_map}" ]]; then
    OBJECT_SPEC_PATH="${default_map}"
    echo "[INFO] Using clip-object URDF map: ${OBJECT_SPEC_PATH}"
  fi
fi

if [[ "${ENABLE_VISER}" == "1" ]]; then
  echo "[INFO] Starting training with live Viser on port ${VISER_PORT}"
  echo "[INFO] Open: http://localhost:${VISER_PORT}"
else
  echo "[INFO] Starting training without Viser"
fi
train_cmd=(
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}"
  src/holosoma/holosoma/train_agent.py
  "exp:${EXP}"
  --training.num_envs="${NUM_ENVS}"
  --training.enable_viser=False
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
  --algo.config.save_interval=500
  logger:wandb
  --logger.video.interval=2000
  "${EXTRA_ARGS[@]}"
)
if [[ "${ENABLE_VISER}" == "1" ]]; then
  train_cmd+=(
    --training.enable_viser=True
    --training.viser_port="${VISER_PORT}"
    --training.viser_env_id="${VISER_ENV_ID}"
    --training.viser_update_hz="${VISER_UPDATE_HZ}"
    --training.viser_sync_to_sim="${VISER_SYNC_TO_SIM}"
    --training.viser_force_dt="${VISER_FORCE_DT}"
    --training.viser_recenter="${VISER_RECENTER}"
    --training.viser_show_scandots="${VISER_SHOW_SCANDOTS}"
  )
fi
if [[ -n "${OBJECT_SPEC_PATH}" ]]; then
  train_cmd+=(--robot.object.object_urdf_path "${OBJECT_SPEC_PATH}")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${train_cmd[@]}"
