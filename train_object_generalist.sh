#!/usr/bin/env bash
set -euo pipefail

# Generalist whole-body tracking training with dynamic object from a motion bank directory.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SIM_ENV_BIN=/home/ubuntu/miniconda3/envs/sim/bin
if ! command -v torchrun >/dev/null 2>&1 && [[ -x "${SIM_ENV_BIN}/torchrun" ]]; then
  export PATH="${SIM_ENV_BIN}:${PATH}"
fi
if [[ -x "${SIM_ENV_BIN}/python" ]]; then
  DEFAULT_PYTHON_BIN="${SIM_ENV_BIN}/python"
else
  DEFAULT_PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN=${PYTHON_BIN:-"${DEFAULT_PYTHON_BIN}"}

DEFAULT_CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}
EXP=${EXP:-g1-29dof-wbt-w-object-generalist}
WANDB_PROJECT=${WANDB_PROJECT:-boxer}
DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml"
MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}
OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-""}
NUM_ENVS=${NUM_ENVS:-65536}
NPROC=${NPROC:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
PHYSX_GPU_MAX_RIGID_PATCH_COUNT=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-655360}

TRAIN_DATASETS=${TRAIN_DATASETS:-"omomo,behave"}
AUTO_PREP_MIXED_BANK=${AUTO_PREP_MIXED_BANK:-0}
MIXED_CLEAN_OUT=${MIXED_CLEAN_OUT:-1}
MIXED_LINK_MODE=${MIXED_LINK_MODE:-symlink}
MIXED_BEHAVE_FILTER=${MIXED_BEHAVE_FILTER:-boxmedium,boxlarge}
MIXED_OMOMO_DIR=${MIXED_OMOMO_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
MIXED_BEHAVE_DIR=${MIXED_BEHAVE_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"}
MIXED_BEHAVE_MAP_FILE=${MIXED_BEHAVE_MAP_FILE:-"${MIXED_BEHAVE_DIR}/_clip_object_urdf_map.json"}

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
VISER_LOAD_URDF=${VISER_LOAD_URDF:-1}
ENABLE_VISER=${ENABLE_VISER:-0}
DEBUG_MODE=${DEBUG_MODE:-${DEBUG_MODEL:-off}}
CURRICULUM=${CURRICULUM:-0}
PERCEPTION=${PERCEPTION:-none}
LEGACY_OBS=${LEGACY_OBS:-0}
GENERALIST_CONTACT_REWARD_ENABLED=${GENERALIST_CONTACT_REWARD_ENABLED:-1}
GENERALIST_CONTACT_REWARD_MODE=${GENERALIST_CONTACT_REWARD_MODE:-tanh}
GENERALIST_CONTACT_REWARD_THRESHOLD=${GENERALIST_CONTACT_REWARD_THRESHOLD:-1.0}
GENERALIST_CONTACT_REWARD_FORCE_SCALE=${GENERALIST_CONTACT_REWARD_FORCE_SCALE:-25.0}
GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT:-0.30}
GENERALIST_ARM_CONTACT_REWARD_WEIGHT=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT:-0.20}
GENERALIST_PALM_CONTACT_REWARD_WEIGHT=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT:-0.10}

SEQUENCE_NAME=${SEQUENCE_NAME:-""}
if [[ "$#" -gt 0 ]]; then
  SEQUENCE_NAME="$1"
  shift
fi
EXTRA_ARGS=("$@")
if [[ -n "${SEQUENCE_NAME}" ]]; then
  echo "[INFO] Sequence name: ${SEQUENCE_NAME}"
fi

datasets_normalized=$(echo "${TRAIN_DATASETS}" | tr '[:upper:]' '[:lower:]' | tr -d '[]')
IFS=',' read -r -a dataset_tokens <<< "${datasets_normalized}"
USE_OMOMO=0
USE_BEHAVE=0
for token in "${dataset_tokens[@]}"; do
  dataset_key=$(echo "${token}" | tr -d '[:space:]')
  if [[ -z "${dataset_key}" ]]; then
    continue
  fi
  case "${dataset_key}" in
    omomo)
      USE_OMOMO=1
      ;;
    behave)
      USE_BEHAVE=1
      ;;
    *)
      echo "[ERROR] Unsupported dataset '${dataset_key}' in TRAIN_DATASETS='${TRAIN_DATASETS}'. Use only omomo,behave." >&2
      exit 2
      ;;
  esac
done
if [[ "${USE_OMOMO}" != "1" && "${USE_BEHAVE}" != "1" ]]; then
  echo "[ERROR] TRAIN_DATASETS='${TRAIN_DATASETS}' selected no datasets. Use omomo and/or behave." >&2
  exit 2
fi

selected_datasets=()
if [[ "${USE_OMOMO}" == "1" ]]; then
  selected_datasets+=("omomo")
fi
if [[ "${USE_BEHAVE}" == "1" ]]; then
  selected_datasets+=("behave")
fi
if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
  if [[ "${USE_OMOMO}" == "1" && "${USE_BEHAVE}" == "1" ]]; then
    MOTION_DIR="${DEFAULT_MOTION_DIR}"
  elif [[ "${USE_OMOMO}" == "1" ]]; then
    MOTION_DIR="${MIXED_OMOMO_DIR}"
  else
    MOTION_DIR="${MIXED_BEHAVE_DIR}"
  fi
fi
echo "[INFO] TRAIN_DATASETS (resolved): $(IFS=,; echo "${selected_datasets[*]}")"
echo "[INFO] MOTION_DIR: ${MOTION_DIR}"

if [[ "${AUTO_PREP_MIXED_BANK}" != "0" ]]; then
  if [[ "${USE_OMOMO}" == "1" && "${USE_BEHAVE}" == "1" ]]; then
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
  else
    echo "[INFO] AUTO_PREP_MIXED_BANK is enabled but skipped for single-dataset training ($(IFS=,; echo "${selected_datasets[*]}"))."
  fi
fi

if [[ -z "${OBJECT_SPEC_PATH}" ]]; then
  default_map="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ -f "${default_map}" ]]; then
    OBJECT_SPEC_PATH="${default_map}"
    echo "[INFO] Using clip-object URDF map: ${OBJECT_SPEC_PATH}"
  elif [[ "${USE_BEHAVE}" == "1" && -f "${MIXED_BEHAVE_MAP_FILE}" ]]; then
    OBJECT_SPEC_PATH="${MIXED_BEHAVE_MAP_FILE}"
    echo "[INFO] Using BEHAVE clip-object URDF map: ${OBJECT_SPEC_PATH}"
  elif [[ "${USE_BEHAVE}" == "1" ]]; then
    echo "[WARN] BEHAVE selected but no clip-object URDF map found. Training may fallback to single-object URDF." >&2
  fi
fi

# BEHAVE requires per-clip URDF mapping; do not silently fall back to a single URDF.
if [[ "${USE_BEHAVE}" == "1" ]]; then
  if [[ -z "${OBJECT_SPEC_PATH}" || ! -f "${OBJECT_SPEC_PATH}" ]]; then
    echo "[ERROR] BEHAVE training requires a valid _clip_object_urdf_map.json, but OBJECT_SPEC_PATH is missing." >&2
    echo "[ERROR] Expected map example: ${MIXED_BEHAVE_DIR}/_clip_object_urdf_map.json" >&2
    exit 2
  fi
  python - <<'PY' "${OBJECT_SPEC_PATH}" || exit 2
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    payload = payload["clips"]
if not isinstance(payload, dict) or not payload:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {path}")
has_urdf = False
for entry in payload.values():
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()
    else:
        urdf = ""
    if urdf:
        has_urdf = True
        break
if not has_urdf:
    raise SystemExit(f"[ERROR] Object map has no valid object_urdf_path entries: {path}")
print(f"[INFO] Validated BEHAVE object map: {path}")
PY
fi

DEBUG_MODE=$(echo "${DEBUG_MODE}" | tr '[:upper:]' '[:lower:]')
case "${DEBUG_MODE}" in
  ""|0|off|none)
    DEBUG_MODE="off"
    ;;
  1|replay)
    DEBUG_MODE="replay"
    ;;
  toy)
    DEBUG_MODE="toy"
    ;;
  *)
    echo "[ERROR] Unsupported DEBUG_MODE='${DEBUG_MODE}'. Use one of: off, replay, toy"
    exit 2
    ;;
esac

if [[ "${DEBUG_MODE}" == "replay" || "${DEBUG_MODE}" == "toy" ]]; then
  if [[ -n "${OBJECT_SPEC_PATH}" && -f "${OBJECT_SPEC_PATH}" ]]; then
    DEBUG_URDF_COUNT=$(python - <<'PY' "${OBJECT_SPEC_PATH}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    payload = payload["clips"]
if not isinstance(payload, dict):
    print(0)
    raise SystemExit(0)

seen = set()
for _, entry in payload.items():
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()
    else:
        urdf = ""
    if urdf:
        seen.add(str(Path(urdf).resolve()))
print(len(seen))
PY
)
    if [[ "${DEBUG_URDF_COUNT}" =~ ^[0-9]+$ ]] && (( DEBUG_URDF_COUNT > 0 )); then
      NUM_ENVS="${DEBUG_URDF_COUNT}"
      echo "[INFO] DEBUG_MODE=${DEBUG_MODE}: using one env per unique URDF => NUM_ENVS=${NUM_ENVS}"
    else
      echo "[WARN] DEBUG_MODE=${DEBUG_MODE}: failed to infer URDF count from ${OBJECT_SPEC_PATH}; keeping NUM_ENVS=${NUM_ENVS}"
    fi
  else
    echo "[WARN] DEBUG_MODE=${DEBUG_MODE}: OBJECT_SPEC_PATH missing; keeping NUM_ENVS=${NUM_ENVS}"
  fi
  ENABLE_VISER=1
  NPROC=1
fi

if [[ "${ENABLE_VISER}" == "1" ]]; then
  echo "[INFO] Starting training with live Viser on port ${VISER_PORT}"
  echo "[INFO] Open: http://localhost:${VISER_PORT}"
  echo "[INFO] Viser runtime source: Isaac Sim state; URDF mesh loading in Viser = ${VISER_LOAD_URDF}"
else
  echo "[INFO] Starting training without Viser"
fi

legacy_obs_normalized=$(echo "${LEGACY_OBS}" | tr '[:upper:]' '[:lower:]')
if [[ "${legacy_obs_normalized}" == "1" || "${legacy_obs_normalized}" == "true" ]]; then
  if [[ "${EXP}" == "g1-29dof-wbt-w-object-generalist" ]]; then
    EXP="g1-29dof-wbt-w-object-generalist-legacy-obs"
  fi
  echo "[INFO] LEGACY_OBS enabled: using legacy actor observation (175-dim, no object velocity terms)."
  echo "[INFO] Resolved EXP: ${EXP}"
fi

contact_reward_enabled_normalized=$(echo "${GENERALIST_CONTACT_REWARD_ENABLED}" | tr '[:upper:]' '[:lower:]')
case "${contact_reward_enabled_normalized}" in
  1|true|yes|on)
    GENERALIST_CONTACT_REWARD_ENABLED_FLAG=1
    ;;
  0|false|no|off|"")
    GENERALIST_CONTACT_REWARD_ENABLED_FLAG=0
    ;;
  *)
    echo "[ERROR] GENERALIST_CONTACT_REWARD_ENABLED must be one of: 0/1/true/false/yes/no/on/off. Got: ${GENERALIST_CONTACT_REWARD_ENABLED}" >&2
    exit 2
    ;;
esac

if [[ "${GENERALIST_CONTACT_REWARD_ENABLED_FLAG}" != "1" ]]; then
  GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=0.0
  GENERALIST_ARM_CONTACT_REWARD_WEIGHT=0.0
  GENERALIST_PALM_CONTACT_REWARD_WEIGHT=0.0
fi

echo "[INFO] Generalist contact reward enabled: ${GENERALIST_CONTACT_REWARD_ENABLED_FLAG}"
echo "[INFO] Generalist contact reward mode=${GENERALIST_CONTACT_REWARD_MODE} threshold=${GENERALIST_CONTACT_REWARD_THRESHOLD} force_scale=${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
echo "[INFO] Generalist contact reward weights torso=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT} arms=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT} palms=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT}"

train_cmd=(
  src/holosoma/holosoma/train_agent.py
  "exp:${EXP}"
  "perception:${PERCEPTION}"
  --training.project="${WANDB_PROJECT}"
  --training.num-envs="${NUM_ENVS}"
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
  --algo.config.save-interval=500
  --simulator.config.sim.physx.gpu-max-rigid-patch-count="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT}"
  --reward.terms.body_contact_reward_torso.weight="${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT}"
  --reward.terms.body_contact_reward_arms.weight="${GENERALIST_ARM_CONTACT_REWARD_WEIGHT}"
  --reward.terms.body_contact_reward_palms.weight="${GENERALIST_PALM_CONTACT_REWARD_WEIGHT}"
  --reward.terms.body_contact_reward_torso.params.reward_mode="${GENERALIST_CONTACT_REWARD_MODE}"
  --reward.terms.body_contact_reward_arms.params.reward_mode="${GENERALIST_CONTACT_REWARD_MODE}"
  --reward.terms.body_contact_reward_palms.params.reward_mode="${GENERALIST_CONTACT_REWARD_MODE}"
  --reward.terms.body_contact_reward_torso.params.threshold="${GENERALIST_CONTACT_REWARD_THRESHOLD}"
  --reward.terms.body_contact_reward_arms.params.threshold="${GENERALIST_CONTACT_REWARD_THRESHOLD}"
  --reward.terms.body_contact_reward_palms.params.threshold="${GENERALIST_CONTACT_REWARD_THRESHOLD}"
  --reward.terms.body_contact_reward_torso.params.force_scale="${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
  --reward.terms.body_contact_reward_arms.params.force_scale="${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
  --reward.terms.body_contact_reward_palms.params.force_scale="${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
)
if [[ "${DEBUG_MODE}" == "replay" || "${DEBUG_MODE}" == "toy" ]]; then
  train_cmd=("${PYTHON_BIN}" "${train_cmd[@]}")
else
  train_cmd=(torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" "${train_cmd[@]}")
fi
if [[ "${DEBUG_MODE}" == "replay" ]]; then
  train_cmd+=(--training.debug=True)
fi
if [[ "${DEBUG_MODE}" == "toy" ]]; then
  train_cmd+=(--training.toy-mode=True)
  train_cmd+=(--training.viser-env-count="${NUM_ENVS}")
fi
if [[ "${ENABLE_VISER}" == "1" ]]; then
  train_cmd+=(
    --training.enable-viser=True
    --training.viser-port="${VISER_PORT}"
    --training.viser-env-id="${VISER_ENV_ID}"
    --training.viser-update-hz="${VISER_UPDATE_HZ}"
    --training.viser-sync-to-sim="${VISER_SYNC_TO_SIM}"
    --training.viser-force-dt="${VISER_FORCE_DT}"
    --training.viser-recenter="${VISER_RECENTER}"
    --training.viser-show-scandots="${VISER_SHOW_SCANDOTS}"
  )
fi
if [[ -n "${OBJECT_SPEC_PATH}" ]]; then
  train_cmd+=(--robot.object.object-urdf-path "${OBJECT_SPEC_PATH}")
fi
if [[ -n "${SEQUENCE_NAME}" ]]; then
  train_cmd+=(--training.name="${SEQUENCE_NAME}")
fi
if [[ "${CURRICULUM}" == "1" || "${CURRICULUM,,}" == "true" ]]; then
  echo "[INFO] Enabling w-object curriculum."
  train_cmd+=(--curriculum.setup-terms.w-object-difficulty-curriculum.params.enabled=True)
fi
train_cmd+=("${EXTRA_ARGS[@]}")
train_cmd+=(logger:wandb)
if [[ -n "${SEQUENCE_NAME}" ]]; then
  train_cmd+=(--logger.name="${SEQUENCE_NAME}")
fi
echo "[INFO] Training video recording disabled."
train_cmd+=(--logger.video.enabled=False)
train_cmd+=(--logger.headless_recording=False)
train_cmd+=(--logger.video.upload_to_wandb=False)
VISER_LOAD_URDF="${VISER_LOAD_URDF}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${train_cmd[@]}"
