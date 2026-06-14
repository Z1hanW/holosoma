#!/usr/bin/env bash
set -euo pipefail

# DS-box object generalist training.
#
# This launcher prepares a trainable motion bank from:
# - raw motion clips in `data/ds_box_data[/scale_mix_all]/train_g1_w_obj`
# - per-clip box geometry in `data/ds_box_data[/scale_mix_all]/train_g1_w_obj_geometry`
#
# The prepared bank augments each clip with:
# - `object_size`
# - `object_name`
# - `object_urdf_path`
# - `_clip_object_urdf_map.json`

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/object_generalist_ds_paths.sh"

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
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
export CUDA_VISIBLE_DEVICES

detect_nproc() {
  local gpu_count=""
  if gpu_count="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null
import torch

print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"; then
    gpu_count="${gpu_count//[[:space:]]/}"
  fi

  if [[ ! "${gpu_count}" =~ ^[0-9]+$ || "${gpu_count}" == "0" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')"
    fi
  fi

  if [[ ! "${gpu_count}" =~ ^[0-9]+$ || "${gpu_count}" == "0" ]]; then
    echo "[ERROR] Failed to detect available CUDA GPUs. Set NPROC explicitly." >&2
    exit 1
  fi

  echo "${gpu_count}"
}

WANDB_PROJECT_FROM_ENV=0
if [[ -n "${WANDB_PROJECT+x}" ]]; then
  WANDB_PROJECT_FROM_ENV=1
fi
WANDB_RESUME_SAME_RUN_FROM_ENV=0
if [[ -n "${WANDB_RESUME_SAME_RUN+x}" ]]; then
  WANDB_RESUME_SAME_RUN_FROM_ENV=1
fi
EXP=${EXP:-g1-29dof-wbt-w-object-generalist}
COMMAND_CONFIG=${COMMAND_CONFIG:-g1-29dof-wbt-w-object-generalist}
REWARD_CONFIG=${REWARD_CONFIG:-g1-29dof-wbt-w-object-generalist}
REWARD_CONFIG_NORMALIZED=$(echo "${REWARD_CONFIG}" | tr '[:upper:]_' '[:lower:]-')
USE_TEACHER_ROLLOUT_REWARD=0
case "${REWARD_CONFIG_NORMALIZED}" in
  *r2s-rollout-reference-guidance*|*r2s-rollout-ref*)
    USE_TEACHER_ROLLOUT_REWARD=1
    ;;
esac
USE_OFFLINE_CONTACT_GUIDANCE=0
case "${REWARD_CONFIG_NORMALIZED}" in
  *offline-contact-guidance*)
    USE_OFFLINE_CONTACT_GUIDANCE=1
    ;;
esac
WANDB_PROJECT=${WANDB_PROJECT:-boxer}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_RUN_ID=${WANDB_RUN_ID:-${RESUME_WANDB_ID:-""}}
WANDB_RESUME=${WANDB_RESUME:-""}
WANDB_RESUME_SAME_RUN=${WANDB_RESUME_SAME_RUN:-auto}
WANDB_MODEL_FILE=${WANDB_MODEL_FILE:-${RESUME_MODEL_FILE:-""}}
RESUME_CKPT=${RESUME_CKPT:-${RESUME_CHECKPOINT:-""}}
POLICY_INIT_CKPT=${POLICY_INIT_CKPT:-${POLICY_INIT_CHECKPOINT:-""}}
RESUME_STEP_RAW=${RESUME_STEP:-""}
DS_DATA_ROOT=${DS_DATA_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}
DS_DATA_ROOT="$(ogds_resolve_data_root "${DS_DATA_ROOT}")"
ASSERT_NEW_DS_DATA=${ASSERT_NEW_DS_DATA:-1}
NEW_DS_DATA_ROOT=${NEW_DS_DATA_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}
NEW_DS_EXPECTED_PREPARED_TOTAL=${NEW_DS_EXPECTED_PREPARED_TOTAL:-408}
NEW_DS_EXPECTED_PREPARED_BOX=${NEW_DS_EXPECTED_PREPARED_BOX:-372}
NEW_DS_EXPECTED_PREPARED_BEHAVE=${NEW_DS_EXPECTED_PREPARED_BEHAVE:-30}
NEW_DS_EXPECTED_PREPARED_LC=${NEW_DS_EXPECTED_PREPARED_LC:-6}
NEW_DS_EXPECTED_PREPARED_OMOMO=${NEW_DS_EXPECTED_PREPARED_OMOMO:-0}
NEW_DS_EXPECTED_MIXED_TOTAL=${NEW_DS_EXPECTED_MIXED_TOTAL:-434}
NEW_DS_EXPECTED_MIXED_BOX=${NEW_DS_EXPECTED_MIXED_BOX:-372}
NEW_DS_EXPECTED_MIXED_BEHAVE=${NEW_DS_EXPECTED_MIXED_BEHAVE:-0}
NEW_DS_EXPECTED_MIXED_LC=${NEW_DS_EXPECTED_MIXED_LC:-0}
NEW_DS_EXPECTED_MIXED_OMOMO=${NEW_DS_EXPECTED_MIXED_OMOMO:-62}
DEFAULT_DS_RAW_MOTION_DIR="$(ogds_default_raw_motion_dir "${DS_DATA_ROOT}")"
DEFAULT_DS_GEOMETRY_DIR="$(ogds_default_geometry_dir "${DS_DATA_ROOT}")"
DEFAULT_DS_PREPARED_MOTION_DIR="$(ogds_default_motion_dir "${DS_DATA_ROOT}" pure-sd)"
DEFAULT_MIX_NAIVE_MOTION_DIR="$(ogds_default_motion_dir "${DS_DATA_ROOT}" mix-naive)"
# Optional strict count checks.
# Leave unset to validate structure/fields only so newer banks with different clip counts still run.
DS_EXPECTED_TOTAL=${DS_EXPECTED_TOTAL:-""}
MIX_NAIVE_EXPECTED_TOTAL=${MIX_NAIVE_EXPECTED_TOTAL:-""}
MIX_NAIVE_EXPECTED_DS=${MIX_NAIVE_EXPECTED_DS:-""}
MIX_NAIVE_EXPECTED_OMOMO=${MIX_NAIVE_EXPECTED_OMOMO:-""}
DATA_SUBSET_MODE=${DATA_SUBSET_MODE:-${SAMPLE_MODE:-}}
DATA_SUBSET_SEED=${DATA_SUBSET_SEED:-0}
DATA_SUBSET_BANK_ROOT=${DATA_SUBSET_BANK_ROOT:-""}
MOTION_DIR_FROM_ENV=0
if [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_DIR_FROM_ENV=1
fi
MOTION_DIR=${MOTION_DIR:-""}
RAW_MOTION_DIR=${RAW_MOTION_DIR:-"${DEFAULT_DS_RAW_MOTION_DIR}"}
OBJ_DIR=${OBJ_DIR:-"${DEFAULT_DS_GEOMETRY_DIR}"}
PREPARED_MOTION_DIR=${PREPARED_MOTION_DIR:-""}
OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-""}
AVAILABLE_GPU_COUNT=$(detect_nproc)
NPROC=${NPROC:-${AVAILABLE_GPU_COUNT}}
if [[ ! "${NPROC}" =~ ^[0-9]+$ || "${NPROC}" == "0" ]]; then
  echo "[ERROR] NPROC must be a positive integer. Got: ${NPROC}" >&2
  exit 2
fi
if (( NPROC > AVAILABLE_GPU_COUNT )); then
  echo "[ERROR] Requested NPROC=${NPROC}, but only ${AVAILABLE_GPU_COUNT} CUDA GPU(s) are available." >&2
  exit 2
fi
PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
if [[ ! "${NNODES}" =~ ^[0-9]+$ || "${NNODES}" == "0" ]]; then
  echo "[ERROR] NNODES must be a positive integer. Got: ${NNODES}" >&2
  exit 2
fi
if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] NODE_RANK must be a non-negative integer. Got: ${NODE_RANK}" >&2
  exit 2
fi
if (( NODE_RANK >= NNODES )); then
  echo "[ERROR] NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}." >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR}" ]]; then
  echo "[ERROR] MASTER_ADDR is required for multi-node torchrun." >&2
  exit 2
fi
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT:-33554432}
PHYSX_GPU_MAX_RIGID_PATCH_COUNT=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-4194304}
PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-134217728}
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-134217728}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-16777216}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-67108864}
PHYSX_GPU_HEAP_CAPACITY=${PHYSX_GPU_HEAP_CAPACITY:-67108864}
PHYSX_GPU_TEMP_BUFFER_CAPACITY=${PHYSX_GPU_TEMP_BUFFER_CAPACITY:-16777216}
HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-1}
OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-${HOLOSOMA_OBJECT_SPAWN_MODE:-single_slot_multi_urdf}}
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-}
DISABLE_ACTOR_HISTORY=${DISABLE_ACTOR_HISTORY:-True}
DISABLE_CRITIC_HISTORY=${DISABLE_CRITIC_HISTORY:-True}
POLICY_HISTORY_LENGTH=${POLICY_HISTORY_LENGTH:-${HISTORY_LENGTH:-}}
TEACHER_ROLLOUT_REFERENCE_ROOT=${TEACHER_ROLLOUT_REFERENCE_ROOT:-}
CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE:-}
CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS=${CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS:-}
CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG=${CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG:-}
PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE=""
if [[ -n "${OBJECT_GEOMETRY_MODE}" ]]; then
  case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on|primitive|primitives|box|cuboid)
      OBJECT_SPAWN_MODE="primitive"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="primitive"
      ;;
    0|false|no|off|mesh|urdf|disable|disabled)
      OBJECT_SPAWN_MODE="urdf"
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="mesh"
      ;;
    *)
      echo "[ERROR] OBJECT_GEOMETRY_MODE must be one of: on/off/primitive/mesh. Got: ${OBJECT_GEOMETRY_MODE}" >&2
      exit 2
      ;;
  esac
else
  case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
    primitive|primitives|box|cuboid)
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="primitive"
      ;;
    urdf|mesh|off|disable|disabled|single_slot_multi_urdf|single-slot-multi-urdf|single_slot|single-slot|heterogeneous_single_slot|heterogeneous-single-slot)
      PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE="mesh"
      ;;
  esac
fi
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK
ACTOR_LR=${ACTOR_LR:-1e-05}
CRITIC_LR=${CRITIC_LR:-1e-05}
NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-7}
CLIP_WEIGHTING_STRATEGY=${CLIP_WEIGHTING_STRATEGY:-success_rate_adaptive}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
TRAINING_SEED=${TRAINING_SEED:-${SEED:-}}
RANDOMIZATION_PRESET=${RANDOMIZATION_PRESET:-${RANDOMIZATION:-}}
INIT_AT_RANDOM_EP_LEN=${INIT_AT_RANDOM_EP_LEN:-}

AUTO_PREP_DS_BANK=${AUTO_PREP_DS_BANK:-auto}
DS_PREP_CLEAN_OUT=${DS_PREP_CLEAN_OUT:-1}
DS_OBJECT_MASS=${DS_OBJECT_MASS:-0.1}
DS_OBJECT_COLOR_RGBA=${DS_OBJECT_COLOR_RGBA:-"0.7 0.8 0.9 1"}
PREP_ONLY=${PREP_ONLY:-0}
DATA_MODE=${DATA_MODE:-pure-sd}
STRICT_DEFAULT_DS_BANK_VALIDATION=${STRICT_DEFAULT_DS_BANK_VALIDATION:-1}
DEFAULT_POSE_PREPEND_ENABLED=${DEFAULT_POSE_PREPEND_ENABLED:-1}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.2}
MIX_CURRICULUM_OMOMO_PREFIXES=${MIX_CURRICULUM_OMOMO_PREFIXES:-'["sub"]'}
MIX_CURRICULUM_STAGE_START_ITERATIONS=${MIX_CURRICULUM_STAGE_START_ITERATIONS:-'[0, 1500, 2000, 2500, 3000, 3500]'}
MIX_CURRICULUM_OMOMO_PROBABILITIES=${MIX_CURRICULUM_OMOMO_PROBABILITIES:-'[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]'}
MIX_NAIVE_FIXED_OMOMO_PREFIXES=${MIX_NAIVE_FIXED_OMOMO_PREFIXES:-'["sub"]'}
MIX_NAIVE_FIXED_STAGE_START_ITERATIONS=${MIX_NAIVE_FIXED_STAGE_START_ITERATIONS:-'[0]'}
MIX_NAIVE_FIXED_OMOMO_PROBABILITIES=${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES:-""}
PURE_REAL_OMOMO_PREFIXES=${PURE_REAL_OMOMO_PREFIXES:-'["sub"]'}
FIX_OMOMO_QUATER_PREFIXES=${FIX_OMOMO_QUATER_PREFIXES:-${FIX_REAL_OMOMO_PREFIXES:-'["sub"]'}}
FIX_OMOMO_QUATER_ENV_FRACTION=${FIX_OMOMO_QUATER_ENV_FRACTION:-${FIX_REAL_OMOMO_ENV_FRACTION:-0.25}}
RESUME_PRESET_MODE=""
RESUME_PRESET_RUN_URL="https://wandb.ai/zihanw22/boxer/runs/sw8scopo"
RESUME_PRESET_STEP=10000
RESUME_PRESET_MIX_NAIVE_OMOMO_PROBABILITIES='[0.3]'

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
PURE_SD_REWARD_PROFILE_RAW=${PURE_SD_REWARD_PROFILE:-default}
PURE_SD_REWARD_PROFILE=$(echo "${PURE_SD_REWARD_PROFILE_RAW}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
GENERALIST_CONTACT_REWARD_ENABLED=${GENERALIST_CONTACT_REWARD_ENABLED:-1}
GENERALIST_CONTACT_REWARD_MODE=${GENERALIST_CONTACT_REWARD_MODE:-tanh}
GENERALIST_CONTACT_REWARD_THRESHOLD=${GENERALIST_CONTACT_REWARD_THRESHOLD:-1.0}
GENERALIST_CONTACT_REWARD_FORCE_SCALE=${GENERALIST_CONTACT_REWARD_FORCE_SCALE:-25.0}
DEFAULT_AS_CONTACT_EXPORT_ROOT="data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513/contact_export_from_retarget"
CONTACT_EXPORT_ROOT=${CONTACT_EXPORT_ROOT:-${AS_CONTACT_EXPORT_ROOT:-}}
CONTACT_EXPORT_CLIPS_ROOT=""
OFFLINE_WRIST_TARGET_GUIDANCE_WEIGHT=${OFFLINE_WRIST_TARGET_GUIDANCE_WEIGHT:-5.0}
OFFLINE_CONTACT_GUIDANCE_WEIGHT=${OFFLINE_CONTACT_GUIDANCE_WEIGHT:-10.0}
OFFLINE_CONTACT_POSITION_SIGMA=${OFFLINE_CONTACT_POSITION_SIGMA:-0.08}
OFFLINE_CONTACT_FORCE_THRESHOLD=${OFFLINE_CONTACT_FORCE_THRESHOLD:-1.0}
OFFLINE_CONTACT_FORCE_SIGMA=${OFFLINE_CONTACT_FORCE_SIGMA:-10.0}
OFFLINE_CONTACT_SCHEDULE_RELAX_STEPS=${OFFLINE_CONTACT_SCHEDULE_RELAX_STEPS:-5}
ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT:-}
REFERENCE_ROOT_POS_W=${REFERENCE_ROOT_POS_W:-0.5}
REFERENCE_ROOT_ORI_W=${REFERENCE_ROOT_ORI_W:-0.5}
REFERENCE_FULL_BODY_POS_W=${REFERENCE_FULL_BODY_POS_W:-1.0}
REFERENCE_FULL_BODY_ORI_W=${REFERENCE_FULL_BODY_ORI_W:-1.0}
REFERENCE_FULL_BODY_LIN_VEL_W=${REFERENCE_FULL_BODY_LIN_VEL_W:-1.0}
REFERENCE_FULL_BODY_ANG_VEL_W=${REFERENCE_FULL_BODY_ANG_VEL_W:-1.0}
REFERENCE_OBJECT_POS_W=${REFERENCE_OBJECT_POS_W:-1.0}
REFERENCE_OBJECT_ORI_W=${REFERENCE_OBJECT_ORI_W:-1.0}
REFERENCE_ACTION_RATE_L2_W=${REFERENCE_ACTION_RATE_L2_W:--0.1}
REFERENCE_ROOT_POS_SIGMA=${REFERENCE_ROOT_POS_SIGMA:-0.3}
REFERENCE_ROOT_ORI_SIGMA=${REFERENCE_ROOT_ORI_SIGMA:-0.4}
REFERENCE_FULL_BODY_POS_SIGMA=${REFERENCE_FULL_BODY_POS_SIGMA:-0.3}
REFERENCE_FULL_BODY_ORI_SIGMA=${REFERENCE_FULL_BODY_ORI_SIGMA:-0.4}
REFERENCE_FULL_BODY_LIN_VEL_SIGMA=${REFERENCE_FULL_BODY_LIN_VEL_SIGMA:-1.0}
REFERENCE_FULL_BODY_ANG_VEL_SIGMA=${REFERENCE_FULL_BODY_ANG_VEL_SIGMA:-3.14}
REFERENCE_OBJECT_POS_SIGMA=${REFERENCE_OBJECT_POS_SIGMA:-0.3}
REFERENCE_OBJECT_ORI_SIGMA=${REFERENCE_OBJECT_ORI_SIGMA:-0.4}
GENERALIST_LIMITS_DOF_POS_WEIGHT=${GENERALIST_LIMITS_DOF_POS_WEIGHT:--10.0}

if [[ -n "${TRAINING_SEED}" ]]; then
  if [[ ! "${TRAINING_SEED}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TRAINING_SEED/SEED must be an integer. Got: ${TRAINING_SEED}" >&2
    exit 2
  fi
fi

if [[ -n "${INIT_AT_RANDOM_EP_LEN}" ]]; then
  case "$(echo "${INIT_AT_RANDOM_EP_LEN}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      INIT_AT_RANDOM_EP_LEN=True
      ;;
    0|false|no|off)
      INIT_AT_RANDOM_EP_LEN=False
      ;;
    *)
      echo "[ERROR] INIT_AT_RANDOM_EP_LEN must be a boolean. Got: ${INIT_AT_RANDOM_EP_LEN}" >&2
      exit 2
      ;;
  esac
fi

case "$(echo "${DISABLE_ACTOR_HISTORY}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    DISABLE_ACTOR_HISTORY=True
    ;;
  0|false|no|off)
    DISABLE_ACTOR_HISTORY=False
    ;;
  *)
    echo "[ERROR] DISABLE_ACTOR_HISTORY must be a boolean. Got: ${DISABLE_ACTOR_HISTORY}" >&2
    exit 2
    ;;
esac
case "$(echo "${DISABLE_CRITIC_HISTORY}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    DISABLE_CRITIC_HISTORY=True
    ;;
  0|false|no|off)
    DISABLE_CRITIC_HISTORY=False
    ;;
  *)
    echo "[ERROR] DISABLE_CRITIC_HISTORY must be a boolean. Got: ${DISABLE_CRITIC_HISTORY}" >&2
    exit 2
    ;;
esac
if [[ -n "${POLICY_HISTORY_LENGTH}" && ( ! "${POLICY_HISTORY_LENGTH}" =~ ^[0-9]+$ || "${POLICY_HISTORY_LENGTH}" == "0" ) ]]; then
  echo "[ERROR] POLICY_HISTORY_LENGTH/HISTORY_LENGTH must be a positive integer. Got: ${POLICY_HISTORY_LENGTH}" >&2
  exit 2
fi

if [[ -n "${RANDOMIZATION_PRESET}" ]]; then
  case "${RANDOMIZATION_PRESET}" in
    none|disabled|t1_29dof|g1_29dof|g1_29dof_wbt|g1_29dof_wbt_with_action_delay|g1_29dof_wbt_w_object|g1_29dof_wbt_w_object_with_action_delay)
      ;;
    *)
      echo "[ERROR] RANDOMIZATION/RANDOMIZATION_PRESET must be one of:" >&2
      echo "[ERROR]   none, disabled, t1_29dof, g1_29dof, g1_29dof_wbt, g1_29dof_wbt_with_action_delay, g1_29dof_wbt_w_object, g1_29dof_wbt_w_object_with_action_delay" >&2
      echo "[ERROR] Got: ${RANDOMIZATION_PRESET}" >&2
      exit 2
      ;;
  esac
fi

reject_legacy_converted_res_path() {
  local name="$1"
  local path_value="$2"
  if [[ -z "${path_value}" ]]; then
    return 0
  fi
  case "${path_value}" in
    *src/holosoma_retargeting/converted_res*)
      echo "[ERROR] ${name} points to legacy converted_res data: ${path_value}" >&2
      echo "[ERROR] Use the DS bank under data/ds_box_data instead." >&2
      exit 2
      ;;
  esac
}

reject_legacy_converted_res_path "MOTION_DIR" "${MOTION_DIR}"
reject_legacy_converted_res_path "RAW_MOTION_DIR" "${RAW_MOTION_DIR}"
reject_legacy_converted_res_path "OBJ_DIR" "${OBJ_DIR}"
reject_legacy_converted_res_path "PREPARED_MOTION_DIR" "${PREPARED_MOTION_DIR}"

normalize_resume_step() {
  local raw="$1"
  local compact="${raw//[[:space:]_]/}"
  if [[ -z "${compact}" ]]; then
    echo ""
    return 0
  fi
  if [[ "${compact}" =~ ^([0-9]+)[kK]$ ]]; then
    echo $((10#${BASH_REMATCH[1]} * 1000))
    return 0
  fi
  if [[ "${compact}" =~ ^[0-9]+$ ]]; then
    echo $((10#${compact}))
    return 0
  fi
  return 1
}

resolve_reward_profile_defaults() {
  local requested_profile="$1"
  local requested_profile_raw="$2"

  case "${requested_profile}" in
    ""|default)
      ACTIVE_REWARD_PROFILE="default"
      DEFAULT_GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=0.30
      DEFAULT_GENERALIST_ARM_CONTACT_REWARD_WEIGHT=0.20
      DEFAULT_GENERALIST_PALM_CONTACT_REWARD_WEIGHT=0.10
      DEFAULT_ROOT_POS_W=0.5
      DEFAULT_ROOT_ORI_W=0.5
      DEFAULT_FULL_BODY_POS_W=1.0
      DEFAULT_FULL_BODY_ORI_W=1.0
      DEFAULT_FULL_BODY_LIN_VEL_W=1.0
      DEFAULT_FULL_BODY_ANG_VEL_W=1.0
      DEFAULT_OBJECT_POS_W=1.0
      DEFAULT_OBJECT_ORI_W=1.0
      DEFAULT_ROOT_POS_SIGMA=0.3
      DEFAULT_ROOT_ORI_SIGMA=0.4
      DEFAULT_FULL_BODY_POS_SIGMA=0.3
      DEFAULT_FULL_BODY_ORI_SIGMA=0.4
      DEFAULT_FULL_BODY_LIN_VEL_SIGMA=1.0
      DEFAULT_FULL_BODY_ANG_VEL_SIGMA=3.14
      DEFAULT_OBJECT_POS_SIGMA=0.3
      DEFAULT_OBJECT_ORI_SIGMA=0.4
      ;;
    loose-cotrack|loose-cotracking|cotrack)
      ACTIVE_REWARD_PROFILE="loose-cotrack"
      DEFAULT_GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=0.30
      DEFAULT_GENERALIST_ARM_CONTACT_REWARD_WEIGHT=0.20
      DEFAULT_GENERALIST_PALM_CONTACT_REWARD_WEIGHT=0.10
      DEFAULT_ROOT_POS_W=0.5
      DEFAULT_ROOT_ORI_W=0.5
      DEFAULT_FULL_BODY_POS_W=1.0
      DEFAULT_FULL_BODY_ORI_W=1.0
      DEFAULT_FULL_BODY_LIN_VEL_W=0.75
      DEFAULT_FULL_BODY_ANG_VEL_W=0.75
      DEFAULT_OBJECT_POS_W=1.5
      DEFAULT_OBJECT_ORI_W=1.25
      DEFAULT_ROOT_POS_SIGMA=0.45
      DEFAULT_ROOT_ORI_SIGMA=0.6
      DEFAULT_FULL_BODY_POS_SIGMA=0.45
      DEFAULT_FULL_BODY_ORI_SIGMA=0.6
      DEFAULT_FULL_BODY_LIN_VEL_SIGMA=1.5
      DEFAULT_FULL_BODY_ANG_VEL_SIGMA=4.5
      DEFAULT_OBJECT_POS_SIGMA=0.45
      DEFAULT_OBJECT_ORI_SIGMA=0.6
      ;;
    *)
      echo "[ERROR] PURE_SD_REWARD_PROFILE must be one of: default, loose-cotrack. Got: ${requested_profile_raw}" >&2
      exit 2
      ;;
  esac
}

force_reference_reward_alignment() {
  ROOT_POS_W="${REFERENCE_ROOT_POS_W}"
  ROOT_ORI_W="${REFERENCE_ROOT_ORI_W}"
  FULL_BODY_POS_W="${REFERENCE_FULL_BODY_POS_W}"
  FULL_BODY_ORI_W="${REFERENCE_FULL_BODY_ORI_W}"
  FULL_BODY_LIN_VEL_W="${REFERENCE_FULL_BODY_LIN_VEL_W}"
  FULL_BODY_ANG_VEL_W="${REFERENCE_FULL_BODY_ANG_VEL_W}"
  OBJECT_POS_W="${REFERENCE_OBJECT_POS_W}"
  OBJECT_ORI_W="${REFERENCE_OBJECT_ORI_W}"
  ACTION_RATE_L2_W="${REFERENCE_ACTION_RATE_L2_W}"

  ROOT_POS_SIGMA="${REFERENCE_ROOT_POS_SIGMA}"
  ROOT_ORI_SIGMA="${REFERENCE_ROOT_ORI_SIGMA}"
  FULL_BODY_POS_SIGMA="${REFERENCE_FULL_BODY_POS_SIGMA}"
  FULL_BODY_ORI_SIGMA="${REFERENCE_FULL_BODY_ORI_SIGMA}"
  FULL_BODY_LIN_VEL_SIGMA="${REFERENCE_FULL_BODY_LIN_VEL_SIGMA}"
  FULL_BODY_ANG_VEL_SIGMA="${REFERENCE_FULL_BODY_ANG_VEL_SIGMA}"
  OBJECT_POS_SIGMA="${REFERENCE_OBJECT_POS_SIGMA}"
  OBJECT_ORI_SIGMA="${REFERENCE_OBJECT_ORI_SIGMA}"
}

refresh_effective_sequence_name() {
  local run_name_suffix=""
  if [[ "${ACTIVE_REWARD_PROFILE:-default}" != "default" ]]; then
    run_name_suffix="-${ACTIVE_REWARD_PROFILE}"
  fi

  if [[ -n "${SEQUENCE_NAME:-}" ]]; then
    EFFECTIVE_SEQUENCE_NAME="${SEQUENCE_NAME}${run_name_suffix}"
  elif [[ -n "${DATA_MODE:-}" && "${AUTO_ATTACH_WANDB_RUN:-0}" != "1" ]]; then
    EFFECTIVE_SEQUENCE_NAME="${DATA_MODE}${run_name_suffix}"
  elif [[ -n "${run_name_suffix}" && "${AUTO_ATTACH_WANDB_RUN:-0}" != "1" ]]; then
    EFFECTIVE_SEQUENCE_NAME="${EXP}-${DATA_MODE}${run_name_suffix}"
  else
    EFFECTIVE_SEQUENCE_NAME=""
  fi
}

RESUME_STEP=""
if ! RESUME_STEP="$(normalize_resume_step "${RESUME_STEP_RAW}")"; then
  echo "[ERROR] RESUME_STEP must be an integer step count or '<N>k'. Got: ${RESUME_STEP_RAW}" >&2
  exit 2
fi

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref="${ref%%\?*}"
  if [[ "${clean_ref}" != https://wandb.ai/*/runs/* ]]; then
    return 1
  fi

  local trimmed="${clean_ref#https://wandb.ai/}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 4 || "${parts[2]}" != "runs" ]]; then
    return 1
  fi

  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[3]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi

  if [[ "${#parts[@]}" -ge 6 && "${parts[4]}" == "files" ]]; then
    explicit_file="${trimmed#${entity}/${project}/runs/${run_id}/files/}"
  fi

  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

parse_wandb_uri() {
  local ref="$1"
  if [[ "${ref}" != wandb://* ]]; then
    return 1
  fi

  local trimmed="${ref#wandb://}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 3 ]]; then
    return 1
  fi

  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[2]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi

  if [[ "${#parts[@]}" -gt 3 ]]; then
    explicit_file="${trimmed#${entity}/${project}/${run_id}/}"
  fi

  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

parse_wandb_reference() {
  local ref="$1"
  parse_wandb_run_url "${ref}" || parse_wandb_uri "${ref}"
}

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"
  local requested_step="${4:-}"

  "${PYTHON_BIN}" - "${entity}" "${project}" "${run_id}" "${requested_step}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path: list[str] = []
for path_entry in sys.path:
    if path_entry in {"", "."}:
        continue
    try:
        if Path(path_entry).resolve() == repo_root:
            continue
    except Exception:
        pass
    sanitized_sys_path.append(path_entry)
sys.path = sanitized_sys_path

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id, requested_step = sys.argv[1:5]
requested_step_int = int(requested_step) if requested_step else None
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
model_pattern = re.compile(r"^model_(\d+)\.pt$")
latest_step = -1
latest_name = ""
for file_obj in run.files():
    name = getattr(file_obj, "name", "")
    match = model_pattern.match(name)
    if not match:
        continue
    step = int(match.group(1))
    if requested_step_int is not None:
        if step == requested_step_int:
            print(name)
            sys.exit(0)
        continue
    if step >= latest_step:
        latest_step = step
        latest_name = name
if latest_name and requested_step_int is None:
    print(latest_name)
PY
}

find_local_checkpoint_by_step() {
  local run_log_dir="$1"
  local requested_step="$2"
  local candidate=""
  local file=""
  local basename=""
  local file_step=""
  while IFS= read -r file; do
    basename="$(basename "${file}")"
    if [[ "${basename}" =~ ^model_0*([0-9]+)\.pt$ ]]; then
      file_step=$((10#${BASH_REMATCH[1]}))
      if (( file_step == requested_step )); then
        candidate="${file}"
        break
      fi
    fi
  done < <(find "${run_log_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V)
  echo "${candidate}"
}

resolve_local_checkpoint_from_wandb_ref() {
  local ref="$1"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local wandb_run_dir=""
  local run_log_dir=""
  local local_ckpt=""

  parsed="$(parse_wandb_reference "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo ""
    return 0
  fi
  IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"

  wandb_run_dir=$(find /data/logs_new -maxdepth 8 -type d -name "run-*-${run_id}" 2>/dev/null | head -n 1 || true)
  if [[ -z "${wandb_run_dir}" ]]; then
    echo ""
    return 0
  fi

  run_log_dir="$(dirname "$(dirname "$(dirname "${wandb_run_dir}")")")"
  if [[ -n "${explicit_file}" && -f "${run_log_dir}/${explicit_file}" ]]; then
    local_ckpt="${run_log_dir}/${explicit_file}"
  elif [[ -n "${WANDB_MODEL_FILE}" && -f "${run_log_dir}/${WANDB_MODEL_FILE}" ]]; then
    local_ckpt="${run_log_dir}/${WANDB_MODEL_FILE}"
  elif [[ -n "${RESUME_STEP}" ]]; then
    local_ckpt="$(find_local_checkpoint_by_step "${run_log_dir}" "${RESUME_STEP}")"
  else
    local_ckpt=$(ls -1 "${run_log_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  fi
  echo "${local_ckpt}"
}

normalize_resume_checkpoint_ref() {
  local ref="$1"
  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file=""

  parsed="$(parse_wandb_reference "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi
  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  model_file="${explicit_file}"

  if [[ -z "${model_file}" && -n "${WANDB_MODEL_FILE}" ]]; then
    model_file="${WANDB_MODEL_FILE}"
    echo "[INFO] Resolved wandb reference to requested checkpoint file: ${model_file}" >&2
  fi

  if [[ -z "${model_file}" && -n "${RESUME_STEP}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}" "${RESUME_STEP}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb reference to step ${RESUME_STEP} checkpoint: ${model_file}" >&2
    else
      echo "[ERROR] Could not find checkpoint step ${RESUME_STEP} for W&B reference: ${ref}" >&2
      exit 1
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}" "")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb reference to latest remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B reference: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL, set WANDB_MODEL_FILE, or set RESUME_STEP." >&2
    exit 1
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

validate_object_spec_map() {
  local map_path="$1"
  "${PYTHON_BIN}" - "${map_path}" "${SCRIPT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[2]).expanduser().resolve()
sys.path.insert(0, str(repo_root / "src" / "holosoma"))
from holosoma.utils.path import resolve_data_file_path

path = Path(sys.argv[1]).expanduser().resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    payload = payload["clips"]
if not isinstance(payload, dict) or not payload:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {path}")

def resolve_urdf(raw: str) -> Path:
    raw = raw.strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or raw.startswith("holosoma/data"):
        resolved_data = Path(resolve_data_file_path(raw)).expanduser().resolve()
        if resolved_data.is_file():
            return resolved_data

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(path.parent / candidate)
        candidates.append(repo_root / candidate)
        candidates.append(repo_root / "src" / "holosoma" / candidate)
    for item in candidates:
        resolved = item.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()

missing = []
for clip_id, entry in payload.items():
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()
    else:
        urdf = ""
    if not urdf:
        missing.append((clip_id, "<missing>"))
        continue
    resolved = resolve_urdf(urdf)
    if not resolved.is_file():
        missing.append((clip_id, str(resolved)))

if missing:
    sample = ", ".join(f"{clip}:{urdf}" for clip, urdf in missing[:10])
    raise SystemExit(f"[ERROR] Object map has missing URDFs in {path}: {sample}")

print(f"[INFO] Validated clip-object URDF map: {path} ({len(payload)} clips)")
PY
}

validate_object_spec_primitives() {
  local map_path="$1"
  PYTHONPATH="${SCRIPT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - "${map_path}" "${SCRIPT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.object_geometry import load_urdf_box_primitive_metadata

path = Path(sys.argv[1]).expanduser().resolve()
repo_root = Path(sys.argv[2]).expanduser().resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    payload = payload["clips"]
if not isinstance(payload, dict) or not payload:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {path}")

def resolve_urdf(raw: str) -> Path:
    raw = raw.strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or raw.startswith("holosoma/data"):
        resolved_data = Path(resolve_data_file_path(raw)).expanduser().resolve()
        if resolved_data.is_file():
            return resolved_data

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(path.parent / candidate)
        candidates.append(repo_root / candidate)
        candidates.append(repo_root / "src" / "holosoma" / candidate)
    for item in candidates:
        resolved = item.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()

unique_urdfs: dict[str, str] = {}
for clip_id, entry in payload.items():
    if isinstance(entry, str):
        urdf = entry.strip()
    elif isinstance(entry, dict):
        urdf = str(entry.get("object_urdf_path", "")).strip()
    else:
        urdf = ""
    if not urdf:
        raise SystemExit(f"[ERROR] Object map entry has no URDF: {clip_id}")
    resolved = resolve_urdf(urdf)
    unique_urdfs[str(resolved)] = clip_id

bad = []
for urdf in sorted(unique_urdfs):
    if load_urdf_box_primitive_metadata(urdf) is None:
        bad.append(urdf)

if bad:
    sample = "\n  ".join(bad[:10])
    raise SystemExit(
        "[ERROR] HOLOSOMA_OBJECT_SPAWN_MODE=primitive requires every object URDF to be "
        f"simple box-like. Failed {len(bad)} URDF(s):\n  {sample}"
    )

print(f"[INFO] Validated primitive object spawning metadata: {len(unique_urdfs)} unique URDF(s)")
PY
}

validate_default_ds_bank() {
  local motion_dir="$1"
  local expected_count="${2:-}"
  "${PYTHON_BIN}" - "${motion_dir}" "${expected_count}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

motion_dir = Path(sys.argv[1]).expanduser().resolve()
expected_raw = sys.argv[2].strip()
expected = int(expected_raw) if expected_raw else None
npz_files = sorted(motion_dir.glob('*.npz'))
if expected is not None and len(npz_files) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} DS motion clips under {motion_dir}, found {len(npz_files)}")
if not npz_files:
    raise SystemExit(f"[ERROR] No DS motion clips found under {motion_dir}")

map_path = motion_dir / '_clip_object_urdf_map.json'
payload = json.loads(map_path.read_text(encoding='utf-8'))
clips = payload['clips'] if isinstance(payload, dict) and isinstance(payload.get('clips'), dict) else payload
if not isinstance(clips, dict):
    raise SystemExit(f"[ERROR] Invalid DS clip-object map payload: {map_path}")
if expected is not None and len(clips) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} map entries in {map_path}, found {len(clips)}")

unique_names = set()
unique_sizes = set()
missing = []
missing_map_entries = []
for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=True)
    object_name = data['object_name'].item() if 'object_name' in data else ''
    object_urdf = data['object_urdf_path'].item() if 'object_urdf_path' in data else ''
    object_size = np.asarray(data['object_size']).reshape(-1).tolist() if 'object_size' in data else None
    if not object_name or not object_urdf or object_size is None or len(object_size) != 3:
        missing.append(npz_path.name)
        continue
    unique_names.add(str(object_name))
    unique_sizes.add(tuple(round(float(v), 6) for v in object_size))
    if npz_path.stem not in clips:
        missing_map_entries.append(npz_path.stem)

if missing:
    preview = ', '.join(missing[:10])
    raise SystemExit(f"[ERROR] DS prepared bank is missing object fields in: {preview}")
if missing_map_entries:
    preview = ', '.join(sorted(missing_map_entries)[:10])
    raise SystemExit(f"[ERROR] DS prepared bank map is missing {len(missing_map_entries)} active clip entries: {preview}")
if expected is not None and len(unique_names) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} unique object_name entries, found {len(unique_names)}")
if expected is not None and len(unique_sizes) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} unique object_size entries, found {len(unique_sizes)}")

print(
    f"[INFO] Validated default DS prepared bank: {motion_dir} ({len(npz_files)} clips, {len(unique_sizes)} unique sizes)"
)
PY
}


validate_mix_naive_bank() {
  local motion_dir="$1"
  local expected_total="${2:-}"
  local expected_ds="${3:-}"
  local expected_omomo="${4:-}"
  "${PYTHON_BIN}" - "${motion_dir}" "${expected_total}" "${expected_ds}" "${expected_omomo}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

motion_dir = Path(sys.argv[1]).expanduser().resolve()
expected_total_raw = sys.argv[2].strip()
expected_ds_raw = sys.argv[3].strip()
expected_omomo_raw = sys.argv[4].strip()
expected_total = int(expected_total_raw) if expected_total_raw else None
expected_ds = int(expected_ds_raw) if expected_ds_raw else None
expected_omomo = int(expected_omomo_raw) if expected_omomo_raw else None
npz_files = sorted(motion_dir.glob('*.npz'))
if expected_total is not None and len(npz_files) != expected_total:
    raise SystemExit(f"[ERROR] Expected {expected_total} mix-naive clips under {motion_dir}, found {len(npz_files)}")
if not npz_files:
    raise SystemExit(f"[ERROR] No mix-naive clips found under {motion_dir}")

map_path = motion_dir / '_clip_object_urdf_map.json'
payload = json.loads(map_path.read_text(encoding='utf-8'))
clips = payload['clips'] if isinstance(payload, dict) and isinstance(payload.get('clips'), dict) else payload
if not isinstance(clips, dict):
    raise SystemExit(f"[ERROR] Invalid mix-naive clip-object map payload: {map_path}")
if expected_total is not None and len(clips) != expected_total:
    raise SystemExit(f"[ERROR] Expected {expected_total} map entries in {map_path}, found {len(clips)}")

missing = []
ds_count = 0
omomo_count = 0
unique_names = set()
missing_map_entries = []
for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=True)
    object_name = data['object_name'].item() if 'object_name' in data else ''
    object_urdf = data['object_urdf_path'].item() if 'object_urdf_path' in data else ''
    object_size = np.asarray(data['object_size']).reshape(-1).tolist() if 'object_size' in data else None
    if not object_name or not object_urdf or object_size is None or len(object_size) != 3:
        missing.append(npz_path.name)
        continue
    unique_names.add(str(object_name))
    if npz_path.stem not in clips:
        missing_map_entries.append(npz_path.stem)
    if npz_path.stem.startswith('sub'):
        omomo_count += 1
    else:
        ds_count += 1

if missing:
    preview = ', '.join(missing[:10])
    raise SystemExit(f"[ERROR] mix-naive bank is missing object fields in: {preview}")
if ds_count == 0 or omomo_count == 0:
    raise SystemExit(
        f"[ERROR] mix-naive bank must contain both DS and OMOMO clips under {motion_dir}, "
        f"but found ds={ds_count}, omomo={omomo_count}"
    )
if expected_ds is not None and ds_count != expected_ds:
    raise SystemExit(f"[ERROR] Expected {expected_ds} DS clips under {motion_dir}, found {ds_count}")
if expected_omomo is not None and omomo_count != expected_omomo:
    raise SystemExit(f"[ERROR] Expected {expected_omomo} OMOMO clips under {motion_dir}, found {omomo_count}")
if missing_map_entries:
    preview = ', '.join(sorted(missing_map_entries)[:10])
    print(
        f"[WARN] mix-naive object map is missing {len(missing_map_entries)} clip entries; "
        f"falling back to per-npz object metadata for: {preview}"
    )
print(
    f"[INFO] Validated mix-naive bank: {motion_dir} ({len(npz_files)} clips = {ds_count} ds + {omomo_count} omomo, {len(unique_names)} unique object names)"
)
PY
}

assert_new_ds_data_layout() {
  local enabled_normalized
  enabled_normalized="$(echo "${ASSERT_NEW_DS_DATA}" | tr '[:upper:]' '[:lower:]')"
  case "${enabled_normalized}" in
    0|false|no|off)
      echo "[WARN] ASSERT_NEW_DS_DATA=0; skipping new DS data guard."
      return 0
      ;;
    1|true|yes|on|"")
      ;;
    *)
      echo "[ERROR] ASSERT_NEW_DS_DATA must be one of: 0/1, true/false, yes/no, on/off. Got: ${ASSERT_NEW_DS_DATA}" >&2
      exit 2
      ;;
  esac

  "${PYTHON_BIN}" - \
    "${DS_DATA_ROOT}" \
    "${NEW_DS_DATA_ROOT}" \
    "${NEW_DS_EXPECTED_PREPARED_TOTAL}" \
    "${NEW_DS_EXPECTED_PREPARED_BOX}" \
    "${NEW_DS_EXPECTED_PREPARED_BEHAVE}" \
    "${NEW_DS_EXPECTED_PREPARED_LC}" \
    "${NEW_DS_EXPECTED_PREPARED_OMOMO}" \
    "${NEW_DS_EXPECTED_MIXED_TOTAL}" \
    "${NEW_DS_EXPECTED_MIXED_BOX}" \
    "${NEW_DS_EXPECTED_MIXED_BEHAVE}" \
    "${NEW_DS_EXPECTED_MIXED_LC}" \
    "${NEW_DS_EXPECTED_MIXED_OMOMO}" <<'PY'
import json
import sys
from pathlib import Path


def resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def count_bank(bank_dir: Path) -> dict[str, int]:
    if not bank_dir.is_dir():
        raise SystemExit(f"[ERROR] Expected motion bank directory is missing: {bank_dir}")
    npz_paths = sorted(bank_dir.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"[ERROR] Motion bank has no .npz clips: {bank_dir}")
    counts = {
        "total": len(npz_paths),
        "box": 0,
        "behave": 0,
        "lc": 0,
        "omomo": 0,
        "other": 0,
    }
    for path in npz_paths:
        stem = path.stem
        if stem.startswith("box_"):
            counts["box"] += 1
        elif stem.startswith("behave_"):
            counts["behave"] += 1
        elif stem.startswith("lc_"):
            counts["lc"] += 1
        elif stem.startswith("sub"):
            counts["omomo"] += 1
        else:
            counts["other"] += 1

    map_path = bank_dir / "_clip_object_urdf_map.json"
    if not map_path.is_file():
        raise SystemExit(f"[ERROR] Motion bank is missing _clip_object_urdf_map.json: {bank_dir}")
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    clips = payload.get("clips", payload) if isinstance(payload, dict) else {}
    if not isinstance(clips, dict):
        raise SystemExit(f"[ERROR] Invalid clip-object map payload: {map_path}")
    if len(clips) != len(npz_paths):
        raise SystemExit(
            f"[ERROR] Clip-object map count does not match .npz count in {bank_dir}: "
            f"map={len(clips)} npz={len(npz_paths)}"
        )
    return counts


def assert_counts(label: str, actual: dict[str, int], expected: dict[str, int]) -> None:
    mismatches = {
        key: (actual.get(key), expected_value)
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: got {got}, expected {expected}"
            for key, (got, expected) in sorted(mismatches.items())
        )
        raise SystemExit(f"[ERROR] New DS data guard failed for {label}: {details}")


ds_root = resolve(sys.argv[1])
expected_root = resolve(sys.argv[2])
if ds_root != expected_root:
    raise SystemExit(
        "[ERROR] New DS data guard failed: training resolved DS_DATA_ROOT to "
        f"{ds_root}, expected repo-local new data at {expected_root}. "
        "Run bash cp_box.sh first, or set ASSERT_NEW_DS_DATA=0 only for an intentional old-data run."
    )
if "scale_mix_all" in ds_root.parts:
    raise SystemExit(f"[ERROR] New DS data guard failed: DS_DATA_ROOT points into old scale_mix_all data: {ds_root}")

prepared_expected = {
    "total": int(sys.argv[3]),
    "box": int(sys.argv[4]),
    "behave": int(sys.argv[5]),
    "lc": int(sys.argv[6]),
    "omomo": int(sys.argv[7]),
    "other": 0,
}
mixed_expected = {
    "total": int(sys.argv[8]),
    "box": int(sys.argv[9]),
    "behave": int(sys.argv[10]),
    "lc": int(sys.argv[11]),
    "omomo": int(sys.argv[12]),
    "other": 0,
}

prepared_dir = ds_root / "train_g1_w_obj_prepared"
mixed_dir = ds_root / "train_g1_w_obj_prepared_plus_omomo_orig"
prepared_counts = count_bank(prepared_dir)
mixed_counts = count_bank(mixed_dir)
assert_counts(str(prepared_dir), prepared_counts, prepared_expected)
assert_counts(str(mixed_dir), mixed_counts, mixed_expected)

print(
    "[INFO] New DS data guard passed: "
    f"root={ds_root}; prepared={prepared_counts}; mixed={mixed_counts}"
)
PY
}


prepare_ds_motion_bank() {
  local raw_motion_dir="$1"
  local geometry_dir="$2"
  local out_dir="$3"
  local clean_out="$4"
  local object_mass="$5"
  local color_rgba="$6"

  if [[ ! -d "${raw_motion_dir}" ]]; then
    echo "[ERROR] RAW_MOTION_DIR does not exist: ${raw_motion_dir}" >&2
    exit 2
  fi
  if [[ ! -d "${geometry_dir}" ]]; then
    echo "[ERROR] OBJ_DIR does not exist: ${geometry_dir}" >&2
    exit 2
  fi

  echo "[INFO] Preparing DS motion bank:"
  echo "[INFO]   RAW_MOTION_DIR=${raw_motion_dir}"
  echo "[INFO]   OBJ_DIR=${geometry_dir}"
  echo "[INFO]   OUT_DIR=${out_dir}"

  "${PYTHON_BIN}" - "${raw_motion_dir}" "${geometry_dir}" "${out_dir}" "${clean_out}" "${object_mass}" "${color_rgba}" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

raw_motion_dir = Path(sys.argv[1]).expanduser().resolve()
geometry_dir = Path(sys.argv[2]).expanduser().resolve()
out_dir = Path(sys.argv[3]).expanduser().resolve()
clean_out = sys.argv[4].strip().lower() not in {"", "0", "false", "no", "off"}
object_mass = float(sys.argv[5])
color_rgba = str(sys.argv[6]).strip()

if out_dir == raw_motion_dir:
    raise SystemExit(f"[ERROR] Refusing to prepare in-place over RAW_MOTION_DIR: {out_dir}")
if out_dir == geometry_dir:
    raise SystemExit(f"[ERROR] Refusing to prepare in-place over OBJ_DIR: {out_dir}")

motion_files = sorted(raw_motion_dir.glob('*.npz'))
if not motion_files:
    raise SystemExit(f"[ERROR] No .npz clips found in {raw_motion_dir}")

generated_urdf_dir = out_dir / '_generated_urdfs'
if clean_out and out_dir.exists():
    for child in out_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
out_dir.mkdir(parents=True, exist_ok=True)
generated_urdf_dir.mkdir(parents=True, exist_ok=True)


def ensure_removed(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_removed(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_obj_extents(obj_path: Path) -> np.ndarray:
    mins = None
    maxs = None
    with obj_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('v '):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            if mins is None:
                mins = xyz.copy()
                maxs = xyz.copy()
            else:
                mins = np.minimum(mins, xyz)
                maxs = np.maximum(maxs, xyz)
    if mins is None or maxs is None:
        raise ValueError(f'No OBJ vertices found in {obj_path}')
    return np.maximum(maxs - mins, 1.0e-4).astype(np.float32)


def build_urdf_text(robot_name: str, mesh_filename: str, mass: float, color: str, extents: np.ndarray) -> str:
    x, y, z = [float(v) for v in extents]
    ixx = mass * (y * y + z * z) / 12.0
    iyy = mass * (x * x + z * z) / 12.0
    izz = mass * (x * x + y * y) / 12.0
    return f"""<?xml version="1.0" ?>
<robot name="{robot_name}">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="baseLink">
    <inertial>
      <mass value="{mass:.8g}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="{ixx:.8g}" ixy="0" ixz="0" iyy="{iyy:.8g}" iyz="0" izz="{izz:.8g}"/>
    </inertial>
    <contact>
      <lateral_friction value="0.9"/>
      <rolling_friction value="0.5"/>
      <stiffness value="30000"/>
      <damping value="1000"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="mat">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


clip_map = {}
missing_geometry = []
for src_npz in motion_files:
    clip_id = src_npz.stem
    obj_src = geometry_dir / f'{clip_id}.obj'
    if not obj_src.is_file():
        missing_geometry.append(clip_id)
        continue

    object_size = parse_obj_extents(obj_src)
    obj_dst = generated_urdf_dir / obj_src.name
    symlink_or_copy(obj_src, obj_dst)

    urdf_path = generated_urdf_dir / f'{clip_id}.urdf'
    urdf_path.write_text(
        build_urdf_text(clip_id, obj_dst.name, object_mass, color_rgba, object_size),
        encoding='utf-8',
    )

    with np.load(src_npz, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload['object_size'] = object_size.astype(np.float32)
    payload['object_name'] = np.array(clip_id)
    payload['object_urdf_path'] = np.array(str(urdf_path.resolve()))
    np.savez_compressed(out_dir / src_npz.name, **payload)

    clip_map[clip_id] = {
        'object_name': clip_id,
        'object_urdf_path': str(urdf_path.resolve()),
        'object_size': [float(v) for v in object_size.tolist()],
    }

if missing_geometry:
    sample = ', '.join(missing_geometry[:10])
    raise SystemExit(f'[ERROR] Missing OBJ geometry for {len(missing_geometry)} clips: {sample}')

map_path = out_dir / '_clip_object_urdf_map.json'
map_path.write_text(json.dumps({'clips': clip_map}, indent=2, sort_keys=True), encoding='utf-8')

print(f'[INFO] Prepared DS motion bank with {len(clip_map)} clips at {out_dir}')
print(f'[INFO] Wrote clip-object map: {map_path}')
PY
}

prepared_motion_bank_ready() {
  local prepared_dir="$1"
  [[ -d "${prepared_dir}" ]] || return 1
  [[ -f "${prepared_dir}/_clip_object_urdf_map.json" ]] || return 1
  compgen -G "${prepared_dir}/*.npz" >/dev/null
}

is_data_subset_mode_alias() {
  local raw_mode="${1:-}"
  local mode
  mode="$(printf '%s' "${raw_mode}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
  case "${mode}" in
    box|pure-box|box128|box-128|ds128|box+ds128|box-ds128|box256|box-256|ds256|box+ds256|box-ds256|box-all|boxall|all|all-data|all-the-data|dsall|1|omomo|pure-omomo|pure-omomo-subset|2|omomo+behave|omomo-behave|3|omomo+behave+ds128|omomo-behave-ds128|4|omomo+behave+ds256|omomo-behave-ds256|5|64+64+dsall|6|omomo+ds64|omomo-ds64|omomo+ds-64|omomo-ds-64|omomo+ds128|omomo-ds128|omomo+ds-128|omomo-ds-128|omomo+ds256|omomo-ds256|omomo+ds-256|omomo-ds-256|omomo+dsall|omomo-dsall|omomo+all|omomo-all|omomo+boxall|omomo-boxall)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_box_data_subset_mode() {
  case "$1" in
    box128|box256|box-all)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_data_subset_mode() {
  local raw_mode="${1:-}"
  local mode
  mode="$(printf '%s' "${raw_mode}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
  case "${mode}" in
    box128|box-128|ds128|box+ds128|box-ds128)
      printf '%s\n' "box128"
      ;;
    box256|box-256|ds256|box+ds256|box-ds256)
      printf '%s\n' "box256"
      ;;
    box|pure-box|box-all|boxall|all|all-data|all-the-data|dsall)
      printf '%s\n' "box-all"
      ;;
    1|omomo|pure-omomo|pure-omomo-subset)
      printf '%s\n' "omomo"
      ;;
    2|omomo+behave|omomo-behave)
      printf '%s\n' "omomo+behave"
      ;;
    3|omomo+behave+ds128|omomo-behave-ds128)
      printf '%s\n' "omomo+behave+ds128"
      ;;
    4|omomo+behave+ds256|omomo-behave-ds256)
      printf '%s\n' "omomo+behave+ds256"
      ;;
    5|64+64+dsall)
      printf '%s\n' "64+64+dsall"
      ;;
    6|omomo+ds128|omomo-ds128|omomo+ds-128|omomo-ds-128)
      printf '%s\n' "omomo+ds128"
      ;;
    omomo+ds64|omomo-ds64|omomo+ds-64|omomo-ds-64)
      printf '%s\n' "omomo+ds64"
      ;;
    omomo+ds256|omomo-ds256|omomo+ds-256|omomo-ds-256)
      printf '%s\n' "omomo+ds256"
      ;;
    omomo+dsall|omomo-dsall|omomo+all|omomo-all|omomo+boxall|omomo-boxall)
      printf '%s\n' "omomo+dsall"
      ;;
    "")
      printf '%s\n' ""
      ;;
    *)
      echo "[ERROR] Unsupported DATA_SUBSET_MODE='${raw_mode}'." >&2
      echo "[ERROR] Use one of: ds128, ds256, all, omomo, omomo+ds64, omomo+ds128, omomo+ds256, omomo+dsall, omomo+behave, omomo+behave+ds128, omomo+behave+ds256, 64+64+dsall" >&2
      return 2
      ;;
  esac
}

data_subset_run_name() {
  case "$1" in
    box128)
      printf '%s\n' "box128"
      ;;
    box256)
      printf '%s\n' "box256"
      ;;
    box-all)
      printf '%s\n' "box-all"
      ;;
    omomo)
      printf '%s\n' "omomo"
      ;;
    omomo+behave)
      printf '%s\n' "omomo+behave"
      ;;
    omomo+behave+ds128)
      printf '%s\n' "omomo+behave+ds128"
      ;;
    omomo+behave+ds256)
      printf '%s\n' "omomo+behave+ds256"
      ;;
    64+64+dsall)
      printf '%s\n' "64+64+dsall"
      ;;
    omomo+ds64)
      printf '%s\n' "omomo+ds64"
      ;;
    omomo+ds128)
      printf '%s\n' "omomo+ds128"
      ;;
    omomo+ds256)
      printf '%s\n' "omomo+ds256"
      ;;
    omomo+dsall)
      printf '%s\n' "omomo+dsall"
      ;;
  esac
}

prepare_data_subset_bank() {
  local source_dir="$1"
  local subset_mode="$2"
  local subset_seed="$3"
  local subset_bank_root="$4"

  "${PYTHON_BIN}" - "${source_dir}" "${subset_mode}" "${subset_seed}" "${subset_bank_root}" <<'PY'
import copy
import json
import os
import random
import shutil
import sys
from pathlib import Path


source_dir = Path(sys.argv[1]).expanduser().resolve()
mode = sys.argv[2].strip()
seed_raw = sys.argv[3].strip()
bank_root_raw = sys.argv[4].strip()

try:
    seed = int(seed_raw)
except ValueError:
    raise SystemExit(f"[ERROR] DATA_SUBSET_SEED must be an integer. Got: {seed_raw}")

if not source_dir.is_dir():
    raise SystemExit(f"[ERROR] DATA subset source bank does not exist: {source_dir}")
map_path = source_dir / "_clip_object_urdf_map.json"
if not map_path.is_file():
    raise SystemExit(f"[ERROR] DATA subset source bank is missing _clip_object_urdf_map.json: {source_dir}")

mode_table = {
    "box128": {
        "slug": f"box128_seed{seed}",
        "wandb_name": "box128",
        "requires_omomo": False,
        "include_omomo": False,
        "include_behave": False,
        "ds_limit": 128,
        "all_ds": False,
    },
    "box256": {
        "slug": f"box256_seed{seed}",
        "wandb_name": "box256",
        "requires_omomo": False,
        "include_omomo": False,
        "include_behave": False,
        "ds_limit": 256,
        "all_ds": False,
    },
    "box-all": {
        "slug": "box_all",
        "wandb_name": "box-all",
        "requires_omomo": False,
        "include_omomo": False,
        "include_behave": False,
        "ds_limit": None,
        "all_ds": True,
    },
    "omomo": {
        "slug": "omomo",
        "wandb_name": "omomo",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": False,
        "ds_limit": 0,
        "all_ds": False,
    },
    "omomo+behave": {
        "slug": "omomo_behave",
        "wandb_name": "omomo+behave",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": True,
        "ds_limit": 0,
        "all_ds": False,
    },
    "omomo+behave+ds128": {
        "slug": f"omomo_behave_ds128_seed{seed}",
        "wandb_name": "omomo+behave+ds128",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": True,
        "ds_limit": 128,
        "all_ds": False,
    },
    "omomo+ds64": {
        "slug": f"omomo_ds64_seed{seed}",
        "wandb_name": "omomo+ds64",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": False,
        "ds_limit": 64,
        "all_ds": False,
    },
    "omomo+ds128": {
        "slug": f"omomo_ds128_seed{seed}",
        "wandb_name": "omomo+ds128",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": False,
        "ds_limit": 128,
        "all_ds": False,
    },
    "omomo+ds256": {
        "slug": f"omomo_ds256_seed{seed}",
        "wandb_name": "omomo+ds256",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": False,
        "ds_limit": 256,
        "all_ds": False,
    },
    "omomo+dsall": {
        "slug": "omomo_dsall",
        "wandb_name": "omomo+dsall",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": False,
        "ds_limit": None,
        "all_ds": True,
    },
    "omomo+behave+ds256": {
        "slug": f"omomo_behave_ds256_seed{seed}",
        "wandb_name": "omomo+behave+ds256",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": True,
        "ds_limit": 256,
        "all_ds": False,
    },
    "64+64+dsall": {
        "slug": "64_64_dsall",
        "wandb_name": "64+64+dsall",
        "requires_omomo": True,
        "include_omomo": True,
        "include_behave": True,
        "ds_limit": None,
        "all_ds": True,
    },
}
spec = mode_table.get(mode)
if spec is None:
    raise SystemExit(f"[ERROR] Unsupported normalized DATA_SUBSET_MODE={mode}")

payload = json.loads(map_path.read_text(encoding="utf-8"))
clips_map = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips_map, dict) or not clips_map:
    raise SystemExit(f"[ERROR] Invalid or empty clip-object map: {map_path}")

npz_by_id = {p.stem: p for p in sorted(source_dir.glob("*.npz"))}
omomo_ids = sorted([clip_id for clip_id in npz_by_id if clip_id.startswith("sub")])
behave_ids = sorted([clip_id for clip_id in npz_by_id if clip_id.startswith("behave_")])
ds_ids = sorted([clip_id for clip_id in npz_by_id if clip_id.startswith("box_")])
other_ids = sorted(set(npz_by_id) - set(omomo_ids) - set(behave_ids) - set(ds_ids))

if spec.get("requires_omomo", True) and not omomo_ids:
    raise SystemExit(f"[ERROR] DATA subset mode {mode} requires OMOMO sub* clips in {source_dir}")
if spec["include_behave"] and not behave_ids:
    raise SystemExit(f"[ERROR] DATA subset mode {mode} requires behave_* clips in {source_dir}")

selected_ds = []
if spec["all_ds"]:
    selected_ds = ds_ids
elif spec["ds_limit"]:
    limit = int(spec["ds_limit"])
    if len(ds_ids) < limit:
        raise SystemExit(f"[ERROR] DATA subset mode {mode} requires {limit} box_* DS clips, found {len(ds_ids)}")
    shuffled = list(ds_ids)
    random.Random(seed).shuffle(shuffled)
    selected_ds = sorted(shuffled[:limit])

selected_ids = list(omomo_ids) if spec.get("include_omomo", True) else []
if spec["include_behave"]:
    selected_ids.extend(behave_ids)
selected_ids.extend(selected_ds)
selected_ids = sorted(selected_ids)

missing_map_entries = [clip_id for clip_id in selected_ids if clip_id not in clips_map]
if missing_map_entries:
    preview = ", ".join(missing_map_entries[:10])
    raise SystemExit(f"[ERROR] Source object map is missing {len(missing_map_entries)} selected entries: {preview}")

if bank_root_raw:
    out_dir = Path(bank_root_raw).expanduser().resolve() / f"{source_dir.name}_{spec['slug']}"
else:
    out_dir = source_dir.parent / f"{source_dir.name}_{spec['slug']}"

expected_metadata = {
    "mode": mode,
    "wandb_name": spec["wandb_name"],
    "source_motion_bank": str(source_dir),
    "data_subset_seed": seed,
    "selected_clip_ids": selected_ids,
}
existing_map = out_dir / "_clip_object_urdf_map.json"
if existing_map.is_file():
    try:
        existing_payload = json.loads(existing_map.read_text(encoding="utf-8"))
        existing_meta = existing_payload.get("clip_subset", {}) if isinstance(existing_payload, dict) else {}
        existing_npz_ids = sorted(p.stem for p in out_dir.glob("*.npz"))
        if existing_meta.get("selected_clip_ids") == selected_ids and existing_npz_ids == selected_ids:
            counts = {
                "omomo": len([x for x in selected_ids if x.startswith("sub")]),
                "behave": len([x for x in selected_ids if x.startswith("behave_")]),
                "ds": len([x for x in selected_ids if x.startswith("box_")]),
                "other": len([x for x in selected_ids if x in other_ids]),
                "total": len(selected_ids),
            }
            print(
                f"[INFO] Reusing DATA_SUBSET_MODE={mode} bank: {out_dir} "
                f"({counts['total']} clips = {counts['omomo']} omomo + {counts['behave']} behave + {counts['ds']} ds)",
                file=sys.stderr,
            )
            print(f"{out_dir}\t{existing_map}\t{spec['wandb_name']}\t{json.dumps(counts, sort_keys=True)}")
            raise SystemExit(0)
    except json.JSONDecodeError:
        pass

if out_dir.exists() or out_dir.is_symlink():
    if out_dir.is_dir() and not out_dir.is_symlink():
        shutil.rmtree(out_dir)
    else:
        out_dir.unlink()
out_dir.mkdir(parents=True, exist_ok=True)

def symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def resolve_entry_paths(entry: dict) -> dict:
    resolved = copy.deepcopy(entry)
    for key in ("object_urdf_path", "object_mesh_path"):
        raw = str(resolved.get(key, "")).strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (source_dir / path).resolve()
        resolved[key] = str(path)
    return resolved

selected_map = {}
for clip_id in selected_ids:
    src_npz = npz_by_id[clip_id]
    symlink_or_copy(src_npz, out_dir / src_npz.name)
    selected_map[clip_id] = resolve_entry_paths(clips_map[clip_id])

counts = {
    "omomo": len([x for x in selected_ids if x.startswith("sub")]),
    "behave": len([x for x in selected_ids if x.startswith("behave_")]),
    "ds": len([x for x in selected_ids if x.startswith("box_")]),
    "other": len([x for x in selected_ids if x in other_ids]),
    "total": len(selected_ids),
}
subset_payload = {
    "clips": selected_map,
    "clip_subset": {
        **expected_metadata,
        "source_total_counts": {
            "omomo": len(omomo_ids),
            "behave": len(behave_ids),
            "ds": len(ds_ids),
            "other": len(other_ids),
            "total": len(npz_by_id),
        },
        "selected_counts": counts,
    },
}
existing_map.write_text(json.dumps(subset_payload, indent=2, sort_keys=True), encoding="utf-8")

print(
    f"[INFO] Prepared DATA_SUBSET_MODE={mode} bank: {out_dir} "
    f"({counts['total']} clips = {counts['omomo']} omomo + {counts['behave']} behave + {counts['ds']} ds)",
    file=sys.stderr,
)
print(f"{out_dir}\t{existing_map}\t{spec['wandb_name']}\t{json.dumps(counts, sort_keys=True)}")
PY
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --data-subset-mode|--sample-mode)
      if [[ "$#" -lt 2 ]]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 2
      fi
      DATA_SUBSET_MODE="$2"
      shift 2
      ;;
    --data-subset-mode=*|--sample-mode=*)
      DATA_SUBSET_MODE="${1#*=}"
      shift
      ;;
    --data-subset-seed)
      if [[ "$#" -lt 2 ]]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 2
      fi
      DATA_SUBSET_SEED="$2"
      shift 2
      ;;
    --data-subset-seed=*)
      DATA_SUBSET_SEED="${1#*=}"
      shift
      ;;
    --data-subset-bank-root)
      if [[ "$#" -lt 2 ]]; then
        echo "[ERROR] $1 requires a value." >&2
        exit 2
      fi
      DATA_SUBSET_BANK_ROOT="$2"
      shift 2
      ;;
    --data-subset-bank-root=*)
      DATA_SUBSET_BANK_ROOT="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$#" -gt 0 ]]; then
  first_arg_normalized=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case "${first_arg_normalized}" in
    pure-sd|pure-ds)
      DATA_MODE="pure-sd"
      shift
      ;;
    pure-real|pure-omomo)
      DATA_MODE="pure-real"
      shift
      ;;
    mix-naive)
      DATA_MODE="mix-naive"
      shift
      ;;
    mix-curriculum|mix-clean-noisy|mix-curr)
      DATA_MODE="mix-curriculum"
      shift
      ;;
    fix-omomo-quater|fix_omomo_quater|fix-real|fixed-real|fix_real|fixed_real)
      DATA_MODE="fix-omomo-quater"
      shift
      ;;
    resume|resume-mix-naive|mix-naive-resume)
      RESUME_PRESET_MODE="mix-naive"
      DATA_MODE="mix-naive"
      shift
      ;;
  esac
fi
if [[ "$#" -gt 0 ]] && is_data_subset_mode_alias "$1"; then
  echo "[ERROR] DATA subset modes are not accepted as the first positional argument because that slot is also used for DATA_MODE/run name/checkpoint." >&2
  echo "[ERROR] Use: bash train_object_generalist_ds.sh --data-subset-mode '$1'" >&2
  echo "[ERROR] Or:  DATA_SUBSET_MODE='$1' bash train_object_generalist_ds.sh" >&2
  exit 2
fi
DATA_MODE=$(echo "${DATA_MODE}" | tr '[:upper:]' '[:lower:]')
case "${DATA_MODE}" in
  pure-ds)
    DATA_MODE="pure-sd"
    ;;
  pure-omomo)
    DATA_MODE="pure-real"
    ;;
  mix-clean-noisy|mix-curr)
    DATA_MODE="mix-curriculum"
    ;;
  fix_omomo_quater|fix-real|fixed-real|fix_real|fixed_real)
    DATA_MODE="fix-omomo-quater"
    ;;
esac
DATA_SUBSET_WANDB_NAME=""
DATA_SUBSET_COUNTS_JSON=""
if [[ -n "${DATA_SUBSET_MODE}" ]]; then
  DATA_SUBSET_MODE="$(normalize_data_subset_mode "${DATA_SUBSET_MODE}")"
  DATA_SUBSET_WANDB_NAME="$(data_subset_run_name "${DATA_SUBSET_MODE}")"
  if [[ ! "${DATA_SUBSET_SEED}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] DATA_SUBSET_SEED must be an integer. Got: ${DATA_SUBSET_SEED}" >&2
    exit 2
  fi
  if is_box_data_subset_mode "${DATA_SUBSET_MODE}"; then
    DATA_MODE="pure-sd"
  else
    DATA_MODE="mix-naive"
  fi
  if [[ -n "${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES}" ]]; then
    echo "[WARN] Ignoring MIX_NAIVE_FIXED_OMOMO_PROBABILITIES for DATA_SUBSET_MODE=${DATA_SUBSET_MODE}; filtered subset bank controls sampled clips."
    MIX_NAIVE_FIXED_OMOMO_PROBABILITIES=""
  fi
fi
case "${DATA_MODE}" in
  pure-sd|pure-real|mix-naive|mix-curriculum|fix-omomo-quater)
    ;;
  *)
    echo "[ERROR] Unsupported DATA_MODE='${DATA_MODE}'. Use one of: pure-sd, pure-real, mix-naive, mix-curriculum, fix-omomo-quater" >&2
    exit 2
    ;;
esac
assert_new_ds_data_layout
if [[ "${DATA_MODE}" == "mix-naive" && -n "${MIX_NAIVE_CLEAN_NOISY_CURRICULUM+x}" ]]; then
  legacy_mix_curriculum_normalized=$(echo "${MIX_NAIVE_CLEAN_NOISY_CURRICULUM}" | tr '[:upper:]' '[:lower:]')
  case "${legacy_mix_curriculum_normalized}" in
    1|true|yes|on)
      echo "[ERROR] MIX_NAIVE_CLEAN_NOISY_CURRICULUM is no longer supported with DATA_MODE=mix-naive." >&2
      echo "[ERROR] Use the third mode directly instead: bash train_object_generalist_ds.sh mix-curriculum" >&2
      exit 2
      ;;
  esac
fi
if [[ "${DATA_MODE}" != "mix-naive" && -n "${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES}" ]]; then
  echo "[ERROR] MIX_NAIVE_FIXED_OMOMO_PROBABILITIES only applies to DATA_MODE=mix-naive." >&2
  exit 2
fi
if [[ "${DATA_MODE}" == "pure-sd" ]]; then
  resolve_reward_profile_defaults "${PURE_SD_REWARD_PROFILE}" "${PURE_SD_REWARD_PROFILE_RAW}"
else
  resolve_reward_profile_defaults "default" "default"
  if [[ -n "${PURE_SD_REWARD_PROFILE}" && "${PURE_SD_REWARD_PROFILE}" != "default" ]]; then
    echo "[WARN] Ignoring PURE_SD_REWARD_PROFILE=${PURE_SD_REWARD_PROFILE_RAW} for DATA_MODE=${DATA_MODE}; reward profile presets only apply to pure-sd."
  fi
fi
GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT:-${DEFAULT_GENERALIST_TORSO_CONTACT_REWARD_WEIGHT}}
GENERALIST_ARM_CONTACT_REWARD_WEIGHT=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT:-${DEFAULT_GENERALIST_ARM_CONTACT_REWARD_WEIGHT}}
GENERALIST_PALM_CONTACT_REWARD_WEIGHT=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT:-${DEFAULT_GENERALIST_PALM_CONTACT_REWARD_WEIGHT}}
ROOT_POS_W=${ROOT_POS_W:-${DEFAULT_ROOT_POS_W}}
ROOT_ORI_W=${ROOT_ORI_W:-${DEFAULT_ROOT_ORI_W}}
FULL_BODY_POS_W=${FULL_BODY_POS_W:-${DEFAULT_FULL_BODY_POS_W}}
FULL_BODY_ORI_W=${FULL_BODY_ORI_W:-${DEFAULT_FULL_BODY_ORI_W}}
FULL_BODY_LIN_VEL_W=${FULL_BODY_LIN_VEL_W:-${DEFAULT_FULL_BODY_LIN_VEL_W}}
FULL_BODY_ANG_VEL_W=${FULL_BODY_ANG_VEL_W:-${DEFAULT_FULL_BODY_ANG_VEL_W}}
OBJECT_POS_W=${OBJECT_POS_W:-${DEFAULT_OBJECT_POS_W}}
OBJECT_ORI_W=${OBJECT_ORI_W:-${DEFAULT_OBJECT_ORI_W}}
ROOT_POS_SIGMA=${ROOT_POS_SIGMA:-${DEFAULT_ROOT_POS_SIGMA}}
ROOT_ORI_SIGMA=${ROOT_ORI_SIGMA:-${DEFAULT_ROOT_ORI_SIGMA}}
FULL_BODY_POS_SIGMA=${FULL_BODY_POS_SIGMA:-${DEFAULT_FULL_BODY_POS_SIGMA}}
FULL_BODY_ORI_SIGMA=${FULL_BODY_ORI_SIGMA:-${DEFAULT_FULL_BODY_ORI_SIGMA}}
FULL_BODY_LIN_VEL_SIGMA=${FULL_BODY_LIN_VEL_SIGMA:-${DEFAULT_FULL_BODY_LIN_VEL_SIGMA}}
FULL_BODY_ANG_VEL_SIGMA=${FULL_BODY_ANG_VEL_SIGMA:-${DEFAULT_FULL_BODY_ANG_VEL_SIGMA}}
OBJECT_POS_SIGMA=${OBJECT_POS_SIGMA:-${DEFAULT_OBJECT_POS_SIGMA}}
OBJECT_ORI_SIGMA=${OBJECT_ORI_SIGMA:-${DEFAULT_OBJECT_ORI_SIGMA}}
force_reference_reward_alignment
SEQUENCE_NAME=${SEQUENCE_NAME:-""}
if [[ "$#" -gt 0 && "$1" != -* ]]; then
  if is_checkpoint_ref "$1"; then
    RESUME_CKPT="$1"
    shift
  else
    SEQUENCE_NAME="$1"
    shift
    if [[ "$#" -gt 0 ]] && is_checkpoint_ref "$1"; then
      RESUME_CKPT="$1"
      shift
    fi
  fi
fi
if [[ -n "${DATA_SUBSET_MODE}" && -z "${SEQUENCE_NAME}" ]]; then
  SEQUENCE_NAME="${DATA_SUBSET_WANDB_NAME}"
fi
EXTRA_ARGS=("$@")

if [[ "${RESUME_PRESET_MODE}" == "mix-naive" ]]; then
  if [[ -z "${RESUME_CKPT}" ]]; then
    RESUME_CKPT="${RESUME_PRESET_RUN_URL}"
    echo "[INFO] Applying resume preset checkpoint source: ${RESUME_PRESET_RUN_URL} (step ${RESUME_PRESET_STEP})"
  fi
  if [[ -z "${RESUME_STEP}" ]]; then
    RESUME_STEP="${RESUME_PRESET_STEP}"
  fi
  if [[ -z "${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES}" ]]; then
    MIX_NAIVE_FIXED_OMOMO_PROBABILITIES="${RESUME_PRESET_MIX_NAIVE_OMOMO_PROBABILITIES}"
  fi
  if [[ "${WANDB_RESUME_SAME_RUN_FROM_ENV}" != "1" ]]; then
    WANDB_RESUME_SAME_RUN=0
  fi
fi

RESUME_WANDB_ENTITY=""
RESUME_WANDB_PROJECT=""
RESUME_WANDB_RUN_ID=""
if [[ -n "${RESUME_CKPT}" && -n "${POLICY_INIT_CKPT}" ]]; then
  echo "[ERROR] RESUME_CKPT and POLICY_INIT_CKPT are mutually exclusive." >&2
  exit 1
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_SOURCE_REF="${RESUME_CKPT}"
  parsed_resume_ref="$(parse_wandb_reference "${RESUME_SOURCE_REF}" || true)"
  if [[ -n "${parsed_resume_ref}" ]]; then
    IFS=$'\t' read -r RESUME_WANDB_ENTITY RESUME_WANDB_PROJECT RESUME_WANDB_RUN_ID _resume_explicit_file <<< "${parsed_resume_ref}"
    LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_wandb_ref "${RESUME_SOURCE_REF}")"
    if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
      RESUME_CKPT="${LOCAL_WANDB_CKPT}"
      echo "[INFO] Resolved wandb reference to local checkpoint: ${RESUME_CKPT}"
    else
      RESUME_CKPT="$(normalize_resume_checkpoint_ref "${RESUME_SOURCE_REF}")"
    fi
  fi

  if [[ "${RESUME_CKPT}" != wandb://* ]] && [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "[ERROR] Resume checkpoint not found: ${RESUME_CKPT}" >&2
    exit 1
  fi
  echo "[INFO] Resume checkpoint: ${RESUME_CKPT}"
else
  echo "[INFO] No resume checkpoint requested; training will start from scratch."
  if [[ -n "${WANDB_RUN_ID}" || -n "${WANDB_RESUME}" ]]; then
    echo "[WARN] WANDB_RUN_ID/WANDB_RESUME is set without RESUME_CKPT. Training still starts without a model checkpoint, but W&B may attach to an existing run."
  fi
fi
if [[ -n "${POLICY_INIT_CKPT}" ]]; then
  POLICY_INIT_SOURCE_REF="${POLICY_INIT_CKPT}"
  parsed_policy_init_ref="$(parse_wandb_reference "${POLICY_INIT_SOURCE_REF}" || true)"
  if [[ -n "${parsed_policy_init_ref}" ]]; then
    LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_wandb_ref "${POLICY_INIT_SOURCE_REF}")"
    if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
      POLICY_INIT_CKPT="${LOCAL_WANDB_CKPT}"
      echo "[INFO] Resolved wandb policy init reference to local checkpoint: ${POLICY_INIT_CKPT}"
    else
      POLICY_INIT_CKPT="$(normalize_resume_checkpoint_ref "${POLICY_INIT_SOURCE_REF}")"
    fi
  fi

  if [[ "${POLICY_INIT_CKPT}" != wandb://* ]] && [[ ! -f "${POLICY_INIT_CKPT}" ]]; then
    echo "[ERROR] Policy init checkpoint not found: ${POLICY_INIT_CKPT}" >&2
    exit 1
  fi
  echo "[INFO] Policy init checkpoint: ${POLICY_INIT_CKPT}"
fi

AUTO_ATTACH_WANDB_RUN=0
resume_same_run_normalized=$(echo "${WANDB_RESUME_SAME_RUN}" | tr '[:upper:]' '[:lower:]')
case "${resume_same_run_normalized}" in
  auto|"")
    if [[ -n "${RESUME_WANDB_RUN_ID}" ]]; then
      AUTO_ATTACH_WANDB_RUN=1
    fi
    ;;
  1|true|yes|on)
    AUTO_ATTACH_WANDB_RUN=1
    ;;
  0|false|no|off)
    AUTO_ATTACH_WANDB_RUN=0
    ;;
  *)
    echo "[ERROR] WANDB_RESUME_SAME_RUN must be one of: auto, 0/1, true/false, yes/no, on/off. Got: ${WANDB_RESUME_SAME_RUN}" >&2
    exit 2
    ;;
esac

if [[ "${AUTO_ATTACH_WANDB_RUN}" == "1" && -n "${RESUME_WANDB_RUN_ID}" ]]; then
  if [[ "${WANDB_PROJECT_FROM_ENV}" != "1" ]]; then
    WANDB_PROJECT="${RESUME_WANDB_PROJECT}"
  fi
  if [[ -z "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY="${RESUME_WANDB_ENTITY}"
  fi
  if [[ -z "${WANDB_RUN_ID}" ]]; then
    WANDB_RUN_ID="${RESUME_WANDB_RUN_ID}"
  fi
  if [[ -z "${WANDB_RESUME}" ]]; then
    WANDB_RESUME="must"
  fi
  echo "[INFO] W&B same-run resume enabled: ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
fi

refresh_effective_sequence_name
echo "[INFO] Data mode: ${DATA_MODE}"
if [[ -n "${SEQUENCE_NAME}" ]]; then
  echo "[INFO] Sequence name: ${SEQUENCE_NAME}"
fi
if [[ -n "${EFFECTIVE_SEQUENCE_NAME}" ]]; then
  echo "[INFO] Effective run name: ${EFFECTIVE_SEQUENCE_NAME}"
fi
if [[ -n "${DATA_SUBSET_MODE}" ]]; then
  echo "[INFO] DATA_SUBSET_MODE=${DATA_SUBSET_MODE} DATA_SUBSET_SEED=${DATA_SUBSET_SEED}"
  echo "[INFO] DATA_SUBSET_MODE builds a filtered bank and trains only on those selected clips."
fi
if [[ "${DATA_MODE}" == "mix-curriculum" ]]; then
  echo "[INFO] DATA_MODE=mix-curriculum enables OMOMO->SD clip-group curriculum on the mixed bank."
  echo "[INFO] OMOMO clip prefixes=${MIX_CURRICULUM_OMOMO_PREFIXES}"
  echo "[INFO] Curriculum stage_start_iterations=${MIX_CURRICULUM_STAGE_START_ITERATIONS}"
  echo "[INFO] Curriculum omomo_probabilities=${MIX_CURRICULUM_OMOMO_PROBABILITIES}"
elif [[ "${DATA_MODE}" == "mix-naive" && -n "${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES}" ]]; then
  echo "[INFO] DATA_MODE=mix-naive with fixed DS/OMOMO clip weighting on the mixed bank."
  echo "[INFO] OMOMO clip prefixes=${MIX_NAIVE_FIXED_OMOMO_PREFIXES}"
  echo "[INFO] Fixed stage_start_iterations=${MIX_NAIVE_FIXED_STAGE_START_ITERATIONS}"
  echo "[INFO] Fixed omomo_probabilities=${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES} (DS probability = 1 - OMOMO)"
elif [[ "${DATA_MODE}" == "pure-real" ]]; then
  echo "[INFO] DATA_MODE=pure-real uses the mixed bank but samples only OMOMO clips."
  echo "[INFO] OMOMO clip prefixes=${PURE_REAL_OMOMO_PREFIXES}"
elif [[ "${DATA_MODE}" == "fix-omomo-quater" ]]; then
  echo "[INFO] DATA_MODE=fix-omomo-quater uses the mixed bank with fixed env groups."
  echo "[INFO] OMOMO clip prefixes=${FIX_OMOMO_QUATER_PREFIXES}"
  echo "[INFO] OMOMO env fraction=${FIX_OMOMO_QUATER_ENV_FRACTION} (remaining envs sample only DS/complement clips)"
fi

case "${DATA_MODE}" in
  pure-sd)
    MODE_DEFAULT_MOTION_DIR="$(ogds_default_motion_dir "${DS_DATA_ROOT}" "${DATA_MODE}")"
    ;;
  pure-real|mix-naive|mix-curriculum|fix-omomo-quater)
    MODE_DEFAULT_MOTION_DIR="$(ogds_default_motion_dir "${DS_DATA_ROOT}" "${DATA_MODE}")"
    ;;
esac

if [[ -z "${PREPARED_MOTION_DIR}" ]]; then
  if [[ "${MOTION_DIR_FROM_ENV}" == "1" ]]; then
    if [[ -d "${MOTION_DIR}" && ! -f "${MOTION_DIR}/_clip_object_urdf_map.json" ]]; then
      PREPARED_MOTION_DIR="${MOTION_DIR%/}_prepared"
    else
      PREPARED_MOTION_DIR="${MOTION_DIR}"
    fi
  else
    PREPARED_MOTION_DIR="${MODE_DEFAULT_MOTION_DIR}"
  fi
fi

if [[ "${DATA_MODE}" == "pure-sd" && "${MOTION_DIR_FROM_ENV}" == "1" && -d "${MOTION_DIR}" && ! -f "${MOTION_DIR}/_clip_object_urdf_map.json" ]]; then
  RAW_MOTION_DIR="${MOTION_DIR}"
  echo "[INFO] MOTION_DIR points to a raw DS bank; using it as RAW_MOTION_DIR source: ${RAW_MOTION_DIR}"
fi

if [[ "${DATA_MODE}" == "pure-sd" ]]; then
  auto_prep_ds_bank_normalized=$(echo "${AUTO_PREP_DS_BANK}" | tr '[:upper:]' '[:lower:]')
  case "${auto_prep_ds_bank_normalized}" in
    auto|"")
      if prepared_motion_bank_ready "${PREPARED_MOTION_DIR}"; then
        echo "[INFO] Reusing existing prepared DS bank: ${PREPARED_MOTION_DIR}"
        MOTION_DIR="${PREPARED_MOTION_DIR}"
      else
        prepare_ds_motion_bank "${RAW_MOTION_DIR}" "${OBJ_DIR}" "${PREPARED_MOTION_DIR}" "${DS_PREP_CLEAN_OUT}" "${DS_OBJECT_MASS}" "${DS_OBJECT_COLOR_RGBA}"
        MOTION_DIR="${PREPARED_MOTION_DIR}"
      fi
      ;;
    1|true|yes|on)
      prepare_ds_motion_bank "${RAW_MOTION_DIR}" "${OBJ_DIR}" "${PREPARED_MOTION_DIR}" "${DS_PREP_CLEAN_OUT}" "${DS_OBJECT_MASS}" "${DS_OBJECT_COLOR_RGBA}"
      MOTION_DIR="${PREPARED_MOTION_DIR}"
      ;;
    0|false|no|off)
      if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
        MOTION_DIR="${PREPARED_MOTION_DIR}"
      fi
      ;;
    *)
      echo "[ERROR] AUTO_PREP_DS_BANK must be one of: auto, 0/1, true/false, yes/no, on/off. Got: ${AUTO_PREP_DS_BANK}" >&2
      exit 2
      ;;
  esac
else
  if [[ "${MOTION_DIR_FROM_ENV}" != "1" ]]; then
    MOTION_DIR="${MODE_DEFAULT_MOTION_DIR}"
  fi
fi

if [[ -n "${DATA_SUBSET_MODE}" ]]; then
  DATA_SUBSET_SOURCE_DIR="${MOTION_DIR}"
  previous_object_spec_path="${OBJECT_SPEC_PATH}"
  subset_info="$(prepare_data_subset_bank "${DATA_SUBSET_SOURCE_DIR}" "${DATA_SUBSET_MODE}" "${DATA_SUBSET_SEED}" "${DATA_SUBSET_BANK_ROOT}")"
  IFS=$'\t' read -r MOTION_DIR OBJECT_SPEC_PATH DATA_SUBSET_WANDB_NAME DATA_SUBSET_COUNTS_JSON <<< "${subset_info}"
  if [[ -n "${previous_object_spec_path}" && "${previous_object_spec_path}" != "${OBJECT_SPEC_PATH}" ]]; then
    echo "[WARN] DATA_SUBSET_MODE overrides OBJECT_SPEC_PATH with the filtered subset map: ${OBJECT_SPEC_PATH}"
  fi
  echo "[INFO] DATA_SUBSET_SOURCE_DIR: ${DATA_SUBSET_SOURCE_DIR}"
  echo "[INFO] DATA_SUBSET_COUNTS: ${DATA_SUBSET_COUNTS_JSON}"
fi

echo "[INFO] RAW_MOTION_DIR: ${RAW_MOTION_DIR}"
echo "[INFO] OBJ_DIR: ${OBJ_DIR}"
echo "[INFO] MOTION_DIR: ${MOTION_DIR}"

if [[ -z "${OBJECT_SPEC_PATH}" ]]; then
  default_map="${MOTION_DIR}/_clip_object_urdf_map.json"
  if [[ -f "${default_map}" ]]; then
    OBJECT_SPEC_PATH="${default_map}"
    echo "[INFO] Using clip-object URDF map: ${OBJECT_SPEC_PATH}"
  fi
fi

if [[ -z "${OBJECT_SPEC_PATH}" || ! -f "${OBJECT_SPEC_PATH}" ]]; then
  echo "[ERROR] DS object generalist training requires a valid _clip_object_urdf_map.json." >&2
  echo "[ERROR] Current MOTION_DIR: ${MOTION_DIR}" >&2
  echo "[ERROR] Either enable AUTO_PREP_DS_BANK=1 to rebuild the bank or point OBJECT_SPEC_PATH to a valid map." >&2
  exit 2
fi
validate_object_spec_map "${OBJECT_SPEC_PATH}"
case "$(echo "${HOLOSOMA_OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  primitive|primitives|box|cuboid)
    validate_object_spec_primitives "${OBJECT_SPEC_PATH}"
    ;;
esac

if [[ "${STRICT_DEFAULT_DS_BANK_VALIDATION}" != "0" ]]; then
  case "${DATA_MODE}" in
    pure-sd)
      if [[ "$(realpath "${MOTION_DIR}")" == "$(realpath "${DEFAULT_DS_PREPARED_MOTION_DIR}")" ]]; then
        validate_default_ds_bank "${MOTION_DIR}" "${DS_EXPECTED_TOTAL}"
      fi
      ;;
    pure-real|mix-naive|mix-curriculum|fix-omomo-quater)
      if [[ "$(realpath "${MOTION_DIR}")" == "$(realpath "${DEFAULT_MIX_NAIVE_MOTION_DIR}")" ]]; then
        validate_mix_naive_bank "${MOTION_DIR}" "${MIX_NAIVE_EXPECTED_TOTAL}" "${MIX_NAIVE_EXPECTED_DS}" "${MIX_NAIVE_EXPECTED_OMOMO}"
      fi
      ;;
  esac
fi

prep_only_normalized=$(echo "${PREP_ONLY}" | tr '[:upper:]' '[:lower:]')
case "${prep_only_normalized}" in
  1|true|yes|on)
    echo "[INFO] PREP_ONLY enabled; skipping training launch."
    exit 0
    ;;
  0|false|no|off|"")
    ;;
  *)
    echo "[ERROR] PREP_ONLY must be one of: 0/1/true/false/yes/no/on/off. Got: ${PREP_ONLY}" >&2
    exit 2
    ;;
esac

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
    DEBUG_URDF_COUNT=$("${PYTHON_BIN}" - <<'PY' "${OBJECT_SPEC_PATH}"
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
if [[ "${USE_OFFLINE_CONTACT_GUIDANCE}" == "1" ]]; then
  GENERALIST_TORSO_CONTACT_REWARD_WEIGHT=0.0
  GENERALIST_ARM_CONTACT_REWARD_WEIGHT=0.0
  GENERALIST_PALM_CONTACT_REWARD_WEIGHT=0.0
fi
CONTACT_REWARD_TERMS=(
  "palms:${GENERALIST_PALM_CONTACT_REWARD_WEIGHT}"
  "arms:${GENERALIST_ARM_CONTACT_REWARD_WEIGHT}"
  "torso:${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT}"
)

if [[ "${USE_OFFLINE_CONTACT_GUIDANCE}" == "1" && -z "${CONTACT_EXPORT_ROOT}" ]]; then
  CONTACT_EXPORT_ROOT="${DEFAULT_AS_CONTACT_EXPORT_ROOT}"
fi
if [[ -n "${CONTACT_EXPORT_ROOT}" ]]; then
  CONTACT_EXPORT_ROOT=$(realpath -m "${CONTACT_EXPORT_ROOT}")
  LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
  case "${CONTACT_EXPORT_ROOT}" in
    /nfs|/nfs/*)
      echo "[ERROR] CONTACT_EXPORT_ROOT must be local, not NFS: ${CONTACT_EXPORT_ROOT}" >&2
      echo "[ERROR] Copy contact_export_from_retarget under data/ds_as_data/ first." >&2
      exit 2
      ;;
  esac
  case "${CONTACT_EXPORT_ROOT}" in
    "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
      ;;
    *)
      echo "[ERROR] CONTACT_EXPORT_ROOT must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
      echo "[ERROR] Got: ${CONTACT_EXPORT_ROOT}" >&2
      exit 2
      ;;
  esac
  if [[ "${USE_OFFLINE_CONTACT_GUIDANCE}" == "1" && ! -d "${CONTACT_EXPORT_ROOT}" ]]; then
    echo "[ERROR] Missing local contact export root: ${CONTACT_EXPORT_ROOT}" >&2
    echo "[ERROR] Expected copied data at: ${DEFAULT_AS_CONTACT_EXPORT_ROOT}" >&2
    exit 2
  fi
  if [[ -d "${CONTACT_EXPORT_ROOT}/clips" ]]; then
    CONTACT_EXPORT_CLIPS_ROOT=$(realpath -m "${CONTACT_EXPORT_ROOT}/clips")
  else
    CONTACT_EXPORT_CLIPS_ROOT="${CONTACT_EXPORT_ROOT}"
  fi
  if [[ -z "${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}" ]]; then
    ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT="${CONTACT_EXPORT_CLIPS_ROOT}"
  fi
fi

default_pose_prepend_enabled_normalized=$(echo "${DEFAULT_POSE_PREPEND_ENABLED}" | tr '[:upper:]' '[:lower:]')
case "${default_pose_prepend_enabled_normalized}" in
  1|true|yes|on)
    DEFAULT_POSE_PREPEND_ENABLED_FLAG=True
    ;;
  0|false|no|off)
    DEFAULT_POSE_PREPEND_ENABLED_FLAG=False
    ;;
  *)
    echo "[ERROR] DEFAULT_POSE_PREPEND_ENABLED must be one of: 0/1/true/false/yes/no/on/off. Got: ${DEFAULT_POSE_PREPEND_ENABLED}" >&2
    exit 2
    ;;
esac

echo "[INFO] Generalist contact reward enabled: ${GENERALIST_CONTACT_REWARD_ENABLED_FLAG}"
echo "[INFO] pure_sd_reward_profile=${ACTIVE_REWARD_PROFILE}"
echo "[INFO] Generalist contact reward mode=${GENERALIST_CONTACT_REWARD_MODE} threshold=${GENERALIST_CONTACT_REWARD_THRESHOLD} force_scale=${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
echo "[INFO] Forcing GT/u5 aligned tracking reward weights and sigmas"
echo "[INFO] Generalist contact reward weights palms=${GENERALIST_PALM_CONTACT_REWARD_WEIGHT} arms=${GENERALIST_ARM_CONTACT_REWARD_WEIGHT} torso=${GENERALIST_TORSO_CONTACT_REWARD_WEIGHT}"
echo "[INFO] Reference tracking reward weights root_pos=${ROOT_POS_W} root_ori=${ROOT_ORI_W} body_pos=${FULL_BODY_POS_W} body_ori=${FULL_BODY_ORI_W} body_lin_vel=${FULL_BODY_LIN_VEL_W} body_ang_vel=${FULL_BODY_ANG_VEL_W}"
echo "[INFO] Box tracking reward weights object_pos=${OBJECT_POS_W} object_ori=${OBJECT_ORI_W}"
echo "[INFO] action_rate_l2 weight=${ACTION_RATE_L2_W}"
echo "[INFO] Reference tracking reward sigmas root_pos=${ROOT_POS_SIGMA} root_ori=${ROOT_ORI_SIGMA} body_pos=${FULL_BODY_POS_SIGMA} body_ori=${FULL_BODY_ORI_SIGMA} body_lin_vel=${FULL_BODY_LIN_VEL_SIGMA} body_ang_vel=${FULL_BODY_ANG_VEL_SIGMA}"
echo "[INFO] Box tracking reward sigmas object_pos=${OBJECT_POS_SIGMA} object_ori=${OBJECT_ORI_SIGMA}"
echo "[INFO] limits_dof_pos weight=${GENERALIST_LIMITS_DOF_POS_WEIGHT}"
echo "[INFO] Motion default-pose prepend enabled: ${DEFAULT_POSE_PREPEND_ENABLED_FLAG}"
echo "[INFO] Motion default-pose prepend duration: ${DEFAULT_POSE_PREPEND_DURATION_S}s"
if [[ -n "${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE}" ]]; then
  echo "[INFO] contact_aware_sparse_root_command_mode=${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE}"
fi
if [[ -n "${CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS}" ]]; then
  echo "[INFO] contact_aware_sparse_root_segment_steps=${CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS}"
fi
if [[ -n "${CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG}" ]]; then
  echo "[INFO] contact_aware_sparse_root_zero_yaw_threshold_deg=${CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG}"
fi
echo "[INFO] PPO learning rates: actor=${ACTOR_LR} critic=${CRITIC_LR}"
echo "[INFO] PPO num_learning_epochs=${NUM_LEARNING_EPOCHS}"
echo "[INFO] PPO num_learning_iterations=${NUM_LEARNING_ITERATIONS}"
echo "[INFO] PPO save_interval=${SAVE_INTERVAL}"
echo "[INFO] Clip weighting strategy: ${CLIP_WEIGHTING_STRATEGY}"
echo "[INFO] Within-clip adaptive timestep sampler: ${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
echo "[INFO] freeze_at_timestep_zero_prob=${FREEZE_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] Termination defaults: BadTracking full 3D + motion_ends"
echo "[INFO] REWARD_CONFIG=${REWARD_CONFIG} rollout_reference_reward=${USE_TEACHER_ROLLOUT_REWARD} offline_contact_guidance=${USE_OFFLINE_CONTACT_GUIDANCE}"
if [[ -n "${CONTACT_EXPORT_ROOT}" ]]; then
  echo "[INFO] contact_export_root=${CONTACT_EXPORT_ROOT}"
  echo "[INFO] contact_export_clips_root=${CONTACT_EXPORT_CLIPS_ROOT}"
  echo "[INFO] offline_contact weights target=${OFFLINE_WRIST_TARGET_GUIDANCE_WEIGHT} force_gated=${OFFLINE_CONTACT_GUIDANCE_WEIGHT} position_sigma=${OFFLINE_CONTACT_POSITION_SIGMA} force_threshold=${OFFLINE_CONTACT_FORCE_THRESHOLD}"
fi
if [[ -n "${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}" ]]; then
  echo "[INFO] adaptive_sampling_contact_interval_root=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}"
fi
echo "[INFO] GPU_SELECTION=all-visible"
echo "[INFO] AVAILABLE_GPU_COUNT=${AVAILABLE_GPU_COUNT}"
echo "[INFO] NPROC=${NPROC} NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[INFO] PER_GPU_ENVS=${PER_GPU_ENVS} NUM_ENVS=${NUM_ENVS}"
echo "[INFO] TRAINING_SEED=${TRAINING_SEED:-<config-default>} RANDOMIZATION=${RANDOMIZATION_PRESET:-<exp-default>} INIT_AT_RANDOM_EP_LEN=${INIT_AT_RANDOM_EP_LEN:-<algo-default>}"
echo "[INFO] actor_history_disabled=${DISABLE_ACTOR_HISTORY} critic_history_disabled=${DISABLE_CRITIC_HISTORY} policy_history_length=${POLICY_HISTORY_LENGTH:-<config-default>}"
if [[ -n "${TEACHER_ROLLOUT_REFERENCE_ROOT}" ]]; then
  echo "[INFO] teacher_rollout_reference_root=${TEACHER_ROLLOUT_REFERENCE_ROOT}"
fi
echo "[INFO] HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE}"
echo "[INFO] HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}"
echo "[INFO] HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS:-<sim-default>}"
if [[ -n "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}" ]]; then
  echo "[INFO] perception_object_geometry_mode=${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}"
fi
echo "[INFO] PhysX gpu_max_rigid_contact_count=${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT} gpu_max_rigid_patch_count=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT} gpu_found_lost_pairs_capacity=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
echo "[INFO] PhysX gpu_found_lost_aggregate_pairs_capacity=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY} gpu_total_aggregate_pairs_capacity=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY} gpu_collision_stack_size=${PHYSX_GPU_COLLISION_STACK_SIZE} gpu_heap_capacity=${PHYSX_GPU_HEAP_CAPACITY} gpu_temp_buffer_capacity=${PHYSX_GPU_TEMP_BUFFER_CAPACITY}"

train_cmd=(
  src/holosoma/holosoma/train_agent.py
  "exp:${EXP}"
  "command:${COMMAND_CONFIG}"
  "reward:${REWARD_CONFIG}"
  "perception:${PERCEPTION}"
  --training.project="${WANDB_PROJECT}"
  --training.num-envs="${NUM_ENVS}"
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy="${CLIP_WEIGHTING_STRATEGY}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob="${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --algo.config.actor_learning_rate="${ACTOR_LR}"
  --algo.config.critic_learning_rate="${CRITIC_LR}"
  --algo.config.num_learning_epochs="${NUM_LEARNING_EPOCHS}"
  --algo.config.num_learning_iterations="${NUM_LEARNING_ITERATIONS}"
  --algo.config.normalize-actor-obs=False
  --algo.config.normalize-critic-obs=False
  --observation-overrides.disable-actor-history "${DISABLE_ACTOR_HISTORY}"
  --observation-overrides.disable-critic-history "${DISABLE_CRITIC_HISTORY}"
  --algo.config.save-interval="${SAVE_INTERVAL}"
  --simulator.config.sim.physx.gpu-max-rigid-contact-count="${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT}"
  --simulator.config.sim.physx.gpu-max-rigid-patch-count="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT}"
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --simulator.config.sim.physx.gpu-heap-capacity="${PHYSX_GPU_HEAP_CAPACITY}"
  --simulator.config.sim.physx.gpu-temp-buffer-capacity="${PHYSX_GPU_TEMP_BUFFER_CAPACITY}"
  --reward.terms.action-rate-l2.weight="${ACTION_RATE_L2_W}"
  --reward.terms.limits-dof-pos.weight="${GENERALIST_LIMITS_DOF_POS_WEIGHT}"
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend="${DEFAULT_POSE_PREPEND_ENABLED_FLAG}"
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s="${DEFAULT_POSE_PREPEND_DURATION_S}"
)
if [[ -n "${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root="${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}"
  )
fi
if [[ -n "${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE}" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-command-mode="${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE}"
  )
fi
if [[ -n "${CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS}" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-segment-steps="${CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS}"
  )
fi
if [[ -n "${CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG}" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-zero-yaw-threshold-deg="${CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG}"
  )
fi
if [[ -n "${POLICY_HISTORY_LENGTH}" ]]; then
  train_cmd+=(
    --observation.groups.actor_obs.history-length "${POLICY_HISTORY_LENGTH}"
    --observation.groups.critic_obs.history-length "${POLICY_HISTORY_LENGTH}"
  )
fi
if [[ "${USE_TEACHER_ROLLOUT_REWARD}" == "1" ]]; then
  train_cmd+=(
    --reward.terms.motion-global-ref-position-error-exp.weight="${ROOT_POS_W}"
    --reward.terms.motion-global-ref-orientation-error-exp.weight="${ROOT_ORI_W}"
    --reward.terms.motion-relative-body-position-error-exp.weight="${FULL_BODY_POS_W}"
    --reward.terms.motion-relative-body-orientation-error-exp.weight="${FULL_BODY_ORI_W}"
    --reward.terms.motion-global-body-lin-vel.weight="${FULL_BODY_LIN_VEL_W}"
    --reward.terms.motion-global-body-ang-vel.weight="${FULL_BODY_ANG_VEL_W}"
    --reward.terms.object-global-ref-position-error-exp.weight="${OBJECT_POS_W}"
    --reward.terms.object-global-ref-orientation-error-exp.weight="${OBJECT_ORI_W}"
    --reward.terms.motion-global-ref-position-error-exp.params.sigma="${ROOT_POS_SIGMA}"
    --reward.terms.motion-global-ref-orientation-error-exp.params.sigma="${ROOT_ORI_SIGMA}"
    --reward.terms.motion-relative-body-position-error-exp.params.sigma="${FULL_BODY_POS_SIGMA}"
    --reward.terms.motion-relative-body-orientation-error-exp.params.sigma="${FULL_BODY_ORI_SIGMA}"
    --reward.terms.motion-global-body-lin-vel.params.sigma="${FULL_BODY_LIN_VEL_SIGMA}"
    --reward.terms.motion-global-body-ang-vel.params.sigma="${FULL_BODY_ANG_VEL_SIGMA}"
    --reward.terms.object-global-ref-position-error-exp.params.sigma="${OBJECT_POS_SIGMA}"
    --reward.terms.object-global-ref-orientation-error-exp.params.sigma="${OBJECT_ORI_SIGMA}"
  )
  if [[ -n "${TEACHER_ROLLOUT_REFERENCE_ROOT}" ]]; then
    train_cmd+=(
      --reward.terms.motion-global-ref-position-error-exp.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.motion-global-ref-orientation-error-exp.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.motion-relative-body-position-error-exp.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.motion-relative-body-orientation-error-exp.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.motion-global-body-lin-vel.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.motion-global-body-ang-vel.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.object-global-ref-position-error-exp.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.object-global-ref-orientation-error-exp.params.rollout-reference-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
      --reward.terms.offline-contact-guidance.params.contact-export-root "${TEACHER_ROLLOUT_REFERENCE_ROOT}"
    )
  fi
else
  train_cmd+=(
    --reward.terms.motion-global-ref-position-error-exp.weight="${ROOT_POS_W}"
    --reward.terms.motion-global-ref-orientation-error-exp.weight="${ROOT_ORI_W}"
    --reward.terms.motion-relative-body-position-error-exp.weight="${FULL_BODY_POS_W}"
    --reward.terms.motion-relative-body-orientation-error-exp.weight="${FULL_BODY_ORI_W}"
    --reward.terms.motion-global-body-lin-vel.weight="${FULL_BODY_LIN_VEL_W}"
    --reward.terms.motion-global-body-ang-vel.weight="${FULL_BODY_ANG_VEL_W}"
    --reward.terms.object-global-ref-position-error-exp.weight="${OBJECT_POS_W}"
    --reward.terms.object-global-ref-orientation-error-exp.weight="${OBJECT_ORI_W}"
    --reward.terms.motion-global-ref-position-error-exp.params.sigma="${ROOT_POS_SIGMA}"
    --reward.terms.motion-global-ref-orientation-error-exp.params.sigma="${ROOT_ORI_SIGMA}"
    --reward.terms.motion-relative-body-position-error-exp.params.sigma="${FULL_BODY_POS_SIGMA}"
    --reward.terms.motion-relative-body-orientation-error-exp.params.sigma="${FULL_BODY_ORI_SIGMA}"
    --reward.terms.motion-global-body-lin-vel.params.sigma="${FULL_BODY_LIN_VEL_SIGMA}"
    --reward.terms.motion-global-body-ang-vel.params.sigma="${FULL_BODY_ANG_VEL_SIGMA}"
    --reward.terms.object-global-ref-position-error-exp.params.sigma="${OBJECT_POS_SIGMA}"
    --reward.terms.object-global-ref-orientation-error-exp.params.sigma="${OBJECT_ORI_SIGMA}"
  )
fi
if [[ "${USE_OFFLINE_CONTACT_GUIDANCE}" == "1" ]]; then
  train_cmd+=(
    --reward.terms.offline-contact-guidance.weight=1.0
    --reward.terms.offline-contact-guidance.params.wrist-weight="${OFFLINE_WRIST_TARGET_GUIDANCE_WEIGHT}"
    --reward.terms.offline-contact-guidance.params.contact-weight="${OFFLINE_CONTACT_GUIDANCE_WEIGHT}"
    --reward.terms.offline-contact-guidance.params.position-sigma="${OFFLINE_CONTACT_POSITION_SIGMA}"
    --reward.terms.offline-contact-guidance.params.force-threshold="${OFFLINE_CONTACT_FORCE_THRESHOLD}"
    --reward.terms.offline-contact-guidance.params.force-sigma="${OFFLINE_CONTACT_FORCE_SIGMA}"
    --reward.terms.offline-contact-guidance.params.contact-schedule-relax-steps="${OFFLINE_CONTACT_SCHEDULE_RELAX_STEPS}"
  )
  if [[ -n "${CONTACT_EXPORT_ROOT}" ]]; then
    train_cmd+=(
      --reward.terms.offline-contact-guidance.params.contact-export-root "${CONTACT_EXPORT_ROOT}"
    )
  fi
fi
if [[ -n "${TRAINING_SEED}" ]]; then
  train_cmd+=(--training.seed="${TRAINING_SEED}")
fi
if [[ -n "${INIT_AT_RANDOM_EP_LEN}" ]]; then
  train_cmd+=(--algo.config.init_at_random_ep_len="${INIT_AT_RANDOM_EP_LEN}")
fi
if [[ "${USE_TEACHER_ROLLOUT_REWARD}" == "1" ]]; then
  echo "[INFO] Rollout-reference reward config active; skipping body-contact-reward term overrides."
elif [[ "${USE_OFFLINE_CONTACT_GUIDANCE}" == "1" ]]; then
  echo "[INFO] Offline contact guidance active; skipping coarse body-contact-reward term overrides."
else
  for reward_spec in "${CONTACT_REWARD_TERMS[@]}"; do
    reward_term="${reward_spec%%:*}"
    reward_weight="${reward_spec#*:}"
    train_cmd+=(
      --reward.terms.body-contact-reward-"${reward_term}".weight="${reward_weight}"
      --reward.terms.body-contact-reward-"${reward_term}".params.reward-mode="${GENERALIST_CONTACT_REWARD_MODE}"
      --reward.terms.body-contact-reward-"${reward_term}".params.threshold="${GENERALIST_CONTACT_REWARD_THRESHOLD}"
      --reward.terms.body-contact-reward-"${reward_term}".params.force-scale="${GENERALIST_CONTACT_REWARD_FORCE_SCALE}"
    )
  done
fi
if [[ "${DEBUG_MODE}" == "replay" || "${DEBUG_MODE}" == "toy" ]]; then
  train_cmd=("${PYTHON_BIN}" "${train_cmd[@]}")
else
  torchrun_args=(torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}")
  if (( NNODES > 1 )); then
    torchrun_args+=(--nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}")
  fi
  train_cmd=("${torchrun_args[@]}" "${train_cmd[@]}")
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
if [[ -n "${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}" ]]; then
  case "$(echo "${PERCEPTION}" | tr '[:upper:]' '[:lower:]')" in
    ""|none|off|disabled|disable)
      ;;
    *)
      train_cmd+=(--perception.object-geometry-mode="${PERCEPTION_OBJECT_GEOMETRY_MODE_OVERRIDE}")
      ;;
  esac
fi
if [[ -n "${EFFECTIVE_SEQUENCE_NAME}" ]]; then
  train_cmd+=(--training.name="${EFFECTIVE_SEQUENCE_NAME}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  train_cmd+=(--training.checkpoint="${RESUME_CKPT}")
fi
if [[ -n "${POLICY_INIT_CKPT}" ]]; then
  train_cmd+=(--training.policy-init-checkpoint="${POLICY_INIT_CKPT}")
fi
if [[ "${CURRICULUM}" == "1" || "${CURRICULUM,,}" == "true" ]]; then
  echo "[INFO] Enabling w-object curriculum."
  train_cmd+=(--curriculum.setup-terms.w-object-difficulty-curriculum.params.enabled=True)
fi
if [[ "${DATA_MODE}" == "mix-curriculum" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-clip-name-prefixes="${MIX_CURRICULUM_OMOMO_PREFIXES}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.stage-start-iterations="${MIX_CURRICULUM_STAGE_START_ITERATIONS}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-group-probabilities="${MIX_CURRICULUM_OMOMO_PROBABILITIES}"
  )
elif [[ "${DATA_MODE}" == "mix-naive" && -n "${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES}" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-clip-name-prefixes="${MIX_NAIVE_FIXED_OMOMO_PREFIXES}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.stage-start-iterations="${MIX_NAIVE_FIXED_STAGE_START_ITERATIONS}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-group-probabilities="${MIX_NAIVE_FIXED_OMOMO_PROBABILITIES}"
  )
elif [[ "${DATA_MODE}" == "pure-real" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-clip-name-prefixes="${PURE_REAL_OMOMO_PREFIXES}"
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.stage-start-iterations='[0]'
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-group-probabilities='[1.0]'
  )
elif [[ "${DATA_MODE}" == "fix-omomo-quater" ]]; then
  train_cmd+=(
    --command.setup-terms.motion-command.params.motion-config.fixed-clip-group-assignment.enabled=True
    --command.setup-terms.motion-command.params.motion-config.fixed-clip-group-assignment.group-clip-name-prefixes="${FIX_OMOMO_QUATER_PREFIXES}"
    --command.setup-terms.motion-command.params.motion-config.fixed-clip-group-assignment.group-env-fraction="${FIX_OMOMO_QUATER_ENV_FRACTION}"
  )
fi
if [[ -n "${RANDOMIZATION_PRESET}" ]]; then
  train_cmd+=("randomization:${RANDOMIZATION_PRESET}")
fi
train_cmd+=("${EXTRA_ARGS[@]}")
train_cmd+=(logger:wandb)
if [[ -n "${WANDB_ENTITY}" ]]; then
  train_cmd+=(--logger.entity="${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  train_cmd+=(--logger.id="${WANDB_RUN_ID}")
fi
if [[ -n "${WANDB_RESUME}" ]]; then
  train_cmd+=(--logger.resume="${WANDB_RESUME}")
fi
if [[ -n "${EFFECTIVE_SEQUENCE_NAME}" ]]; then
  train_cmd+=(--logger.name="${EFFECTIVE_SEQUENCE_NAME}")
fi
echo "[INFO] Training video recording disabled."
train_cmd+=(--logger.video.enabled=False)
train_cmd+=(--logger.headless_recording=False)
train_cmd+=(--logger.video.upload_to_wandb=False)
if [[ "${PRINT_TRAIN_CMD:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
  printf '[INFO] Final train command:'
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi
VISER_LOAD_URDF="${VISER_LOAD_URDF}" "${train_cmd[@]}"
