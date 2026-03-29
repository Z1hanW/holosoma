#!/usr/bin/env bash
set -euo pipefail

# Extendable object co-tracking launcher.
#
# Goals:
# - Keep train_object_base.sh untouched for the legacy single-trajectory setup.
# - Expose a modular "decoupled" training path where lower-body tracking, torso/waist
#   priors, upper-body weak tracking, contact-support rewards, and scale curriculum
#   can be composed from shell-level knobs.
# - Support staged scale curriculum by chaining checkpoints across runs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SIM_ENV_BIN_CANDIDATES=(
  /home/ubuntu/miniconda3/envs/sim/bin
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin
)

resolve_env_executable() {
  local binary_name=$1
  if command -v "${binary_name}" >/dev/null 2>&1; then
    command -v "${binary_name}"
    return 0
  fi

  local candidate_dir
  for candidate_dir in "${SIM_ENV_BIN_CANDIDATES[@]}"; do
    if [[ -x "${candidate_dir}/${binary_name}" ]]; then
      echo "${candidate_dir}/${binary_name}"
      return 0
    fi
  done
  return 1
}

TORCHRUN_BIN=${TORCHRUN_BIN:-$(resolve_env_executable torchrun || true)}
PYTHON_BIN=${PYTHON_BIN:-$(resolve_env_executable python || true)}

if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "[ERROR] Unable to resolve torchrun from PATH or known simulator environments." >&2
  exit 2
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Unable to resolve python from PATH or known simulator environments." >&2
  exit 2
fi

DEFAULT_CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}
NPROC=${NPROC:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
PHYSX_GPU_MAX_RIGID_PATCH_COUNT=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-655360}

LEGACY_DEFAULT_EXP=g1-29dof-wbt-w-object-extend
COMMAND_CURRICULUM_DEFAULT_EXP=g1-29dof-wbt-w-object-command-curriculum
DEFAULT_EXP=${DEFAULT_EXP:-${COMMAND_CURRICULUM_DEFAULT_EXP}}
LEGACY_DEFAULT_TRAINING_NAME=g1_29dof_wbt_w_object_extend
COMMAND_CURRICULUM_DEFAULT_TRAINING_NAME=g1_29dof_wbt_w_object_command_curriculum
DEFAULT_TRAINING_NAME=${DEFAULT_TRAINING_NAME:-${COMMAND_CURRICULUM_DEFAULT_TRAINING_NAME}}

EXP_FROM_ENV=0
[[ -n "${EXP+x}" ]] && EXP_FROM_ENV=1
TRAINING_NAME_FROM_ENV=0
[[ -n "${TRAINING_NAME+x}" ]] && TRAINING_NAME_FROM_ENV=1

EXP=${EXP:-${DEFAULT_EXP}}
WANDB_PROJECT=${WANDB_PROJECT:-boxer}
LOGGER_BASE_DIR=${LOGGER_BASE_DIR:-/data/logs_new}
TRAINING_NAME=${TRAINING_NAME:-${DEFAULT_TRAINING_NAME}}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
RUN_GROUP=${RUN_GROUP:-${TRAINING_NAME}_${RUN_TAG}}

DEFAULT_MOTION_SOURCE="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
if [[ -n "${MOTION_SOURCE+x}" ]]; then
  :
elif [[ -n "${MOTION_FILE+x}" ]]; then
  MOTION_SOURCE="${MOTION_FILE}"
elif [[ -n "${MOTION_DIR+x}" ]]; then
  MOTION_SOURCE="${MOTION_DIR}"
else
  MOTION_SOURCE="${DEFAULT_MOTION_SOURCE}"
fi

OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH:-""}
if [[ -z "${OBJECT_SPEC_PATH}" && -d "${MOTION_SOURCE}" ]]; then
  for candidate in "${MOTION_SOURCE}/_clip_object_urdf_map.json" "${MOTION_SOURCE}/clip_object_urdf_map.json"; do
    if [[ -f "${candidate}" ]]; then
      OBJECT_SPEC_PATH="${candidate}"
      break
    fi
  done
fi

MOTION_CLIP_ID=${MOTION_CLIP_ID:-""}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-""}
PERCEPTION=${PERCEPTION:-none}
NUM_ENVS=${NUM_ENVS:-81920}
SAVE_INTERVAL=${SAVE_INTERVAL:-1500}
NUM_ITERS=${NUM_ITERS:-""}
CHECKPOINT=${CHECKPOINT:-""}

COMMAND_CURRICULUM_MODE=${COMMAND_CURRICULUM_MODE:-1}
COMMAND_CURRICULUM_PERCEPTION_INTO_POLICY_MODULES=${COMMAND_CURRICULUM_PERCEPTION_INTO_POLICY_MODULES:-True}
COMMAND_CURRICULUM_MAX_EPISODE_LENGTH_S=${COMMAND_CURRICULUM_MAX_EPISODE_LENGTH_S:-12.0}
COMMAND_CURRICULUM_HYBRID_START_ITER=${COMMAND_CURRICULUM_HYBRID_START_ITER:-2000}
COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START=${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START:-0.0}
COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END=${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END:-0.6}
COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START_ITER=${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START_ITER:-${COMMAND_CURRICULUM_HYBRID_START_ITER}}
COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END_ITER=${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END_ITER:-8000}
COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_RAMP_RESETS=${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_RAMP_RESETS:-120000}
COMMAND_CURRICULUM_EVAL_COMMAND_ONLY_ENV_PROB=${COMMAND_CURRICULUM_EVAL_COMMAND_ONLY_ENV_PROB:-${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END}}
COMMAND_CURRICULUM_CLIP_GOAL_DELTA_MIN_STEPS=${COMMAND_CURRICULUM_CLIP_GOAL_DELTA_MIN_STEPS:-45}
COMMAND_CURRICULUM_CLIP_GOAL_DELTA_MAX_STEPS=${COMMAND_CURRICULUM_CLIP_GOAL_DELTA_MAX_STEPS:-120}
COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START=${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START:-0.0}
COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END=${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END:-0.25}
COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START_ITER=${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START_ITER:-${COMMAND_CURRICULUM_HYBRID_START_ITER}}
COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END_ITER=${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END_ITER:-9000}
COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_RAMP_RESETS=${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_RAMP_RESETS:-140000}
COMMAND_CURRICULUM_EVAL_CARRY_EXTENSION_PROB=${COMMAND_CURRICULUM_EVAL_CARRY_EXTENSION_PROB:-${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END}}
COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_RAMP_RESETS=${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_RAMP_RESETS:-${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_RAMP_RESETS}}
COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_START_ITER=${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_START_ITER:-${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START_ITER}}
COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_END_ITER=${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_END_ITER:-${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END_ITER}}
COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MIN_START=${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MIN_START:-"[0.10, -0.05, 0.0]"}
COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MAX_START=${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MAX_START:-"[0.25, 0.05, 0.0]"}
COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MIN=${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MIN:-"[0.15, -0.20, 0.0]"}
COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MAX=${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MAX:-"[0.60, 0.20, 0.0]"}
COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MIN_START=${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MIN_START:-"[0.0, 0.0, -0.20]"}
COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MAX_START=${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MAX_START:-"[0.0, 0.0, 0.20]"}
COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MIN=${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MIN:-"[0.0, 0.0, -0.80]"}
COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MAX=${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MAX:-"[0.0, 0.0, 0.80]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START=${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START:-0.0}
COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END=${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END:-0.15}
COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START_ITER=${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START_ITER:-${COMMAND_CURRICULUM_HYBRID_START_ITER}}
COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END_ITER=${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END_ITER:-13000}
COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_RAMP_RESETS=${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_RAMP_RESETS:-220000}
COMMAND_CURRICULUM_EVAL_EXTERNAL_GOAL_PROB=${COMMAND_CURRICULUM_EVAL_EXTERNAL_GOAL_PROB:-${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END}}
COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_RAMP_RESETS=${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_RAMP_RESETS:-${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_RAMP_RESETS}}
COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_START_ITER=${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_START_ITER:-${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START_ITER}}
COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_END_ITER=${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_END_ITER:-${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END_ITER}}
COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MIN_START=${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MIN_START:-"[0.40, -0.20, 0.185]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MAX_START=${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MAX_START:-"[0.65, 0.20, 0.185]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MIN=${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MIN:-"[0.25, -0.75, 0.185]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MAX=${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MAX:-"[1.00, 0.75, 0.185]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MIN_START=${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MIN_START:-"[0.0, 0.0, -0.80]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MAX_START=${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MAX_START:-"[0.0, 0.0, 0.80]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MIN=${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MIN:-"[0.0, 0.0, -3.1415926]"}
COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MAX=${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MAX:-"[0.0, 0.0, 3.1415926]"}

PROFILE=$(echo "${PROFILE:-decoupled}" | tr '[:upper:]' '[:lower:]')
PROFILE_BLEND=${PROFILE_BLEND:-${TRANSITION_BLEND:-""}}
PROFILE_BLEND_STAGES=${PROFILE_BLEND_STAGES:-${TRANSITION_CURRICULUM_STAGES:-""}}

OBJECT_SCALE=${OBJECT_SCALE:-1.0}
SCALE_CURRICULUM_STAGES=${SCALE_CURRICULUM_STAGES:-""}

ENABLE_VISER=${ENABLE_VISER:-0}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}

ASSIST_CURRICULUM=${ASSIST_CURRICULUM:-0}
ASSIST_INITIAL_LAMBDA=${ASSIST_INITIAL_LAMBDA:-0.0}
ASSIST_LAMBDA_STEP_UP=${ASSIST_LAMBDA_STEP_UP:-0.01}
ASSIST_LAMBDA_STEP_DOWN=${ASSIST_LAMBDA_STEP_DOWN:-0.01}
ASSIST_EARLY_TERM_THRESHOLD=${ASSIST_EARLY_TERM_THRESHOLD:-0.30}
ASSIST_SIMILARITY_THRESHOLD=${ASSIST_SIMILARITY_THRESHOLD:-0.60}
ASSIST_BETA_MAX=${ASSIST_BETA_MAX:-1.0}

CONTACT_THRESHOLD=${CONTACT_THRESHOLD:-1.0}
CONTACT_FORCE_SCALE=${CONTACT_FORCE_SCALE:-25.0}
CONTACT_MODE=${CONTACT_MODE:-binary}

ROOT_POS_W=${ROOT_POS_W:-}
ROOT_ORI_W=${ROOT_ORI_W:-}
FULL_BODY_POS_W=${FULL_BODY_POS_W:-}
FULL_BODY_ORI_W=${FULL_BODY_ORI_W:-}
FULL_BODY_LIN_VEL_W=${FULL_BODY_LIN_VEL_W:-}
FULL_BODY_ANG_VEL_W=${FULL_BODY_ANG_VEL_W:-}
LOWER_BODY_POS_W=${LOWER_BODY_POS_W:-}
LOWER_BODY_ORI_W=${LOWER_BODY_ORI_W:-}
LOWER_BODY_LIN_VEL_W=${LOWER_BODY_LIN_VEL_W:-}
LOWER_BODY_ANG_VEL_W=${LOWER_BODY_ANG_VEL_W:-}
TORSO_BODY_POS_W=${TORSO_BODY_POS_W:-}
TORSO_BODY_ORI_W=${TORSO_BODY_ORI_W:-}
UPPER_BODY_POS_W=${UPPER_BODY_POS_W:-}
UPPER_BODY_ORI_W=${UPPER_BODY_ORI_W:-}
UPPER_BODY_LIN_VEL_W=${UPPER_BODY_LIN_VEL_W:-}
UPPER_BODY_ANG_VEL_W=${UPPER_BODY_ANG_VEL_W:-}
LOWER_JOINT_POS_W=${LOWER_JOINT_POS_W:-}
LOWER_JOINT_VEL_W=${LOWER_JOINT_VEL_W:-}
WAIST_JOINT_POS_W=${WAIST_JOINT_POS_W:-}
WAIST_JOINT_VEL_W=${WAIST_JOINT_VEL_W:-}
UPPER_JOINT_POS_W=${UPPER_JOINT_POS_W:-}
UPPER_JOINT_VEL_W=${UPPER_JOINT_VEL_W:-}
PALM_CONTACT_W=${PALM_CONTACT_W:-}
ARM_CONTACT_W=${ARM_CONTACT_W:-}
TORSO_CONTACT_W=${TORSO_CONTACT_W:-}
OBJECT_POS_W=${OBJECT_POS_W:-}
OBJECT_ORI_W=${OBJECT_ORI_W:-}

EXTRA_ARGS=("$@")

WEIGHT_KNOBS=(
  ROOT_POS_W
  ROOT_ORI_W
  FULL_BODY_POS_W
  FULL_BODY_ORI_W
  FULL_BODY_LIN_VEL_W
  FULL_BODY_ANG_VEL_W
  LOWER_BODY_POS_W
  LOWER_BODY_ORI_W
  LOWER_BODY_LIN_VEL_W
  LOWER_BODY_ANG_VEL_W
  TORSO_BODY_POS_W
  TORSO_BODY_ORI_W
  UPPER_BODY_POS_W
  UPPER_BODY_ORI_W
  UPPER_BODY_LIN_VEL_W
  UPPER_BODY_ANG_VEL_W
  LOWER_JOINT_POS_W
  LOWER_JOINT_VEL_W
  WAIST_JOINT_POS_W
  WAIST_JOINT_VEL_W
  UPPER_JOINT_POS_W
  UPPER_JOINT_VEL_W
  PALM_CONTACT_W
  ARM_CONTACT_W
  TORSO_CONTACT_W
  OBJECT_POS_W
  OBJECT_ORI_W
)

declare -A EXPLICIT_WEIGHT_OVERRIDES=()

capture_explicit_weight_overrides() {
  local knob
  for knob in "${WEIGHT_KNOBS[@]}"; do
    if [[ -n "${!knob}" ]]; then
      EXPLICIT_WEIGHT_OVERRIDES["${knob}"]=1
    fi
  done
}

profile_default_value() {
  local profile=$1
  local knob=$2
  case "${profile}:${knob}" in
    baseline:ROOT_POS_W|baseline:ROOT_ORI_W|baseline:FULL_BODY_POS_W|baseline:FULL_BODY_ORI_W|baseline:FULL_BODY_LIN_VEL_W|baseline:FULL_BODY_ANG_VEL_W|baseline:OBJECT_POS_W|baseline:OBJECT_ORI_W)
      echo 1.0
      ;;
    baseline:LOWER_BODY_POS_W|baseline:LOWER_BODY_ORI_W|baseline:LOWER_BODY_LIN_VEL_W|baseline:LOWER_BODY_ANG_VEL_W|baseline:TORSO_BODY_POS_W|baseline:TORSO_BODY_ORI_W|baseline:UPPER_BODY_POS_W|baseline:UPPER_BODY_ORI_W|baseline:UPPER_BODY_LIN_VEL_W|baseline:UPPER_BODY_ANG_VEL_W|baseline:LOWER_JOINT_POS_W|baseline:LOWER_JOINT_VEL_W|baseline:WAIST_JOINT_POS_W|baseline:WAIST_JOINT_VEL_W|baseline:UPPER_JOINT_POS_W|baseline:UPPER_JOINT_VEL_W|baseline:PALM_CONTACT_W|baseline:ARM_CONTACT_W|baseline:TORSO_CONTACT_W)
      echo 0.0
      ;;
    decoupled:ROOT_POS_W|decoupled:ROOT_ORI_W|decoupled:OBJECT_POS_W|decoupled:OBJECT_ORI_W|custom:ROOT_POS_W|custom:ROOT_ORI_W|custom:OBJECT_POS_W|custom:OBJECT_ORI_W)
      echo 1.0
      ;;
    decoupled:FULL_BODY_POS_W|decoupled:FULL_BODY_ORI_W|decoupled:FULL_BODY_LIN_VEL_W|decoupled:FULL_BODY_ANG_VEL_W|custom:FULL_BODY_POS_W|custom:FULL_BODY_ORI_W|custom:FULL_BODY_LIN_VEL_W|custom:FULL_BODY_ANG_VEL_W)
      echo 0.0
      ;;
    decoupled:LOWER_BODY_POS_W|custom:LOWER_BODY_POS_W)
      echo 1.5
      ;;
    decoupled:LOWER_BODY_ORI_W|decoupled:LOWER_BODY_LIN_VEL_W|decoupled:LOWER_BODY_ANG_VEL_W|decoupled:LOWER_JOINT_POS_W|custom:LOWER_BODY_ORI_W|custom:LOWER_BODY_LIN_VEL_W|custom:LOWER_BODY_ANG_VEL_W|custom:LOWER_JOINT_POS_W)
      echo 1.0
      ;;
    decoupled:TORSO_BODY_POS_W|decoupled:TORSO_BODY_ORI_W|custom:TORSO_BODY_POS_W|custom:TORSO_BODY_ORI_W)
      echo 0.5
      ;;
    decoupled:UPPER_BODY_POS_W|decoupled:UPPER_BODY_ORI_W|custom:UPPER_BODY_POS_W|custom:UPPER_BODY_ORI_W)
      echo 0.10
      ;;
    decoupled:UPPER_BODY_LIN_VEL_W|decoupled:UPPER_BODY_ANG_VEL_W|decoupled:WAIST_JOINT_VEL_W|custom:UPPER_BODY_LIN_VEL_W|custom:UPPER_BODY_ANG_VEL_W|custom:WAIST_JOINT_VEL_W)
      echo 0.05
      ;;
    decoupled:LOWER_JOINT_VEL_W|custom:LOWER_JOINT_VEL_W)
      echo 0.25
      ;;
    decoupled:WAIST_JOINT_POS_W|custom:WAIST_JOINT_POS_W)
      echo 0.25
      ;;
    decoupled:UPPER_JOINT_POS_W|custom:UPPER_JOINT_POS_W)
      echo 0.05
      ;;
    decoupled:UPPER_JOINT_VEL_W|custom:UPPER_JOINT_VEL_W)
      echo 0.01
      ;;
    decoupled:PALM_CONTACT_W|custom:PALM_CONTACT_W)
      echo 0.50
      ;;
    decoupled:ARM_CONTACT_W|custom:ARM_CONTACT_W)
      echo 0.20
      ;;
    decoupled:TORSO_CONTACT_W|custom:TORSO_CONTACT_W)
      # Aggressively favor chest-brace carries in the extend profile.
      echo 0.60
      ;;
    *)
      echo "[ERROR] Unsupported profile/weight combination '${profile}:${knob}'." >&2
      exit 2
      ;;
  esac
}

validate_blend_value() {
  local blend_value=$1
  if ! awk -v t="${blend_value}" 'BEGIN { exit !(t >= 0.0 && t <= 1.0) }'; then
    echo "[ERROR] PROFILE_BLEND must be within [0, 1]. Got '${blend_value}'." >&2
    exit 2
  fi
}

lerp_float() {
  local start_value=$1
  local end_value=$2
  local alpha=$3
  awk -v a="${start_value}" -v b="${end_value}" -v t="${alpha}" 'BEGIN { printf "%.6f", a + (b - a) * t }'
}

set_weight_if_not_explicit() {
  local knob=$1
  local value=$2
  if [[ -n "${EXPLICIT_WEIGHT_OVERRIDES["${knob}"]+x}" ]]; then
    return
  fi
  printf -v "${knob}" '%s' "${value}"
}

apply_profile_weights() {
  local stage_profile=$1
  local stage_blend=${2:-}
  local knob

  case "${stage_profile}" in
    baseline|decoupled|custom)
      for knob in "${WEIGHT_KNOBS[@]}"; do
        set_weight_if_not_explicit "${knob}" "$(profile_default_value "${stage_profile}" "${knob}")"
      done
      ;;
    blend|transition)
      if [[ -z "${stage_blend}" ]]; then
        echo "[ERROR] PROFILE='${stage_profile}' requires PROFILE_BLEND or PROFILE_BLEND_STAGES." >&2
        exit 2
      fi
      validate_blend_value "${stage_blend}"
      local start_value end_value blended_value
      for knob in "${WEIGHT_KNOBS[@]}"; do
        start_value=$(profile_default_value baseline "${knob}")
        end_value=$(profile_default_value decoupled "${knob}")
        blended_value=$(lerp_float "${start_value}" "${end_value}" "${stage_blend}")
        set_weight_if_not_explicit "${knob}" "${blended_value}"
      done
      ;;
    *)
      echo "[ERROR] Unsupported PROFILE='${stage_profile}'. Use one of: baseline, decoupled, custom, blend." >&2
      exit 2
      ;;
  esac
}

validate_profile_configuration() {
  case "${PROFILE}" in
    baseline|decoupled|custom)
      if [[ -n "${PROFILE_BLEND}" || -n "${PROFILE_BLEND_STAGES}" ]]; then
        echo "[ERROR] PROFILE_BLEND / PROFILE_BLEND_STAGES require PROFILE=blend." >&2
        exit 2
      fi
      ;;
    blend|transition)
      if [[ -z "${PROFILE_BLEND}" && -z "${PROFILE_BLEND_STAGES}" ]]; then
        echo "[ERROR] PROFILE=blend requires PROFILE_BLEND or PROFILE_BLEND_STAGES." >&2
        exit 2
      fi
      if [[ -n "${PROFILE_BLEND}" ]]; then
        validate_blend_value "${PROFILE_BLEND}"
      fi
      ;;
    *)
      echo "[ERROR] Unsupported PROFILE='${PROFILE}'. Use one of: baseline, decoupled, custom, blend." >&2
      exit 2
      ;;
  esac
}

is_truthy() {
  local value="${1,,}"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

validate_probability_range() {
  local label=$1
  local value=$2
  if ! awk -v t="${value}" 'BEGIN { exit !(t >= 0.0 && t <= 1.0) }'; then
    echo "[ERROR] ${label} must be within [0, 1]. Got '${value}'." >&2
    exit 2
  fi
}

validate_nonnegative_integer() {
  local label=$1
  local value=$2
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ${label} must be a non-negative integer. Got '${value}'." >&2
    exit 2
  fi
}

validate_iter_schedule() {
  local label=$1
  local start_iter=$2
  local end_iter=$3
  validate_nonnegative_integer "${label}_start_iter" "${start_iter}"
  validate_nonnegative_integer "${label}_end_iter" "${end_iter}"
  if (( end_iter < start_iter )); then
    echo "[ERROR] ${label} end_iter must be >= start_iter. Got start=${start_iter}, end=${end_iter}." >&2
    exit 2
  fi
}

validate_command_curriculum_probability_budget() {
  local label=$1
  local command_prob=$2
  local carry_prob=$3
  local external_prob=$4
  validate_probability_range "${label} command_only_env_prob" "${command_prob}"
  validate_probability_range "${label} carry_extension_prob" "${carry_prob}"
  validate_probability_range "${label} external_goal_prob" "${external_prob}"
  if ! awk -v c="${command_prob}" 'BEGIN { exit !(c <= 0.6) }'; then
    echo "[ERROR] ${label} command_only_env_prob exceeds 0.60: command=${command_prob}." >&2
    echo "[ERROR] Command curriculum requires at least 40% co-tracking environments throughout training." >&2
    exit 2
  fi
  if ! awk -v c="${command_prob}" -v a="${carry_prob}" -v b="${external_prob}" 'BEGIN { exit !((a + b) <= c) }'; then
    echo "[ERROR] ${label} carry/external budget exceeds command-only budget: command=${command_prob}, carry=${carry_prob}, external=${external_prob}." >&2
    exit 2
  fi
}

validate_command_curriculum_iteration_schedule() {
  validate_nonnegative_integer "command_curriculum_hybrid_start_iter" "${COMMAND_CURRICULUM_HYBRID_START_ITER}"
  validate_iter_schedule \
    "command_only_env_prob" \
    "${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START_ITER}" \
    "${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END_ITER}"
  validate_iter_schedule \
    "carry_extension_prob" \
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START_ITER}" \
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END_ITER}"
  validate_iter_schedule \
    "carry_extension_range" \
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_START_ITER}" \
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_END_ITER}"
  validate_iter_schedule \
    "external_goal_prob" \
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START_ITER}" \
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END_ITER}"
  validate_iter_schedule \
    "external_goal_range" \
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_START_ITER}" \
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_END_ITER}"
}

apply_command_curriculum_defaults() {
  if ! is_truthy "${COMMAND_CURRICULUM_MODE}"; then
    return
  fi

  if [[ "${EXP_FROM_ENV}" == "1" && "${EXP}" == "${LEGACY_DEFAULT_EXP}" ]]; then
    echo "[ERROR] COMMAND_CURRICULUM_MODE is enabled, but EXP was explicitly set to legacy extend: '${EXP}'." >&2
    echo "[ERROR] Remove EXP or set EXP='${COMMAND_CURRICULUM_DEFAULT_EXP}'." >&2
    exit 2
  fi
  if [[ "${EXP}" == "${LEGACY_DEFAULT_EXP}" ]]; then
    EXP="${COMMAND_CURRICULUM_DEFAULT_EXP}"
  fi
  if [[ "${TRAINING_NAME_FROM_ENV}" == "1" && "${TRAINING_NAME}" == "${LEGACY_DEFAULT_TRAINING_NAME}" ]]; then
    echo "[ERROR] COMMAND_CURRICULUM_MODE is enabled, but TRAINING_NAME was explicitly set to legacy extend: '${TRAINING_NAME}'." >&2
    echo "[ERROR] Remove TRAINING_NAME or use a command-curriculum training name." >&2
    exit 2
  fi
  if [[ "${TRAINING_NAME}" == "${LEGACY_DEFAULT_TRAINING_NAME}" ]]; then
    TRAINING_NAME="${COMMAND_CURRICULUM_DEFAULT_TRAINING_NAME}"
  fi
  if [[ "${RUN_GROUP}" == "${LEGACY_DEFAULT_TRAINING_NAME}_${RUN_TAG}" ]]; then
    RUN_GROUP="${TRAINING_NAME}_${RUN_TAG}"
  fi
  if [[ "${PERCEPTION}" == "none" ]]; then
    PERCEPTION="camera_depth_d435i_17x17"
  fi
}

validate_command_curriculum_configuration() {
  if ! is_truthy "${COMMAND_CURRICULUM_MODE}"; then
    return
  fi

  if [[ "${EXP}" != "${COMMAND_CURRICULUM_DEFAULT_EXP}" ]]; then
    echo "[ERROR] COMMAND_CURRICULUM_MODE requires EXP='${COMMAND_CURRICULUM_DEFAULT_EXP}'. Got EXP='${EXP}'." >&2
    echo "[ERROR] Set EXP='${COMMAND_CURRICULUM_DEFAULT_EXP}' or disable COMMAND_CURRICULUM_MODE for legacy extend." >&2
    exit 2
  fi

  if [[ -n "${SCALE_CURRICULUM_STAGES}" || -n "${PROFILE_BLEND_STAGES}" ]]; then
    echo "[ERROR] COMMAND_CURRICULUM_MODE must run as a single torchrun stage." >&2
    echo "[ERROR] Leave SCALE_CURRICULUM_STAGES and PROFILE_BLEND_STAGES empty; the curriculum is handled inside the run." >&2
    exit 2
  fi

  validate_command_curriculum_probability_budget \
    "COMMAND_CURRICULUM start" \
    "${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START}" \
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START}" \
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START}"
  validate_command_curriculum_probability_budget \
    "COMMAND_CURRICULUM end" \
    "${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END}" \
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END}" \
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END}"
  validate_command_curriculum_iteration_schedule
}

append_scalar_arg() {
  local cmd_name=$1
  local -n cmd_ref="${cmd_name}"
  local _flag=$2
  local _value=$3
  cmd_ref+=("${_flag}=${_value}")
}

append_list_flag() {
  local cmd_name=$1
  local -n cmd_ref="${cmd_name}"
  local _flag=$2
  local _spec=$3
  if [[ -z "${_spec}" ]]; then
    return
  fi
  IFS=',' read -r -a _parts <<<"${_spec}"
  cmd_ref+=("${_flag}")
  local _trimmed
  for _part in "${_parts[@]}"; do
    _trimmed=$(echo "${_part}" | xargs)
    if [[ -n "${_trimmed}" ]]; then
      cmd_ref+=("${_trimmed}")
    fi
  done
}

scale_is_identity() {
  local spec=$1
  local normalized
  normalized=$(echo "${spec}" | tr -d '[:space:]')
  case "${normalized}" in
    ""|1|1.0|1.00|1.000|1.0000|1,1,1|1.0,1.0,1.0|1.00,1.00,1.00|1.000,1.000,1.000)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

sanitize_scale_tag() {
  local spec=$1
  local cleaned
  cleaned=$(echo "${spec}" | tr -d '[:space:]')
  cleaned=${cleaned//,/x}
  cleaned=${cleaned//./p}
  cleaned=${cleaned//-/m}
  echo "${cleaned}"
}

compact_stage_specs() {
  local src_name=$1
  local dst_name=$2
  local -n src_ref="${src_name}"
  local -n dst_ref="${dst_name}"
  dst_ref=()

  local raw_spec trimmed_spec
  for raw_spec in "${src_ref[@]}"; do
    trimmed_spec=$(echo "${raw_spec}" | xargs)
    if [[ -n "${trimmed_spec}" ]]; then
      dst_ref+=("${trimmed_spec}")
    fi
  done
}

validate_stage_spec_count() {
  local label=$1
  local count=$2
  local stage_count=$3
  if (( count != 0 && count != 1 && count != stage_count )); then
    echo "[ERROR] ${label} must provide either 1 spec or ${stage_count} specs. Got ${count}." >&2
    exit 2
  fi
}

get_stage_spec_value() {
  local array_name=$1
  local stage_index=$2
  local default_value=$3
  local output_var=$4
  local -n array_ref="${array_name}"

  local selected="${default_value}"
  if (( ${#array_ref[@]} > 0 )); then
    if (( ${#array_ref[@]} == 1 )); then
      selected="${array_ref[0]}"
    else
      selected="${array_ref[stage_index]}"
    fi
  fi
  printf -v "${output_var}" '%s' "${selected}"
}

parse_stage_spec() {
  local raw_spec=$1
  local default_value=$2
  local value_var=$3
  local iters_var=$4

  local value="${default_value}"
  local iters=""
  if [[ -n "${raw_spec}" ]]; then
    value="${raw_spec}"
    if [[ "${raw_spec}" == *"@"* ]]; then
      value="${raw_spec%@*}"
      iters="${raw_spec#*@}"
    fi
  fi

  value=$(echo "${value}" | xargs)
  iters=$(echo "${iters}" | xargs)
  printf -v "${value_var}" '%s' "${value}"
  printf -v "${iters_var}" '%s' "${iters}"
}

resolve_stage_iters() {
  local scale_iters=$1
  local blend_iters=$2
  local resolved_iters="${NUM_ITERS}"

  if [[ -n "${scale_iters}" && -n "${blend_iters}" && "${scale_iters}" != "${blend_iters}" ]]; then
    echo "[ERROR] Stage iteration mismatch: scale curriculum requested ${scale_iters}, blend curriculum requested ${blend_iters}." >&2
    exit 2
  fi
  if [[ -n "${blend_iters}" ]]; then
    resolved_iters="${blend_iters}"
  elif [[ -n "${scale_iters}" ]]; then
    resolved_iters="${scale_iters}"
  fi
  echo "${resolved_iters}"
}

find_latest_stage_dir() {
  local stage_name=$1
  local project_dir="${LOGGER_BASE_DIR}/${WANDB_PROJECT}"
  if [[ ! -d "${project_dir}" ]]; then
    return 1
  fi
  find "${project_dir}" -maxdepth 1 -mindepth 1 -type d -name "*-${stage_name}-*" | sort | tail -n1
}

find_latest_checkpoint_for_stage() {
  local stage_name=$1
  local stage_dir
  stage_dir=$(find_latest_stage_dir "${stage_name}") || return 1
  find "${stage_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort | tail -n1
}

append_common_training_args() {
  local cmd_name=$1
  local -n cmd_ref="${cmd_name}"
  local motion_source=$2
  local stage_name=$3
  local stage_iters=$4
  local checkpoint_path=$5

  cmd_ref+=(
    "exp:${EXP}"
    "perception:${PERCEPTION}"
    --training.project="${WANDB_PROJECT}"
    --training.name="${stage_name}"
    --training.num_envs="${NUM_ENVS}"
    --simulator.config.sim.physx.gpu-max-rigid-patch-count="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT}"
    --command.setup_terms.motion_command.params.motion_config.motion_file
    "${motion_source}"
    --algo.config.save_interval="${SAVE_INTERVAL}"
    logger:wandb
    --logger.base_dir="${LOGGER_BASE_DIR}"
    --logger.group="${RUN_GROUP}"
    --logger.name="${stage_name}"
    --logger.video.enabled=False
    --logger.headless_recording=False
    --logger.video.upload_to_wandb=False
  )

  if [[ -n "${stage_iters}" ]]; then
    append_scalar_arg "${cmd_name}" --algo.config.num_learning_iterations "${stage_iters}"
  fi
  if [[ -n "${checkpoint_path}" ]]; then
    cmd_ref+=(--training.checkpoint="${checkpoint_path}")
  fi
  if [[ -n "${OBJECT_SPEC_PATH}" ]]; then
    cmd_ref+=(--robot.object.object_urdf_path "${OBJECT_SPEC_PATH}")
  fi
  if [[ -n "${MOTION_CLIP_ID}" ]]; then
    append_scalar_arg "${cmd_name}" --command.setup_terms.motion_command.params.motion_config.motion_clip_id "${MOTION_CLIP_ID}"
  fi
  if [[ -n "${MOTION_CLIP_NAME}" ]]; then
    cmd_ref+=(
      --command.setup_terms.motion_command.params.motion_config.motion_clip_name
      "${MOTION_CLIP_NAME}"
    )
  fi
  if [[ "${ENABLE_VISER}" == "1" ]]; then
    cmd_ref+=(
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
}

append_object_scale_args() {
  local cmd_name=$1
  local -n cmd_ref="${cmd_name}"
  local scale_spec=$2
  if scale_is_identity "${scale_spec}"; then
    return
  fi
  append_list_flag "${cmd_name}" --robot.object.scale "${scale_spec}"
  append_list_flag "${cmd_name}" --command.setup_terms.motion_command.params.motion_config.object_size_scale "${scale_spec}"
}

append_curriculum_args() {
  local cmd_name=$1
  local -n cmd_ref="${cmd_name}"
  if [[ "${ASSIST_CURRICULUM}" == "1" || "${ASSIST_CURRICULUM,,}" == "true" ]]; then
    cmd_ref+=(
      --curriculum.setup_terms.w_object_difficulty_curriculum.params.enabled=True
      --curriculum.setup_terms.w_object_difficulty_curriculum.params.initial_lambda="${ASSIST_INITIAL_LAMBDA}"
      --curriculum.setup_terms.w_object_difficulty_curriculum.params.lambda_step_up="${ASSIST_LAMBDA_STEP_UP}"
      --curriculum.setup_terms.w_object_difficulty_curriculum.params.lambda_step_down="${ASSIST_LAMBDA_STEP_DOWN}"
      --curriculum.setup_terms.w_object_difficulty_curriculum.params.early_termination_threshold="${ASSIST_EARLY_TERM_THRESHOLD}"
      --curriculum.setup_terms.w_object_difficulty_curriculum.params.similarity_threshold="${ASSIST_SIMILARITY_THRESHOLD}"
      --curriculum.setup_terms.w_object_difficulty_curriculum.params.assist_beta_max="${ASSIST_BETA_MAX}"
    )
  else
    cmd_ref+=(--curriculum.setup_terms.w_object_difficulty_curriculum.params.enabled=False)
  fi
}

append_command_curriculum_args() {
  local cmd_name=$1
  local -n cmd_ref="${cmd_name}"

  if ! is_truthy "${COMMAND_CURRICULUM_MODE}"; then
    return
  fi

  cmd_ref+=(
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.enabled=True
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command-only-env-prob-start="${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command-only-env-prob-end="${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command-only-env-prob-start-iter="${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command-only-env-prob-end-iter="${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.command-only-env-prob-ramp-resets="${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_RAMP_RESETS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval-command-only-env-prob="${COMMAND_CURRICULUM_EVAL_COMMAND_ONLY_ENV_PROB}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.clip-goal-delta-min-steps="${COMMAND_CURRICULUM_CLIP_GOAL_DELTA_MIN_STEPS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.clip-goal-delta-max-steps="${COMMAND_CURRICULUM_CLIP_GOAL_DELTA_MAX_STEPS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-prob-start="${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-prob-end="${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-prob-start-iter="${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-prob-end-iter="${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-prob-ramp-resets="${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_RAMP_RESETS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval-carry-extension-prob="${COMMAND_CURRICULUM_EVAL_CARRY_EXTENSION_PROB}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-range-start-iter="${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-range-end-iter="${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-range-ramp-resets="${COMMAND_CURRICULUM_CARRY_EXTENSION_RANGE_RAMP_RESETS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-prob-start="${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-prob-end="${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-prob-start-iter="${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-prob-end-iter="${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-prob-ramp-resets="${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_RAMP_RESETS}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.eval-external-goal-prob="${COMMAND_CURRICULUM_EVAL_EXTERNAL_GOAL_PROB}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-range-start-iter="${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_START_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-range-end-iter="${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_END_ITER}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-range-ramp-resets="${COMMAND_CURRICULUM_EXTERNAL_GOAL_RANGE_RAMP_RESETS}"
    --simulator.config.sim.max_episode_length_s="${COMMAND_CURRICULUM_MAX_EPISODE_LENGTH_S}"
  )

  cmd_ref+=(
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-pos-local-min-start
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MIN_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-pos-local-max-start
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MAX_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-pos-local-min
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MIN}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-pos-local-max
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_POS_LOCAL_MAX}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-rpy-min-start
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MIN_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-rpy-max-start
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MAX_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-rpy-min
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MIN}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.carry-extension-rpy-max
    "${COMMAND_CURRICULUM_CARRY_EXTENSION_RPY_MAX}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-pos-local-min-start
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MIN_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-pos-local-max-start
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MAX_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-pos-local-min
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MIN}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-pos-local-max
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_POS_LOCAL_MAX}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-rpy-min-start
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MIN_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-rpy-max-start
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MAX_START}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-rpy-min
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MIN}"
    --command.setup_terms.motion_command.params.motion_config.sparse_object_goal.external-goal-rpy-max
    "${COMMAND_CURRICULUM_EXTERNAL_GOAL_RPY_MAX}"
  )
}

append_reward_args() {
  local cmd_name=$1
  local -n cmd_ref="${cmd_name}"
  cmd_ref+=(
    --reward.terms.motion_global_ref_position_error_exp.weight="${ROOT_POS_W}"
    --reward.terms.motion_global_ref_orientation_error_exp.weight="${ROOT_ORI_W}"
    --reward.terms.motion_relative_body_position_error_exp.weight="${FULL_BODY_POS_W}"
    --reward.terms.motion_relative_body_orientation_error_exp.weight="${FULL_BODY_ORI_W}"
    --reward.terms.motion_global_body_lin_vel.weight="${FULL_BODY_LIN_VEL_W}"
    --reward.terms.motion_global_body_ang_vel.weight="${FULL_BODY_ANG_VEL_W}"
    --reward.terms.motion_relative_body_position_error_lower.weight="${LOWER_BODY_POS_W}"
    --reward.terms.motion_relative_body_orientation_error_lower.weight="${LOWER_BODY_ORI_W}"
    --reward.terms.motion_global_body_lin_vel_lower.weight="${LOWER_BODY_LIN_VEL_W}"
    --reward.terms.motion_global_body_ang_vel_lower.weight="${LOWER_BODY_ANG_VEL_W}"
    --reward.terms.motion_relative_body_position_error_torso.weight="${TORSO_BODY_POS_W}"
    --reward.terms.motion_relative_body_orientation_error_torso.weight="${TORSO_BODY_ORI_W}"
    --reward.terms.motion_relative_body_position_error_upper.weight="${UPPER_BODY_POS_W}"
    --reward.terms.motion_relative_body_orientation_error_upper.weight="${UPPER_BODY_ORI_W}"
    --reward.terms.motion_global_body_lin_vel_upper.weight="${UPPER_BODY_LIN_VEL_W}"
    --reward.terms.motion_global_body_ang_vel_upper.weight="${UPPER_BODY_ANG_VEL_W}"
    --reward.terms.motion_joint_position_error_lower.weight="${LOWER_JOINT_POS_W}"
    --reward.terms.motion_joint_velocity_error_lower.weight="${LOWER_JOINT_VEL_W}"
    --reward.terms.motion_joint_position_error_waist.weight="${WAIST_JOINT_POS_W}"
    --reward.terms.motion_joint_velocity_error_waist.weight="${WAIST_JOINT_VEL_W}"
    --reward.terms.motion_joint_position_error_upper.weight="${UPPER_JOINT_POS_W}"
    --reward.terms.motion_joint_velocity_error_upper.weight="${UPPER_JOINT_VEL_W}"
    --reward.terms.object_global_ref_position_error_exp.weight="${OBJECT_POS_W}"
    --reward.terms.object_global_ref_orientation_error_exp.weight="${OBJECT_ORI_W}"
    --reward.terms.body_contact_reward_palms.weight="${PALM_CONTACT_W}"
    --reward.terms.body_contact_reward_arms.weight="${ARM_CONTACT_W}"
    --reward.terms.body_contact_reward_torso.weight="${TORSO_CONTACT_W}"
    --reward.terms.body_contact_reward_palms.params.threshold="${CONTACT_THRESHOLD}"
    --reward.terms.body_contact_reward_palms.params.force_scale="${CONTACT_FORCE_SCALE}"
    --reward.terms.body_contact_reward_palms.params.reward_mode="${CONTACT_MODE}"
    --reward.terms.body_contact_reward_arms.params.threshold="${CONTACT_THRESHOLD}"
    --reward.terms.body_contact_reward_arms.params.force_scale="${CONTACT_FORCE_SCALE}"
    --reward.terms.body_contact_reward_arms.params.reward_mode="${CONTACT_MODE}"
    --reward.terms.body_contact_reward_torso.params.threshold="${CONTACT_THRESHOLD}"
    --reward.terms.body_contact_reward_torso.params.force_scale="${CONTACT_FORCE_SCALE}"
    --reward.terms.body_contact_reward_torso.params.reward_mode="${CONTACT_MODE}"
  )
}

run_stage() {
  local stage_index=$1
  local scale_spec=$2
  local stage_iters=$3
  local checkpoint_path=$4
  local stage_profile=$5
  local stage_blend=$6
  local stage_tag
  stage_tag=$(sanitize_scale_tag "${scale_spec}")
  local stage_name="${TRAINING_NAME}_${RUN_TAG}_s$(printf '%02d' "${stage_index}")_scale_${stage_tag}"

  apply_profile_weights "${stage_profile}" "${stage_blend}"

  local train_cmd=(
    src/holosoma/holosoma/train_agent.py
  )
  append_common_training_args train_cmd "${MOTION_SOURCE}" "${stage_name}" "${stage_iters}" "${checkpoint_path}"
  append_object_scale_args train_cmd "${scale_spec}"
  append_curriculum_args train_cmd
  append_command_curriculum_args train_cmd
  append_reward_args train_cmd
  train_cmd+=("${EXTRA_ARGS[@]}")

  echo "[INFO] Stage ${stage_index}: profile=${stage_profile} scale=${scale_spec} stage_name=${stage_name}"
  echo "[INFO] Stage ${stage_index}: num_envs=${NUM_ENVS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} physx_gpu_max_rigid_patch_count=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT}"
  if [[ "${stage_profile}" == "blend" || "${stage_profile}" == "transition" ]]; then
    echo "[INFO] Stage ${stage_index}: profile_blend=${stage_blend} (0=full tracking, 1=decoupled/contact-oriented)"
  fi
  if [[ -n "${stage_iters}" ]]; then
    echo "[INFO] Stage ${stage_index}: additional learning iterations=${stage_iters}"
  fi
  if [[ -n "${checkpoint_path}" ]]; then
    echo "[INFO] Stage ${stage_index}: resuming from ${checkpoint_path}"
  fi
  if is_truthy "${COMMAND_CURRICULUM_MODE}"; then
    echo "[INFO] Stage ${stage_index}: command_curriculum=True"
    echo "[INFO] Stage ${stage_index}: co_tracking_fraction schedule 1.00 -> $(awk -v c="${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END}" 'BEGIN { printf "%.2f", 1.0 - c }')"
    echo "[INFO] Stage ${stage_index}: hybrid_start_iter=${COMMAND_CURRICULUM_HYBRID_START_ITER}"
    echo "[INFO] Stage ${stage_index}: command_only_fraction ${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START} -> ${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END} iter=${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START_ITER} -> ${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END_ITER}"
    echo "[INFO] Stage ${stage_index}: carry_extension_prob ${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START} -> ${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END} iter=${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START_ITER} -> ${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END_ITER}"
    echo "[INFO] Stage ${stage_index}: external_goal_prob ${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START} -> ${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END} iter=${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START_ITER} -> ${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END_ITER}"
  fi

  local launch_env=(env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
  if is_truthy "${COMMAND_CURRICULUM_MODE}"; then
    launch_env+=("HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=${COMMAND_CURRICULUM_PERCEPTION_INTO_POLICY_MODULES}")
  fi

  "${launch_env[@]}" "${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    "${train_cmd[@]}"

  local latest_ckpt
  latest_ckpt=$(find_latest_checkpoint_for_stage "${stage_name}") || {
    echo "[ERROR] Failed to resolve checkpoint for stage ${stage_name} under ${LOGGER_BASE_DIR}/${WANDB_PROJECT}" >&2
    exit 3
  }
  echo "[INFO] Stage ${stage_index}: latest checkpoint=${latest_ckpt}"
  LAST_STAGE_NAME="${stage_name}"
  LAST_CHECKPOINT="${latest_ckpt}"
}

apply_command_curriculum_defaults
capture_explicit_weight_overrides
validate_command_curriculum_configuration
validate_profile_configuration

echo "[INFO] EXP: ${EXP}"
echo "[INFO] PROFILE: ${PROFILE}"
echo "[INFO] MOTION_SOURCE: ${MOTION_SOURCE}"
echo "[INFO] OBJECT_SPEC_PATH: ${OBJECT_SPEC_PATH:-<exp default>}"
echo "[INFO] RUN_GROUP: ${RUN_GROUP}"
if is_truthy "${COMMAND_CURRICULUM_MODE}"; then
  echo "[INFO] COMMAND_CURRICULUM_MODE: True"
  echo "[INFO] COMMAND_CURRICULUM_PERCEPTION: ${PERCEPTION}"
  echo "[INFO] COMMAND_CURRICULUM_EPISODE_LENGTH_S: ${COMMAND_CURRICULUM_MAX_EPISODE_LENGTH_S}"
  echo "[INFO] COMMAND_CURRICULUM_HYBRID_START_ITER: ${COMMAND_CURRICULUM_HYBRID_START_ITER}"
  echo "[INFO] COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB: ${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START} -> ${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END} iter=${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_START_ITER} -> ${COMMAND_CURRICULUM_COMMAND_ONLY_ENV_PROB_END_ITER}"
  echo "[INFO] COMMAND_CURRICULUM_CARRY_EXTENSION_PROB: ${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START} -> ${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END} iter=${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_START_ITER} -> ${COMMAND_CURRICULUM_CARRY_EXTENSION_PROB_END_ITER}"
  echo "[INFO] COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB: ${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START} -> ${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END} iter=${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_START_ITER} -> ${COMMAND_CURRICULUM_EXTERNAL_GOAL_PROB_END_ITER}"
fi
if [[ -n "${PROFILE_BLEND}" ]]; then
  echo "[INFO] PROFILE_BLEND: ${PROFILE_BLEND}"
fi

LAST_STAGE_NAME=""
LAST_CHECKPOINT="${CHECKPOINT}"

RAW_SCALE_STAGE_SPECS=()
RAW_BLEND_STAGE_SPECS=()
CLEAN_SCALE_STAGE_SPECS=()
CLEAN_BLEND_STAGE_SPECS=()

if [[ -n "${SCALE_CURRICULUM_STAGES}" ]]; then
  IFS=';' read -r -a RAW_SCALE_STAGE_SPECS <<<"${SCALE_CURRICULUM_STAGES}"
  compact_stage_specs RAW_SCALE_STAGE_SPECS CLEAN_SCALE_STAGE_SPECS
fi

if [[ -n "${PROFILE_BLEND_STAGES}" ]]; then
  IFS=';' read -r -a RAW_BLEND_STAGE_SPECS <<<"${PROFILE_BLEND_STAGES}"
  compact_stage_specs RAW_BLEND_STAGE_SPECS CLEAN_BLEND_STAGE_SPECS
fi

stage_count=1
if (( ${#CLEAN_SCALE_STAGE_SPECS[@]} > stage_count )); then
  stage_count=${#CLEAN_SCALE_STAGE_SPECS[@]}
fi
if (( ${#CLEAN_BLEND_STAGE_SPECS[@]} > stage_count )); then
  stage_count=${#CLEAN_BLEND_STAGE_SPECS[@]}
fi

validate_stage_spec_count "SCALE_CURRICULUM_STAGES" "${#CLEAN_SCALE_STAGE_SPECS[@]}" "${stage_count}"
validate_stage_spec_count "PROFILE_BLEND_STAGES" "${#CLEAN_BLEND_STAGE_SPECS[@]}" "${stage_count}"

if [[ -n "${SCALE_CURRICULUM_STAGES}" && "${#CLEAN_SCALE_STAGE_SPECS[@]}" -eq 0 ]]; then
  echo "[ERROR] SCALE_CURRICULUM_STAGES='${SCALE_CURRICULUM_STAGES}' did not contain any non-empty stages." >&2
  exit 2
fi
if [[ -n "${PROFILE_BLEND_STAGES}" && "${#CLEAN_BLEND_STAGE_SPECS[@]}" -eq 0 ]]; then
  echo "[ERROR] PROFILE_BLEND_STAGES='${PROFILE_BLEND_STAGES}' did not contain any non-empty stages." >&2
  exit 2
fi

for ((stage_idx = 0; stage_idx < stage_count; stage_idx++)); do
  get_stage_spec_value CLEAN_SCALE_STAGE_SPECS "${stage_idx}" "${OBJECT_SCALE}" raw_scale_spec
  get_stage_spec_value CLEAN_BLEND_STAGE_SPECS "${stage_idx}" "${PROFILE_BLEND}" raw_blend_spec

  parse_stage_spec "${raw_scale_spec}" "${OBJECT_SCALE}" scale_spec stage_iters_from_scale
  parse_stage_spec "${raw_blend_spec}" "${PROFILE_BLEND}" stage_blend stage_iters_from_blend
  stage_iters=$(resolve_stage_iters "${stage_iters_from_scale}" "${stage_iters_from_blend}")

  if [[ "${PROFILE}" != "blend" && "${PROFILE}" != "transition" ]]; then
    stage_blend=""
  elif [[ -z "${stage_blend}" ]]; then
    echo "[ERROR] Stage $((stage_idx + 1)) is missing a blend value. Set PROFILE_BLEND or PROFILE_BLEND_STAGES." >&2
    exit 2
  fi

  run_stage "$((stage_idx + 1))" "${scale_spec}" "${stage_iters}" "${LAST_CHECKPOINT}" "${PROFILE}" "${stage_blend}"
done
