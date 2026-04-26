#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-wandb://zihanw22/boxer/u5lguxvl/latest.onnx}"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-${ROOT_DIR}/outputs/motion_bank_success_box_0_92_0p3}"

usage() {
  cat <<EOF
Usage:
  bash mj_env_generalist_success.sh [clip_name|motion.npz] [model.onnx|wandb://...]

Purpose:
  Launch the generalist success-rollout MuJoCo env + scene web with a single
  S trigger. There is no manual Q/W/E/S/A/D root command UI in this mode.

Defaults:
  model      = ${DEFAULT_MODEL_REF}
  motion dir = ${DEFAULT_MOTION_DIR}
  web port   = 6060
  trigger    = S -> rollout_start (Space + ]), R/Backspace -> reset rollout

Examples:
  bash mj_env_generalist_success.sh
  bash mj_env_generalist_success.sh box_74
  COMMAND_WEB_PORT=7070 VISER_PORT=2984 bash mj_env_generalist_success.sh box_74
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

export DEFAULT_MODEL_REF
export DEFAULT_MOTION_DIR

is_model_ref() {
  local value="${1:-}"
  [[ "${value}" == wandb://* || "${value}" == https://wandb.ai/* || "${value}" == *.onnx || "${value}" == *.pt ]]
}

LAUNCH_MODEL_REF="${MODEL_REF:-${MODEL_PATH:-${MODEL_INPUT:-${DEFAULT_MODEL_REF}}}}"
ARGS=("$@")
for ((idx = 0; idx < ${#ARGS[@]}; idx++)); do
  arg="${ARGS[$idx]}"
  case "$arg" in
    --)
      break
      ;;
    --model-ref|--model|--model-path)
      if (( idx + 1 < ${#ARGS[@]} )); then
        LAUNCH_MODEL_REF="${ARGS[$((idx + 1))]}"
      fi
      ;;
    --model-ref=*|--model=*|--model-path=*)
      LAUNCH_MODEL_REF="${arg#*=}"
      ;;
    -*)
      ;;
    *)
      if is_model_ref "$arg"; then
        LAUNCH_MODEL_REF="$arg"
      fi
      ;;
  esac
done
export MODEL_REF="${LAUNCH_MODEL_REF}"
export MODEL_INPUT="${LAUNCH_MODEL_REF}"
export MJ_TRACK_LAUNCHER="${ROOT_DIR}/mj_track_generalist_success.sh"
export MJ_POLICY_HINT_SCRIPT="${ROOT_DIR}/mj_policy_generalist_success.sh"
export COMMAND_WEB_PORT="${COMMAND_WEB_PORT:-6060}"
export COMMAND_WEB_LOG="${COMMAND_WEB_LOG:-${ROOT_DIR}/logs/live_debug/mj_command_web_6060.log}"
export VISER_PORT="${VISER_PORT:-3984}"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-6655}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-6657}"
export PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-6658}"
export SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-6659}"
export SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-6661}"
export POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-6662}"
export POLICY_OVERLAY_PORT="${POLICY_OVERLAY_PORT:-6663}"
export HOLOSOMA_POLICY_OVERLAY_PORT="${HOLOSOMA_POLICY_OVERLAY_PORT:-${POLICY_OVERLAY_PORT}}"
export COMMAND_WEB_TRACK_ONLY=1
export COMMAND_MANUAL_ENABLED=0
export SHOW_MOTION_ROBOT="${SHOW_MOTION_ROBOT:-1}"
export SHOW_MOTION_OBJECT="${SHOW_MOTION_OBJECT:-1}"
if [[ -z "${GT_MUJOCO_PHYSICS+x}" && -z "${HOLOSOMA_GT_MUJOCO_PHYSICS+x}" ]]; then
  export GT_MUJOCO_PHYSICS=0
  export HOLOSOMA_GT_MUJOCO_PHYSICS=0
fi
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
export SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-1}"
export HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION="${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION:-1}"
export HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS:-1}"
export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-1.1}"
export HOLOSOMA_SIM_STATE_INCLUDE_KEY_BODY_STATES="${HOLOSOMA_SIM_STATE_INCLUDE_KEY_BODY_STATES:-1}"
export HOLOSOMA_SIM_STATE_KEY_BODY_NAMES="${HOLOSOMA_SIM_STATE_KEY_BODY_NAMES:-torso_link,pelvis,left_shoulder_roll_link,right_shoulder_roll_link,left_elbow_link,right_elbow_link,left_wrist_yaw_link,right_wrist_yaw_link,left_hip_roll_link,right_hip_roll_link,left_knee_link,right_knee_link,left_ankle_roll_link,right_ankle_roll_link}"
export MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-0}"
export SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-training_default_pose}"

exec bash "${ROOT_DIR}/mj_env.sh" "$@"
