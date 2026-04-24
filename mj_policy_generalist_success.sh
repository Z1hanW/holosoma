#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-wandb://zihanw22/boxer/u5lguxvl/latest.onnx}"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-${ROOT_DIR}/outputs/motion_bank_success_box_0_92_0p3}"

usage() {
  cat <<EOF
Usage:
  bash mj_policy_generalist_success.sh [clip_name|motion.npz] [model.onnx|wandb://...]

Purpose:
  Launch the policy side for the generalist success-rollout MuJoCo demo.
  Pair this with mj_env_generalist_success.sh.

Defaults:
  model      = ${DEFAULT_MODEL_REF}
  motion dir = ${DEFAULT_MOTION_DIR}

Examples:
  bash mj_policy_generalist_success.sh
  bash mj_policy_generalist_success.sh box_74
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
export MODEL_REF="${MODEL_REF:-${MODEL_PATH:-${MODEL_INPUT:-${DEFAULT_MODEL_REF}}}}"
export MODEL_INPUT="${MODEL_INPUT:-${MODEL_REF}}"
export MJ_TRACK_LAUNCHER="${ROOT_DIR}/mj_track_generalist_success.sh"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-6655}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-6657}"
export PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-6658}"
export SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-6659}"
export SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-6661}"
export POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-6662}"
export POLICY_OVERLAY_PORT="${POLICY_OVERLAY_PORT:-6663}"
export HOLOSOMA_POLICY_OVERLAY_PORT="${HOLOSOMA_POLICY_OVERLAY_PORT:-${POLICY_OVERLAY_PORT}}"
export HOLOSOMA_SKIP_STIFF_PROMPT="${HOLOSOMA_SKIP_STIFF_PROMPT:-1}"
export MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-1}"
export HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE="${HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE:-1.0}"

exec bash "${ROOT_DIR}/mj_policy.sh" "$@"
