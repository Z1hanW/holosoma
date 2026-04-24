#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-wandb://zihanw22/boxer/u5lguxvl/model_17000.onnx}"
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
  trigger    = S -> rollout_start (Space + ])

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
export MODEL_REF="${MODEL_REF:-${MODEL_PATH:-${MODEL_INPUT:-${DEFAULT_MODEL_REF}}}}"
export MODEL_INPUT="${MODEL_INPUT:-${MODEL_REF}}"
export MJ_TRACK_LAUNCHER="${ROOT_DIR}/mj_track_generalist_success.sh"
export MJ_POLICY_HINT_SCRIPT="${ROOT_DIR}/mj_policy_generalist_success.sh"
export COMMAND_WEB_TRACK_ONLY=1
export COMMAND_MANUAL_ENABLED=0

exec bash "${ROOT_DIR}/mj_env.sh" "$@"
