#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CLIP="${DEFAULT_CLIP:-box_74}"

usage() {
  cat <<EOF
Usage:
  bash mj_one.sh [rendered|warp] [clip_name|motion.npz] [model.onnx|wandb://...]

Defaults:
  clip                  = ${DEFAULT_CLIP}
  command web          = on
  manual command        = on
  command value         = 0
  yaw command degrees   = 0
  motion init           = raw_motion

Examples:
  bash mj_one.sh
  bash mj_one.sh box_74
  VISER_PORT=12984 COMMAND_WEB_PORT=18080 bash mj_one.sh box_74
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ $# -eq 0 ]]; then
  set -- "$DEFAULT_CLIP"
fi

export COMMAND_WEB="${COMMAND_WEB:-1}"
export COMMAND_MANUAL_ENABLED="${COMMAND_MANUAL_ENABLED:-1}"
export COMMAND_VALUE="${COMMAND_VALUE:-0}"
export COMMAND_YAW_DEGREES="${COMMAND_YAW_DEGREES:-0}"
export COMMAND_MODE="${COMMAND_MODE:-manual}"
export ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-1}"

export SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-raw_motion}"
export RUN_SECONDS="${RUN_SECONDS:-0}"
export HOLOSOMA_MJ_TRACK_RUN_FOREVER="${HOLOSOMA_MJ_TRACK_RUN_FOREVER:-1}"

echo "[INFO] mj_one: launch env-only rollout from initialized state"
echo "[INFO] mj_one: command web=${COMMAND_WEB}, manual=${COMMAND_MANUAL_ENABLED}, xy=${COMMAND_VALUE}, yaw_deg=${COMMAND_YAW_DEGREES}"
echo "[INFO] mj_one: motion_init=${SIM_MOTION_INIT_MODE}"

exec bash "$ROOT_DIR/mj_env.sh" "$@"
