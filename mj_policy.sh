#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_INPUT="${ROOT_DIR}/logs/wandb_runs/shoo7sr1/model_18500.onnx"

usage() {
  cat <<EOF
Usage:
  bash mj_policy.sh [rendered|warp] [clip_name|motion.npz] [model.onnx|wandb://...]

Examples:
  bash mj_policy.sh box_74
  bash mj_policy.sh rendered box_74 ${DEFAULT_MODEL_INPUT}

Environment:
  MODEL_INPUT / MODEL_PATH        default: ${DEFAULT_MODEL_INPUT}
  SIM_CLOCK_PORT                  default: 5655
  SIM_STATE_PORT                  default: 5657
  PERCEPTION_OBS_PORT             default: 5658
  SIM_CONTROL_PORT                default: 5659
  SPARSE_ROOT_COMMAND_PORT        default: 5661
  POLICY_CONTROL_PORT             default: 5662; command web start/stop/init channel
  MJ_POLICY_TERMINAL_KEYS=1       use terminal W/S/A/D/Q/E instead of web command
  HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE         default: 0.5 for W/S/A/D x/y
  HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES   default: 17 for Q/E yaw
  HOLOSOMA_ZMQ_ACTIVE_BEFORE_POLICY_START=1    allow active lowcmd before pressing ] (default: inactive)
  HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART=1
                                  keep old non-TTY auto-start behavior even with policy control enabled

Interactive keys:
  Enter                           acknowledge startup prompt
  ]                               start policy actions
  o                               stop policy actions
  i                               move to init state
EOF
}

is_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

export MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${DEFAULT_MODEL_INPUT}}}"
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-rendered}"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
export PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
export SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
export SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-5661}"
export POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-5662}"
export HOLOSOMA_POLICY_CONTROL_PORT="${HOLOSOMA_POLICY_CONTROL_PORT:-$POLICY_CONTROL_PORT}"
export ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-1}"
export RUN_SECONDS="${RUN_SECONDS:-0}"
export POLICY_STDIO="${POLICY_STDIO:-inherit}"
export POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-1}"
export MJ_TRACK_MODE=policy
export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1

export HOLOSOMA_DISABLE_AUTO_RESET="${HOLOSOMA_DISABLE_AUTO_RESET:-1}"
export HOLOSOMA_DISABLE_MOTION_END_RESET="${HOLOSOMA_DISABLE_MOTION_END_RESET:-1}"
export HOLOSOMA_DISABLE_CLIP_END_RESET="${HOLOSOMA_DISABLE_CLIP_END_RESET:-1}"
export HOLOSOMA_DISABLE_BAD_TRACKING_RESET="${HOLOSOMA_DISABLE_BAD_TRACKING_RESET:-1}"

if is_truthy "${MJ_POLICY_TERMINAL_KEYS:-0}"; then
  export HOLOSOMA_POLICY_TTY_INPUT="${HOLOSOMA_POLICY_TTY_INPUT:-1}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND="${HOLOSOMA_KEYBOARD_ROOT_COMMAND:-1}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE:-0.5}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES:-${HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEG:-17}}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE:-manual}"
else
  unset HOLOSOMA_POLICY_TTY_INPUT
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE
fi

echo "[INFO] launching policy only"
echo "[INFO] model=${MODEL_INPUT}"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT} sparse_root=${SPARSE_ROOT_COMMAND_PORT} policy_control=${HOLOSOMA_POLICY_CONTROL_PORT}"
echo "[INFO] web sparse-root command=${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND}"
echo "[INFO] web policy control=${HOLOSOMA_POLICY_CONTROL_PORT}"
if [[ "${POLICY_STDIO}" == "inherit" ]]; then
  echo "[INFO] interactive policy keys: press Enter at the prompt, then ] to start policy, o to stop"
fi

if is_truthy "${DRY_RUN:-0}"; then
  echo "[INFO] DRY_RUN=1; not launching policy."
  exit 0
fi

exec bash "${ROOT_DIR}/mj_box_depth_track.sh" "$@"
