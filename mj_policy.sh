#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-https://wandb.ai/zihanw22/boxer/runs/w5qostjn}"
TRACK_LAUNCHER="${MJ_TRACK_LAUNCHER:-${ROOT_DIR}/mj_box_depth_track.sh}"

usage() {
  cat <<EOF
Usage:
  bash mj_policy.sh [rendered|rendered848|warp] [clip_name|motion.npz] [model.onnx|wandb://...]

Examples:
  bash mj_policy.sh box_74
  bash mj_policy.sh rendered box_74 ${DEFAULT_MODEL_INPUT}
  bash mj_policy.sh rendered848 box_74 ${DEFAULT_MODEL_INPUT}

Environment:
  MODEL_INPUT / MODEL_PATH / MODEL_REF
                                  default: ${DEFAULT_MODEL_INPUT}
  SIM_CLOCK_PORT                  default: 5655
  SIM_STATE_PORT                  default: 5657
  PERCEPTION_OBS_PORT             default: 5658
  SIM_CONTROL_PORT                default: 5659
  SPARSE_ROOT_COMMAND_PORT        default: 5661
  POLICY_CONTROL_PORT             default: 5662 for web ]/Space/Stop/Init policy control
  MJ_POLICY_KILL_STALE            default: 1; terminate same-port policy leftovers before launch
  MJ_POLICY_TERMINAL_KEYS=1       use terminal W/S/A/D/Q/E instead of web command
  HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE         default: 0.2 for W/S/A/D x/y
  HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES   default: 17 for Q/E yaw
  HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE    default: hold; set latch to tap W once and hold command
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

export MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${MODEL_REF:-${DEFAULT_MODEL_INPUT}}}}"
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-far_tracking_warp}"
export HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE="${HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE:-1}"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
export PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
export SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
export SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-5661}"
export POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-5662}"
export POLICY_OVERLAY_PORT="${POLICY_OVERLAY_PORT:-5663}"
export HOLOSOMA_POLICY_CONTROL_PORT="${HOLOSOMA_POLICY_CONTROL_PORT:-$POLICY_CONTROL_PORT}"
export HOLOSOMA_POLICY_OVERLAY_PORT="${HOLOSOMA_POLICY_OVERLAY_PORT:-$POLICY_OVERLAY_PORT}"
export HOLOSOMA_SKIP_STIFF_PROMPT="${HOLOSOMA_SKIP_STIFF_PROMPT:-1}"
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
MJ_POLICY_KILL_STALE="${MJ_POLICY_KILL_STALE:-1}"

same_port_policy_pids() {
  ps -eww -o pid=,args= | awk \
    -v self="$$" \
    -v root="${ROOT_DIR}" \
    -v state="${SIM_STATE_PORT}" \
    -v control="${SIM_CONTROL_PORT}" '
      $1 == self { next }
      (index($0, "python") || index($0, "python3")) &&
      index($0, root "/src/holosoma_inference/holosoma_inference/run_policy.py") &&
      index($0, "--task.sim-state-port " state) &&
      index($0, "--task.sim-control-port " control) {
        print $1
      }
    ' | sort -n -u
}

terminate_pids() {
  local pids="$1"
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  sleep 1
  local pid
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

format_pids() {
  printf '%s\n' "$1" | tr '\n' ' ' | sed 's/[[:space:]]*$//'
}

STALE_POLICY_PIDS="$(same_port_policy_pids || true)"
if [[ -n "$STALE_POLICY_PIDS" ]]; then
  STALE_POLICY_PIDS_ONE_LINE="$(format_pids "$STALE_POLICY_PIDS")"
  if is_truthy "$MJ_POLICY_KILL_STALE"; then
    echo "[WARN] terminating stale policy process(es) on state=${SIM_STATE_PORT} control=${SIM_CONTROL_PORT}: ${STALE_POLICY_PIDS_ONE_LINE}" >&2
    terminate_pids "$STALE_POLICY_PIDS"
  else
    echo "[ERROR] stale policy process(es) already target state=${SIM_STATE_PORT} control=${SIM_CONTROL_PORT}: ${STALE_POLICY_PIDS_ONE_LINE}" >&2
    echo "[ERROR] Stop them first, or set MJ_POLICY_KILL_STALE=1." >&2
    exit 1
  fi
fi

if is_truthy "${MJ_POLICY_TERMINAL_KEYS:-0}"; then
  export HOLOSOMA_POLICY_TTY_INPUT="${HOLOSOMA_POLICY_TTY_INPUT:-1}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND="${HOLOSOMA_KEYBOARD_ROOT_COMMAND:-1}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE:-0.2}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES:-${HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEG:-17}}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE:-manual}"
  export HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE:-hold}"
else
  unset HOLOSOMA_POLICY_TTY_INPUT
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE
  unset HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE
fi

echo "[INFO] launching policy only"
echo "[INFO] model=${MODEL_INPUT}"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT} sparse_root=${SPARSE_ROOT_COMMAND_PORT} policy_control=${HOLOSOMA_POLICY_CONTROL_PORT} policy_overlay=${HOLOSOMA_POLICY_OVERLAY_PORT}"
echo "[INFO] web sparse-root command=${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND}"
echo "[INFO] terminal sparse-root command=${HOLOSOMA_KEYBOARD_ROOT_COMMAND:-0} input_mode=${HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE:-hold}"
echo "[INFO] web policy controls: ] starts policy, Space starts motion clip, Stop, Init"

if is_truthy "${DRY_RUN:-0}"; then
  echo "[INFO] DRY_RUN=1; not launching policy."
  exit 0
fi

exec bash "${TRACK_LAUNCHER}" "$@"
