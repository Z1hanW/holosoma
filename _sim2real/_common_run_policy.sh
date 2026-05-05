#!/usr/bin/env bash
set -euo pipefail

SIM2REAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SIM2REAL_DIR}/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<EOF
Usage:
  UNITREE_INTERFACE=<iface> bash _sim2real/<policy>.sh [extra run_policy args...]

Required:
  UNITREE_INTERFACE          real Unitree network interface, for example eth0 or enp3s0

Common environment:
  UNITREE_DOMAIN_ID          default: 0
  PERCEPTION_OBS_PORT        default: 5658; real depth/perception_obs ZMQ publisher
  MODEL_INPUT                override W&B run URL or local .onnx path
  INFERENCE_CONFIG           override inference config
  HOLOSOMA_JOYSTICK_ROOT_COMMAND_VALUE
                            default: 0.2 full-stick xy command
  HOLOSOMA_JOYSTICK_ROOT_COMMAND_XY_MAX
                            default: 0.5; clamps x/y to [-0.5, 0.5]
  HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_DEGREES
                            default: 17 full-stick yaw command
  HOLOSOMA_JOYSTICK_ROOT_COMMAND_X_SIGN/Y_SIGN/YAW_SIGN
                            defaults: 1, -1, -1
  DRY_RUN=1                  print the command without running it
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

resolve_motion_file() {
  local input="$1"
  local stem="${input%.npz}"
  local candidate
  local candidates=()

  if [[ "$input" = /* ]]; then
    candidates+=("$input")
  else
    candidates+=(
      "$ROOT_DIR/$input"
      "$ROOT_DIR/${input}.npz"
      "$ROOT_DIR/data_demo/$input"
      "$ROOT_DIR/data_demo/${stem}.npz"
      "$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/$input"
      "$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/${stem}.npz"
    )
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      readlink -f "$candidate"
      return 0
    fi
  done

  echo "[ERROR] Motion file not found for '${input}'." >&2
  echo "        Try box_75 or an explicit /path/to/motion.npz." >&2
  return 1
}

print_command() {
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

: "${SIM2REAL_POLICY_NAME:?SIM2REAL_POLICY_NAME is required}"
: "${SIM2REAL_MODEL_REF:?SIM2REAL_MODEL_REF is required}"
: "${SIM2REAL_INFERENCE_CONFIG:?SIM2REAL_INFERENCE_CONFIG is required}"

EXTRA_ARGS=("$@")
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  echo "[INFO] Ignoring positional motion argument '${1}' for sim2real sparse-root deployment."
  shift
  EXTRA_ARGS=("$@")
fi

MOTION_FILE=""
if [[ -n "${SIM2REAL_MOTION_FILE:-}" ]]; then
  MOTION_FILE="$(resolve_motion_file "$SIM2REAL_MOTION_FILE")"
fi
MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${MODEL_REF:-${SIM2REAL_MODEL_REF}}}}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-${SIM2REAL_INFERENCE_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UNITREE_INTERFACE_RESOLVED="${UNITREE_INTERFACE:-${INTERFACE_NAME:-${INTERFACE:-}}}"
UNITREE_DOMAIN_ID="${UNITREE_DOMAIN_ID:-${DOMAIN_ID:-0}}"
PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
POLICY_RL_RATE="${POLICY_RL_RATE:-50}"
POLICY_ACTION_SCALE="${POLICY_ACTION_SCALE:-1.0}"
WANDB_DOWNLOAD_DIR="${WANDB_DOWNLOAD_DIR:-$ROOT_DIR/logs/wandb_runs/sim2real/${SIM2REAL_POLICY_NAME}}"

if [[ -z "$UNITREE_INTERFACE_RESOLVED" ]]; then
  echo "[ERROR] UNITREE_INTERFACE is required for sim2real." >&2
  echo "        Example: UNITREE_INTERFACE=eth0 bash _sim2real/${SIM2REAL_POLICY_NAME}.sh" >&2
  exit 2
fi

if [[ "$UNITREE_INTERFACE_RESOLVED" == "lo" ]] && ! is_truthy "${ALLOW_LOOPBACK_INTERFACE:-0}"; then
  echo "[ERROR] Refusing to use loopback interface 'lo' for sim2real." >&2
  echo "        Set UNITREE_INTERFACE to the robot network card, or ALLOW_LOOPBACK_INTERFACE=1 for command testing only." >&2
  exit 2
fi

export PYTHONSAFEPATH="${PYTHONSAFEPATH:-1}"
export PYTHONPATH="$ROOT_DIR/src/holosoma_inference:$ROOT_DIR/src/holosoma${PYTHONPATH:+:$PYTHONPATH}"

export HOLOSOMA_KEYBOARD_ROOT_COMMAND="${HOLOSOMA_KEYBOARD_ROOT_COMMAND:-0}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND="${HOLOSOMA_JOYSTICK_ROOT_COMMAND:-1}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_MODE="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_MODE:-manual}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_VALUE="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_VALUE:-0.2}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_XY_MAX="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_XY_MAX:-0.5}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_DEGREES="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_DEGREES:-17}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_DEADZONE="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_DEADZONE:-0.1}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_X_SIGN="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_X_SIGN:-1}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_Y_SIGN="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_Y_SIGN:--1}"
export HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_SIGN="${HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_SIGN:--1}"
export HOLOSOMA_FORCE_MOTION_ALIGNMENT="${HOLOSOMA_FORCE_MOTION_ALIGNMENT:-1}"
export HOLOSOMA_SPARSE_ROOT_COMMAND_WITHOUT_MOTION="${HOLOSOMA_SPARSE_ROOT_COMMAND_WITHOUT_MOTION:-1}"

if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ -n "${POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET="$POLICY_MOTION_INDEX_OFFSET"
  elif [[ "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-contact-aware-depth-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-mocap-distill" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1
  else
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
  fi
fi

CMD=(
  "$PYTHON_BIN" -u "$ROOT_DIR/src/holosoma_inference/holosoma_inference/run_policy.py"
  "inference:${INFERENCE_CONFIG}"
  --task.model-path "$MODEL_INPUT"
  --task.interface "$UNITREE_INTERFACE_RESOLVED"
  --task.domain-id "$UNITREE_DOMAIN_ID"
  --task.use-joystick
  --task.no-use-sim-time
  --task.no-use-sim-state
  --task.no-prefer-sim-ref-from-sim-state
  --task.no-restart-motion-on-clock-reset
  --task.no-restart-sim-on-motion-end
  --task.no-use-zmq-lowcmd
  --task.use-split-perception-obs
  --task.no-use-split-perception-obs-shm
  --task.perception-obs-port "$PERCEPTION_OBS_PORT"
  --task.no-auto-start-policy
  --task.no-auto-start-motion
  --task.no-auto-start-motion-clip
  --task.no-defer-policy-start-until-valid-state
  --task.use-root-reference-at-clip-start
  --task.policy-action-scale "$POLICY_ACTION_SCALE"
  --task.rl-rate "$POLICY_RL_RATE"
  --task.wandb-download-dir "$WANDB_DOWNLOAD_DIR"
)

if [[ -n "$MOTION_FILE" ]]; then
  CMD+=(--task.motion-file "$MOTION_FILE")
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] _sim2real policy=${SIM2REAL_POLICY_NAME}"
echo "[INFO] _sim2real model=${MODEL_INPUT}"
echo "[INFO] _sim2real inference_config=${INFERENCE_CONFIG}"
echo "[INFO] _sim2real motion=${MOTION_FILE:-none, sparse-root-only}"
echo "[INFO] _sim2real interface=${UNITREE_INTERFACE_RESOLVED} domain=${UNITREE_DOMAIN_ID}"
echo "[INFO] _sim2real perception_obs_port=${PERCEPTION_OBS_PORT}"
echo "[INFO] _sim2real joystick xy=${HOLOSOMA_JOYSTICK_ROOT_COMMAND_VALUE} xy_max=${HOLOSOMA_JOYSTICK_ROOT_COMMAND_XY_MAX} yaw_deg=${HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_DEGREES}"
print_command "${CMD[@]}"

if is_truthy "${DRY_RUN:-0}"; then
  exit 0
fi

exec "${CMD[@]}"
