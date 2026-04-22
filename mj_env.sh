#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_INPUT="${ROOT_DIR}/logs/wandb_runs/shoo7sr1/model_18500.onnx"
COMMAND_WEB_LOG="${COMMAND_WEB_LOG:-${ROOT_DIR}/logs/live_debug/mj_command_web.log}"

usage() {
  cat <<EOF
Usage:
  bash mj_env.sh [rendered|warp] [clip_name|motion.npz] [model.onnx|wandb://...]

Examples:
  bash mj_env.sh box_74
  HEADLESS=False LAUNCH_VISER=0 bash mj_env.sh box_74
  COMMAND_WEB_PORT=8080 VISER_PORT=2984 bash mj_env.sh rendered box_74

Environment:
  MODEL_INPUT / MODEL_PATH        default: ${DEFAULT_MODEL_INPUT}
  HEADLESS                        auto by default; False opens native MuJoCo GUI when available
  LAUNCH_VISER                    auto by default; enabled when HEADLESS=True or no DISPLAY
  COMMAND_WEB                     default: 1
  COMMAND_WEB_PORT                default: 8080
  COMMAND_MANUAL_ENABLED          default: 0; unchecked uses motion-derived command
  COMMAND_VALUE                   default: 0.5 for W/S/A/D x/y
  COMMAND_YAW_DEGREES             default: 17 for Q/E yaw
  COMMAND_MODE                    default: manual when manual mode is enabled
  POLICY_CONTROL_PORT             default: 5662 for web start/stop/init policy control
  MJ_ENV_KILL_STALE_POLICY        default: 0; set 1 to terminate same-port policy leftovers before launch
  SIM_MOTION_INIT_MODE            default: raw_motion (first motion init pose)
  VISER_PORT                      default: 2984 when viser is launched
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

normalize_bool_flag() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      printf 'True\n'
      ;;
    0|false|no|off)
      printf 'False\n'
      ;;
    *)
      echo "[ERROR] Expected boolean True/False/1/0, got: $1" >&2
      exit 2
      ;;
  esac
}

resolve_python() {
  local configured="$1"
  shift
  if [[ -n "$configured" && -x "$configured" ]]; then
    printf '%s\n' "$configured"
    return
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "[ERROR] No usable python3 found." >&2
  exit 1
}

resolve_headless() {
  local raw="${HEADLESS:-${headless:-auto}}"
  if [[ -n "${TRAINING_HEADLESS+x}" ]]; then
    raw="${TRAINING_HEADLESS}"
  fi
  case "$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')" in
    auto|"")
      if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
        printf 'False\n'
      else
        printf 'True\n'
      fi
      ;;
    *)
      normalize_bool_flag "$raw"
      ;;
  esac
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

INFER_PY="$(resolve_python "${INFER_PY:-}" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"
COMMAND_WEB_PY="$(resolve_python "${COMMAND_WEB_PY:-}" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python)"

export MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${DEFAULT_MODEL_INPUT}}}"
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-rendered}"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
export PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
export SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
export SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-5661}"
export POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-5662}"
export ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-1}"
export RUN_SECONDS="${RUN_SECONDS:-0}"
export HOLOSOMA_MJ_TRACK_RUN_FOREVER="${HOLOSOMA_MJ_TRACK_RUN_FOREVER:-1}"
export MJ_TRACK_MODE=env
export SKIP_POLICY=1

export HOLOSOMA_DISABLE_AUTO_RESET="${HOLOSOMA_DISABLE_AUTO_RESET:-1}"
export HOLOSOMA_DISABLE_MOTION_END_RESET="${HOLOSOMA_DISABLE_MOTION_END_RESET:-1}"
export HOLOSOMA_DISABLE_CLIP_END_RESET="${HOLOSOMA_DISABLE_CLIP_END_RESET:-1}"
export HOLOSOMA_DISABLE_BAD_TRACKING_RESET="${HOLOSOMA_DISABLE_BAD_TRACKING_RESET:-1}"
export SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-raw_motion}"

HEADLESS_FLAG="$(resolve_headless)"
export TRAINING_HEADLESS="$HEADLESS_FLAG"
if [[ "$HEADLESS_FLAG" == "False" && "$PERCEPTION_CAMERA_SOURCE" == "rendered" && -z "${MUJOCO_GL:-}" ]]; then
  export MUJOCO_GL=glfw
fi

COMMAND_WEB="${COMMAND_WEB:-1}"
COMMAND_WEB_PORT="${COMMAND_WEB_PORT:-8080}"
COMMAND_MANUAL_ENABLED="${COMMAND_MANUAL_ENABLED:-0}"
COMMAND_VALUE="${COMMAND_VALUE:-0.5}"
COMMAND_YAW_DEGREES="${COMMAND_YAW_DEGREES:-${COMMAND_YAW_DEG:-17}}"
COMMAND_MODE="${COMMAND_MODE:-manual}"
VISER_PORT_RESOLVED="${VISER_PORT:-2984}"
MJ_ENV_KILL_STALE_POLICY="${MJ_ENV_KILL_STALE_POLICY:-0}"
COMMAND_WEB_PID=""

same_port_policy_pids() {
  ps -eo pid=,args= | awk \
    -v root="${ROOT_DIR}" \
    -v state="${SIM_STATE_PORT}" \
    -v control="${SIM_CONTROL_PORT}" '
      index($0, root "/src/holosoma_inference/holosoma_inference/run_policy.py") &&
      index($0, "--task.sim-state-port " state) &&
      index($0, "--task.sim-control-port " control) {
        print $1
      }
    '
}

STALE_POLICY_PIDS="$(same_port_policy_pids || true)"
if [[ -n "$STALE_POLICY_PIDS" ]]; then
  if is_truthy "$MJ_ENV_KILL_STALE_POLICY"; then
    echo "[WARN] terminating stale policy process(es) on state=${SIM_STATE_PORT} control=${SIM_CONTROL_PORT}: ${STALE_POLICY_PIDS}" >&2
    # shellcheck disable=SC2086
    kill -TERM $STALE_POLICY_PIDS 2>/dev/null || true
    sleep 1
    for pid in $STALE_POLICY_PIDS; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  else
    echo "[ERROR] stale policy process(es) already target state=${SIM_STATE_PORT} control=${SIM_CONTROL_PORT}: ${STALE_POLICY_PIDS}" >&2
    echo "[ERROR] Stop them first, or set MJ_ENV_KILL_STALE_POLICY=1 to terminate them before launching env." >&2
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${COMMAND_WEB_PID:-}" ]] && kill -0 "$COMMAND_WEB_PID" 2>/dev/null; then
    kill "$COMMAND_WEB_PID" 2>/dev/null || true
    wait "$COMMAND_WEB_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if is_truthy "$COMMAND_WEB" && ! is_truthy "${DRY_RUN:-0}"; then
  mkdir -p "$(dirname "$COMMAND_WEB_LOG")"
  : >"$COMMAND_WEB_LOG"
  COMMAND_WEB_ENABLED_ARG=(--no-enabled)
  if is_truthy "$COMMAND_MANUAL_ENABLED"; then
    COMMAND_WEB_ENABLED_ARG=(--enabled)
  fi
  PYTHONPATH="${ROOT_DIR}/src/holosoma:${ROOT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}" \
    "$COMMAND_WEB_PY" -u "$ROOT_DIR/src/holosoma/holosoma/mj_command_web.py" \
      --port "$COMMAND_WEB_PORT" \
      --sparse-root-command-port "$SPARSE_ROOT_COMMAND_PORT" \
      --control-port "$SIM_CONTROL_PORT" \
      --policy-control-port "$POLICY_CONTROL_PORT" \
      --value "$COMMAND_VALUE" \
      --yaw-degrees "$COMMAND_YAW_DEGREES" \
      --mode "$COMMAND_MODE" \
      "${COMMAND_WEB_ENABLED_ARG[@]}" \
      --scene-proxy-url "http://127.0.0.1:${VISER_PORT_RESOLVED}" \
      >"$COMMAND_WEB_LOG" 2>&1 &
  COMMAND_WEB_PID=$!
  sleep 0.4
  if ! kill -0 "$COMMAND_WEB_PID" 2>/dev/null; then
    echo "[ERROR] command web failed to start. See $COMMAND_WEB_LOG" >&2
    tail -n 40 "$COMMAND_WEB_LOG" >&2 || true
    exit 1
  fi
fi

LAUNCH_VISER_RAW="${LAUNCH_VISER:-auto}"
case "$(printf '%s' "$LAUNCH_VISER_RAW" | tr '[:upper:]' '[:lower:]')" in
  auto|"")
    if [[ "$HEADLESS_FLAG" == "True" || -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
      LAUNCH_VISER_RESOLVED=1
    else
      LAUNCH_VISER_RESOLVED=0
    fi
    ;;
  *)
    if is_truthy "$LAUNCH_VISER_RAW"; then
      LAUNCH_VISER_RESOLVED=1
    else
      LAUNCH_VISER_RESOLVED=0
    fi
    ;;
esac

echo "[INFO] launching MuJoCo environment only"
echo "[INFO] model=${MODEL_INPUT}"
echo "[INFO] headless=${TRAINING_HEADLESS} launch_viser=${LAUNCH_VISER_RESOLVED}"
echo "[INFO] motion init=${SIM_MOTION_INIT_MODE}"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT} sparse_root=${SPARSE_ROOT_COMMAND_PORT} policy_control=${POLICY_CONTROL_PORT}"
if is_truthy "$COMMAND_WEB"; then
  echo "[INFO] command+scene web: http://localhost:${COMMAND_WEB_PORT} (log: ${COMMAND_WEB_LOG})"
  echo "[INFO] web policy controls: Start/Stop/Init via policy_control=${POLICY_CONTROL_PORT}"
  echo "[INFO] command source: motion-derived by default; manual_mode_initial=${COMMAND_MANUAL_ENABLED}"
  echo "[INFO] manual command scale: xy=${COMMAND_VALUE}, yaw=${COMMAND_YAW_DEGREES} deg"
fi
if [[ "$LAUNCH_VISER_RESOLVED" == "1" ]]; then
  echo "[INFO] raw viser scene: http://localhost:${VISER_PORT_RESOLVED}"
fi
echo "[INFO] launch policy with: bash ${ROOT_DIR}/mj_policy.sh $*"

if is_truthy "${DRY_RUN:-0}"; then
  echo "[INFO] DRY_RUN=1; not launching MuJoCo."
  exit 0
fi

if [[ "$LAUNCH_VISER_RESOLVED" == "1" ]]; then
  VISER_ARGS=(
    --launch-env-only
    --manual-root-mode "$COMMAND_MODE"
  )
  if is_truthy "$COMMAND_WEB"; then
    VISER_ARGS+=(--no-manual-root-publisher-enabled)
  else
    VISER_ARGS+=(--manual-root-publisher-enabled)
  fi
  if [[ "$TRAINING_HEADLESS" == "True" ]]; then
    VISER_ARGS+=(--training-headless)
  else
    VISER_ARGS+=(--no-training-headless)
  fi
  if [[ -n "${VISER_PORT_RESOLVED}" ]]; then
    VISER_ARGS+=(--port "${VISER_PORT_RESOLVED}")
  fi
  bash "${ROOT_DIR}/mj_box_depth_track.sh" "$@" "${VISER_ARGS[@]}"
else
  export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1
  bash "${ROOT_DIR}/mj_box_depth_track.sh" "$@"
fi
