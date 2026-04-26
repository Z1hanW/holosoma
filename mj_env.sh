#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-https://wandb.ai/zihanw22/boxer/runs/shoo7sr1/model_29999.onnx}"
COMMAND_WEB_LOG="${COMMAND_WEB_LOG:-${ROOT_DIR}/logs/live_debug/mj_command_web.log}"
TRACK_LAUNCHER="${MJ_TRACK_LAUNCHER:-${ROOT_DIR}/mj_box_depth_track.sh}"
POLICY_HINT_SCRIPT="${MJ_POLICY_HINT_SCRIPT:-${ROOT_DIR}/mj_policy.sh}"
# Avoid repo-root source checkouts (notably `mujoco/`) shadowing installed
# Python packages; launchers pass Holosoma sources through PYTHONPATH instead.
export PYTHONSAFEPATH="${PYTHONSAFEPATH:-1}"

usage() {
  cat <<EOF
Usage:
  bash mj_env.sh [rendered|warp] [clip_name|motion.npz] [model.onnx|wandb://...]

Examples:
  bash mj_env.sh box_74
  HEADLESS=False LAUNCH_VISER=0 bash mj_env.sh box_74
  COMMAND_WEB_PORT=7777 VISER_PORT=2984 bash mj_env.sh rendered box_74

Environment:
  MODEL_INPUT / MODEL_PATH / MODEL_REF
                                  default: ${DEFAULT_MODEL_INPUT}
  HEADLESS                        auto by default; False opens native MuJoCo GUI when available
  LAUNCH_VISER                    auto by default; enabled when HEADLESS=True or no DISPLAY
  COMMAND_WEB                     default: 1
  COMMAND_WEB_TRACK_ONLY          default: 0; use single-button S rollout-start web UI
  MJ_ENV_AUTO_LAUNCH_POLICY       default: 1; start policy in the same rollout, gated by web ]
  COMMAND_WEB_PORT                default: first free port in [4477, 4499]
  COMMAND_WEB_PORT_BASE           default: 4477
  COMMAND_WEB_PORT_MAX            default: 4499
  COMMAND_MANUAL_ENABLED          default: 0; unchecked uses motion-derived command
  COMMAND_VALUE                   default: 0.5 for W/S/A/D x/y
  COMMAND_YAW_DEGREES             default: 17 for Q/E yaw
  COMMAND_MODE                    default: manual; only used when manual mode is enabled
  SHOW_MOTION_ROBOT               default: 0; show tracked motion robot overlay in scene
  SHOW_MOTION_OBJECT              default: 0; show tracked motion object overlay in scene
  GT_MUJOCO_PHYSICS=1             force GT-style object/G1/floor MuJoCo physics
  POLICY_CONTROL_PORT             default: 5662 for web start/space/stop/init policy control
  MJ_ENV_KILL_STALE_ENV           default: 1; terminate same-port env/web/viser leftovers before launch
  MJ_ENV_KILL_STALE_POLICY        default: 1; terminate same-port policy leftovers before launch
  SIM_MOTION_INIT_MODE            default: raw_motion (first motion init pose); raw_motion_grounded keeps motion joints/object and uses training root height
  VISER_PORT                      default: 2984 when viser is launched
  HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH
                                  default: 1; select cuda:0 for far_tracking_warp when SIM_DEVICE is unset
  MJ_ENV_RECORD_ROLLOUT_VIDEO     default: 0; after launch, auto reset/start and save a debug mp4
  MJ_ENV_RECORD_DURATION          default: 12 seconds
  MJ_ENV_RECORD_OUTPUT            default: logs/live_debug/mujoco_rollout_<timestamp>.mp4
  MJ_ENV_RECORD_MUJOCO_XML         default: logs/live_debug/mujoco_rollout_<timestamp>.xml for rendered MP4
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

python_has_modules() {
  local python_bin="$1"
  shift
  "$python_bin" - "$@" <<'PY' >/dev/null 2>&1
import importlib
import sys

for module_name in sys.argv[1:]:
    importlib.import_module(module_name)
raise SystemExit(0)
PY
}

resolve_python_with_modules() {
  local modules_csv="$1"
  local modules=()
  read -r -a modules <<<"$modules_csv"
  shift
  local candidate
  for candidate in "$@"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if python_has_modules "$candidate" "${modules[@]}"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "[ERROR] No usable python found with modules: ${modules_csv}" >&2
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

find_free_port() {
  local start_port="$1"
  local end_port="$2"
  local port
  for ((port = start_port; port <= end_port; port++)); do
    if "$COMMAND_WEB_PY" - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        raise SystemExit(1)
raise SystemExit(0)
PY
    then
      printf '%s\n' "$port"
      return 0
    fi
  done
  echo "[ERROR] No free command web port in range ${start_port}-${end_port}" >&2
  exit 1
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
export HOLOSOMA_POLICY_OVERLAY_PORT="${HOLOSOMA_POLICY_OVERLAY_PORT:-$POLICY_OVERLAY_PORT}"
export HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART="${HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART:-0}"
export ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-1}"
export RUN_SECONDS="${RUN_SECONDS:-0}"
export HOLOSOMA_MJ_TRACK_RUN_FOREVER="${HOLOSOMA_MJ_TRACK_RUN_FOREVER:-1}"
MJ_ENV_AUTO_LAUNCH_POLICY="${MJ_ENV_AUTO_LAUNCH_POLICY:-1}"
if is_truthy "$MJ_ENV_AUTO_LAUNCH_POLICY"; then
  export MJ_TRACK_MODE=both
  export SKIP_POLICY=0
else
  export MJ_TRACK_MODE=env
  export SKIP_POLICY=1
fi

export HOLOSOMA_DISABLE_AUTO_RESET="${HOLOSOMA_DISABLE_AUTO_RESET:-1}"
export HOLOSOMA_DISABLE_MOTION_END_RESET="${HOLOSOMA_DISABLE_MOTION_END_RESET:-1}"
export HOLOSOMA_DISABLE_CLIP_END_RESET="${HOLOSOMA_DISABLE_CLIP_END_RESET:-1}"
export HOLOSOMA_DISABLE_BAD_TRACKING_RESET="${HOLOSOMA_DISABLE_BAD_TRACKING_RESET:-1}"
export HOLOSOMA_MUJOCO_APPLY_TRAINING_JOINT_DYNAMICS="${HOLOSOMA_MUJOCO_APPLY_TRAINING_JOINT_DYNAMICS:-1}"
export HOLOSOMA_W_OBJECT_URDF="${HOLOSOMA_W_OBJECT_URDF:-g1/g1_29dof.urdf}"
export SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-raw_motion}"
export HOLOSOMA_ZMQ_LOWCMD_KP_SCALE="${HOLOSOMA_ZMQ_LOWCMD_KP_SCALE:-1.0}"
export HOLOSOMA_ZMQ_LOWCMD_KD_SCALE="${HOLOSOMA_ZMQ_LOWCMD_KD_SCALE:-1.0}"
export HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE="${HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE:-1.0}"
export HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST="0"
export HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST="0"
export HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST="0"
export HOLOSOMA_MUJOCO_WRIST_ORIGIN_CONTACT_SPHERES="0"
export HOLOSOMA_MUJOCO_PALM_CONTACT_SPHERES="0"
export HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION="${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION:-0}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
export HOLOSOMA_MUJOCO_RESET_NOISE="0"
export PERCEPTION_CAMERA_WARP_EDGE_NOISE="False"
export PERCEPTION_CAMERA_WARP_ENABLE_HOLES="False"
export PERCEPTION_CAMERA_APPLY_SENSOR_NOISE="False"

HEADLESS_FLAG="$(resolve_headless)"
export TRAINING_HEADLESS="$HEADLESS_FLAG"
if [[ "$HEADLESS_FLAG" == "False" && "$PERCEPTION_CAMERA_SOURCE" == "rendered" && -z "${MUJOCO_GL:-}" ]]; then
  export MUJOCO_GL=glfw
fi

COMMAND_WEB="${COMMAND_WEB:-1}"
COMMAND_WEB_PORT="${COMMAND_WEB_PORT:-}"
if [[ -z "$COMMAND_WEB_PORT" ]]; then
  COMMAND_WEB_PORT_BASE="${COMMAND_WEB_PORT_BASE:-4477}"
  COMMAND_WEB_PORT_MAX="${COMMAND_WEB_PORT_MAX:-4499}"
  COMMAND_WEB_PORT="$(find_free_port "$COMMAND_WEB_PORT_BASE" "$COMMAND_WEB_PORT_MAX")"
fi
COMMAND_MANUAL_ENABLED="${COMMAND_MANUAL_ENABLED:-0}"
COMMAND_VALUE="${COMMAND_VALUE:-0.5}"
COMMAND_YAW_DEGREES="${COMMAND_YAW_DEGREES:-${COMMAND_YAW_DEG:-17}}"
COMMAND_MODE="${COMMAND_MODE:-manual}"
COMMAND_WEB_TRACK_ONLY="${COMMAND_WEB_TRACK_ONLY:-0}"
SHOW_MOTION_ROBOT="${SHOW_MOTION_ROBOT:-0}"
SHOW_MOTION_OBJECT="${SHOW_MOTION_OBJECT:-0}"
VISER_PORT_RESOLVED="${VISER_PORT:-2984}"
MJ_ENV_KILL_STALE_ENV="${MJ_ENV_KILL_STALE_ENV:-1}"
MJ_ENV_KILL_STALE_POLICY="${MJ_ENV_KILL_STALE_POLICY:-1}"
GT_MUJOCO_PHYSICS="${GT_MUJOCO_PHYSICS:-${HOLOSOMA_GT_MUJOCO_PHYSICS:-0}}"
if is_truthy "$GT_MUJOCO_PHYSICS"; then
  export GT_MUJOCO_PHYSICS=1
  export HOLOSOMA_GT_MUJOCO_PHYSICS=1
  export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS=0
else
  export GT_MUJOCO_PHYSICS=0
  export HOLOSOMA_GT_MUJOCO_PHYSICS=0
fi
COMMAND_WEB_PID=""

same_port_env_pids() {
  ps -eww -o pid=,args= | awk \
    -v self="$$" \
    -v root="${ROOT_DIR}" \
    -v state="${SIM_STATE_PORT}" \
    -v control="${SIM_CONTROL_PORT}" \
    -v sparse="${SPARSE_ROOT_COMMAND_PORT}" \
    -v policy="${POLICY_CONTROL_PORT}" \
    -v cmdweb="${COMMAND_WEB_PORT}" \
    -v viser="${VISER_PORT_RESOLVED}" '
      $1 == self { next }
      {
        is_python = index($0, "python") || index($0, "python3")
        is_sim = index($0, root "/src/holosoma/holosoma/run_sim.py")
        is_web = index($0, root "/src/holosoma/holosoma/mj_command_web.py") || index($0, root "/src/holosoma/holosoma/mj_track_trigger_web.py")
        is_viser = index($0, root "/src/holosoma/holosoma/viser_mujoco_sim_state.py")
        web_port_match = index($0, "--port " cmdweb) || index($0, "--sparse-root-command-port " sparse) || index($0, "--control-port " control) || index($0, "--policy-control-port " policy)
        viser_port_match = index($0, "--port " viser) || index($0, "--state-port " state) || index($0, "--control-port " control) || index($0, "--sparse-root-command-port " sparse)
        if (is_python && is_sim && index($0, "--simulator.config.bridge.sim-state-port " state) && index($0, "--simulator.config.bridge.control-port " control)) {
          print $1
        }
        if (is_python && is_web && web_port_match) {
          print $1
        }
        if (is_python && is_viser && viser_port_match) {
          print $1
        }
      }
    ' | sort -n -u
}

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

if ! is_truthy "${DRY_RUN:-0}"; then
  STALE_ENV_PIDS="$(same_port_env_pids || true)"
  if [[ -n "$STALE_ENV_PIDS" ]]; then
    STALE_ENV_PIDS_ONE_LINE="$(format_pids "$STALE_ENV_PIDS")"
    if is_truthy "$MJ_ENV_KILL_STALE_ENV"; then
      echo "[WARN] terminating stale env/web/viser process(es) on matching ports: ${STALE_ENV_PIDS_ONE_LINE}" >&2
      terminate_pids "$STALE_ENV_PIDS"
    else
      echo "[ERROR] stale env/web/viser process(es) already target matching ports: ${STALE_ENV_PIDS_ONE_LINE}" >&2
      echo "[ERROR] Stop them first, or set MJ_ENV_KILL_STALE_ENV=1." >&2
      exit 1
    fi
  fi

  STALE_POLICY_PIDS="$(same_port_policy_pids || true)"
  if [[ -n "$STALE_POLICY_PIDS" ]]; then
    STALE_POLICY_PIDS_ONE_LINE="$(format_pids "$STALE_POLICY_PIDS")"
    if is_truthy "$MJ_ENV_KILL_STALE_POLICY"; then
      echo "[WARN] terminating stale policy process(es) on state=${SIM_STATE_PORT} control=${SIM_CONTROL_PORT}: ${STALE_POLICY_PIDS_ONE_LINE}" >&2
      terminate_pids "$STALE_POLICY_PIDS"
    else
      echo "[ERROR] stale policy process(es) already target state=${SIM_STATE_PORT} control=${SIM_CONTROL_PORT}: ${STALE_POLICY_PIDS_ONE_LINE}" >&2
      echo "[ERROR] Stop them first, or set MJ_ENV_KILL_STALE_POLICY=1." >&2
      exit 1
    fi
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
  COMMAND_WEB_SCRIPT="${ROOT_DIR}/src/holosoma/holosoma/mj_command_web.py"
  COMMAND_WEB_ARGS=(
    --port "$COMMAND_WEB_PORT"
    --sparse-root-command-port "$SPARSE_ROOT_COMMAND_PORT"
    --control-port "$SIM_CONTROL_PORT"
    --policy-control-port "$POLICY_CONTROL_PORT"
    --policy-overlay-port "$HOLOSOMA_POLICY_OVERLAY_PORT"
    --scene-proxy-url "http://127.0.0.1:${VISER_PORT_RESOLVED}"
  )
  if is_truthy "$COMMAND_WEB_TRACK_ONLY"; then
    COMMAND_WEB_SCRIPT="${ROOT_DIR}/src/holosoma/holosoma/mj_track_trigger_web.py"
  else
    COMMAND_WEB_ARGS+=(
      --value "$COMMAND_VALUE"
      --yaw-degrees "$COMMAND_YAW_DEGREES"
      --mode "$COMMAND_MODE"
      "${COMMAND_WEB_ENABLED_ARG[@]}"
    )
  fi
  PYTHONPATH="${ROOT_DIR}/src/holosoma:${ROOT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}" \
    "$COMMAND_WEB_PY" -u "$COMMAND_WEB_SCRIPT" \
      "${COMMAND_WEB_ARGS[@]}" \
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

if [[ "$LAUNCH_VISER_RESOLVED" == "1" && -z "${PYTHON_BIN:-}" ]] && ! is_truthy "${DRY_RUN:-0}"; then
  PYTHON_BIN="$(resolve_python_with_modules "holosoma.viser_mujoco_sim_state" \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)")"
  export PYTHON_BIN
fi

if is_truthy "$MJ_ENV_AUTO_LAUNCH_POLICY"; then
  echo "[INFO] launching MuJoCo environment + policy"
else
  echo "[INFO] launching MuJoCo environment only"
fi
echo "[INFO] model=${MODEL_INPUT}"
echo "[INFO] headless=${TRAINING_HEADLESS} launch_viser=${LAUNCH_VISER_RESOLVED}"
echo "[INFO] robot_urdf=${HOLOSOMA_W_OBJECT_URDF}"
echo "[INFO] motion init=${SIM_MOTION_INIT_MODE}"
echo "[INFO] gt_mujoco_physics=${GT_MUJOCO_PHYSICS} zero_passive_dynamics=${HOLOSOMA_GT_MUJOCO_ZERO_PASSIVE_DYNAMICS:-0}"
echo "[INFO] lowcmd scales kp=${HOLOSOMA_ZMQ_LOWCMD_KP_SCALE} kd=${HOLOSOMA_ZMQ_LOWCMD_KD_SCALE} torque_limit=${HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE}"
echo "[INFO] disabled assists root=${HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST} dof=${HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST} object=${HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST}; contact_spheres wrist=${HOLOSOMA_MUJOCO_WRIST_ORIGIN_CONTACT_SPHERES} palm=${HOLOSOMA_MUJOCO_PALM_CONTACT_SPHERES}; reset_noise=${HOLOSOMA_MUJOCO_RESET_NOISE}"
echo "[INFO] rubber_hand_contacts=${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS} keep_reference_hand_collision=${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION}"
echo "[INFO] auto_launch_policy=${MJ_ENV_AUTO_LAUNCH_POLICY} mj_track_mode=${MJ_TRACK_MODE} skip_policy=${SKIP_POLICY}"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT} sparse_root=${SPARSE_ROOT_COMMAND_PORT} policy_control=${POLICY_CONTROL_PORT} policy_overlay=${HOLOSOMA_POLICY_OVERLAY_PORT}"
if is_truthy "$COMMAND_WEB"; then
  echo "[INFO] command+scene web: http://localhost:${COMMAND_WEB_PORT} (log: ${COMMAND_WEB_LOG})"
  if is_truthy "$COMMAND_WEB_TRACK_ONLY"; then
    echo "[INFO] web policy control: S starts rollout (Space + ]), R/Backspace resets rollout via control=${SIM_CONTROL_PORT}, policy_control=${POLICY_CONTROL_PORT}"
  else
    echo "[INFO] web policy controls: ] start policy, Space start motion clip, Stop, Init via policy_control=${POLICY_CONTROL_PORT}"
    echo "[INFO] command source: motion-derived by default; manual_mode_initial=${COMMAND_MANUAL_ENABLED}"
    echo "[INFO] manual command scale: xy=${COMMAND_VALUE}, yaw=${COMMAND_YAW_DEGREES} deg"
  fi
fi
if is_truthy "$MJ_ENV_AUTO_LAUNCH_POLICY"; then
  echo "[INFO] policy is launched by this rollout and waits for web ] before sending lowcmd."
else
  echo "[INFO] launch policy with: bash ${POLICY_HINT_SCRIPT} $*"
fi
if [[ "$LAUNCH_VISER_RESOLVED" == "1" ]] && ! is_truthy "$COMMAND_WEB"; then
  echo "[INFO] raw viser scene: http://localhost:${VISER_PORT_RESOLVED}"
fi

if is_truthy "${DRY_RUN:-0}"; then
  echo "[INFO] DRY_RUN=1; not launching MuJoCo."
  exit 0
fi

LAUNCH_CMD=()
if [[ "$LAUNCH_VISER_RESOLVED" == "1" ]]; then
  if is_truthy "$COMMAND_WEB"; then
    export HOLOSOMA_VISER_ANNOUNCE_URL="http://localhost:${COMMAND_WEB_PORT}"
    export HOLOSOMA_VISER_SUPPRESS_DIRECT_URL=1
  else
    unset HOLOSOMA_VISER_ANNOUNCE_URL
    unset HOLOSOMA_VISER_SUPPRESS_DIRECT_URL
  fi
  VISER_ARGS=(
    --manual-root-mode "$COMMAND_MODE"
    --state-port "$SIM_STATE_PORT"
    --perception-obs-port "$PERCEPTION_OBS_PORT"
    --perception-obs-shm-name "${PERCEPTION_OBS_SHM_NAME:-depth_img_shm}"
    --control-port "$SIM_CONTROL_PORT"
    --sparse-root-command-port "$SPARSE_ROOT_COMMAND_PORT"
    --policy-overlay-port "$HOLOSOMA_POLICY_OVERLAY_PORT"
  )
  if ! is_truthy "$MJ_ENV_AUTO_LAUNCH_POLICY"; then
    VISER_ARGS+=(--launch-env-only)
  fi
  if is_truthy "$SHOW_MOTION_ROBOT"; then
    VISER_ARGS+=(--show-motion-robot)
  fi
  if is_truthy "$SHOW_MOTION_OBJECT"; then
    VISER_ARGS+=(--show-motion-object)
  fi
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
  LAUNCH_CMD=(bash "${TRACK_LAUNCHER}" "$@" "${VISER_ARGS[@]}")
else
  export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1
  LAUNCH_CMD=(bash "${TRACK_LAUNCHER}" "$@")
fi

if is_truthy "${MJ_ENV_RECORD_ROLLOUT_VIDEO:-0}"; then
  if ! is_truthy "$COMMAND_WEB"; then
    echo "[ERROR] MJ_ENV_RECORD_ROLLOUT_VIDEO=1 requires COMMAND_WEB=1 for reset/start control." >&2
    exit 2
  fi
  RECORD_PY="$(resolve_python_with_modules "cv2 numpy zmq loguru mujoco" \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)")"
  RECORD_STAMP="$(date +%Y%m%d_%H%M%S)"
  RECORD_OUTPUT="${MJ_ENV_RECORD_OUTPUT:-${ROOT_DIR}/logs/live_debug/mujoco_rollout_$(date +%Y%m%d_%H%M%S).mp4}"
  RECORD_MUJOCO_XML="${HOLOSOMA_MUJOCO_EXPORT_XML_PATH:-${MJ_ENV_RECORD_MUJOCO_XML:-${ROOT_DIR}/logs/live_debug/mujoco_rollout_${RECORD_STAMP}.xml}}"
  export HOLOSOMA_MUJOCO_EXPORT_XML_PATH="$RECORD_MUJOCO_XML"
  RECORD_DURATION="${MJ_ENV_RECORD_DURATION:-12}"
  RECORD_SIM_DURATION="${MJ_ENV_RECORD_SIM_DURATION:-}"
  RECORD_FPS="${MJ_ENV_RECORD_FPS:-30}"
  echo "[INFO] auto rollout recording enabled: output=${RECORD_OUTPUT} xml=${RECORD_MUJOCO_XML} duration=${RECORD_DURATION}s sim_duration=${RECORD_SIM_DURATION:-wall-only}s fps=${RECORD_FPS}"
  "${LAUNCH_CMD[@]}" &
  rollout_pid=$!
  trap 'kill "$rollout_pid" 2>/dev/null || true' EXIT
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${COMMAND_WEB_PORT}/state" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  for _ in $(seq 1 60); do
    if [[ -s "$RECORD_MUJOCO_XML" ]]; then
      break
    fi
    sleep 1
  done
  "$RECORD_PY" "${ROOT_DIR}/src/holosoma/holosoma/record_mujoco_rollout_video.py" \
    --web-port "$COMMAND_WEB_PORT" \
    --state-port "$SIM_STATE_PORT" \
    --depth-shm-name "${PERCEPTION_OBS_SHM_NAME:-depth_img_shm}" \
    --duration "$RECORD_DURATION" \
    $( [[ -n "$RECORD_SIM_DURATION" ]] && printf '%s %s' "--sim-duration" "$RECORD_SIM_DURATION" ) \
    --fps "$RECORD_FPS" \
    --output "$RECORD_OUTPUT" \
    --mujoco-xml "$RECORD_MUJOCO_XML" \
    $( [[ "${MJ_ENV_RECORD_NO_AUTO_START:-0}" == "1" ]] && printf '%s' "--no-auto-start" ) || echo "[WARN] rollout video recording failed" >&2
  echo "[INFO] rollout recording finished: ${RECORD_OUTPUT}"
  wait "$rollout_pid"
else
  "${LAUNCH_CMD[@]}"
fi
