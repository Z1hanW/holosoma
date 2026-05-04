#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-https://wandb.ai/zihanw22/boxer/runs/w5qostjn}"
COMMAND_WEB_LOG="${COMMAND_WEB_LOG:-${ROOT_DIR}/logs/live_debug/mj_command_web.log}"
TRACK_LAUNCHER="${MJ_TRACK_LAUNCHER:-${ROOT_DIR}/mj_box_depth_track.sh}"
POLICY_HINT_SCRIPT="${MJ_POLICY_HINT_SCRIPT:-${ROOT_DIR}/mj_policy.sh}"
# Avoid repo-root source checkouts (notably `mujoco/`) shadowing installed
# Python packages; launchers pass Holosoma sources through PYTHONPATH instead.
export PYTHONSAFEPATH="${PYTHONSAFEPATH:-1}"
export PYTHONPATH="${ROOT_DIR}/src/holosoma:${ROOT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<EOF
Usage:
  bash mj_env.sh [rendered|rendered848|warp] [clip_name|motion.npz] [model.onnx|wandb://...]

Examples:
  bash mj_env.sh box_75
  HEADLESS=False LAUNCH_VISER=0 bash mj_env.sh box_75
  COMMAND_WEB_PORT=7777 VISER_PORT=2984 bash mj_env.sh rendered box_75
  COMMAND_WEB_PORT=7777 VISER_PORT=2984 bash mj_env_mujoco_render_848.sh box_75

Environment:
  MODEL_INPUT / MODEL_PATH / MODEL_REF
                                  default: ${DEFAULT_MODEL_INPUT}
  HEADLESS                        auto by default; False opens native MuJoCo GUI when available
  LAUNCH_VISER                    auto by default; enabled when HEADLESS=True or no DISPLAY
  COMMAND_WEB                     default: 0; native MuJoCo + terminal controls
  COMMAND_WEB_TRACK_ONLY          default: 0; use single-button S rollout-start web UI
  MJ_ENV_AUTO_LAUNCH_POLICY       default: 1; start policy in the same rollout
  POLICY_STDIO                    default: inherit; policy receives terminal keys
  HOLOSOMA_KEYBOARD_ROOT_COMMAND  default: 1; W/S/A/D/Q/E sparse-root terminal commands
  HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE
                                  default: hold; set latch to tap W once and hold x=0.2
  COMMAND_WEB_PORT                default: first free port in [4477, 4499]
  COMMAND_WEB_PORT_BASE           default: 4477
  COMMAND_WEB_PORT_MAX            default: 4499
  COMMAND_MANUAL_ENABLED          default: 0; unchecked uses motion-derived command
  COMMAND_VALUE                   default: 0.2 for W/S/A/D x/y
  COMMAND_YAW_DEGREES             default: 17 for Q/E yaw
  COMMAND_MODE                    default: manual; only used when manual command is enabled
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

first_motion_clip_arg() {
  local positional_mode=1
  local arg
  for arg in "$@"; do
    if [[ "$positional_mode" == "0" ]]; then
      break
    fi
    case "$arg" in
      depth|rendered|render|mujoco|rendered848|render848|mujoco848|mujoco_render_848x480|warp|far_tracking_warp)
        ;;
      wandb://*|https://*|*.onnx|*.pt)
        ;;
      --*)
        positional_mode=0
        ;;
      *.npz)
        if [[ "$arg" != */* ]]; then
          printf '%s\n' "${arg%.npz}"
          return 0
        fi
        ;;
      *)
        printf '%s\n' "${arg%.npz}"
        return 0
        ;;
    esac
  done
  return 1
}

first_model_arg() {
  local positional_mode=1
  local arg
  for arg in "$@"; do
    if [[ "$positional_mode" == "0" ]]; then
      break
    fi
    case "$arg" in
      wandb://*|https://*|*.onnx|*.pt)
        printf '%s\n' "$arg"
        return 0
        ;;
      --*)
        positional_mode=0
        ;;
    esac
  done
  return 1
}

apply_model_perception_env_defaults() {
  local model_ref="$1"
  [[ -n "$model_ref" ]] || return 0

  local override_lines
  if ! override_lines="$(
    "$INFER_PY" - <<'PY' "$model_ref" "$ROOT_DIR/logs/wandb_runs"
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import urlparse

import onnx
from holosoma_inference.utils.wandb import load_checkpoint


def download_dir_for_ref(ref: str, root: Path) -> Path:
    download_dir = root / "box_depth"
    if ref.startswith("wandb://"):
        parts = ref[len("wandb://") :].split("/", 3)
        if len(parts) >= 3:
            download_dir = root / parts[2]
    elif ref.startswith("https://"):
        parts = [part for part in urlparse(ref).path.split("/") if part]
        if len(parts) >= 4 and parts[2] == "runs":
            download_dir = root / parts[3]
    return download_dir


def resolve_model(ref: str, root: Path) -> Path:
    if ref.startswith("wandb://") and ref.endswith(".pt"):
        ref = f"{ref[:-3]}.onnx"
    local = Path(ref).expanduser()
    if local.is_file():
        path = local.resolve()
    else:
        with redirect_stdout(sys.stderr):
            path = Path(load_checkpoint(None, ref, str(download_dir_for_ref(ref, root)))).expanduser().resolve()
    if path.suffix == ".pt":
        sibling_onnx = path.with_suffix(".onnx")
        if sibling_onnx.is_file():
            path = sibling_onnx
    return path


model = onnx.load(resolve_model(sys.argv[1], Path(sys.argv[2])))
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

perception_cfg = metadata.get("experiment_config", {}).get("perception", {})
if not isinstance(perception_cfg, dict):
    raise SystemExit(0)

field_map = {
    "update_hz": "PERCEPTION_UPDATE_HZ",
    "camera_fps": "PERCEPTION_CAMERA_FPS",
    "camera_width": "PERCEPTION_CAMERA_WIDTH",
    "camera_height": "PERCEPTION_CAMERA_HEIGHT",
    "camera_pitch_deg": "PERCEPTION_CAMERA_PITCH_DEG",
    "camera_vfov_deg": "PERCEPTION_CAMERA_VFOV_DEG",
    "camera_hfov_deg": "PERCEPTION_CAMERA_HFOV_DEG",
    "camera_include_robot_mesh": "PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH",
    "camera_near": "PERCEPTION_CAMERA_NEAR",
    "camera_far": "PERCEPTION_CAMERA_FAR",
    "max_distance": "PERCEPTION_MAX_DISTANCE",
    "camera_warp_crop_top": "PERCEPTION_CAMERA_WARP_CROP_TOP",
    "camera_warp_crop_bottom": "PERCEPTION_CAMERA_WARP_CROP_BOTTOM",
    "camera_warp_crop_left": "PERCEPTION_CAMERA_WARP_CROP_LEFT",
    "camera_warp_crop_right": "PERCEPTION_CAMERA_WARP_CROP_RIGHT",
    "camera_warp_min_valid_depth": "PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH",
    "camera_warp_buffer_len": "PERCEPTION_CAMERA_WARP_BUFFER_LEN",
    "camera_warp_latency_frame": "PERCEPTION_CAMERA_WARP_LATENCY_FRAME",
    "camera_warp_edge_noise": "PERCEPTION_CAMERA_WARP_EDGE_NOISE",
    "camera_warp_edge_border": "PERCEPTION_CAMERA_WARP_EDGE_BORDER",
    "camera_warp_edge_shuffle_prob": "PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB",
    "camera_warp_edge_empty_prob": "PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB",
    "camera_warp_edge_thresh_primary": "PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY",
    "camera_warp_edge_thresh_secondary": "PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY",
    "camera_warp_edge_far_depth_thresh": "PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH",
    "camera_warp_enable_holes": "PERCEPTION_CAMERA_WARP_ENABLE_HOLES",
    "camera_warp_hole_prob": "PERCEPTION_CAMERA_WARP_HOLE_PROB",
    "camera_apply_sensor_noise": "PERCEPTION_CAMERA_APPLY_SENSOR_NOISE",
}

for src_key, env_key in field_map.items():
    value = perception_cfg.get(src_key)
    if value is None:
        continue
    if isinstance(value, bool):
        print(f"{env_key}={value}")
    elif isinstance(value, int):
        print(f"{env_key}={value}")
    elif isinstance(value, float):
        print(f"{env_key}={value:g}")
    else:
        print(f"{env_key}={value}")

noise_requested = any(bool(perception_cfg.get(key)) for key in (
    "camera_warp_edge_noise",
    "camera_warp_enable_holes",
    "camera_apply_sensor_noise",
))
print(f"HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE={noise_requested}")

randomization_cfg = metadata.get("experiment_config", {}).get("randomization", {})
if isinstance(randomization_cfg, dict):
    reset_terms = randomization_cfg.get("reset_terms", {})
    if isinstance(reset_terms, dict):
        camera_randomizer = reset_terms.get("randomize_camera_raycast", {})
        if isinstance(camera_randomizer, dict):
            params = camera_randomizer.get("params", {})
            if isinstance(params, dict) and params.get("enabled") is not None:
                print(f"HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT={bool(params.get('enabled'))}")
PY
  )"; then
    echo "[WARN] could not read ONNX perception metadata for depth preview; using launch defaults." >&2
    return 0
  fi

  local key value
  while IFS='=' read -r key value; do
    [[ -n "${key:-}" ]] || continue
    case "$key" in
      PERCEPTION_*|HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE|HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT)
        if [[ -z "${!key:-}" ]]; then
          printf -v "$key" '%s' "$value"
          export "$key"
        fi
        ;;
    esac
  done <<< "$override_lines"
}

maybe_use_native_data_demo_motion_dir() {
  [[ -z "${MOTION_DIR:-}" && -z "${MOTION_FILE:-}" ]] || return 0

  local demo_dir="$ROOT_DIR/data_demo"
  local clip_name=""
  clip_name="$(first_motion_clip_arg "$@" || true)"
  [[ -n "$clip_name" ]] || return 0
  [[ -f "$demo_dir/${clip_name}.npz" ]] || return 0

  export MOTION_DIR="$demo_dir"
  if [[ -z "${OBJECT_URDF:-}" && -f "$demo_dir/_clip_object_urdf_map.json" ]]; then
    export OBJECT_URDF="$demo_dir/_clip_object_urdf_map.json"
  fi
  echo "[INFO] using data_demo motion assets for ${clip_name}: ${MOTION_DIR}"
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ $# -eq 0 ]]; then
  set -- box_75
else
  case "${1:-}" in
    wandb://*|https://*|*.onnx|*.pt)
      set -- box_75 "$@"
      ;;
  esac
fi

INFER_PY="$(resolve_python "${INFER_PY:-}" \
  /home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/sim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"
export INFER_PY
COMMAND_WEB_PY="$(resolve_python "${COMMAND_WEB_PY:-}" \
  /home/user/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/sim/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python)"
DEPTH_PREVIEW_PY="$(resolve_python "${DEPTH_PREVIEW_PY:-}" \
  /home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/user/.holosoma_deps/miniconda3/envs/sim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  "$(command -v python3 2>/dev/null || true)" \
  "$(command -v python 2>/dev/null || true)")"

MODEL_ARG_INPUT="$(first_model_arg "$@" || true)"
export MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${MODEL_REF:-${DEFAULT_MODEL_INPUT}}}}"
if [[ -n "$MODEL_ARG_INPUT" ]]; then
  export MODEL_INPUT="$MODEL_ARG_INPUT"
fi
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-far_tracking_warp}"
export PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
export HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE="${HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE:-1}"
apply_model_perception_env_defaults "$MODEL_INPUT"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
export PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
export SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
export SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-5661}"
export POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-5662}"
export POLICY_OVERLAY_PORT="${POLICY_OVERLAY_PORT:-5663}"
export HOLOSOMA_POLICY_OVERLAY_PORT="${HOLOSOMA_POLICY_OVERLAY_PORT:-$POLICY_OVERLAY_PORT}"
export PERCEPTION_OBS_SHM_NAME="${PERCEPTION_OBS_SHM_NAME:-depth_img_shm_${SIM_STATE_PORT}}"
export HOLOSOMA_POLICY_PERCEPTION_OBS_SHM_NAME="${HOLOSOMA_POLICY_PERCEPTION_OBS_SHM_NAME:-policy_${PERCEPTION_OBS_SHM_NAME}}"
export HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART="${HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART:-0}"
export ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-0}"
export RUN_SECONDS="${RUN_SECONDS:-0}"
if [[ -z "${HOLOSOMA_MJ_TRACK_RUN_FOREVER+x}" && "$RUN_SECONDS" == "0" ]]; then
  export HOLOSOMA_MJ_TRACK_RUN_FOREVER=1
fi
MJ_ENV_AUTO_LAUNCH_POLICY="${MJ_ENV_AUTO_LAUNCH_POLICY:-1}"
if is_truthy "$MJ_ENV_AUTO_LAUNCH_POLICY"; then
  export MJ_TRACK_MODE=both
  export SKIP_POLICY=0
  export PERCEPTION_OBS_TRANSPORT="${PERCEPTION_OBS_TRANSPORT:-shm}"
  export POLICY_USE_PERCEPTION_OBS_SHM="${POLICY_USE_PERCEPTION_OBS_SHM:-1}"
  export MJ_ENV_DEPTH_PREVIEW_SOURCE="${MJ_ENV_DEPTH_PREVIEW_SOURCE:-policy}"
else
  export MJ_TRACK_MODE=env
  export SKIP_POLICY=1
  export PERCEPTION_OBS_TRANSPORT="${PERCEPTION_OBS_TRANSPORT:-shm}"
  export MJ_ENV_DEPTH_PREVIEW_SOURCE="${MJ_ENV_DEPTH_PREVIEW_SOURCE:-producer}"
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
export HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY="${HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY:-1}"
export HOLOSOMA_ZMQ_LOWCMD_MATCH_TOLERANCE_MS="${HOLOSOMA_ZMQ_LOWCMD_MATCH_TOLERANCE_MS:-2}"
export HOLOSOMA_POLICY_SUPPRESS_DUP_SIM_TIME_LOWCMD="${HOLOSOMA_POLICY_SUPPRESS_DUP_SIM_TIME_LOWCMD:-1}"
export HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES="${HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES:-0}"
export HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST="${HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST:-0}"
export HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST="${HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST:-0}"
export HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST="${HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST:-0}"
export HOLOSOMA_MUJOCO_WRIST_ORIGIN_CONTACT_SPHERES="0"
export HOLOSOMA_MUJOCO_PALM_CONTACT_SPHERES="0"
export HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE="${HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE:-0}"
export HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT="${HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT:-0}"
export PERCEPTION_CAMERA_WARP_EDGE_NOISE="${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-False}"
export PERCEPTION_CAMERA_WARP_ENABLE_HOLES="${PERCEPTION_CAMERA_WARP_ENABLE_HOLES:-False}"
export PERCEPTION_CAMERA_APPLY_SENSOR_NOISE="${PERCEPTION_CAMERA_APPLY_SENSOR_NOISE:-False}"
export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-}"
export SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-}"
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-}"
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-}"
export HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION="${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION:-0}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
export HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS:-0}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS:-}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION:-}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION:-}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION:-}"
export HOLOSOMA_MUJOCO_REPLACE_CYLINDERS_WITH_CAPSULES="${HOLOSOMA_MUJOCO_REPLACE_CYLINDERS_WITH_CAPSULES:-0}"
export HOLOSOMA_MUJOCO_RESET_NOISE="0"
export PERCEPTION_CAMERA_WARP_EDGE_NOISE="${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-}"
export PERCEPTION_CAMERA_WARP_ENABLE_HOLES="${PERCEPTION_CAMERA_WARP_ENABLE_HOLES:-}"
export PERCEPTION_CAMERA_APPLY_SENSOR_NOISE="${PERCEPTION_CAMERA_APPLY_SENSOR_NOISE:-}"

HEADLESS_FLAG="$(resolve_headless)"
export TRAINING_HEADLESS="$HEADLESS_FLAG"
if [[ "$HEADLESS_FLAG" == "False" && "$PERCEPTION_CAMERA_SOURCE" == "rendered" && -z "${MUJOCO_GL:-}" ]]; then
  export MUJOCO_GL=glfw
fi

COMMAND_WEB="${COMMAND_WEB:-0}"
COMMAND_WEB_PORT="${COMMAND_WEB_PORT:-}"
if [[ -z "$COMMAND_WEB_PORT" ]]; then
  COMMAND_WEB_PORT_BASE="${COMMAND_WEB_PORT_BASE:-4477}"
  COMMAND_WEB_PORT_MAX="${COMMAND_WEB_PORT_MAX:-4499}"
  COMMAND_WEB_PORT="$(find_free_port "$COMMAND_WEB_PORT_BASE" "$COMMAND_WEB_PORT_MAX")"
fi
COMMAND_MANUAL_ENABLED="${COMMAND_MANUAL_ENABLED:-0}"
COMMAND_VALUE="${COMMAND_VALUE:-0.2}"
COMMAND_YAW_DEGREES="${COMMAND_YAW_DEGREES:-${COMMAND_YAW_DEG:-17}}"
COMMAND_MODE="${COMMAND_MODE:-manual}"
COMMAND_WEB_TRACK_ONLY="${COMMAND_WEB_TRACK_ONLY:-0}"
export POLICY_STDIO="${POLICY_STDIO:-inherit}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND="${HOLOSOMA_KEYBOARD_ROOT_COMMAND:-1}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE:-manual}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE:-hold}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE:-${COMMAND_VALUE}}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES="${HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES:-${COMMAND_YAW_DEGREES}}"
export HOLOSOMA_MUJOCO_BACKSPACE_POLICY_CONTROL="${HOLOSOMA_MUJOCO_BACKSPACE_POLICY_CONTROL:-1}"
export HOLOSOMA_MUJOCO_BACKSPACE_AUTORESTART_POLICY="${HOLOSOMA_MUJOCO_BACKSPACE_AUTORESTART_POLICY:-1}"
export HOLOSOMA_MUJOCO_GUARD_DEFAULT_RESET="${HOLOSOMA_MUJOCO_GUARD_DEFAULT_RESET:-1}"
export HOLOSOMA_MUJOCO_POLICY_COMMAND_OVERLAY="${HOLOSOMA_MUJOCO_POLICY_COMMAND_OVERLAY:-1}"
SHOW_MOTION_ROBOT="${SHOW_MOTION_ROBOT:-0}"
SHOW_MOTION_OBJECT="${SHOW_MOTION_OBJECT:-0}"
VISER_PORT_RESOLVED="${VISER_PORT:-2984}"
MJ_ENV_DEPTH_PREVIEW="${MJ_ENV_DEPTH_PREVIEW:-auto}"
MJ_ENV_DEPTH_PREVIEW_LOG="${MJ_ENV_DEPTH_PREVIEW_LOG:-${ROOT_DIR}/logs/live_debug/mj_depth_preview.log}"
MJ_ENV_KILL_STALE_ENV="${MJ_ENV_KILL_STALE_ENV:-1}"
MJ_ENV_KILL_STALE_POLICY="${MJ_ENV_KILL_STALE_POLICY:-1}"
GT_MUJOCO_PHYSICS="${GT_MUJOCO_PHYSICS:-${HOLOSOMA_GT_MUJOCO_PHYSICS:-1}}"
if is_truthy "$GT_MUJOCO_PHYSICS"; then
  export GT_MUJOCO_PHYSICS=1
  export HOLOSOMA_GT_MUJOCO_PHYSICS=1
  export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS=0
  export SIM_USE_TRAINING_URDF_OBJECT_SCENE=1
  export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML=0
  export SIM_COPY_TENDONS_FROM_ROBOT_XML=0
  export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML=0
  export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML=0
  export MUJOCO_OBJECT_MASS_SCALE=""
  export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-1.4}"
  export MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-[0.6,0.02,0.005]}"
  export MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-[0.6,0.02,0.005]}"
  export MUJOCO_OBJECT_LATERAL_FRICTION=""
  export MUJOCO_OBJECT_ROLLING_FRICTION=""
  export MUJOCO_OBJECT_CONTACT_STIFFNESS=""
  export MUJOCO_OBJECT_CONTACT_DAMPING=""
else
  export GT_MUJOCO_PHYSICS=0
  export HOLOSOMA_GT_MUJOCO_PHYSICS=0
fi
COMMAND_WEB_PID=""
DEPTH_PREVIEW_PID=""

same_port_env_pids() {
  ps -eww -o pid=,args= | awk \
    -v self="$$" \
    -v root="${ROOT_DIR}" \
    -v state="${SIM_STATE_PORT}" \
    -v control="${SIM_CONTROL_PORT}" \
    -v sparse="${SPARSE_ROOT_COMMAND_PORT}" \
    -v policy="${POLICY_CONTROL_PORT}" \
    -v cmdweb="${COMMAND_WEB_PORT}" \
    -v viser="${VISER_PORT_RESOLVED}" \
    -v perception_shm="${PERCEPTION_OBS_SHM_NAME}" \
    -v policy_perception_shm="${HOLOSOMA_POLICY_PERCEPTION_OBS_SHM_NAME}" '
      $1 == self { next }
      {
        is_python = index($0, "python") || index($0, "python3")
        is_sim = index($0, root "/src/holosoma/holosoma/run_sim.py")
        is_web = index($0, root "/src/holosoma/holosoma/mj_command_web.py") || index($0, root "/src/holosoma/holosoma/mj_track_trigger_web.py")
        is_viser = index($0, root "/src/holosoma/holosoma/viser_mujoco_sim_state.py")
        is_depth_preview = index($0, root "/src/holosoma/holosoma/depth_preview_window.py")
        web_port_match = index($0, "--port " cmdweb) || index($0, "--sparse-root-command-port " sparse) || index($0, "--control-port " control) || index($0, "--policy-control-port " policy)
        viser_port_match = index($0, "--port " viser) || index($0, "--state-port " state) || index($0, "--control-port " control) || index($0, "--sparse-root-command-port " sparse)
        depth_preview_match = index($0, "--shm-name " perception_shm) || index($0, "--shm-name " policy_perception_shm)
        if (is_python && is_sim && index($0, "--simulator.config.bridge.sim-state-port " state) && index($0, "--simulator.config.bridge.control-port " control)) {
          print $1
        }
        if (is_python && is_web && web_port_match) {
          print $1
        }
        if (is_python && is_viser && viser_port_match) {
          print $1
        }
        if (is_python && is_depth_preview && depth_preview_match) {
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
  if [[ -n "${DEPTH_PREVIEW_PID:-}" ]] && kill -0 "$DEPTH_PREVIEW_PID" 2>/dev/null; then
    kill "$DEPTH_PREVIEW_PID" 2>/dev/null || true
    wait "$DEPTH_PREVIEW_PID" 2>/dev/null || true
  fi
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

maybe_use_native_data_demo_motion_dir "$@"

if [[ "$LAUNCH_VISER_RESOLVED" == "1" && -z "${PYTHON_BIN:-}" ]] && ! is_truthy "${DRY_RUN:-0}"; then
  PYTHON_BIN="$(resolve_python_with_modules "holosoma.viser_mujoco_sim_state" \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    /home/user/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    /home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
    /home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
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
echo "[INFO] target assists root=${HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST} dof=${HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST} object=${HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST}; contact_spheres wrist=${HOLOSOMA_MUJOCO_WRIST_ORIGIN_CONTACT_SPHERES} palm=${HOLOSOMA_MUJOCO_PALM_CONTACT_SPHERES}; reset_noise=${HOLOSOMA_MUJOCO_RESET_NOISE}"
echo "[INFO] reference_xml_copy joint_defaults=${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML} tendons=${SIM_COPY_TENDONS_FROM_ROBOT_XML} collision_geoms=${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML} contact_pairs=${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML}"
echo "[INFO] web_demo_object_contacts=${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS} training_object_contact_pairs=${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS} keep_reference_hand_collision=${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION} carry_arm_object_contacts=${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS}"
echo "[INFO] mujoco contact material lateral=${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION} spin=${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION} rolling=${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION}; replace_cylinders_with_capsules=${HOLOSOMA_MUJOCO_REPLACE_CYLINDERS_WITH_CAPSULES}; zero_motion_vel=${HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES}"
echo "[INFO] auto_launch_policy=${MJ_ENV_AUTO_LAUNCH_POLICY} mj_track_mode=${MJ_TRACK_MODE} skip_policy=${SKIP_POLICY}"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT} sparse_root=${SPARSE_ROOT_COMMAND_PORT} policy_control=${POLICY_CONTROL_PORT} policy_overlay=${HOLOSOMA_POLICY_OVERLAY_PORT}"
echo "[INFO] native command HUD=${HOLOSOMA_MUJOCO_POLICY_COMMAND_OVERLAY} command_scale_xy=${COMMAND_VALUE} yaw_deg=${COMMAND_YAW_DEGREES}"
case "$(printf '%s' "$MJ_ENV_DEPTH_PREVIEW" | tr '[:upper:]' '[:lower:]')" in
  auto|"")
    if [[ "$TRAINING_HEADLESS" == "False" && -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
      MJ_ENV_DEPTH_PREVIEW_RESOLVED=1
    else
      MJ_ENV_DEPTH_PREVIEW_RESOLVED=0
    fi
    ;;
  *)
    if is_truthy "$MJ_ENV_DEPTH_PREVIEW"; then
      MJ_ENV_DEPTH_PREVIEW_RESOLVED=1
    else
      MJ_ENV_DEPTH_PREVIEW_RESOLVED=0
    fi
    ;;
esac
MJ_ENV_DEPTH_PREVIEW_SOURCE_NORMALIZED="$(printf '%s' "$MJ_ENV_DEPTH_PREVIEW_SOURCE" | tr '[:upper:]' '[:lower:]')"
case "$MJ_ENV_DEPTH_PREVIEW_SOURCE_NORMALIZED" in
  policy|policy_input|policy-input|accepted|accepted_policy|accepted-policy)
    DEPTH_PREVIEW_SOURCE_LABEL="policy"
    DEPTH_PREVIEW_SHM_NAME="${HOLOSOMA_POLICY_PERCEPTION_OBS_SHM_NAME}"
    ;;
  raw|producer|camera|sim|simulator)
    DEPTH_PREVIEW_SOURCE_LABEL="producer"
    DEPTH_PREVIEW_SHM_NAME="${PERCEPTION_OBS_SHM_NAME}"
    ;;
  *)
    echo "[WARN] unknown MJ_ENV_DEPTH_PREVIEW_SOURCE=${MJ_ENV_DEPTH_PREVIEW_SOURCE}; using producer depth shm." >&2
    DEPTH_PREVIEW_SOURCE_LABEL="producer"
    DEPTH_PREVIEW_SHM_NAME="${PERCEPTION_OBS_SHM_NAME}"
    ;;
esac
echo "[INFO] depth preview=${MJ_ENV_DEPTH_PREVIEW_RESOLVED} source=${DEPTH_PREVIEW_SOURCE_LABEL} shm=${DEPTH_PREVIEW_SHM_NAME} near=${PERCEPTION_CAMERA_NEAR:-0.3} far=${PERCEPTION_CAMERA_FAR:-3.0} log=${MJ_ENV_DEPTH_PREVIEW_LOG}"
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
  if is_truthy "$COMMAND_WEB"; then
    echo "[INFO] policy is launched by this rollout and waits for web ] before sending lowcmd."
  elif [[ "${POLICY_STDIO:-}" == "inherit" ]]; then
    if is_truthy "${HOLOSOMA_KEYBOARD_ROOT_COMMAND:-0}"; then
      echo "[INFO] policy is launched by this rollout with terminal controls: ] start policy, Space start motion clip, W/S/A/D/Q/E sparse-root command input_mode=${HOLOSOMA_KEYBOARD_ROOT_COMMAND_INPUT_MODE}."
    else
      echo "[INFO] policy is launched by this rollout with terminal controls: ] start policy, Space start motion clip."
    fi
    echo "[INFO] MuJoCo window controls: Backspace/R coordinated sim + policy reset."
  else
    echo "[INFO] policy is launched by this rollout in noninteractive debug mode; it auto-starts policy/motion according to policy task flags."
    echo "[INFO] MuJoCo window controls: Backspace/R coordinated sim + policy reset."
  fi
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

if [[ "$MJ_ENV_DEPTH_PREVIEW_RESOLVED" == "1" ]]; then
  mkdir -p "$(dirname "$MJ_ENV_DEPTH_PREVIEW_LOG")"
  : >"$MJ_ENV_DEPTH_PREVIEW_LOG"
  if [[ "$DEPTH_PREVIEW_SOURCE_LABEL" == "policy" ]]; then
    rm -f "/dev/shm/${DEPTH_PREVIEW_SHM_NAME#/}" 2>/dev/null || true
  fi
  DEPTH_PREVIEW_RAW_WIDTH="${DEPTH_PREVIEW_RAW_WIDTH:-${PERCEPTION_CAMERA_WIDTH:-106}}"
  DEPTH_PREVIEW_RAW_HEIGHT="${DEPTH_PREVIEW_RAW_HEIGHT:-${PERCEPTION_CAMERA_HEIGHT:-60}}"
  DEPTH_PREVIEW_CROP_TOP="${DEPTH_PREVIEW_CROP_TOP:-${PERCEPTION_CAMERA_WARP_CROP_TOP:-2}}"
  DEPTH_PREVIEW_CROP_BOTTOM="${DEPTH_PREVIEW_CROP_BOTTOM:-${PERCEPTION_CAMERA_WARP_CROP_BOTTOM:-0}}"
  DEPTH_PREVIEW_CROP_LEFT="${DEPTH_PREVIEW_CROP_LEFT:-${PERCEPTION_CAMERA_WARP_CROP_LEFT:-4}}"
  DEPTH_PREVIEW_CROP_RIGHT="${DEPTH_PREVIEW_CROP_RIGHT:-${PERCEPTION_CAMERA_WARP_CROP_RIGHT:-4}}"
  DEPTH_PREVIEW_CROP="top=${DEPTH_PREVIEW_CROP_TOP} bottom=${DEPTH_PREVIEW_CROP_BOTTOM} left=${DEPTH_PREVIEW_CROP_LEFT} right=${DEPTH_PREVIEW_CROP_RIGHT}"
  DEPTH_PREVIEW_NEAR="${DEPTH_PREVIEW_NEAR:-${PERCEPTION_CAMERA_NEAR:-0.3}}"
  DEPTH_PREVIEW_FAR="${DEPTH_PREVIEW_FAR:-${PERCEPTION_CAMERA_FAR:-3.0}}"
  PYTHONPATH="${ROOT_DIR}/src/holosoma:${ROOT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}" \
    "$DEPTH_PREVIEW_PY" -u "${ROOT_DIR}/src/holosoma/holosoma/depth_preview_window.py" \
      --shm-name "${DEPTH_PREVIEW_SHM_NAME}" \
      --dim "${DEPTH_PREVIEW_DIM:-5046}" \
      --shape "${DEPTH_PREVIEW_SHAPE:-58x87}" \
      --source "${PERCEPTION_CAMERA_SOURCE}" \
      --preset "${PERCEPTION_PRESET:-camera_depth_d435i}" \
      --object-geometry-mode "${PERCEPTION_OBJECT_GEOMETRY_MODE:-primitive}" \
      --raw-size "${DEPTH_PREVIEW_RAW_WIDTH}x${DEPTH_PREVIEW_RAW_HEIGHT}" \
      --crop "${DEPTH_PREVIEW_CROP}" \
      --near "${DEPTH_PREVIEW_NEAR}" \
      --far "${DEPTH_PREVIEW_FAR}" \
      --input-mode "${DEPTH_PREVIEW_INPUT_MODE:-normalized}" \
      >"$MJ_ENV_DEPTH_PREVIEW_LOG" 2>&1 &
  DEPTH_PREVIEW_PID=$!
  sleep 0.2
  if ! kill -0 "$DEPTH_PREVIEW_PID" 2>/dev/null; then
    echo "[WARN] depth preview failed to start. See $MJ_ENV_DEPTH_PREVIEW_LOG" >&2
    DEPTH_PREVIEW_PID=""
  fi
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
    /home/user/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    /home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
    /home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
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
    --pre-trigger-wait-s "${MJ_ENV_RECORD_PRE_TRIGGER_WAIT_S:-2.0}" \
    $( [[ "${MJ_ENV_RECORD_SKIP_RESET:-0}" == "1" ]] && printf '%s' "--skip-reset" ) \
    $( [[ "${MJ_ENV_RECORD_NO_AUTO_START:-0}" == "1" ]] && printf '%s' "--no-auto-start" ) || echo "[WARN] rollout video recording failed" >&2
  echo "[INFO] rollout recording finished: ${RECORD_OUTPUT}"
  wait "$rollout_pid"
else
  "${LAUNCH_CMD[@]}"
fi
