#!/usr/bin/env bash
set -euo pipefail

# Split MuJoCo depth rollout launcher for box policies.
#
# This is intentionally a thin wrapper around mj_track.sh:
# - hsmujoco runs the real MuJoCo simulator.
# - hsinference runs holosoma_inference/run_policy.py.
# - The policy receives split perception_obs published by the MuJoCo process.
#
# Defaults are aligned with the web MuJoCo depth demo's fixed object setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEFAULT_MOTION_DIR="${ROOT_DIR}/data/ds_box_data/train_g1_w_obj_prepared"
DEFAULT_MOTION_CLIP="box_74"
DEFAULT_MOTION_FILE="${DEFAULT_MOTION_DIR}/${DEFAULT_MOTION_CLIP}.npz"
DEFAULT_WEB_MOTION_FILE="${ROOT_DIR}/.tmp_teacher_box_contacts_part2/motion_bank/${DEFAULT_MOTION_CLIP}.npz"
DEFAULT_OBJECT_MAP="${DEFAULT_MOTION_DIR}/_clip_object_urdf_map.json"
DEFAULT_OBJECT_URDF="${DEFAULT_MOTION_DIR}/_generated_urdfs/${DEFAULT_MOTION_CLIP}.urdf"
DEFAULT_DEPTH_BUNDLE_DIR="${ROOT_DIR}/mujoco-web-wobj-depth-demo/public/demo-assets/clips/${DEFAULT_MOTION_CLIP}"
DEFAULT_DEPTH_ONNX="${DEFAULT_DEPTH_BUNDLE_DIR}/policy.onnx"
DEFAULT_TEACHER_MODEL="${ROOT_DIR}/.teacher_checkpoints/model_07000.onnx"
DEFAULT_WANDB_MODEL="wandb://zihanw22/boxer/shoo7sr1/model_07000.onnx"

usage() {
  cat <<EOF
Usage:
  bash mujoco_depth_joystick.sh [rendered|warp] [clip_name|motion.npz] [model.onnx|wandb://...]

Examples:
  bash mujoco_depth_joystick.sh
  HEADLESS=False bash mujoco_depth_joystick.sh rendered box_74
  LAUNCH_VISER=1 VISER_PORT=18080 bash mujoco_depth_joystick.sh rendered box_74

Environment:
  MOTION_DIR                  default: ${DEFAULT_MOTION_DIR}
  OBJECT_URDF                 default: ${DEFAULT_OBJECT_URDF} for box_74, otherwise ${DEFAULT_OBJECT_MAP}
                              may be a map json or a single URDF
  MODEL_INPUT / MODEL_PATH    default: ${DEFAULT_DEPTH_ONNX} if present, then ${DEFAULT_TEACHER_MODEL},
                              otherwise ${DEFAULT_WANDB_MODEL}
  HEADLESS                    default: True; set False to open the native MuJoCo viewer
  WEB_DEMO_MATCH              default: 1; use web-demo motion/reset/manual-command/depth timing
  LAUNCH_VISER                default: 0; set 1 for the Viser split-state monitor
  PERCEPTION_CAMERA_SOURCE    default: rendered; options: rendered, far_tracking_warp
  ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND
                              default: 1; lets Viser override sparse root command
  MUJOCO_OBJECT_MASS_OVERRIDE default: 1.4
  MUJOCO_OBJECT_GEOM_FRICTION default: [0.6,0.02,0.005]
  MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION
                              default: [0.6,0.02,0.005]
  AUTO_PORTS                  default: 1; choose free ZMQ ports unless ports are explicitly set
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

find_free_ports() {
  local count="$1"
  python3 - "$count" <<'PY'
import socket
import sys

count = int(sys.argv[1])
sockets = []
ports = []
try:
    for _ in range(count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
        ports.append(sock.getsockname()[1])
    print(" ".join(str(port) for port in ports))
finally:
    for sock in sockets:
        sock.close()
PY
}

resolve_python() {
  local configured="$1"
  shift
  if [[ -n "${configured}" && -x "${configured}" ]]; then
    printf '%s\n' "${configured}"
    return
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
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

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

INFER_PY="$(resolve_python "${INFER_PY:-}" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"
MUJOCO_PY="$(resolve_python "${MUJOCO_PY:-}" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"

export INFER_PY
export MUJOCO_PY

MOTION_DIR="${MOTION_DIR:-${DEFAULT_MOTION_DIR}}"
DEFAULT_SELECTED_MOTION_FILE="${MOTION_DIR}/${DEFAULT_MOTION_CLIP}.npz"
DEFAULT_SELECTED_OBJECT_URDF="${MOTION_DIR}/_generated_urdfs/${DEFAULT_MOTION_CLIP}.urdf"
OBJECT_MAP_INPUT="${OBJECT_URDF:-}"
WEB_DEMO_MATCH="${WEB_DEMO_MATCH:-1}"
if [[ -f "${DEFAULT_DEPTH_ONNX}" ]]; then
  MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${DEFAULT_DEPTH_ONNX}}}"
elif [[ -f "${DEFAULT_TEACHER_MODEL}" ]]; then
  MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${DEFAULT_TEACHER_MODEL}}}"
else
  MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${DEFAULT_WANDB_MODEL}}}"
fi
MOTION_FILE="${MOTION_FILE:-}"
MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-${MOTION_CLIP:-}}"
PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-rendered}"

for arg in "$@"; do
  case "${arg}" in
    rendered|render|mujoco|depth)
      PERCEPTION_CAMERA_SOURCE="rendered"
      ;;
    warp|far_tracking_warp)
      PERCEPTION_CAMERA_SOURCE="far_tracking_warp"
      ;;
    *.npz)
      MOTION_FILE="${arg}"
      ;;
    wandb://*|https://*|*.onnx|*.pt)
      MODEL_INPUT="${arg}"
      ;;
    --*)
      echo "[ERROR] This wrapper is configured by environment variables, not extra tyro args: ${arg}" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -z "${MOTION_CLIP_NAME}" ]]; then
        MOTION_CLIP_NAME="${arg}"
      else
        echo "[ERROR] Unexpected positional argument: ${arg}" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${MOTION_FILE}" ]]; then
  if [[ -n "${MOTION_CLIP_NAME}" ]]; then
    if is_truthy "${WEB_DEMO_MATCH}" && [[ "${MOTION_CLIP_NAME%.npz}" == "${DEFAULT_MOTION_CLIP}" && -f "${DEFAULT_WEB_MOTION_FILE}" ]]; then
      MOTION_FILE="${DEFAULT_WEB_MOTION_FILE}"
    else
      MOTION_FILE="${MOTION_DIR}/${MOTION_CLIP_NAME%.npz}.npz"
    fi
  elif is_truthy "${WEB_DEMO_MATCH}" && [[ -f "${DEFAULT_WEB_MOTION_FILE}" ]]; then
    MOTION_FILE="${DEFAULT_WEB_MOTION_FILE}"
  elif [[ -f "${DEFAULT_SELECTED_MOTION_FILE}" ]]; then
    MOTION_FILE="${DEFAULT_SELECTED_MOTION_FILE}"
  elif [[ -f "${DEFAULT_MOTION_FILE}" ]]; then
    MOTION_FILE="${DEFAULT_MOTION_FILE}"
  else
    MOTION_FILE="$(find "${MOTION_DIR}" -maxdepth 1 -name '*.npz' | sort | head -n 1)"
  fi
fi

if [[ -z "${MOTION_FILE}" || ! -f "${MOTION_FILE}" ]]; then
  echo "[ERROR] Motion clip not found: ${MOTION_FILE:-<empty>}" >&2
  exit 1
fi
MOTION_FILE="$(cd "$(dirname "${MOTION_FILE}")" && pwd)/$(basename "${MOTION_FILE}")"

if [[ -z "${OBJECT_MAP_INPUT}" ]]; then
  if [[ "$(basename "${MOTION_FILE}")" == "${DEFAULT_MOTION_CLIP}.npz" && -f "${DEFAULT_SELECTED_OBJECT_URDF}" ]]; then
    OBJECT_MAP_INPUT="${DEFAULT_SELECTED_OBJECT_URDF}"
  elif [[ "$(basename "${MOTION_FILE}")" == "${DEFAULT_MOTION_CLIP}.npz" && -f "${DEFAULT_OBJECT_URDF}" ]]; then
    OBJECT_MAP_INPUT="${DEFAULT_OBJECT_URDF}"
  else
    OBJECT_MAP_INPUT="${DEFAULT_OBJECT_MAP}"
  fi
fi

if [[ "${MODEL_INPUT}" == wandb://*.pt ]]; then
  MODEL_INPUT="${MODEL_INPUT%.pt}.onnx"
fi

MODEL_LOCAL="$(
  "${INFER_PY}" - <<'PY' "${MODEL_INPUT}" "${ROOT_DIR}/logs/sim2sim_remote_models"
import sys
from pathlib import Path

from holosoma_inference.utils.wandb import load_checkpoint

model = sys.argv[1]
root = Path(sys.argv[2])
download_dir = root / "box_depth_joystick"
if model.startswith("wandb://"):
    parts = model[len("wandb://") :].split("/", 3)
    if len(parts) >= 3:
        download_dir = root / parts[0] / parts[1] / parts[2]
path = load_checkpoint(None, model, str(download_dir))
path = Path(path).expanduser().resolve()
if path.suffix == ".pt":
    candidate = path.with_suffix(".onnx")
    if not candidate.is_file():
        raise FileNotFoundError(f"Expected sibling ONNX next to checkpoint: {candidate}")
    path = candidate
if not path.is_file():
    raise FileNotFoundError(path)
print(path)
PY
)"
MODEL_LOCAL="$(printf '%s\n' "${MODEL_LOCAL}" | tail -n 1)"

OBJECT_URDF_RESOLVED="$(
  "${INFER_PY}" - <<'PY' "${OBJECT_MAP_INPUT}" "${MOTION_FILE}"
import json
import sys
from pathlib import Path

import numpy as np

raw = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
stem = motion_path.stem
candidate = Path(raw).expanduser() if raw else None

if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".json":
    data = json.loads(candidate.read_text())
    clips = data.get("clips", data) if isinstance(data, dict) else {}
    entry = clips.get(stem) if isinstance(clips, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"Object map has no entry for clip '{stem}': {candidate}")
    path = entry.get("object_urdf_path") or entry.get("urdf_path")
    if not path:
        raise SystemExit(f"Object map entry for clip '{stem}' has no object_urdf_path")
    print(Path(path).expanduser().resolve())
elif candidate is not None and str(candidate):
    print(candidate.expanduser().resolve())
else:
    with np.load(motion_path, allow_pickle=True) as data:
        if "object_urdf_path" not in data:
            raise SystemExit(f"No OBJECT_URDF map provided and motion has no object_urdf_path: {motion_path}")
        print(Path(str(np.asarray(data["object_urdf_path"]).item())).expanduser().resolve())
PY
)"

HEADLESS_FLAG="$(normalize_bool_flag "${HEADLESS:-${headless:-True}}")"
LAUNCH_VISER="${LAUNCH_VISER:-0}"
AUTO_PORTS="${AUTO_PORTS:-1}"

if is_truthy "${AUTO_PORTS}"; then
  if [[ -z "${SIM_CLOCK_PORT+x}" && -z "${SIM_STATE_PORT+x}" && -z "${PERCEPTION_OBS_PORT+x}" && -z "${SIM_CONTROL_PORT+x}" && -z "${SPARSE_ROOT_COMMAND_PORT+x}" ]]; then
    read -r SIM_CLOCK_PORT SIM_STATE_PORT PERCEPTION_OBS_PORT SIM_CONTROL_PORT SPARSE_ROOT_COMMAND_PORT < <(find_free_ports 5)
  fi
fi

export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
export PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
export SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
export SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-5661}"
export TRAINING_HEADLESS="${HEADLESS_FLAG}"
export HOLOSOMA_MJ_TRACK_RUN_FOREVER="${HOLOSOMA_MJ_TRACK_RUN_FOREVER:-1}"
export OBJECT_URDF="${OBJECT_URDF_RESOLVED}"
export ENABLE_SPLIT_PERCEPTION_OBS="${ENABLE_SPLIT_PERCEPTION_OBS:-1}"
export PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
export PERCEPTION_CAMERA_SOURCE
export PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
if is_truthy "${WEB_DEMO_MATCH}"; then
  export SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-training_default_pose}"
  export PERCEPTION_UPDATE_HZ="${PERCEPTION_UPDATE_HZ:-50.0}"
  export PERCEPTION_CAMERA_FPS="${PERCEPTION_CAMERA_FPS:-50.0}"
  export PERCEPTION_CAMERA_WARP_EDGE_NOISE="${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-False}"
  export PERCEPTION_CAMERA_WARP_BUFFER_LEN="${PERCEPTION_CAMERA_WARP_BUFFER_LEN:-1}"
  export PERCEPTION_CAMERA_WARP_LATENCY_FRAME="${PERCEPTION_CAMERA_WARP_LATENCY_FRAME:-0}"
fi
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-object-distill}"
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
export SIM_ADD_DEFAULT_OBJECT_ACTUATORS="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS:-1}"
export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
export SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
export MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-[\"torso\",\"shoulder\",\"elbow\",\"wrist\",\"hand\",\"rubber_hand\"]}"
export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-1.4}"
export MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-[0.6,0.02,0.005]}"
export MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-[0.6,0.02,0.005]}"
export SIM_DEBUG_VIZ="${SIM_DEBUG_VIZ:-True}"
export SIM_USE_ZMQ_LOWCMD="${SIM_USE_ZMQ_LOWCMD:-1}"
export ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-1}"
export USE_SIM_TIME="${USE_SIM_TIME:-1}"
export USE_ROOT_REFERENCE_AT_CLIP_START="${USE_ROOT_REFERENCE_AT_CLIP_START:-1}"
export PREFER_SIM_REF_FROM_SIM_STATE="${PREFER_SIM_REF_FROM_SIM_STATE:-1}"
export POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-1}"
export HOLOSOMA_MUJOCO_OBJECT_GEOM_SNAPSHOT_PATH="${HOLOSOMA_MUJOCO_OBJECT_GEOM_SNAPSHOT_PATH:-${ROOT_DIR}/logs/live_debug/mujoco_depth_joystick_object_geoms.json}"

if [[ "${PERCEPTION_CAMERA_SOURCE}" == "far_tracking_warp" ]]; then
  export SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
fi
if [[ "${PERCEPTION_CAMERA_SOURCE}" == "rendered" ]]; then
  export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_RENDERED_DEPTH_FLIPUD="${HOLOSOMA_MUJOCO_RENDERED_DEPTH_FLIPUD:-0}"
  if [[ "${HEADLESS_FLAG}" == "True" && -z "${MUJOCO_GL:-}" ]]; then
    export MUJOCO_GL=egl
  fi
fi

echo "[INFO] MuJoCo depth joystick split rollout"
echo "[INFO] motion_file=${MOTION_FILE}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] model=${MODEL_LOCAL}"
echo "[INFO] headless=${HEADLESS_FLAG}"
echo "[INFO] web_demo_match=${WEB_DEMO_MATCH} motion_init=${SIM_MOTION_INIT_MODE:-raw_motion}"
echo "[INFO] perception=${PERCEPTION_PRESET} source=${PERCEPTION_CAMERA_SOURCE}"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT} sparse_root=${SPARSE_ROOT_COMMAND_PORT}"
echo "[INFO] external_sparse_root_command=${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND}"
echo "[INFO] object_params mass=${MUJOCO_OBJECT_MASS_OVERRIDE} geom_friction=${MUJOCO_OBJECT_GEOM_FRICTION} terrain_pair=${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION}"
echo "[INFO] hsmujoco=${MUJOCO_PY}"
echo "[INFO] hsinference=${INFER_PY}"

if is_truthy "${DRY_RUN:-0}"; then
  echo "[INFO] DRY_RUN=1; not launching."
  exit 0
fi

if is_truthy "${LAUNCH_VISER}"; then
  MANUAL_ROOT_ENABLED_FLAG="$(normalize_bool_flag "${MANUAL_ROOT_ENABLED:-${WEB_DEMO_MANUAL_ROOT_ENABLED:-${WEB_DEMO_MATCH}}}")"
  MANUAL_ROOT_MODE="${MANUAL_ROOT_MODE:-manual}"
  MANUAL_ROOT_DX="${MANUAL_ROOT_DX:-0.0}"
  MANUAL_ROOT_DY="${MANUAL_ROOT_DY:-0.0}"
  MANUAL_ROOT_DYAW="${MANUAL_ROOT_DYAW:-0.0}"
  VISER_ARGS=(
    --state-port "${SIM_STATE_PORT}"
    --perception-obs-port "${PERCEPTION_OBS_PORT}"
    --control-port "${SIM_CONTROL_PORT}"
    --sparse-root-command-port "${SPARSE_ROOT_COMMAND_PORT}"
    --manual-root-mode "${MANUAL_ROOT_MODE}"
    --manual-root-dx "${MANUAL_ROOT_DX}"
    --manual-root-dy "${MANUAL_ROOT_DY}"
    --manual-root-dyaw "${MANUAL_ROOT_DYAW}"
  )
  if [[ "${HEADLESS_FLAG}" == "True" ]]; then
    VISER_ARGS+=(--training-headless)
  else
    VISER_ARGS+=(--no-training-headless)
  fi
  if [[ "${MANUAL_ROOT_ENABLED_FLAG}" == "True" ]]; then
    VISER_ARGS+=(--manual-root-enabled)
  else
    VISER_ARGS+=(--no-manual-root-enabled)
  fi
  if [[ -n "${VISER_PORT:-}" ]]; then
    VISER_ARGS+=(--port "${VISER_PORT}")
  fi
  exec bash "${ROOT_DIR}/mj_track.sh" "${MOTION_FILE}" "${MODEL_LOCAL}" "${VISER_ARGS[@]}"
fi

export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1
exec bash "${ROOT_DIR}/mj_track.sh" "${MOTION_FILE}" "${MODEL_LOCAL}"
