#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

motion="${1:-box_75}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  shift
fi

MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${MODEL_REF:-https://wandb.ai/zihanw22/boxer/runs/w5qostjn}}}"
if [[ "${1:-}" == wandb://* || "${1:-}" == https://* || "${1:-}" == *.onnx || "${1:-}" == *.pt ]]; then
  MODEL_INPUT="$1"
  shift
fi

INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-box-w5qostjn}"
INTERFACE="${INTERFACE:-lo}"
MUJOCO_PYTHON_BIN="${MUJOCO_PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"

export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}"

resolve_metadata_model() {
  local input="$1"
  if [[ "$input" == *.onnx && -f "$input" ]]; then
    readlink -f "$input"
    return 0
  fi

  local run_id=""
  local file_name="model_20000.onnx"
  if [[ "$input" =~ /runs/([^/?#]+) ]]; then
    run_id="${BASH_REMATCH[1]}"
  elif [[ "$input" =~ wandb://[^/]+/[^/]+/([^/]+)/ ]]; then
    run_id="${BASH_REMATCH[1]}"
  fi
  if [[ "$input" =~ /([^/]+\.onnx)(\?.*)?$ ]]; then
    file_name="${BASH_REMATCH[1]}"
  fi
  [[ -n "$run_id" ]] || return 1

  local candidate
  for candidate in \
    "$ROOT_DIR/logs/wandb_runs/$run_id/$file_name" \
    "$ROOT_DIR/logs/wandb_runs/$run_id/model_20000.onnx" \
    "/home/user/FAR/holosoma/logs/wandb_runs/$run_id/$file_name" \
    "/home/user/FAR/holosoma/logs/wandb_runs/$run_id/model_20000.onnx"; do
    if [[ -f "$candidate" ]]; then
      readlink -f "$candidate"
      return 0
    fi
  done
  return 1
}

apply_training_perception_overrides() {
  local metadata_model
  metadata_model="$(resolve_metadata_model "$MODEL_INPUT" 2>/dev/null || true)"
  [[ -n "$metadata_model" ]] || return 0

  local exports
  exports="$("$POLICY_PYTHON_BIN" - "$metadata_model" <<'PY'
import json
import shlex
import sys

try:
    import onnx
except Exception:
    raise SystemExit(0)

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

perception = (metadata.get("experiment_config") or {}).get("perception") or {}
field_map = {
    "camera_width": "PERCEPTION_CAMERA_WIDTH",
    "camera_height": "PERCEPTION_CAMERA_HEIGHT",
    "camera_hfov_deg": "PERCEPTION_CAMERA_HFOV_DEG",
    "camera_vfov_deg": "PERCEPTION_CAMERA_VFOV_DEG",
    "camera_pitch_deg": "PERCEPTION_CAMERA_PITCH_DEG",
    "camera_fps": "PERCEPTION_CAMERA_FPS",
    "camera_near": "PERCEPTION_CAMERA_NEAR",
    "camera_far": "PERCEPTION_CAMERA_FAR",
    "camera_warp_crop_top": "PERCEPTION_CAMERA_WARP_CROP_TOP",
    "camera_warp_crop_bottom": "PERCEPTION_CAMERA_WARP_CROP_BOTTOM",
    "camera_warp_crop_left": "PERCEPTION_CAMERA_WARP_CROP_LEFT",
    "camera_warp_crop_right": "PERCEPTION_CAMERA_WARP_CROP_RIGHT",
    "camera_warp_latency_frame": "PERCEPTION_CAMERA_WARP_LATENCY_FRAME",
    "camera_warp_buffer_len": "PERCEPTION_CAMERA_WARP_BUFFER_LEN",
}
for key, env_name in field_map.items():
    value = perception.get(key)
    if value is None:
        continue
    if isinstance(value, bool):
        value = "True" if value else "False"
    print(f"export {env_name}={shlex.quote(str(value))}")
PY
)"
  [[ -n "$exports" ]] || return 0
  eval "$exports"
}

apply_training_perception_overrides

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/logs/sim2sim_runs/${motion%.*}__myholosoma}"
SIM_LOG="${SIM_LOG:-${RUN_DIR}/mujoco.log}"
mkdir -p "$RUN_DIR"
: >"$SIM_LOG"

MOTION_FILE="${MOTION_FILE:-${HOLOSOMA_MJ_MOTION:-}}"
if [[ -z "$MOTION_FILE" ]]; then
  if [[ "$motion" == *.npz || "$motion" == */* ]]; then
    MOTION_FILE="$motion"
  elif [[ -f "$ROOT_DIR/data_demo/${motion}.npz" ]]; then
    MOTION_FILE="$ROOT_DIR/data_demo/${motion}.npz"
  else
    MOTION_FILE="/home/user/FAR/holosoma/data_demo/${motion}.npz"
  fi
fi

PYTHON_BIN="$MUJOCO_PYTHON_BIN" bash "$ROOT_DIR/mj_env.sh" \
  --motion-init.enabled=True \
  --motion-init.motion-file "$MOTION_FILE" \
  --motion-init.mode "${SIM_MOTION_INIT_MODE:-raw_motion}" \
  --motion-init.object-name object \
  "$@" >"$SIM_LOG" 2>&1 &
sim_pid=$!

cleanup() {
  kill "$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

until grep -q "Starting direct simulation loop" "$SIM_LOG"; do
  if ! kill -0 "$sim_pid" 2>/dev/null; then
    tail -n 80 "$SIM_LOG" >&2 || true
    exit 1
  fi
  sleep 0.5
done

"$POLICY_PYTHON_BIN" -u src/holosoma_inference/holosoma_inference/run_policy.py "inference:${INFERENCE_CONFIG}" \
  --task.interface "$INTERFACE" \
  --task.model-path "$MODEL_INPUT"
