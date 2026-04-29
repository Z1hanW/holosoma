#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-$ROOT_DIR/data_demo}"
DEFAULT_CLIP="${DEFAULT_CLIP:-box_75}"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-$ROOT_DIR/logs/wandb_runs/shoo7sr1/model_29999.onnx}"
DEFAULT_MODEL_FALLBACK="https://wandb.ai/zihanw22/boxer/runs/shoo7sr1/model_29999.onnx"
DEFAULT_INFER_PY="/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python"

export PYTHONSAFEPATH="${PYTHONSAFEPATH:-1}"

usage() {
  cat <<EOF
Usage:
  bash mj_rollout.sh [rendered848|rendered] [clip_name|motion.npz] [model.onnx|wandb://...]

Purpose:
  Launch only the policy rollout process against an already-running mj_launch.sh
  native MuJoCo simulator. This script intentionally uses only hsinference.

Examples:
  bash mj_rollout.sh box_75
  bash mj_rollout.sh rendered848 box_75 logs/wandb_runs/shoo7sr1/model_29999.onnx

Controls:
  ]      start policy
  Space  start motion clip
  w/s    x +/-
  a/d    y +/-
  q/e    yaw +/-
  o      stop policy
  i      init/default pose
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

resolve_hsinference_python() {
  local configured="${INFER_PY:-$DEFAULT_INFER_PY}"
  if [[ ! -x "$configured" ]]; then
    echo "[ERROR] INFER_PY is not executable: $configured" >&2
    exit 1
  fi
  if ! python_has_modules "$configured" holosoma_inference onnx onnxruntime numpy; then
    echo "[ERROR] INFER_PY must be the hsinference env with holosoma_inference/onnx/onnxruntime/numpy: $configured" >&2
    exit 1
  fi
  printf '%s\n' "$configured"
}

resolve_motion_file() {
  local motion_dir="$1"
  local clip="$2"
  if [[ "$clip" == *.npz ]]; then
    if [[ -f "$clip" ]]; then
      realpath "$clip"
      return 0
    fi
    if [[ -f "$motion_dir/${clip##*/}" ]]; then
      realpath "$motion_dir/${clip##*/}"
      return 0
    fi
  elif [[ -f "$motion_dir/${clip}.npz" ]]; then
    realpath "$motion_dir/${clip}.npz"
    return 0
  fi
  echo "[ERROR] Motion clip not found: $clip (MOTION_DIR=$motion_dir)" >&2
  exit 1
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

MODE="rendered848"
MOTION_CLIP="${MOTION_CLIP_NAME:-${MOTION_CLIP:-$DEFAULT_CLIP}}"
MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-}}"
POSITIONAL_MODE=1
for arg in "$@"; do
  if [[ "$POSITIONAL_MODE" == "0" ]]; then
    continue
  fi
  case "$arg" in
    rendered848|render848|mujoco848|mujoco_render_848x480)
      MODE="rendered848"
      ;;
    rendered|render|mujoco)
      MODE="rendered"
      ;;
    --*)
      POSITIONAL_MODE=0
      ;;
    *.onnx|*.pt|wandb://*|https://*)
      MODEL_INPUT="$arg"
      ;;
    *.npz)
      MOTION_FILE="$arg"
      ;;
    *)
      MOTION_CLIP="$arg"
      ;;
  esac
done

INFER_PY="$(resolve_hsinference_python)"
MOTION_DIR="$(realpath "${MOTION_DIR:-$DEFAULT_MOTION_DIR}")"
MOTION_FILE="${MOTION_FILE:-}"
if [[ -n "$MOTION_FILE" ]]; then
  MOTION_FILE="$(resolve_motion_file "$MOTION_DIR" "$MOTION_FILE")"
else
  MOTION_FILE="$(resolve_motion_file "$MOTION_DIR" "$MOTION_CLIP")"
fi
MOTION_STEM="$(basename "${MOTION_FILE%.npz}")"

if [[ -z "$MODEL_INPUT" ]]; then
  if [[ -f "$DEFAULT_MODEL_INPUT" ]]; then
    MODEL_INPUT="$DEFAULT_MODEL_INPUT"
  else
    MODEL_INPUT="$DEFAULT_MODEL_FALLBACK"
  fi
fi

SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-5661}"
POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-5662}"
POLICY_OVERLAY_PORT="${POLICY_OVERLAY_PORT:-5663}"
PERCEPTION_OBS_SHM_NAME="${PERCEPTION_OBS_SHM_NAME:-depth_img_shm_${SIM_STATE_PORT}}"
INTERFACE_NAME="${INTERFACE_NAME:-lo}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-object-distill}"
PATCH_DIR="${PATCH_DIR:-/tmp/holosoma_mj_rollout}"
POLICY_RL_RATE="${POLICY_RL_RATE:-50}"
POLICY_ACTION_SCALE="${POLICY_ACTION_SCALE:-1.0}"
AUTO_START_STIFF_HOLD_SEC="${AUTO_START_STIFF_HOLD_SEC:-0.0}"
AUTO_START_STIFF_MAX_WAIT_SEC="${AUTO_START_STIFF_MAX_WAIT_SEC:-0.0}"
AUTO_START_STIFF_POSE_TOL="${AUTO_START_STIFF_POSE_TOL:-0.12}"
RUN_SECONDS="${RUN_SECONDS:-0}"

export PYTHONPATH="$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
export HOLOSOMA_POLICY_CONTROL_PORT="${HOLOSOMA_POLICY_CONTROL_PORT:-$POLICY_CONTROL_PORT}"
export HOLOSOMA_POLICY_OVERLAY_PORT="${HOLOSOMA_POLICY_OVERLAY_PORT:-$POLICY_OVERLAY_PORT}"
export HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART="${HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART:-0}"
export HOLOSOMA_SKIP_STIFF_PROMPT="${HOLOSOMA_SKIP_STIFF_PROMPT:-1}"
export HOLOSOMA_DISABLE_AUTO_RESET="${HOLOSOMA_DISABLE_AUTO_RESET:-1}"
export HOLOSOMA_DISABLE_MOTION_END_RESET="${HOLOSOMA_DISABLE_MOTION_END_RESET:-1}"
export HOLOSOMA_DISABLE_CLIP_END_RESET="${HOLOSOMA_DISABLE_CLIP_END_RESET:-1}"
export HOLOSOMA_DISABLE_BAD_TRACKING_RESET="${HOLOSOMA_DISABLE_BAD_TRACKING_RESET:-1}"
export HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY="${HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY:-1}"
export HOLOSOMA_ZMQ_LOWCMD_MATCH_TOLERANCE_MS="${HOLOSOMA_ZMQ_LOWCMD_MATCH_TOLERANCE_MS:-2}"
export HOLOSOMA_ZMQ_LOWCMD_KP_SCALE="${HOLOSOMA_ZMQ_LOWCMD_KP_SCALE:-1.0}"
export HOLOSOMA_ZMQ_LOWCMD_KD_SCALE="${HOLOSOMA_ZMQ_LOWCMD_KD_SCALE:-1.0}"
export HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE="${HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE:-1.0}"

if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ "$MOTION_STEM" == "box_75" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=5
  else
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
  fi
fi

if is_truthy "${MJ_ROLLOUT_TERMINAL_KEYS:-1}"; then
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

MODEL_LOCAL="$(
  "$INFER_PY" - <<'PY' "$MODEL_INPUT" "$ROOT_DIR/logs/wandb_runs"
import sys
from pathlib import Path
from urllib.parse import urlparse

from holosoma_inference.utils.wandb import load_checkpoint

model = sys.argv[1]
root = Path(sys.argv[2])
path = Path(model).expanduser()
if path.is_file():
    print(path.resolve())
    raise SystemExit(0)
download_dir = root / "box_depth"
if model.startswith("wandb://"):
    parts = model[len("wandb://") :].split("/", 3)
    if len(parts) >= 3:
        download_dir = root / parts[2]
elif model.startswith("https://"):
    parts = [part for part in urlparse(model).path.split("/") if part]
    if len(parts) >= 4 and parts[2] == "runs":
        download_dir = root / parts[3]
resolved = load_checkpoint(None, model, str(download_dir))
print(Path(resolved).expanduser().resolve())
PY
)"
MODEL_LOCAL="$(printf '%s\n' "$MODEL_LOCAL" | tail -n 1)"

mkdir -p "$PATCH_DIR"
MODEL_STEM="$(basename "${MODEL_LOCAL%.*}")"
PATCHED_ONNX="$PATCH_DIR/${MODEL_STEM}__${MOTION_STEM}.onnx"
"$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/patch_motion_onnx.py" \
  --model-path "$MODEL_LOCAL" \
  --motion-file "$MOTION_FILE" \
  --output-path "$PATCHED_ONNX"

CMD=(
  "$INFER_PY" -u "$ROOT_DIR/src/holosoma_inference/holosoma_inference/run_policy.py"
  "inference:${INFERENCE_CONFIG}"
  --task.model-path "$PATCHED_ONNX"
  --task.motion-file "$MOTION_FILE"
  --task.interface "$INTERFACE_NAME"
  --task.use-sim-state
  --task.sim-clock-port "$SIM_CLOCK_PORT"
  --task.sim-state-port "$SIM_STATE_PORT"
  --task.sim-control-port "$SIM_CONTROL_PORT"
  --task.no-auto-start-motion
  --task.auto-start-stiff-hold-sec "$AUTO_START_STIFF_HOLD_SEC"
  --task.auto-start-stiff-max-wait-sec "$AUTO_START_STIFF_MAX_WAIT_SEC"
  --task.auto-start-stiff-pose-tolerance "$AUTO_START_STIFF_POSE_TOL"
  --task.policy-action-scale "$POLICY_ACTION_SCALE"
  --task.rl-rate "$POLICY_RL_RATE"
  --task.sim-object-name object
  --task.use-zmq-lowcmd
  --task.use-split-perception-obs
  --task.perception-obs-port "$PERCEPTION_OBS_PORT"
  --task.use-split-perception-obs-shm
  --task.perception-obs-shm-name "$PERCEPTION_OBS_SHM_NAME"
  --task.use-external-sparse-root-command
  --task.sparse-root-command-port "$SPARSE_ROOT_COMMAND_PORT"
  --task.use-sim-time
  --task.prefer-sim-ref-from-sim-state
  --task.defer-policy-start-until-valid-state
)

echo "[INFO] launching policy rollout only"
echo "[INFO] python=$INFER_PY"
echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] model=$MODEL_LOCAL"
echo "[INFO] patched_onnx=$PATCHED_ONNX"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT} sparse_root=${SPARSE_ROOT_COMMAND_PORT}"
echo "[INFO] controls: ] start policy, Space start motion, W/S/A/D/Q/E command"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[DRY_RUN] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "$RUN_SECONDS" == "0" ]]; then
  exec "${CMD[@]}"
else
  exec timeout --kill-after=5s --signal=INT "${RUN_SECONDS}s" "${CMD[@]}"
fi
