#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash mj_depth.sh [--depth-source rendered|warp] [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/.../files] [extra infer_box_joystick args...]

Purpose:
  Launch the depth distill box-carry policy in MuJoCo and visualize/control it through the
  existing inference-side viser workflow.

Defaults:
  checkpoint = wandb://zihanw22/boxer/0z2aggr2/model_05000.pt
  depth_source = warp
  simulator  = mujoco
  backend    = classic
  dataset    = omomo
  num_envs   = 1
  headless   = True

Useful env vars:
  DEPTH_CHECKPOINT_DEFAULT  Override the default checkpoint.
  MJ_DEPTH_CAMERA_SOURCE    rendered|warp (default: warp)
  INFER_DATASET             omomo|behave|mixed (default: omomo)
  MOTION_DIR                Optional motion dir/file override.
  OBJECT_URDF               Optional object urdf override.
  NUM_ENVS                  Default: 1
  HEADLESS                  Default: True
  VISER_PORT                Optional fixed viser port.
  MUJOCO_BACKEND            classic|warp (default: classic)
  USE_HW_JOYSTICK_BRIDGE    True/False
  VISER_MANUAL_USE_HW_JOYSTICK
  VISER_MANUAL_HW_BACKEND
  VISER_MANUAL_HW_DEVICE
  VISER_MANUAL_HW_TYPE

Examples:
  bash mj_depth.sh
  bash mj_depth.sh --depth-source warp
  VISER_PORT=18080 bash mj_depth.sh
  HEADLESS=False bash mj_depth.sh /abs/path/model_05000.pt
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_SIM_PY="/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python"
if [[ -z "${PYTHON_BIN+x}" || -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${DEFAULT_SIM_PY}" ]]; then
    export PYTHON_BIN="${DEFAULT_SIM_PY}"
  else
    export PYTHON_BIN="python"
  fi
fi
if [[ "${PYTHON_BIN}" == "${DEFAULT_SIM_PY}" ]]; then
  export PATH="$(dirname "${DEFAULT_SIM_PY}"):${PATH}"
fi

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

CKPT_ARG=""
EXTRA_ARGS=()
DEPTH_SOURCE_RAW="${MJ_DEPTH_CAMERA_SOURCE:-warp}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --depth-source)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --depth-source requires a value: rendered|warp" >&2
        exit 2
      fi
      DEPTH_SOURCE_RAW="$2"
      shift 2
      ;;
    --depth-source=*)
      DEPTH_SOURCE_RAW="${1#*=}"
      shift
      ;;
    wandb://*|https://wandb.ai/*|/*|./*|../*|*.pt)
      if [[ -z "${CKPT_ARG}" ]]; then
        CKPT_ARG="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

DEPTH_SOURCE="$(echo "${DEPTH_SOURCE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${DEPTH_SOURCE}" in
  rendered|warp) ;;
  *)
    echo "[ERROR] depth source must be one of: rendered|warp. Got: ${DEPTH_SOURCE_RAW}" >&2
    exit 2
    ;;
esac

export MJ_DEPTH_CAMERA_SOURCE="${DEPTH_SOURCE}"
export DEPTH_CHECKPOINT_DEFAULT="${DEPTH_CHECKPOINT_DEFAULT:-wandb://zihanw22/boxer/0z2aggr2/model_05000.pt}"
export INFER_DATASET="${INFER_DATASET:-omomo}"
export NUM_ENVS="${NUM_ENVS:-1}"
export HEADLESS="${HEADLESS:-True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
export VISER_ENABLE_CLIP_GUI="${VISER_ENABLE_CLIP_GUI:-0}"
export VISER_ENABLE_MANUAL_GUI="${VISER_ENABLE_MANUAL_GUI:-1}"
export VISER_MANUAL_CONTROL_DEFAULT="${VISER_MANUAL_CONTROL_DEFAULT:-0}"
export VISER_ENABLE_OBJECT_RESET_OVERRIDE="${VISER_ENABLE_OBJECT_RESET_OVERRIDE:-0}"
export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-1}"
export VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
export VISER_PERCEPTION_DEPTH_SOURCE="${VISER_PERCEPTION_DEPTH_SOURCE:-obs}"
export VISER_PERCEPTION_FLIP_VERTICAL="${VISER_PERCEPTION_FLIP_VERTICAL:-0}"
export START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-1.0}"
export FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}"

case "${DEPTH_SOURCE}" in
  rendered)
    export DEPTH_PERCEPTION_PRESET="${DEPTH_PERCEPTION_PRESET:-checkpoint}"
    export HOLOSOMA_ENABLE_CAMERA_TERRAIN_PROXY="${HOLOSOMA_ENABLE_CAMERA_TERRAIN_PROXY:-0}"
    PERCEPTION_ARGS=(
      --perception.camera_source rendered
    )
    ;;
  warp)
    export DEPTH_PERCEPTION_PRESET="${DEPTH_PERCEPTION_PRESET:-checkpoint}"
    export HOLOSOMA_ENABLE_CAMERA_TERRAIN_PROXY="${HOLOSOMA_ENABLE_CAMERA_TERRAIN_PROXY:-0}"
    PERCEPTION_ARGS=(
      --perception.camera_source far_tracking_warp
    )
    ;;
esac

MUJOCO_BACKEND_RAW="${MUJOCO_BACKEND:-CLASSIC}"
MUJOCO_BACKEND="$(echo "${MUJOCO_BACKEND_RAW}" | tr '[:lower:]' '[:upper:]')"
case "${MUJOCO_BACKEND}" in
  CLASSIC|WARP) ;;
  *)
    echo "[ERROR] MUJOCO_BACKEND must be one of: CLASSIC|WARP. Got: ${MUJOCO_BACKEND_RAW}" >&2
    exit 2
    ;;
esac

CMD=(bash "${SCRIPT_DIR}/infer_box_joystick.sh" depth)
if [[ -n "${CKPT_ARG}" ]]; then
  CMD+=("${CKPT_ARG}")
fi
CMD+=(
  simulator:mujoco
  --simulator.config.mujoco-backend "${MUJOCO_BACKEND}"
  --simulator.config.robot-mjcf-filter.enable True
  --simulator.config.robot-mjcf-filter.remove-lights True
  --simulator.config.robot-mjcf-filter.remove-ground True
  --robot.object.mujoco-use-training-urdf-scene True
  --robot.object.mujoco-add-default-actuators True
  --robot.object.mujoco-copy-joint-defaults-from-robot-xml True
  --robot.object.mujoco-copy-tendons-from-robot-xml True
  --robot.object.mujoco-copy-collision-geoms-from-robot-xml True
  --robot.object.mujoco-copy-contact-pairs-from-robot-xml True
  "${PERCEPTION_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

echo "[INFO] launcher=mj_depth.sh"
echo "[INFO] checkpoint_default=${DEPTH_CHECKPOINT_DEFAULT}"
echo "[INFO] depth_source=${MJ_DEPTH_CAMERA_SOURCE}"
echo "[INFO] depth_perception_preset=${DEPTH_PERCEPTION_PRESET}"
echo "[INFO] infer_dataset=${INFER_DATASET}"
echo "[INFO] num_envs=${NUM_ENVS}"
echo "[INFO] headless=${HEADLESS}"
echo "[INFO] mujoco_backend=${MUJOCO_BACKEND}"
echo "[INFO] python_bin=${PYTHON_BIN}"
echo "[INFO] command=${CMD[*]}"

exec "${CMD[@]}"
