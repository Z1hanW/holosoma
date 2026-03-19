#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash sim_joystick.sh [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/.../files] [extra infer_box_joystick args...]

Purpose:
  Launch the depth distill box-carry policy in MuJoCo and visualize/control it through the
  existing inference-side viser workflow.

Defaults:
  checkpoint = wandb://zihanw22/boxer/0z2aggr2/model_05000.pt
  simulator  = mujoco
  backend    = classic
  dataset    = omomo
  num_envs   = 1
  headless   = True

Useful env vars:
  DEPTH_CHECKPOINT_DEFAULT  Override the default checkpoint.
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
  bash sim_joystick.sh
  VISER_PORT=18080 bash sim_joystick.sh
  HEADLESS=False bash sim_joystick.sh /abs/path/model_05000.pt
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
if [[ $# -gt 0 ]]; then
  case "$1" in
    wandb://*|https://wandb.ai/*|/*|./*|../*|*.pt)
      CKPT_ARG="$1"
      shift
      ;;
  esac
fi

export DEPTH_CHECKPOINT_DEFAULT="${DEPTH_CHECKPOINT_DEFAULT:-wandb://zihanw22/boxer/0z2aggr2/model_05000.pt}"
export INFER_DATASET="${INFER_DATASET:-omomo}"
export NUM_ENVS="${NUM_ENVS:-1}"
export HEADLESS="${HEADLESS:-True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export DEPTH_PERCEPTION_PRESET="${DEPTH_PERCEPTION_PRESET:-checkpoint}"
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
  --perception.camera_source rendered
  --robot.object.mujoco-use-training-urdf-scene True
  --robot.object.mujoco-add-default-actuators True
  --robot.object.mujoco-copy-joint-defaults-from-robot-xml True
  --robot.object.mujoco-copy-tendons-from-robot-xml True
  --robot.object.mujoco-copy-collision-geoms-from-robot-xml True
  --robot.object.mujoco-copy-contact-pairs-from-robot-xml True
  "$@"
)

echo "[INFO] launcher=sim_joystick.sh"
echo "[INFO] checkpoint_default=${DEPTH_CHECKPOINT_DEFAULT}"
echo "[INFO] infer_dataset=${INFER_DATASET}"
echo "[INFO] num_envs=${NUM_ENVS}"
echo "[INFO] headless=${HEADLESS}"
echo "[INFO] mujoco_backend=${MUJOCO_BACKEND}"
echo "[INFO] python_bin=${PYTHON_BIN}"
echo "[INFO] command=${CMD[*]}"

exec "${CMD[@]}"
