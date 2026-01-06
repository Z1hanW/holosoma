#!/usr/bin/env bash
set -euo pipefail

# Run inference with Viser visualization (headless sim recommended).
#
# Usage:
#   MODEL_PATH=/ABS/PATH/to/model.onnx ./vis_viser.sh
#
# Optional overrides:
#   INFER_CFG=inference:g1-29dof-wbt
#   INTERFACE=lo RL_RATE=50 USE_SIM_TIME=True USE_JOYSTICK=False DOMAIN_ID=0
#   VISER_PORT=6060 VISER_UPDATE=1 VISER_URDF=/ABS/PATH/to/robot.urdf
#   VISER_SHOW_MESHES=True VISER_GRID=True VISER_GRID_SIZE=10.0

source scripts/source_inference_setup.sh

MODEL_PATH=${MODEL_PATH:-"/ABS/PATH/to/your_wbt_policy.onnx"}
INFER_CFG=${INFER_CFG:-"inference:g1-29dof-wbt"}
INTERFACE=${INTERFACE:-"lo"}
RL_RATE=${RL_RATE:-"50"}
USE_SIM_TIME=${USE_SIM_TIME:-"True"}
USE_JOYSTICK=${USE_JOYSTICK:-"False"}
DOMAIN_ID=${DOMAIN_ID:-"0"}
POLICY_ACTION_SCALE=${POLICY_ACTION_SCALE:-"0.25"}

VISER_PORT=${VISER_PORT:-"6060"}
VISER_UPDATE=${VISER_UPDATE:-"1"}
VISER_URDF=${VISER_URDF:-""}
VISER_SHOW_MESHES=${VISER_SHOW_MESHES:-"True"}
VISER_GRID=${VISER_GRID:-"True"}
VISER_GRID_SIZE=${VISER_GRID_SIZE:-"10.0"}

if [[ "${MODEL_PATH}" == "/ABS/PATH/to/your_wbt_policy.onnx" ]]; then
  echo "Set MODEL_PATH to your ONNX model path." >&2
  exit 1
fi

cmd=(
  python3 src/holosoma_inference/holosoma_inference/run_policy.py
  "${INFER_CFG}"
  --task.model-path "${MODEL_PATH}"
  --task.interface "${INTERFACE}"
  --task.rl-rate "${RL_RATE}"
  --task.use-sim-time "${USE_SIM_TIME}"
  --task.use-joystick "${USE_JOYSTICK}"
  --task.domain-id "${DOMAIN_ID}"
  --task.policy-action-scale "${POLICY_ACTION_SCALE}"
  --viser.enabled True
  --viser.port "${VISER_PORT}"
  --viser.update-interval "${VISER_UPDATE}"
  --viser.show-meshes "${VISER_SHOW_MESHES}"
  --viser.add-grid "${VISER_GRID}"
  --viser.grid-size "${VISER_GRID_SIZE}"
)

if [[ -n "${VISER_URDF}" ]]; then
  cmd+=(--viser.urdf-path "${VISER_URDF}")
fi

"${cmd[@]}"
