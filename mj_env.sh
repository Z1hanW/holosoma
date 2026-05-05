#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

motion=""
if [[ $# -gt 0 && "${1:-}" != --* && "${1:-}" != *:* && "${1:-}" != wandb://* && "${1:-}" != https://* && "${1:-}" != *.onnx && "${1:-}" != *.pt ]]; then
  motion="$1"
  shift
fi

motion_args=()
if [[ -n "$motion" ]]; then
  if [[ "$motion" == *.npz || "$motion" == */* ]]; then
    motion_file="$motion"
  elif [[ -f "$ROOT_DIR/data_demo/${motion}.npz" ]]; then
    motion_file="$ROOT_DIR/data_demo/${motion}.npz"
  else
    motion_file="/home/user/FAR/holosoma/data_demo/${motion}.npz"
  fi
  motion_args=(
    --motion-init.enabled=True
    --motion-init.motion-file "$motion_file"
    --motion-init.mode "${SIM_MOTION_INIT_MODE:-raw_motion}"
    --motion-init.object-name object
  )
fi

ROBOT_CONFIG="${ROBOT_CONFIG:-g1-29dof-w-object}"
CAMERA_CONFIG="${CAMERA_CONFIG:-single_d435i_depth}"
IMAGE_SERVER_CONFIG="${IMAGE_SERVER_CONFIG:-mujoco_d435i}"
IMAGE_DEPTH_SOURCE="${IMAGE_DEPTH_SOURCE:-depth}"
IMAGE_ENABLE_GUM="${IMAGE_ENABLE_GUM:-False}"
IMAGE_ENABLE_RGB="${IMAGE_ENABLE_RGB:-False}"
IMAGE_VISUALIZE="${IMAGE_VISUALIZE:-True}"

exec "$PYTHON_BIN" src/holosoma/holosoma/run_sim.py \
  "robot:${ROBOT_CONFIG}" \
  "camera:${CAMERA_CONFIG}" \
  "image_server:${IMAGE_SERVER_CONFIG}" \
  --simulator.config.bridge.enabled=True \
  --image-server.depth-source "$IMAGE_DEPTH_SOURCE" \
  --image-server.enable-gum-depth-prediction "$IMAGE_ENABLE_GUM" \
  --image-server.enable-rgb "$IMAGE_ENABLE_RGB" \
  --image-server.visualize-images "$IMAGE_VISUALIZE" \
  "${motion_args[@]}" \
  "$@"
