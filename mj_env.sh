#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}"

export GT_MUJOCO_PHYSICS="${GT_MUJOCO_PHYSICS:-1}"
export HOLOSOMA_GT_MUJOCO_PHYSICS="${HOLOSOMA_GT_MUJOCO_PHYSICS:-$GT_MUJOCO_PHYSICS}"
export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-1.4}"
export MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-[0.6,0.02,0.005]}"
export MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-[0.6,0.02,0.005]}"

ROBOT_CONFIG="${ROBOT_CONFIG:-g1-29dof-w-object}"
CAMERA_CONFIG="${CAMERA_CONFIG:-single_d435i_depth}"
IMAGE_SERVER_CONFIG="${IMAGE_SERVER_CONFIG:-mujoco_d435i}"
IMAGE_DEPTH_SOURCE="${IMAGE_DEPTH_SOURCE:-depth}"
IMAGE_ENABLE_GUM="${IMAGE_ENABLE_GUM:-False}"
IMAGE_ENABLE_RGB="${IMAGE_ENABLE_RGB:-False}"
IMAGE_VISUALIZE="${IMAGE_VISUALIZE:-True}"

if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  echo "[WARN] No DISPLAY/WAYLAND_DISPLAY is set; the OpenCV depth window cannot open." >&2
fi

echo "[INFO] robot=${ROBOT_CONFIG}"
echo "[INFO] camera=${CAMERA_CONFIG}"
echo "[INFO] image_server=${IMAGE_SERVER_CONFIG} depth_source=${IMAGE_DEPTH_SOURCE}"
echo "[INFO] gt_mujoco_physics=${GT_MUJOCO_PHYSICS} object_mass=${MUJOCO_OBJECT_MASS_OVERRIDE}"
echo "[INFO] object_friction=${MUJOCO_OBJECT_GEOM_FRICTION}"

exec "$PYTHON_BIN" src/holosoma/holosoma/run_sim.py \
  "robot:${ROBOT_CONFIG}" \
  "camera:${CAMERA_CONFIG}" \
  "image_server:${IMAGE_SERVER_CONFIG}" \
  --simulator.config.bridge.enabled=True \
  --image-server.depth-source "$IMAGE_DEPTH_SOURCE" \
  --image-server.enable-gum-depth-prediction "$IMAGE_ENABLE_GUM" \
  --image-server.enable-rgb "$IMAGE_ENABLE_RGB" \
  --image-server.visualize-images "$IMAGE_VISUALIZE" \
  "$@"
