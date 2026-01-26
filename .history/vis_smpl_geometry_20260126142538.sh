#!/usr/bin/env bash
set -euo pipefail

# Viser: SMPL human mesh + optional geometry viewer
#
# Usage:
#   MOTION_DIR=/abs/path/to/motions GEOMETRY_DIR=/abs/path/to/geometry_or_obj ./vis_smpl_geometry.sh
#
# Optional overrides:
#   SMPL_MODEL_PATH=/abs/path/to/SMPLX_NEUTRAL.pkl (or model root dir)
#   SMPL_MODEL_TYPE=smplx|smpl|smplh
#   PORT=#### START_CLIP=name FPS=30 AUTOPLAY=True LOOP=True PRELOAD=True
#   SHOW_MESH=True SHOW_JOINTS=False SHOW_GEOMETRY=True GRID=True GRID_SIZE=10.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MOTION_DIR=${MOTION_DIR:-"/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene/vmm_25/gv/hmr"}
GEOMETRY_DIR=${GEOMETRY_DIR:-"/home/ubuntu/FAR/holosoma/crisp/vmm_data/geo/obj/vmm_25/scene_mesh_sqs.obj"}
SMPL_MODEL_PATH=${SMPL_MODEL_PATH:-"/home/ubuntu/FAR/CRISP-Real2Sim/prep/data/smplx/models/smplx/SMPLX_NEUTRAL.pkl"}
SMPL_MODEL_TYPE=${SMPL_MODEL_TYPE:-"smplx"}
PORT=${PORT:-"$((RANDOM % 8976 + 1024))"}
START_CLIP=${START_CLIP:-""}
FPS=${FPS:-""}
AUTOPLAY=${AUTOPLAY:-"True"}
LOOP=${LOOP:-"True"}
PRELOAD=${PRELOAD:-"True"}
SHOW_MESH=${SHOW_MESH:-"True"}
SHOW_JOINTS=${SHOW_JOINTS:-"False"}
SHOW_GEOMETRY=${SHOW_GEOMETRY:-"True"}
GRID=${GRID:-"True"}
GRID_SIZE=${GRID_SIZE:-"10.0"}

cmd=(
  python -m holosoma.visualize smpl
  --motion-dir "${MOTION_DIR}"
  --geometry-dir "${GEOMETRY_DIR}"
  --smpl-model-path "${SMPL_MODEL_PATH}"
  --smpl-model-type "${SMPL_MODEL_TYPE}"
  --port "${PORT}"
  --autoplay "${AUTOPLAY}"
  --loop "${LOOP}"
  --preload "${PRELOAD}"
  --show-mesh "${SHOW_MESH}"
  --show-joints "${SHOW_JOINTS}"
  --show-geometry "${SHOW_GEOMETRY}"
  --add-grid "${GRID}"
  --grid-size "${GRID_SIZE}"
)

if [[ -n "${START_CLIP}" ]]; then
  cmd+=(--start-clip "${START_CLIP}")
fi

if [[ -n "${FPS}" ]]; then
  cmd+=(--fps "${FPS}")
fi

echo "[INFO] Viser port: ${PORT}"
"${cmd[@]}"
