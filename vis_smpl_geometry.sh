#!/usr/bin/env bash
set -euo pipefail

# Viser: joint positions viewer (global_joint_positions) + geometry
#
# Usage:
#   MOTION_DIR=/abs/path/to/joint_npz_folder GEOMETRY_DIR=/abs/path/to/obj_or_dir ./vis_smpl_geometry.sh
#
# Optional overrides:
#   PORT=#### START_CLIP=name FPS=30 AUTOPLAY=True LOOP=True PRELOAD=True
#   POINT_SIZE=0.07 POINT_SHAPE=circle SHOW_GEOMETRY=True GRID=True GRID_SIZE=10.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MOTION_DIR=${MOTION_DIR:-"/home/ubuntu/FAR/holosoma/crisp/vmm_data/motion/vmm_24.npz"}
GEOMETRY_DIR=${GEOMETRY_DIR:-"/home/ubuntu/FAR/holosoma/crisp/vmm_data/geo/obj/vmm_24/scene_mesh_sqs.obj"}
PORT=${PORT:-"$((RANDOM % 8976 + 1024))"}
START_CLIP=${START_CLIP:-""}
FPS=${FPS:-""}
AUTOPLAY=${AUTOPLAY:-"True"}
LOOP=${LOOP:-"True"}
PRELOAD=${PRELOAD:-"True"}
POINT_SIZE=${POINT_SIZE:-"0.07"}
POINT_SHAPE=${POINT_SHAPE:-"circle"}
SHOW_GEOMETRY=${SHOW_GEOMETRY:-"True"}
GRID=${GRID:-"True"}
GRID_SIZE=${GRID_SIZE:-"10.0"}

cmd=(
  python -m holosoma.visualize joints
  --motion-dir "${MOTION_DIR}"
  --geometry-dir "${GEOMETRY_DIR}"
  --port "${PORT}"
  --autoplay "${AUTOPLAY}"
  --loop "${LOOP}"
  --preload "${PRELOAD}"
  --point-size "${POINT_SIZE}"
  --point-shape "${POINT_SHAPE}"
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
