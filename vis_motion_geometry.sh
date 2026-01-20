#!/usr/bin/env bash
set -euo pipefail

# Viser: motion + geometry viewer
#
# Usage:
#   MOTION_DIR=/ABS/PATH/to/motions GEOMETRY_DIR=/ABS/PATH/to/geometry ./vis_motion_geometry.sh
#
# Optional overrides:
#   ROBOT=g1_29dof PORT=6060 START_CLIP=clip_name FPS=30 AUTOPLAY=True LOOP=True PRELOAD=True
#   SHOW_MESHES=True SHOW_GEOMETRY=True GRID=True GRID_SIZE=10.0

MOTION_DIR=${MOTION_DIR:-"/ABS/PATH/to/motions"}
GEOMETRY_DIR=${GEOMETRY_DIR:-"/ABS/PATH/to/geometry"}
ROBOT=${ROBOT:-"g1_29dof"}
PORT=${PORT:-"6060"}
START_CLIP=${START_CLIP:-""}
FPS=${FPS:-""}
AUTOPLAY=${AUTOPLAY:-"True"}
LOOP=${LOOP:-"True"}
PRELOAD=${PRELOAD:-"True"}
SHOW_MESHES=${SHOW_MESHES:-"True"}
SHOW_GEOMETRY=${SHOW_GEOMETRY:-"True"}
GRID=${GRID:-"True"}
GRID_SIZE=${GRID_SIZE:-"10.0"}

if [[ "${MOTION_DIR}" == "/ABS/PATH/to/motions" || "${GEOMETRY_DIR}" == "/ABS/PATH/to/geometry" ]]; then
  echo "Set MOTION_DIR and GEOMETRY_DIR." >&2
  exit 1
fi

cmd=(
  python3 src/holosoma/holosoma/viser_motion_geometry.py
  --motion-dir "${MOTION_DIR}"
  --geometry-dir "${GEOMETRY_DIR}"
  --robot "${ROBOT}"
  --port "${PORT}"
  --autoplay "${AUTOPLAY}"
  --loop "${LOOP}"
  --preload "${PRELOAD}"
  --show-meshes "${SHOW_MESHES}"
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

"${cmd[@]}"
