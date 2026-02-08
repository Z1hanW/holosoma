#!/usr/bin/env bash
set -euo pipefail

# Viser: motion + geometry viewer
#
# Usage:
#   MOTION_DIR=/ABS/PATH/to/motions GEOMETRY_DIR=/ABS/PATH/to/geometry ./vis_motion_geometry.sh
#
# Optional overrides:
#   ROBOT=g1_29dof PORT=#### START_CLIP=clip_name FPS=30 AUTOPLAY=True LOOP=True PRELOAD=True
#   SHOW_MESHES=True SHOW_GEOMETRY=True GRID=True GRID_SIZE=10.0

MOTION_DIR=${MOTION_DIR:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_data_trans"}
GEOMETRY_DIR=${GEOMETRY_DIR:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_geo_trans"}
ROBOT=${ROBOT:-"g1_29dof"}
PORT=${PORT:-"$((RANDOM % 8976 + 1024))"}
START_CLIP=${START_CLIP:-""}
FPS=${FPS:-""}
AUTOPLAY=${AUTOPLAY:-"True"}
LOOP=${LOOP:-"True"}
PRELOAD=${PRELOAD:-"True"}
SHOW_MESHES=${SHOW_MESHES:-"True"}
SHOW_GEOMETRY=${SHOW_GEOMETRY:-"True"}
GRID=${GRID:-"True"}
GRID_SIZE=${GRID_SIZE:-"10.0"}

if [[ -z "${GEOMETRY_DIR}" ]]; then
  SHOW_GEOMETRY="False"
else
  if [[ ! -d "${GEOMETRY_DIR}" ]]; then
    echo "[WARN] geometry dir not found: ${GEOMETRY_DIR} (falling back to ground)"
    GEOMETRY_DIR=""
    SHOW_GEOMETRY="False"
  else
    shopt -s nullglob
    _geom_files=("${GEOMETRY_DIR}"/*.obj "${GEOMETRY_DIR}"/*.OBJ)
    shopt -u nullglob
    if (( ${#_geom_files[@]} == 0 )); then
      echo "[WARN] geometry dir is empty: ${GEOMETRY_DIR} (falling back to ground)"
      GEOMETRY_DIR=""
      SHOW_GEOMETRY="False"
    fi
  fi
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
