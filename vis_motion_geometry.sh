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

MOTION_DIR=${MOTION_DIR:-"/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo"}
GEOMETRY_DIR=${GEOMETRY_DIR:-""}
OBJECT_URDF_DIR=${OBJECT_URDF_DIR:-"/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/largebox/largebox.urdf"}

OBJECT_URDF=${OBJECT_URDF:-""}
ROBOT=${ROBOT:-"g1_29dof"}
PORT=${PORT:-"$((RANDOM % 8976 + 1024))"}
START_CLIP=${START_CLIP:-""}
FPS=${FPS:-""}
AUTOPLAY=${AUTOPLAY:-"True"}
LOOP=${LOOP:-"True"}
PRELOAD=${PRELOAD:-"True"}
SHOW_MESHES=${SHOW_MESHES:-"True"}
SHOW_GEOMETRY=${SHOW_GEOMETRY:-"True"}
SHOW_OBJECT=${SHOW_OBJECT:-"True"}
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

if [[ -n "${OBJECT_URDF_DIR}" ]]; then
  if [[ -f "${OBJECT_URDF_DIR}" ]]; then
    # Single URDF provided via OBJECT_URDF_DIR; treat as common object.
    OBJECT_URDF="${OBJECT_URDF_DIR}"
    OBJECT_URDF_DIR=""
  elif [[ ! -d "${OBJECT_URDF_DIR}" ]]; then
    echo "[WARN] object urdf dir not found: ${OBJECT_URDF_DIR} (disabling object)"
    OBJECT_URDF_DIR=""
    SHOW_OBJECT="False"
  else
    shopt -s nullglob
    _urdf_files=("${OBJECT_URDF_DIR}"/*.urdf "${OBJECT_URDF_DIR}"/*.URDF)
    shopt -u nullglob
    if (( ${#_urdf_files[@]} == 0 )); then
      echo "[WARN] object urdf dir is empty: ${OBJECT_URDF_DIR} (disabling object)"
      OBJECT_URDF_DIR=""
      SHOW_OBJECT="False"
    elif (( ${#_urdf_files[@]} == 1 )); then
      # Single URDF in directory; treat as common object.
      OBJECT_URDF="${_urdf_files[0]}"
      OBJECT_URDF_DIR=""
    fi
  fi
fi

if [[ -n "${OBJECT_URDF}" ]]; then
  if [[ ! -f "${OBJECT_URDF}" ]]; then
    echo "[WARN] object urdf not found: ${OBJECT_URDF} (disabling object)"
    OBJECT_URDF=""
    SHOW_OBJECT="False"
  fi
fi



cmd=(
  python3 src/holosoma/holosoma/viser_motion_geometry.py
  --motion-dir "${MOTION_DIR}"
  --geometry-dir "${GEOMETRY_DIR}"
  --object-urdf "${OBJECT_URDF}"
  --object-urdf-dir "${OBJECT_URDF_DIR}"
  --robot "${ROBOT}"
  --port "${PORT}"
  --autoplay "${AUTOPLAY}"
  --loop "${LOOP}"
  --preload "${PRELOAD}"
  --show-meshes "${SHOW_MESHES}"
  --show-geometry "${SHOW_GEOMETRY}"
  --show-object "${SHOW_OBJECT}"
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
