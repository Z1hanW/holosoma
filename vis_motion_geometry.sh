#!/usr/bin/env bash
set -euo pipefail

# Viser: motion + geometry viewer
#
# Usage:
#   DATASET_KNOB=crisp ./vis_motion_geometry.sh
#   DATASET_KNOB=omomo ./vis_motion_geometry.sh
#   DATASET_KNOB=behave ./vis_motion_geometry.sh
#   MOTION_DIR=/ABS/PATH/to/motions GEOMETRY_DIR=/ABS/PATH/to/geometry ./vis_motion_geometry.sh
#
# Optional overrides:
#   DATASET_KNOB=crisp|omomo|behave (default: behave)
#   ROBOT=g1_29dof PORT=#### START_CLIP=clip_name FPS=30 AUTOPLAY=True LOOP=True PRELOAD=True
#   SHOW_MESHES=True SHOW_GEOMETRY=True GRID=True GRID_SIZE=10.0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_KNOB=${DATASET_KNOB:-"behave"}

case "${DATASET_KNOB}" in
  crisp)
    DEFAULT_MOTION_DIR="${SCRIPT_DIR}/crisp/vmm_data/___crisp_motion"
    DEFAULT_GEOMETRY_DIR="${SCRIPT_DIR}/crisp/vmm_data/___crisp_geometry"
    DEFAULT_OBJECT_URDF_DIR="${SCRIPT_DIR}/crisp/vmm_data/___crisp_object_urdf"
    DEFAULT_OBJECT_URDF_MODE="stem"
    DEFAULT_OBJECT_URDF=""
    ;;
  omomo)
    DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry"
    DEFAULT_GEOMETRY_DIR=""
    DEFAULT_OBJECT_URDF_DIR=""
    DEFAULT_OBJECT_URDF_MODE="stem"
    DEFAULT_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma_retargeting/models/largebox/largebox.urdf"
    ;;
  behave)
    DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/demo_results/g1/object_interaction/behave_zup"
    DEFAULT_GEOMETRY_DIR=""
    DEFAULT_OBJECT_URDF_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/models/behave_objects"
    DEFAULT_OBJECT_URDF_MODE="behave"
    DEFAULT_OBJECT_URDF=""
    ;;
  *)
    echo "[ERROR] Unknown DATASET_KNOB=${DATASET_KNOB}. Use crisp|omomo|behave." >&2
    exit 1
    ;;
esac

# Make dataset knob authoritative for motion/geometry/object source.
MOTION_DIR="${DEFAULT_MOTION_DIR}"
GEOMETRY_DIR="${DEFAULT_GEOMETRY_DIR}"
OBJECT_URDF_DIR="${DEFAULT_OBJECT_URDF_DIR}"
OBJECT_URDF_MODE="${DEFAULT_OBJECT_URDF_MODE}"
OBJECT_URDF="${DEFAULT_OBJECT_URDF}"
echo "[INFO] DATASET_KNOB=${DATASET_KNOB} motion=${MOTION_DIR} geometry=${GEOMETRY_DIR} object_urdf=${OBJECT_URDF} object_urdf_dir=${OBJECT_URDF_DIR} object_mode=${OBJECT_URDF_MODE}"

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

object_mode_lc=$(printf "%s" "${OBJECT_URDF_MODE}" | tr '[:upper:]' '[:lower:]')
object_mode_recursive="false"
if [[ "${object_mode_lc}" == "recursive" || "${object_mode_lc}" == "behave" ]]; then
  object_mode_recursive="true"
fi

if [[ -n "${OBJECT_URDF_DIR}" ]]; then
  if [[ ! -d "${OBJECT_URDF_DIR}" ]]; then
    echo "[WARN] object urdf dir not found: ${OBJECT_URDF_DIR} (disabling object)"
    OBJECT_URDF_DIR=""
    SHOW_OBJECT="False"
  else
    if [[ "${object_mode_recursive}" == "true" ]]; then
      _urdf_files=()
      while IFS= read -r -d '' _f; do
        _urdf_files+=("$_f")
      done < <(find "${OBJECT_URDF_DIR}" -type f \( -name "*.urdf" -o -name "*.URDF" \) -print0)
    else
      shopt -s nullglob
      _urdf_files=("${OBJECT_URDF_DIR}"/*.urdf "${OBJECT_URDF_DIR}"/*.URDF)
      shopt -u nullglob
    fi
    if (( ${#_urdf_files[@]} == 0 )); then
      echo "[WARN] object urdf dir is empty: ${OBJECT_URDF_DIR} (disabling object)"
      OBJECT_URDF_DIR=""
      SHOW_OBJECT="False"
    elif (( ${#_urdf_files[@]} == 1 )) && [[ "${object_mode_recursive}" == "false" ]]; then
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
  --object-urdf-mode "${OBJECT_URDF_MODE}"
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
