#!/usr/bin/env bash
set -euo pipefail

# Slope-specific inference wrapper.
# Defaults to the training slope terrain, slope motion, and the latest local
# slope checkpoint. All args/env vars are still forwarded to
# infer_terrain_tracking.sh and can override these defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_SLOPE_RUN_DIR="${DEFAULT_SLOPE_RUN_DIR:-/data/logs_new/terrain-aware/20260412_041344-g1_wbt_sanity_singleterrain_4gpu_1024per_20000_wandb_20260412-locomotion}"
DEFAULT_SLOPE_MOTION_DIR="${DEFAULT_SLOPE_MOTION_DIR:-${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking}"
DEFAULT_SLOPE_GEOMETRY_DIR="${DEFAULT_SLOPE_GEOMETRY_DIR:-${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj}"

find_latest_model_in_dir() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name 'model_*.pt' 2>/dev/null | sort -V | tail -n1
}

DEFAULT_SLOPE_CHECKPOINT="${DEFAULT_SLOPE_CHECKPOINT:-$(find_latest_model_in_dir "${DEFAULT_SLOPE_RUN_DIR}")}"
if [[ -z "${DEFAULT_SLOPE_CHECKPOINT}" || ! -f "${DEFAULT_SLOPE_CHECKPOINT}" ]]; then
  echo "[ERROR] slope checkpoint not found under: ${DEFAULT_SLOPE_RUN_DIR}" >&2
  exit 1
fi

export DEFAULT_TERRAIN_CHECKPOINT="${DEFAULT_TERRAIN_CHECKPOINT:-${CHECKPOINT:-${DEFAULT_SLOPE_CHECKPOINT}}}"
export DEFAULT_TERRAIN_MOTION_DIR="${DEFAULT_TERRAIN_MOTION_DIR:-${DEFAULT_SLOPE_MOTION_DIR}}"
export DEFAULT_TERRAIN_GEOMETRY_DIR="${DEFAULT_TERRAIN_GEOMETRY_DIR:-${DEFAULT_SLOPE_GEOMETRY_DIR}}"

export MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-motion_crawl_slope}"
export TERRAIN_SINGLE_GEOMETRY="${TERRAIN_SINGLE_GEOMETRY:-0}"

# Let infer_terrain_tracking.sh inherit the checkpoint default unless the caller
# explicitly overrides this for a special debug run.
if [[ -n "${PAIR_TERRAIN_WITH_MOTION+x}" ]]; then
  export PAIR_TERRAIN_WITH_MOTION
fi

# Keep the viewer compatible with the current IsaacSim environment.
export VISER_SHOW_SCANDOTS="${VISER_SHOW_SCANDOTS:-False}"
export VISER_DISABLE_CONTACT_FORCE_VIZ="${VISER_DISABLE_CONTACT_FORCE_VIZ:-1}"

exec "${SCRIPT_DIR}/infer_terrain_tracking.sh" "$@"
