#!/usr/bin/env bash
set -euo pipefail

# One-step terrain-aware debug replay (2x2) with sampled points in Isaac Sim.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export HEADLESS="${HEADLESS:-False}"
export NUM_ENVS="${NUM_ENVS:-4}"
export TERRAIN_NUM_ROWS="${TERRAIN_NUM_ROWS:-2}"
export TERRAIN_NUM_COLS="${TERRAIN_NUM_COLS:-2}"
export VISER_ENV_ID="${VISER_ENV_ID:-0}"
export VISER_SHOW_SCANDOTS="${VISER_SHOW_SCANDOTS:-True}"
export ISAAC_SHOW_SCANDOTS="${ISAAC_SHOW_SCANDOTS:-True}"
export ISAAC_SCANDOTS_INCLUDE_MISSES="${ISAAC_SCANDOTS_INCLUDE_MISSES:-0}"
export SCANDOTS_STRIDE="${SCANDOTS_STRIDE:-1}"
export MOTION_PAIR_TERRAIN_WITH_MOTION="${MOTION_PAIR_TERRAIN_WITH_MOTION:-False}"

exec bash "${SCRIPT_DIR}/vis_depth_replay.sh" "$@"
