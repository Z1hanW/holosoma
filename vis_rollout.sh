#!/usr/bin/env bash
set -euo pipefail

# Visualize physics rollouts with Viser (config-first like eval_agent.py).
#
# Usage:
#   ROLLOUT_DIR=/ABS/PATH/to/rollouts ./vis_rollout.sh
#
# Optional overrides:
#   EXP_CFG=exp:g1-29dof-wbt-videomimic-mlp
#   HOLOSOMA_VISER_PORT=6060
#   ROLLOUT_FILE=/ABS/PATH/to/rollout_0001.npz
#   TERRAIN_OBJ_PATH=/ABS/PATH/to/obj_dir
#   RECENTER=False

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_CFG=${EXP_CFG:-"exp:g1-29dof-wbt-videomimic-mlp"}
ROLLOUT_DIR=${ROLLOUT_DIR:-"/ABS/PATH/to/rollouts"}

if [[ "${ROLLOUT_DIR}" == "/ABS/PATH/to/rollouts" ]]; then
  echo "Set ROLLOUT_DIR to your rollout directory." >&2
  exit 1
fi

cmd=(
  python src/holosoma/holosoma/viser_rollout.py
  "${EXP_CFG}"
  --rollout-dir "${ROLLOUT_DIR}"
  # --rollout-file "${ROLLOUT_FILE}"
  # --terrain-obj-path "${TERRAIN_OBJ_PATH}"
  # --recenter "${RECENTER}"
)

"${cmd[@]}"
