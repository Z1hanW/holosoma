#!/usr/bin/env bash
set -euo pipefail

# Physics rollout with Viser (checkpoint + motion, optional geometry).
#
# Required:
#   CKPT=/abs/path/to/model.pt
#   MOTION_DIR=/abs/path/to/motion_folder
#
# Optional:
#   GEOMETRY_DIR=/abs/path/to/obj_dir_or_obj_file
#   GEOMETRY_META=/abs/path/to/metadata.json
#   NUM_ENVS=1
#   HEADLESS=True
#   NUM_ROWS=1
#   NUM_COLS=
#   PAIR_TERRAIN=True
#   VISER_PORT=####
#   VISER_ENV_ID=0
#   VISER_UPDATE_HZ=30
#   VISER_RECENTER=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT=${CKPT:-"/ABS/PATH/to/model.pt"}
MOTION_DIR=${MOTION_DIR:-"/ABS/PATH/to/motion_folder"}
GEOMETRY_DIR=${GEOMETRY_DIR:-""}
GEOMETRY_META=${GEOMETRY_META:-""}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-True}
NUM_ROWS=${NUM_ROWS:-1}
NUM_COLS=${NUM_COLS:-}
PAIR_TERRAIN=${PAIR_TERRAIN:-True}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}

if [[ "${CKPT}" == "/ABS/PATH/to/model.pt" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi
if [[ "${MOTION_DIR}" == "/ABS/PATH/to/motion_folder" ]]; then
  echo "Set MOTION_DIR to your motion folder path." >&2
  exit 1
fi

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS}"
  --num-rows "${NUM_ROWS}"
  --pair-terrain-with-motion "${PAIR_TERRAIN}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
)

if [[ -n "${NUM_COLS}" ]]; then
  cmd+=(--num-cols "${NUM_COLS}")
fi
if [[ -n "${GEOMETRY_DIR}" ]]; then
  cmd+=(--geometry-dir "${GEOMETRY_DIR}")
fi
if [[ -n "${GEOMETRY_META}" ]]; then
  cmd+=(--geometry-metadata "${GEOMETRY_META}")
fi

"${cmd[@]}"
