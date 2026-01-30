#!/usr/bin/env bash
set -euo pipefail

# Physics rollout with Viser using VideoMimic-style heightmap perception.
#
# Required:
#   CKPT=/abs/path/to/model.pt
#   MOTION_DIR=/abs/path/to/motion_folder_or_file
#
# Optional:
#   GEOMETRY_DIR=/abs/path/to/obj_dir_or_obj_file
#   GEOMETRY_META=/abs/path/to/metadata.json
#   NUM_ENVS=4
#   HEADLESS=True
#   NUM_ROWS=1
#   NUM_COLS=
#   ENV_SPACING=0.0
#   PAIR_TERRAIN=True
#   VISER_PORT=####
#   VISER_ENV_ID=0
#   VISER_UPDATE_HZ=30
#   VISER_RECENTER=False
#   VISER_SHOW_SCANDOTS=True
#
# Heightmap (VideoMimic-style):
#   HEIGHTMAP_SIZE=1.0        # meters (length, width)
#   HEIGHTMAP_RESOLUTION=0.1  # meters (=> 11x11 when size=1.0)
#   RAY_START_HEIGHT=0.0
#   MAX_DISTANCE=5.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT=${CKPT:-"../Store/model_52000.pt"}
MOTION_DIR=${MOTION_DIR:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_data_trans"}
GEOMETRY_DIR=${GEOMETRY_DIR:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_geo_trans"}
GEOMETRY_META=${GEOMETRY_META:-""}
NUM_ENVS=${NUM_ENVS:-4}
HEADLESS=${HEADLESS:-True}
NUM_ROWS=${NUM_ROWS:-1}
NUM_COLS=${NUM_COLS:-}
ENV_SPACING=${ENV_SPACING:-0.0}
PAIR_TERRAIN=${PAIR_TERRAIN:-True}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-False}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-True}
HEIGHTMAP_SIZE=${HEIGHTMAP_SIZE:-1.0}
HEIGHTMAP_RESOLUTION=${HEIGHTMAP_RESOLUTION:-0.1}
RAY_START_HEIGHT=${RAY_START_HEIGHT:-0.0}
MAX_DISTANCE=${MAX_DISTANCE:-5.0}

if [[ "${CKPT}" == "/ABS/PATH/to/model.pt" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi
if [[ "${MOTION_DIR}" == "/ABS/PATH/to/motion_folder_or_file" ]]; then
  echo "Set MOTION_DIR to your motion folder path." >&2
  exit 1
fi

echo "[INFO] Viser port: ${VISER_PORT}"

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS}"
  --num-rows "${NUM_ROWS}"
  --simulator.config.scene.env_spacing "${ENV_SPACING}"
  --pair-terrain-with-motion "${PAIR_TERRAIN}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --training.viser-show-scandots "${VISER_SHOW_SCANDOTS}"
  perception:heightmap
  --perception.heightmap_size "(${HEIGHTMAP_SIZE}, ${HEIGHTMAP_SIZE})"
  --perception.heightmap_resolution "${HEIGHTMAP_RESOLUTION}"
  --perception.ray_start_height "${RAY_START_HEIGHT}"
  --perception.max_distance "${MAX_DISTANCE}"
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
