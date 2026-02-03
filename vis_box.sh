#!/usr/bin/env bash
set -euo pipefail

# Visualize an object-interaction policy with Viser.
# Usage:
#   CKPT=/abs/path/to/model.pt ./vis_box.sh

CKPT=${CKPT:-"/ABS/PATH/to/model.pt"}
MOTION_DIR=${MOTION_DIR:-"src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"}
DEPTH_IMPL=${DEPTH_IMPL:-none}
NUM_ENVS=${NUM_ENVS:-1}

if [[ "${CKPT}" == "/ABS/PATH/to/model.pt" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi

CKPT="${CKPT}" \
MOTION_DIR="${MOTION_DIR}" \
DEPTH_IMPL="${DEPTH_IMPL}" \
NUM_ENVS="${NUM_ENVS}" \
./vis.sh
