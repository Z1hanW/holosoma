#!/usr/bin/env bash
set -euo pipefail

# Evaluate a multi-perception policy on a single motion + single terrain (Isaac Sim GUI).
#
# Required:
#   CKPT=/abs/path/to/model.pt
#
# Optional overrides:
#   MOTION_FILE=../far_data/multi-motion/test/motion.npz
#   OBJ_DIR=../far_data/multi-terrain/test/terrain.obj

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT=${CKPT:-""}
if [[ -z "${CKPT}" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi

MOTION_FILE=${MOTION_FILE:-"../far_data/multi-motion/test/motion.npz"}
OBJ_DIR=${OBJ_DIR:-"../far_data/multi-terrain/test/terrain.obj"}

python src/holosoma/holosoma/eval_agent.py \
  --checkpoint "${CKPT}" \
  exp:g1-29dof-wbt-videomimic-mlp \
  --training.num_envs=1 \
  --training.headless=False \
  --simulator.config.scene.env_spacing=0.0 \
  --algo.config.load_optimizer=False \
  --algo.config.save_interval=10000 \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}" \
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=False \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path "${OBJ_DIR}"
