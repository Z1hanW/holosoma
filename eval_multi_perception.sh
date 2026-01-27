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
  perception:camera-depth-d435i-scandots \
  --perception.camera_width=128 \
  --perception.camera_height=72 \
  --simulator.config.sim.physx.gpu_collision_stack_size=4294967295 \
  --training.num_envs=1 \
  --training.headless=False \
  --training.enable_viser=True \
  --training.viser_show_scandots=True \
  --simulator.config.scene.env_spacing=0.0 \
  --algo.config.load_optimizer=False \
  --algo.config.save_interval=10000 \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}" \
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=False \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path "${OBJ_DIR}" \
  --terrain.terrain-term.obj-metadata-path ""
