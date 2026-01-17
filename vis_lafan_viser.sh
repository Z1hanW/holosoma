#!/usr/bin/env bash
set -euo pipefail

# Visualize multi-clip LAFAN motions in Viser (motion replay, no physics).
#
# Usage:
#   ./vis_lafan_viser.sh
#
# Optional overrides:
#   EXP_CFG=exp:g1-29dof-wbt-videomimic-mlp
#   MOTION_DIR=/ABS/PATH/to/holosoma/src/holosoma_retargeting/converted_res/robot_only/lafan
#   HOLOSOMA_VISER_PORT=6060

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_CFG=${EXP_CFG:-"exp:g1-29dof-wbt-videomimic-mlp"}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/robot_only/lafan"}

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi

cmd=(
  python src/holosoma/holosoma/viser_replay.py
  "${EXP_CFG}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
  # --command.setup_terms.motion_command.params.motion_config.motion_clip_name clip_0001
  # --command.setup_terms.motion_command.params.motion_config.motion_clip_id 0
  # --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob 0.05
  # --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False
  # --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s 0
  # --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False
  # --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s 0
  # --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale 0.01
  # terrain:terrain-load-obj
  # --terrain.terrain-term.obj-file-path /abs/path/to/obj_dir
)

"${cmd[@]}"
