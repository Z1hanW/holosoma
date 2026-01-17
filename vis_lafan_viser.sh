#!/usr/bin/env bash
set -euo pipefail

# Run Viser motion replay on LAFAN clips (no physics).
#
# Usage:
#   ./vis_lafan_viser.sh
#
# Optional overrides:
#   EXP_CFG=exp:g1-29dof-wbt-videomimic-mlp
#   MOTION_DIR=/abs/path/to/holosoma/src/holosoma_retargeting/converted_res/robot_only/lafan
#   MOTION_CLIP_NAME=clip_0001
#   MOTION_CLIP_ID=0
#   START_AT_TIMESTEP_ZERO_PROB=0.05
#   ENABLE_DEFAULT_POSE_APPEND=False
#   DEFAULT_POSE_APPEND_DURATION_S=0
#   ENABLE_DEFAULT_POSE_PREPEND=False
#   DEFAULT_POSE_PREPEND_DURATION_S=0
#   NOISE_TO_INITIAL_POSE=0.01
#   TERRAIN_OBJ_PATH=/abs/path/to/obj_dir

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP_CFG=${EXP_CFG:-"exp:g1-29dof-wbt-videomimic-mlp"}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/robot_only/lafan"}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-""}
MOTION_CLIP_ID=${MOTION_CLIP_ID:-""}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-""}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-""}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-""}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-""}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-""}
NOISE_TO_INITIAL_POSE=${NOISE_TO_INITIAL_POSE:-""}
TERRAIN_OBJ_PATH=${TERRAIN_OBJ_PATH:-""}

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "Set MOTION_DIR to a valid directory of LAFAN .npz clips." >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

cmd=(
  python src/holosoma/holosoma/viser_replay.py
  "${EXP_CFG}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}")
fi
if [[ -n "${MOTION_CLIP_ID}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_id "${MOTION_CLIP_ID}")
fi
if [[ -n "${START_AT_TIMESTEP_ZERO_PROB}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob
    "${START_AT_TIMESTEP_ZERO_PROB}"
  )
fi
if [[ -n "${ENABLE_DEFAULT_POSE_APPEND}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append
    "${ENABLE_DEFAULT_POSE_APPEND}"
  )
fi
if [[ -n "${DEFAULT_POSE_APPEND_DURATION_S}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s
    "${DEFAULT_POSE_APPEND_DURATION_S}"
  )
fi
if [[ -n "${ENABLE_DEFAULT_POSE_PREPEND}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend
    "${ENABLE_DEFAULT_POSE_PREPEND}"
  )
fi
if [[ -n "${DEFAULT_POSE_PREPEND_DURATION_S}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s
    "${DEFAULT_POSE_PREPEND_DURATION_S}"
  )
fi
if [[ -n "${NOISE_TO_INITIAL_POSE}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale
    "${NOISE_TO_INITIAL_POSE}"
  )
fi
if [[ -n "${TERRAIN_OBJ_PATH}" ]]; then
  cmd+=(terrain:terrain-load-obj --terrain.terrain-term.obj-file-path "${TERRAIN_OBJ_PATH}")
fi

"${cmd[@]}"
