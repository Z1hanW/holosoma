#!/usr/bin/env bash
set -euo pipefail

# Slope-specific single-card training launcher.
# Defaults mirror the current slope sanity run, except action_scale is bumped
# to 1.0 for a direct force/authority ablation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-g1-29dof-wbt}"
export PERCEPTION_PRESET="${PERCEPTION_PRESET:-none}"
export WANDB_PROJECT="${WANDB_PROJECT:-terrain-aware}"
export TRAINING_NAME="${TRAINING_NAME:-g1_wbt_slope_actionscale1_1024_singlecard}"
export LOGGER_NAME="${LOGGER_NAME:-g1_wbt_slope_actionscale1_1024_singlecard}"

export NPROC="${NPROC:-1}"
export PER_GPU_ENVS="${PER_GPU_ENVS:-1024}"
export NUM_ENVS="${NUM_ENVS:-1024}"

export MOTION_DIR="${MOTION_DIR:-${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz}"
export OBJ_SOURCE="${OBJ_SOURCE:-${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj}"
export OBJ_META_PATH="${OBJ_META_PATH:-}"
export NUM_ROWS="${NUM_ROWS:-10}"
export NUM_COLS="${NUM_COLS:-20}"

export PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION:-False}"
export USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-True}"
export ADD_GROUND_PLANE_COLLISION="${ADD_GROUND_PLANE_COLLISION:-False}"
export START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-0.2}"
export FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.95}"
export ENABLE_DEFAULT_POSE_APPEND="${ENABLE_DEFAULT_POSE_APPEND:-True}"
export DEFAULT_POSE_APPEND_DURATION_S="${DEFAULT_POSE_APPEND_DURATION_S:-2.0}"
export ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND:-True}"
export DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S:-2.0}"

export ACTOR_LR="${ACTOR_LR:-1e-3}"
export CRITIC_LR="${CRITIC_LR:-1e-3}"
export NORMALIZE_ACTOR_OBS="${NORMALIZE_ACTOR_OBS:-True}"
export NORMALIZE_CRITIC_OBS="${NORMALIZE_CRITIC_OBS:-True}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-4000}"
export LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-False}"

export BAD_TRACKING_REF_POS_THRESHOLD="${BAD_TRACKING_REF_POS_THRESHOLD:-0.5}"
export BAD_TRACKING_REF_ORI_THRESHOLD="${BAD_TRACKING_REF_ORI_THRESHOLD:-0.8}"
export BAD_TRACKING_BODY_POS_THRESHOLD="${BAD_TRACKING_BODY_POS_THRESHOLD:-0.25}"

export HEADLESS="${HEADLESS:-True}"
export VISER_SHOW_SCANDOTS="${VISER_SHOW_SCANDOTS:-False}"
export VISER_DISABLE_CONTACT_FORCE_VIZ="${VISER_DISABLE_CONTACT_FORCE_VIZ:-1}"
export DISABLE_CAMERA_RANDOMIZATION="${DISABLE_CAMERA_RANDOMIZATION:-1}"

exec "${SCRIPT_DIR}/train_terrain_generalist.sh" "${PERCEPTION_PRESET}" \
  --robot.control.action_scale=1.0 \
  --robot.control.action_scales_by_effort_limit_over_p_gain=True
