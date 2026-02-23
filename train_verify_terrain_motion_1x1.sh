#!/usr/bin/env bash
set -euo pipefail

# Minimal IsaacSim training launcher for hit-point verification:
# - 1 environment
# - terrain grid 1x1
# - one motion file
# - IsaacSim red dots + Viser scandots enabled by default
#
# Usage:
#   ./train_verify_terrain_motion_1x1.sh
#   ./train_verify_terrain_motion_1x1.sh camera_depth_d435i
#   MOTION_FILE=... TERRAIN_OBJ=... ./train_verify_terrain_motion_1x1.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}"

PERCEPTION=${PERCEPTION:-camera_depth_d435i}
if [[ $# -gt 0 ]]; then
  case "$1" in
    heightmap|camera_depth_d435i)
      PERCEPTION="$1"
      shift
      ;;
    perception:heightmap|perception:camera_depth_d435i)
      PERCEPTION="${1#perception:}"
      shift
      ;;
  esac
fi

MOTION_FILE=${MOTION_FILE:-src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz}
TERRAIN_OBJ=${TERRAIN_OBJ:-src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj}
NUM_ENVS=${NUM_ENVS:-1}
TERRAIN_NUM_ROWS=${TERRAIN_NUM_ROWS:-1}
TERRAIN_NUM_COLS=${TERRAIN_NUM_COLS:-1}

HEADLESS=${HEADLESS:-False}
ENABLE_VISER=${ENABLE_VISER:-True}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-True}
ISAAC_SHOW_SCANDOTS=${ISAAC_SHOW_SCANDOTS:-True}

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "[ERROR] MOTION_FILE not found: ${MOTION_FILE}" >&2
  exit 1
fi
if [[ ! -f "${TERRAIN_OBJ}" ]]; then
  echo "[ERROR] TERRAIN_OBJ not found: ${TERRAIN_OBJ}" >&2
  exit 1
fi

# Show only true hits (no miss placeholders) in both IsaacSim and Viser.
export ISAAC_SCANDOTS_INCLUDE_MISSES="${ISAAC_SCANDOTS_INCLUDE_MISSES:-0}"
export VISER_SCANDOTS_INCLUDE_MISSES="${VISER_SCANDOTS_INCLUDE_MISSES:-0}"
export VISER_STRICT_CAMERA_RAYS="${VISER_STRICT_CAMERA_RAYS:-1}"

# Keep camera ray direction aligned with strict warp chain during verification.
export HOLOSOMA_CAMERA_STRICT_WARP="${HOLOSOMA_CAMERA_STRICT_WARP:-1}"
export HOLOSOMA_CAMERA_AUTOFIX_BACKWARD="${HOLOSOMA_CAMERA_AUTOFIX_BACKWARD:-0}"
export HOLOSOMA_CAMERA_DISABLE_OFFSETS="${HOLOSOMA_CAMERA_DISABLE_OFFSETS:-1}"

# Avoid external logging requirement during local verification.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[INFO] Train verify 1x1x1 launch"
echo "[INFO] PERCEPTION=${PERCEPTION}"
echo "[INFO] MOTION_FILE=${MOTION_FILE}"
echo "[INFO] TERRAIN_OBJ=${TERRAIN_OBJ}"
echo "[INFO] NUM_ENVS=${NUM_ENVS}"
echo "[INFO] TERRAIN_NUM_ROWS=${TERRAIN_NUM_ROWS}"
echo "[INFO] TERRAIN_NUM_COLS=${TERRAIN_NUM_COLS}"
echo "[INFO] HEADLESS=${HEADLESS}"
echo "[INFO] ISAAC_SHOW_SCANDOTS=${ISAAC_SHOW_SCANDOTS}"
echo "[INFO] VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS}"
echo "[INFO] VISER=http://localhost:${VISER_PORT}"

cmd=(
  torchrun
  --nproc_per_node=1
  --master_port="$((29500 + RANDOM % 1000))"
  src/holosoma/holosoma/train_agent.py
  exp:g1-29dof-wbt-motion-tracking-transformer
  "perception:${PERCEPTION}"
  --training.num_envs="${NUM_ENVS}"
  --training.headless="${HEADLESS}"
  --training.enable_viser="${ENABLE_VISER}"
  --training.viser_port="${VISER_PORT}"
  --training.viser_env_id=0
  --training.viser_update_hz=30
  --training.viser_recenter=True
  --training.viser_show_scandots="${VISER_SHOW_SCANDOTS}"
  --training.isaac_show_scandots="${ISAAC_SHOW_SCANDOTS}"
  --training.isaac_scandots_point_size=3.0
  --simulator.config.scene.env_spacing=0.0
  terrain:terrain-load-obj
  --terrain.terrain-term.obj-file-path "${TERRAIN_OBJ}"
  --terrain.terrain-term.num-rows "${TERRAIN_NUM_ROWS}"
  --terrain.terrain-term.num-cols "${TERRAIN_NUM_COLS}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}"
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler=False
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0
  logger:wandb
)

if [[ "${PERCEPTION}" == "camera_depth_d435i" ]]; then
  cmd+=(
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled=False
    --randomization.reset_terms.randomize_camera_raycast.params.enabled=False
  )
fi

cmd+=("$@")
"${cmd[@]}"

