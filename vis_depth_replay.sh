#!/usr/bin/env bash
set -euo pipefail

# Minimal Isaac Sim kinematic replay for depth-camera sanity check.
# Default behavior:
# - Kinematic replay (motion command)
# - Scene scandots in Isaac Sim
# - Perception camera/frustum + depth image in Viser GUI (realtime)
#
# Usage:
#   ./vis_depth_replay.sh
#   MOTION_FILE=/abs/or/rel/path/to/motion_or_dir TERRAIN_OBJ=/abs/or/rel/path/to/scene.obj ./vis_depth_replay.sh
#   DEPTH_IMPL=rendered ./vis_depth_replay.sh   # rendered|depth_sensor|raycast|scandots

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP=${EXP:-"g1-29dof-wbt-videomimic-mlp"}
MOTION_FILE=${MOTION_FILE:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_data/far_robot/far_robot/far_robot.npz"}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-""}
TERRAIN_OBJ=${TERRAIN_OBJ:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj"}

DEPTH_IMPL=${DEPTH_IMPL:-scandots}
case "${DEPTH_IMPL}" in
  scandots)
    PERCEPTION_PRESET="camera_depth_d435i_scandots"
    ;;
  rendered)
    PERCEPTION_PRESET="camera_depth_d435i_rendered"
    ;;
  depth_sensor)
    PERCEPTION_PRESET="camera_depth_d435i_depth_sensor"
    ;;
  raycast)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  *)
    echo "[ERROR] Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use scandots|rendered|depth_sensor|raycast." >&2
    exit 1
    ;;
esac

IMAGE_WIDTH=${IMAGE_WIDTH:-640}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-360}
SCANDOTS_STRIDE=${SCANDOTS_STRIDE:-4}
CAMERA_BODY_NAME=${CAMERA_BODY_NAME:-d435_joint}
HEADLESS=${HEADLESS:-False}
NUM_ENVS=${NUM_ENVS:-1}

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}

if [[ ! -e "${MOTION_FILE}" ]]; then
  echo "[ERROR] MOTION_FILE not found: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ -n "${TERRAIN_OBJ}" ]] && [[ ! -e "${TERRAIN_OBJ}" ]]; then
  echo "[ERROR] TERRAIN_OBJ not found: ${TERRAIN_OBJ}" >&2
  exit 1
fi

if [[ "${DEPTH_IMPL}" != "scandots" ]]; then
  echo "[WARN] DEPTH_IMPL=${DEPTH_IMPL}: scene scan points may disappear. Use DEPTH_IMPL=scandots for scan-point debugging."
fi

echo "[INFO] EXP=exp:${EXP}"
echo "[INFO] MOTION_FILE=${MOTION_FILE}"
echo "[INFO] TERRAIN_OBJ=${TERRAIN_OBJ}"
if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  echo "[INFO] MOTION_CLIP_NAME=${MOTION_CLIP_NAME}"
fi
echo "[INFO] CAMERA_BODY_NAME=${CAMERA_BODY_NAME}"
echo "[INFO] PERCEPTION_PRESET=${PERCEPTION_PRESET}"
echo "[INFO] VISER=http://localhost:${VISER_PORT}"

cmd=(
  python src/holosoma/holosoma/replay.py
  "exp:${EXP}"
  "perception:${PERCEPTION_PRESET}"
  --training.headless="${HEADLESS}"
  --training.num_envs="${NUM_ENVS}"
  --training.enable_viser=True
  --training.viser_port="${VISER_PORT}"
  --training.viser_show_scandots=True
  --training.isaac_show_scandots=True
  --training.isaac_scandots_point_size=3.0
  --simulator.config.debug_viz=True
  --simulator.config.scene.env_spacing=0.0
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}"
  --perception.camera_width="${IMAGE_WIDTH}"
  --perception.camera_height="${IMAGE_HEIGHT}"
  --perception.camera_scandots_stride="${SCANDOTS_STRIDE}"
  --perception.camera_body_name "${CAMERA_BODY_NAME}"
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}")
fi

if [[ -n "${TERRAIN_OBJ}" ]]; then
  cmd+=(
    terrain:terrain-load-obj
    --terrain.terrain-term.obj-file-path "${TERRAIN_OBJ}"
    --terrain.terrain-term.num_rows=1
    --terrain.terrain-term.num_cols=1
  )
fi

cmd+=("$@")
"${cmd[@]}"
