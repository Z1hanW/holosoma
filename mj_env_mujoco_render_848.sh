#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/mujoco_perception_env.sh"

export PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i_mujoco_render_848x480}"
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-rendered}"
export PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="${PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN:-mujoco848}"
holosoma_set_launcher_default PERCEPTION_CAMERA_WARP_EDGE_NOISE False
holosoma_set_launcher_default PERCEPTION_CAMERA_WARP_ENABLE_HOLES False
holosoma_set_launcher_default PERCEPTION_CAMERA_APPLY_SENSOR_NOISE False

exec bash "${ROOT_DIR}/mj_env.sh" rendered848 "$@"
