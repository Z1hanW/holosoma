#!/usr/bin/env bash
set -euo pipefail

# Minimal Viser visualization for object-interaction policy.
# Fill in your checkpoint path and run: bash vis.sh

CKPT="/ABS/PATH/to/model.pt"
MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
NUM_ENVS=1
HEADLESS=True
DEPTH_IMPL=none
VISER_PORT=$((RANDOM % 8976 + 1024))

if [[ "${CKPT}" == "/ABS/PATH/to/model.pt" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi

case "${DEPTH_IMPL}" in
  none|"")
    PERCEPTION_PRESET=""
    ;;
  rendered)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  depth_sensor)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  raycast)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  scandots)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  *)
    echo "Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use rendered|depth_sensor|raycast|scandots|none." >&2
    exit 1
    ;;
 esac

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS}"
  --viser-port "${VISER_PORT}"
)

if [[ -n "${PERCEPTION_PRESET}" ]]; then
  cmd+=("perception:${PERCEPTION_PRESET}")
fi

"${cmd[@]}"
