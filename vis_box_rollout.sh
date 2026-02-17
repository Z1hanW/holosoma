#!/usr/bin/env bash
set -euo pipefail

# Live simulator rollout + Viser with clip dropdown for object tracking.
#
# Required inputs (positional or env):
#   1) checkpoint
#   2) motion dir/file
#   3) object urdf
#
# Example:
#   bash vis_box_rollout.sh /abs/path/model.pt \
#     src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking \
#     src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf
#
# Notes:
# - GUI has "Motion -> Clip" dropdown + "Apply clip" button.
# - State shown in Viser is synchronized from the simulator (not offline replay).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT=${1:-${CKPT:-"/ABS/PATH/to/model.pt"}}
MOTION_DIR=${2:-${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"}}
OBJECT_URDF=${3:-${OBJECT_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}}

NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-True}
DEPTH_IMPL=${DEPTH_IMPL:-none}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_INTERVAL=${VISER_UPDATE_INTERVAL:-1}
VISER_SHOW_MESHES=${VISER_SHOW_MESHES:-True}
VISER_GRID=${VISER_GRID:-True}
VISER_GRID_SIZE=${VISER_GRID_SIZE:-10.0}
AUTO_REAPPLY_CLIP=${AUTO_REAPPLY_CLIP:-True}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}

if [[ "${CKPT}" == "/ABS/PATH/to/model.pt" ]]; then
  echo "Set CKPT (or pass arg1) to your checkpoint path." >&2
  exit 1
fi
if [[ "${CKPT}" != wandb://* ]] && [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi

case "${DEPTH_IMPL}" in
  none|"")
    PERCEPTION_PRESET=""
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
  scandots)
    PERCEPTION_PRESET="camera_depth_d435i_scandots"
    ;;
  *)
    echo "Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use rendered|depth_sensor|raycast|scandots|none." >&2
    exit 1
    ;;
esac

echo "[INFO] CKPT=${CKPT}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_URDF=${OBJECT_URDF}"
echo "[INFO] VISER_PORT=${VISER_PORT}"

cmd=(
  python vis_scripts/eval_agent_viser_clip.py
  --checkpoint "${CKPT}"
  --training.headless "${HEADLESS}"
  --training.num_envs "${NUM_ENVS}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --port "${VISER_PORT}"
  --env-index "${VISER_ENV_ID}"
  --update-interval "${VISER_UPDATE_INTERVAL}"
  --show-meshes "${VISER_SHOW_MESHES}"
  --add-grid "${VISER_GRID}"
  --grid-size "${VISER_GRID_SIZE}"
  --auto-reapply-clip "${AUTO_REAPPLY_CLIP}"
)

if [[ -n "${PERCEPTION_PRESET}" ]]; then
  cmd+=("perception:${PERCEPTION_PRESET}")
fi

"${cmd[@]}"
