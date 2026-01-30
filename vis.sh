#!/usr/bin/env bash
set -euo pipefail

# Physics rollout with Viser (checkpoint + motion, optional geometry).
#
# Required:
#   CKPT=/abs/path/to/model.pt
#   MOTION_DIR=/abs/path/to/motion_folder
#
# Optional:
#   GEOMETRY_DIR=/abs/path/to/obj_dir_or_obj_file
#   GEOMETRY_META=/abs/path/to/metadata.json
#   NUM_ENVS=4
#   HEADLESS=True
#   NUM_ROWS=1
#   NUM_COLS=
#   ENV_SPACING=0.0
#   PAIR_TERRAIN=True
#   DEPTH_IMPL=rendered|depth_sensor|raycast|scandots
#   PERCEPTION_PRESET=camera_depth_d435i_rendered
#   IMAGE_WIDTH=128
#   IMAGE_HEIGHT=72
#   SCANDOTS_GRID=3
#   SCANDOTS_STRIDE=
#   PHYSX_GPU_COLLISION_STACK_SIZE=4294967295
#   START_AT_TIMESTEP_ZERO_PROB=0.05
#   ENABLE_DEFAULT_POSE_APPEND=False
#   DEFAULT_POSE_APPEND_DURATION_S=0
#   ENABLE_DEFAULT_POSE_PREPEND=False
#   DEFAULT_POSE_PREPEND_DURATION_S=0
#   LOAD_OPTIMIZER=False
#   VISER_PORT=####
#   VISER_ENV_ID=0
#   VISER_UPDATE_HZ=30
#   VISER_RECENTER=True
#   VISER_GLOBAL_FRAME_QUAT_WXYZ="0 1 0 0"  # wxyz, x-axis 180deg
#   VISER_SHOW_SCANDOTS=True
#   CONTACT_FORCE_VIZ=True
#   CONTACT_FORCE_VIZ_SCALE=0.001
#   CONTACT_FORCE_VIZ_THRESHOLD=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT=${CKPT:-"../Store/model_52000.pt"}
MOTION_DIR=${MOTION_DIR:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_data_trans"}
GEOMETRY_DIR=${GEOMETRY_DIR:-"/home/ubuntu/FAR/Store/vmm_data/___zero_pad_geo_trans"}
GEOMETRY_META=${GEOMETRY_META:-""}
NUM_ENVS=${NUM_ENVS:-4}
HEADLESS=${HEADLESS:-True}
NUM_ROWS=${NUM_ROWS:-1}
NUM_COLS=${NUM_COLS:-}
ENV_SPACING=${ENV_SPACING:-0.0}
PAIR_TERRAIN=${PAIR_TERRAIN:-True}
DEPTH_IMPL=${DEPTH_IMPL:-scandots}
PERCEPTION_PRESET=${PERCEPTION_PRESET:-""}
IMAGE_WIDTH=${IMAGE_WIDTH:-128}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-72}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-4294967295}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0}
LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-False}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-False}
VISER_GLOBAL_FRAME_QUAT_WXYZ=${VISER_GLOBAL_FRAME_QUAT_WXYZ:-"0 1 0 0"}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-True}
CONTACT_FORCE_VIZ=${CONTACT_FORCE_VIZ:-True}
CONTACT_FORCE_VIZ_SCALE=${CONTACT_FORCE_VIZ_SCALE:-0.001}
CONTACT_FORCE_VIZ_THRESHOLD=${CONTACT_FORCE_VIZ_THRESHOLD:-1.0}
SCANDOTS_GRID=${SCANDOTS_GRID:-3}
SCANDOTS_STRIDE=${SCANDOTS_STRIDE:-}

if [[ "${CKPT}" == "/ABS/PATH/to/model.pt" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi
if [[ "${MOTION_DIR}" == "/ABS/PATH/to/motion_folder" ]]; then
  echo "Set MOTION_DIR to your motion folder path." >&2
  exit 1
fi

if [[ -z "${PERCEPTION_PRESET}" ]]; then
  case "${DEPTH_IMPL}" in
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
    ""|none)
      PERCEPTION_PRESET=""
      ;;
    *)
      echo "Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use rendered|depth_sensor|raycast|scandots|none." >&2
      exit 1
      ;;
  esac
fi

echo "[INFO] Viser port: ${VISER_PORT}"

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS}"
  --num-rows "${NUM_ROWS}"
  --simulator.config.scene.env_spacing "${ENV_SPACING}"
  --pair-terrain-with-motion "${PAIR_TERRAIN}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --training.viser-show-scandots "${VISER_SHOW_SCANDOTS}"
  --simulator.config.contact_force_viz "${CONTACT_FORCE_VIZ}"
  --simulator.config.contact_force_viz_scale "${CONTACT_FORCE_VIZ_SCALE}"
  --simulator.config.contact_force_viz_threshold "${CONTACT_FORCE_VIZ_THRESHOLD}"
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append "${ENABLE_DEFAULT_POSE_APPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s "${DEFAULT_POSE_APPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend "${ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s "${DEFAULT_POSE_PREPEND_DURATION_S}"
  --algo.config.load_optimizer "${LOAD_OPTIMIZER}"
)

if [[ -n "${NUM_COLS}" ]]; then
  cmd+=(--num-cols "${NUM_COLS}")
fi
if [[ -n "${GEOMETRY_DIR}" ]]; then
  cmd+=(--geometry-dir "${GEOMETRY_DIR}")
fi
if [[ -n "${GEOMETRY_META}" ]]; then
  cmd+=(--geometry-metadata "${GEOMETRY_META}")
fi
if [[ -n "${PERCEPTION_PRESET}" ]]; then
  cmd+=(
    "perception:${PERCEPTION_PRESET}"
    --perception.camera_width "${IMAGE_WIDTH}"
    --perception.camera_height "${IMAGE_HEIGHT}"
  )
  if [[ -n "${SCANDOTS_GRID}" ]]; then
    if [[ -n "${SCANDOTS_STRIDE}" ]]; then
      cmd+=(
        --perception.camera_scandots_stride "${SCANDOTS_STRIDE}"
      )
    else
      cmd+=(
        --perception.camera_scandots_width "${SCANDOTS_GRID}"
        --perception.camera_scandots_height "${SCANDOTS_GRID}"
      )
    fi
  fi
fi

if [[ -n "${VISER_GLOBAL_FRAME_QUAT_WXYZ}" ]]; then
  cmd+=(--training.viser-global-frame-quat-wxyz ${VISER_GLOBAL_FRAME_QUAT_WXYZ})
fi

"${cmd[@]}"
