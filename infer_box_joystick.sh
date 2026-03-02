#!/usr/bin/env bash
set -euo pipefail

# Unified IsaacSim + Viser interactive inference for distilled box-carry policies.
#
# Features:
# - Two branches: mocap | depth
# - Viser clip selection GUI
# - Viser manual command GUI (torso-frame XY + yaw), VideoMimic-style workflow
# - Optional hardware joystick via pygame/bridge backend
#
# Usage:
#   bash infer_joystick.sh <mocap|depth> [checkpoint.pt] [extra tyro args...]
#
# Examples:
#   bash infer_joystick.sh mocap /abs/path/model.pt
#   bash infer_joystick.sh depth /abs/path/model.pt --viser-port 18080

usage() {
  cat <<'EOF'
Usage:
  bash infer_joystick.sh <mocap|depth> [checkpoint.pt] [extra tyro args...]

Modes:
  mocap   Distilled policy with box-state (mocap) actor observation
  depth   Distilled policy with camera depth perception (D435i)

Optional env vars:
  MOTION_DIR              (default: src/holosoma_retargeting/converted_res/object_interaction/omomo_carry)
  OBJECT_URDF             (default: src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf)
  GEOMETRY_DIR            (optional; OBJ file/dir for terrain visualization)
  PAIR_TERRAIN_WITH_MOTION (default: False)
  NUM_ENVS                (default: 1)
  HEADLESS                (default: True)
  VISER_PORT              (default: random)
  VISER_ENV_ID            (default: 0)
  VISER_UPDATE_HZ         (default: 30)
  VISER_RECENTER          (default: True)

Hardware joystick (optional):
  VISER_MANUAL_USE_HW_JOYSTICK=1
  VISER_MANUAL_HW_BACKEND=auto|pygame|bridge
  VISER_MANUAL_HW_DEVICE=0
  VISER_MANUAL_HW_TYPE=xbox|switch
  USE_HW_JOYSTICK_BRIDGE=True   # optional legacy bridge joystick path (requires real joystick)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

case "${MODE}" in
  mocap|depth) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[ERROR] mode must be 'mocap' or 'depth', got: ${MODE}" >&2
    exit 1
    ;;
esac

LOG_ROOT="${SCRIPT_DIR}/logs/WholeBodyTracking"
MOCAP_TRAINING_NAME_DEFAULT="g1_29dof_wbt_w_object_distill_box_mocap_access_to_mocap_data"
DEPTH_TRAINING_NAME_DEFAULT="g1_29dof_wbt_w_object_distill_box_perception_access_to_depth"
MOCAP_CHECKPOINT_DEFAULT=${MOCAP_CHECKPOINT_DEFAULT:-"/home/ubuntu/FAR/holosoma/logs/WholeBodyTracking/20260228_055847-g1_29dof_wbt_w_object_distill_box_mocap_access_to_mocap_data-locomotion/model_00600.pt"}
DEPTH_CHECKPOINT_DEFAULT=${DEPTH_CHECKPOINT_DEFAULT:-"wandb://zihanw22/WholeBodyTracking/xplmudrp/model_01000.pt"}

find_latest_ckpt() {
  local training_name="$1"
  local latest_run=""
  local latest_ckpt=""

  latest_run=$(ls -dt "${LOG_ROOT}"/*-"${training_name}"* 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_run}" ]]; then
    echo ""
    return 0
  fi

  latest_ckpt=$(ls -1 "${latest_run}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  echo "${latest_ckpt}"
}

CKPT=""
if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    CKPT="$1"
    shift
  fi
fi

if [[ -z "${CKPT}" ]]; then
  if [[ "${MODE}" == "mocap" ]]; then
    if [[ "${MOCAP_CHECKPOINT_DEFAULT}" == wandb://* ]] || [[ -f "${MOCAP_CHECKPOINT_DEFAULT}" ]]; then
      CKPT="${MOCAP_CHECKPOINT_DEFAULT}"
    else
      CKPT="$(find_latest_ckpt "${MOCAP_TRAINING_NAME_DEFAULT}")"
    fi
  else
    if [[ "${DEPTH_CHECKPOINT_DEFAULT}" == wandb://* ]] || [[ -f "${DEPTH_CHECKPOINT_DEFAULT}" ]]; then
      CKPT="${DEPTH_CHECKPOINT_DEFAULT}"
    else
      CKPT="$(find_latest_ckpt "${DEPTH_TRAINING_NAME_DEFAULT}")"
    fi
  fi
fi

if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] Could not auto-resolve checkpoint. Please pass checkpoint path explicitly." >&2
  exit 1
fi
if [[ "${CKPT}" != wandb://* ]] && [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
OBJECT_URDF=${OBJECT_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}
GEOMETRY_DIR=${GEOMETRY_DIR:-}

PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-True}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}

DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}

IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
CAMERA_NEAR=${CAMERA_NEAR:-0.001}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi
if [[ -n "${GEOMETRY_DIR}" && ! -e "${GEOMETRY_DIR}" ]]; then
  echo "[ERROR] GEOMETRY_DIR not found: ${GEOMETRY_DIR}" >&2
  exit 1
fi

# Viser GUI defaults aligned with VideoMimic-style manual + clip control.
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-1}
export VISER_MANUAL_CONTROL_DEFAULT=${VISER_MANUAL_CONTROL_DEFAULT:-1}
export VISER_FORCE_MANUAL_CONTROL=${VISER_FORCE_MANUAL_CONTROL:-0}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
export VISER_START_PAUSED=${VISER_START_PAUSED:-0}
export VISER_MANUAL_USE_HW_JOYSTICK=${VISER_MANUAL_USE_HW_JOYSTICK:-0}
export VISER_MANUAL_HW_DEADZONE=${VISER_MANUAL_HW_DEADZONE:-0.08}
export VISER_CLIP_LOCK_DEFAULT=${VISER_CLIP_LOCK_DEFAULT:-1}

# VideoMimic-style default:
# - manual control comes from Viser GUI toggles.
# - hardware joystick is optional and should not crash startup when no device exists.
USE_HW_JOYSTICK_BRIDGE=${USE_HW_JOYSTICK_BRIDGE:-False}
export VISER_MANUAL_HW_BACKEND=${VISER_MANUAL_HW_BACKEND:-auto}
export VISER_MANUAL_HW_DEVICE=${VISER_MANUAL_HW_DEVICE:-0}
export VISER_MANUAL_HW_TYPE=${VISER_MANUAL_HW_TYPE:-xbox}

EXTRA_ARGS=("$@")

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${RESET_NOISE_SCALE}"
  --algo.config.distill.bc_loss_coef 0.0
  --algo.config.distill.loss_coef 0.0
  --algo.config.distill.switch_to_rl_after -1
  --algo.config.distill.take_teacher_actions False
  --algo.config.distill.teacher_action_mix_ratio 0.0
)

if [[ -n "${GEOMETRY_DIR}" ]]; then
  cmd+=(--geometry-dir "${GEOMETRY_DIR}")
fi

if [[ "${DISABLE_RANDOMIZATION}" == "True" || "${DISABLE_RANDOMIZATION}" == "true" ]]; then
  cmd+=(
    --randomization.setup_terms.push_randomizer_state.params.enabled False
    --randomization.reset_terms.randomize_push_schedule.params.enabled False
    --randomization.step_terms.apply_pushes.params.enabled False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_pd_gain False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_rfi_lim False
    --randomization.setup_terms.setup_action_delay_buffers.params.enabled False
    --randomization.reset_terms.randomize_action_delay.params.enabled False
    --randomization.setup_terms.randomize_robot_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_base_com_startup.params.enabled False
    --randomization.setup_terms.setup_dof_pos_bias.params.enabled False
    --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias False
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled False
    --randomization.reset_terms.randomize_camera_raycast.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_mass_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_inertia_startup.params.enabled False
  )
fi

if [[ "${USE_HW_JOYSTICK_BRIDGE}" == "True" || "${USE_HW_JOYSTICK_BRIDGE}" == "true" || "${VISER_MANUAL_HW_BACKEND:-auto}" == "bridge" ]]; then
  cmd+=(
    --simulator.config.bridge.enabled True
    --simulator.config.bridge.use_joystick True
    --simulator.config.bridge.joystick_device "${VISER_MANUAL_HW_DEVICE}"
    --simulator.config.bridge.joystick_type "${VISER_MANUAL_HW_TYPE}"
  )
fi

if [[ "${MODE}" == "mocap" ]]; then
  cmd+=(perception:none)
else
  # Depth branch explicitly follows D435i perception setup.
  export VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE:-depth}
  export VISER_SHOW_PERCEPTION_FRUSTUM=${VISER_SHOW_PERCEPTION_FRUSTUM:-1}
  cmd+=(
    perception:camera-depth-d435i
    --perception.camera_width "${IMAGE_WIDTH}"
    --perception.camera_height "${IMAGE_HEIGHT}"
    --perception.camera_near "${CAMERA_NEAR}"
    --perception.camera_far "${CAMERA_FAR}"
    --perception.max_distance "${CAMERA_MAX_DISTANCE}"
  )
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] mode=${MODE}"
echo "[INFO] checkpoint=${CKPT}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
if [[ -n "${GEOMETRY_DIR}" ]]; then
  echo "[INFO] geometry_dir=${GEOMETRY_DIR}"
fi
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] manual_gui=${VISER_ENABLE_MANUAL_GUI} clip_gui=${VISER_ENABLE_CLIP_GUI}"
echo "[INFO] manual_control_default=${VISER_MANUAL_CONTROL_DEFAULT} force_manual=${VISER_FORCE_MANUAL_CONTROL}"
echo "[INFO] hw_joystick=${VISER_MANUAL_USE_HW_JOYSTICK}"
echo "[INFO] hw_backend=${VISER_MANUAL_HW_BACKEND:-auto} bridge_joystick=${USE_HW_JOYSTICK_BRIDGE}"

"${cmd[@]}"
