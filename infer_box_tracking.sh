#!/usr/bin/env bash
set -euo pipefail

# Teacher-policy inference for box tracking on the same motion-box pairs used by
# train_object_generalist.sh, with Isaac Sim <-> Viser sync enabled.
#
# Usage:
#   bash infer_box_tracking.sh [teacher_checkpoint.pt|wandb://...] [extra tyro args...]
#
# Optional env vars:
#   TEACHER_CHECKPOINT        (default: distill_box teacher default)
#   MOTION_DIR                (default: src/holosoma_retargeting/converted_res/object_interaction/omomo_carry)
#   MOTION_CLIP_NAME          (optional: pin a single clip)
#   OBJECT_URDF               (default: src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf)
#   NUM_ENVS                  (default: 1)
#   HEADLESS                  (default: False; set True for headless eval)
#   VISER_PORT                (default: random)
#   VISER_ENV_ID              (default: 0)
#   VISER_UPDATE_HZ           (default: 30)
#   VISER_RECENTER            (default: True)
#   VISER_SYNC_TO_SIM         (default: True)
#   VISER_FORCE_DT            (default: True)
#   DISABLE_RANDOMIZATION     (default: True)

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_tracking.sh [teacher_checkpoint.pt|wandb://...] [extra tyro args...]

Examples:
  bash infer_box_tracking.sh
  bash infer_box_tracking.sh /abs/path/to/model_17000.pt
  MOTION_CLIP_NAME=sub3_largebox_003_mj_w_obj bash infer_box_tracking.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

DEFAULT_TEACHER_CHECKPOINT="/home/ubuntu/FAR/holosoma/logs/WholeBodyTracking/20260216_214200-g1_29dof_wbt_w_object_generalist-locomotion/model_17000.pt"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${CKPT:-${DEFAULT_TEACHER_CHECKPOINT}}}"

if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    shift
  fi
fi

MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-}
OBJECT_URDF=${OBJECT_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}

NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-False}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}
SIM_ENV_SPACING=${SIM_ENV_SPACING:-0.0}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}

# Useful defaults for interactive motion/clip inspection.
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-0}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
export VISER_START_PAUSED=${VISER_START_PAUSED:-0}

if [[ "${TEACHER_CHECKPOINT}" != wandb://* ]] && [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi

if [[ -d "${MOTION_DIR}" && -n "${MOTION_CLIP_NAME}" && ! -f "${MOTION_DIR}/${MOTION_CLIP_NAME}.npz" ]]; then
  echo "[WARN] MOTION_CLIP_NAME not found in MOTION_DIR: ${MOTION_CLIP_NAME}.npz"
  echo "[WARN] Falling back to random clip selection."
  MOTION_CLIP_NAME=""
fi

EXTRA_ARGS=("$@")

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${TEACHER_CHECKPOINT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --training.viser_sync_to_sim "${VISER_SYNC_TO_SIM}"
  --training.viser_force_dt "${VISER_FORCE_DT}"
  --training.viser_show_scandots "${VISER_SHOW_SCANDOTS}"
  --simulator.config.scene.env_spacing "${SIM_ENV_SPACING}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${RESET_NOISE_SCALE}"
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(
    --command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}"
  )
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

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] motion_clip_name=${MOTION_CLIP_NAME:-<auto>}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] headless=${HEADLESS}"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] viser_sync_to_sim=${VISER_SYNC_TO_SIM} viser_force_dt=${VISER_FORCE_DT}"
echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"

"${cmd[@]}"
