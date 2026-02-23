#!/usr/bin/env bash
set -euo pipefail

# Sim rollout + Viser for distilled torso-box policy.
# - Uses the physics visualizer path (ViserLive) so manual control + target keypoints work.
# - Checkpoint is fixed by default to the requested model path.
#
# Usage:
#   bash sim_box_js.sh
#
# Optional overrides:
#   CKPT=/abs/path/model.pt
#   MOTION_DIR=/abs/path/to/motion_dir_or_file
#   OBJECT_URDF=/abs/path/to/objects_largebox.urdf
#   VISER_PORT=18080
#   USE_HW_JOYSTICK=True JOYSTICK_DEVICE=0 JOYSTICK_TYPE=xbox
#   USE_HW_JOYSTICK_BRIDGE=True  # optional legacy bridge mode (injects bridge torques)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT=${CKPT:-"/home/ubuntu/FAR/holosoma/logs/WholeBodyTracking/20260222_081034-g1_29dof_wbt_w_object_distill_torso_box_stage1-locomotion/model_01000.pt"}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-sub3_largebox_003_mj_w_obj}
OBJECT_URDF=${OBJECT_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}

NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-True}
PAIR_TERRAIN=${PAIR_TERRAIN:-False}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
# JS/manual-control mode runs in non-tracking mode by default: keep clip at frame 0.
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-1.0}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}
DISABLE_TERMINATION=${DISABLE_TERMINATION:-True}
DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}

# Viser GUI toggles:
# - Clip GUI: clip selection + apply
# - Manual GUI: joystick-like XY/yaw command controls
# - Target keypoints: target pose markers
export VISER_ENABLE_CLIP_GUI=1
export VISER_ENABLE_MANUAL_GUI=1
export VISER_MANUAL_CONTROL_DEFAULT=1
export VISER_FORCE_MANUAL_CONTROL=1
export VISER_MANUAL_USE_HW_JOYSTICK=${VISER_MANUAL_USE_HW_JOYSTICK:-0}
export VISER_MANUAL_HW_DEADZONE=${VISER_MANUAL_HW_DEADZONE:-0.08}
export VISER_SHOW_TARGET_KEYPOINTS=1
export VISER_START_PAUSED=1

# Optional physical joystick/gamepad control.
# Default path uses direct Viser joystick polling (pygame) and keeps simulator bridge disabled.
USE_HW_JOYSTICK=${USE_HW_JOYSTICK:-False}
USE_HW_JOYSTICK_BRIDGE=${USE_HW_JOYSTICK_BRIDGE:-False}
JOYSTICK_DEVICE=${JOYSTICK_DEVICE:-0}
JOYSTICK_TYPE=${JOYSTICK_TYPE:-xbox}
VISER_MANUAL_HW_BACKEND=${VISER_MANUAL_HW_BACKEND:-auto}

if [[ "${USE_HW_JOYSTICK}" == "True" || "${USE_HW_JOYSTICK}" == "true" ]]; then
  export VISER_MANUAL_USE_HW_JOYSTICK=1
  export VISER_MANUAL_HW_BACKEND="${VISER_MANUAL_HW_BACKEND}"
  export VISER_MANUAL_HW_DEVICE="${VISER_MANUAL_HW_DEVICE:-${JOYSTICK_DEVICE}}"
  export VISER_MANUAL_HW_TYPE="${VISER_MANUAL_HW_TYPE:-${JOYSTICK_TYPE}}"
fi

# Distill checkpoint eval defaults: run student-only inference (no teacher dependency at eval time).
DISTILL_EVAL_STUDENT_ONLY=${DISTILL_EVAL_STUDENT_ONLY:-True}
DISTILL_BC_LOSS_COEF=${DISTILL_BC_LOSS_COEF:-0.0}
DISTILL_LOSS_COEF=${DISTILL_LOSS_COEF:-0.0}
DISTILL_SWITCH_TO_RL_AFTER=${DISTILL_SWITCH_TO_RL_AFTER:--1}

if [[ ! -f "${CKPT}" ]]; then
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
if [[ -d "${MOTION_DIR}" && -n "${MOTION_CLIP_NAME}" ]]; then
  if [[ ! -f "${MOTION_DIR}/${MOTION_CLIP_NAME}.npz" ]]; then
    echo "[WARN] MOTION_CLIP_NAME not found in MOTION_DIR: ${MOTION_CLIP_NAME}. Falling back to random clip."
    MOTION_CLIP_NAME=""
  fi
fi

echo "[INFO] CKPT=${CKPT}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-<auto>}"
echo "[INFO] OBJECT_URDF=${OBJECT_URDF}"
echo "[INFO] VISER: http://localhost:${VISER_PORT}"
echo "[INFO] HW joystick input: ${USE_HW_JOYSTICK}"
echo "[INFO] HW joystick backend: ${VISER_MANUAL_HW_BACKEND}"
echo "[INFO] HW joystick bridge torque path: ${USE_HW_JOYSTICK_BRIDGE}"
echo "[INFO] Distill student-only eval: ${DISTILL_EVAL_STUDENT_ONLY}"

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS}"
  --pair-terrain-with-motion "${PAIR_TERRAIN}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
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

if [[ "${DISTILL_EVAL_STUDENT_ONLY}" == "True" || "${DISTILL_EVAL_STUDENT_ONLY}" == "true" ]]; then
  cmd+=(
    --algo.config.distill.bc_loss_coef "${DISTILL_BC_LOSS_COEF}"
    --algo.config.distill.loss_coef "${DISTILL_LOSS_COEF}"
    --algo.config.distill.switch_to_rl_after "${DISTILL_SWITCH_TO_RL_AFTER}"
    --algo.config.distill.take_teacher_actions False
  )
fi

# Manager-based envs require non-null termination/randomization configs.
# Do NOT set `termination:none` / `randomization:none` here.
if [[ "${DISABLE_TERMINATION}" == "True" || "${DISABLE_TERMINATION}" == "true" ]]; then
  cmd+=(
    --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
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

if [[ "${USE_HW_JOYSTICK_BRIDGE}" == "True" || "${USE_HW_JOYSTICK_BRIDGE}" == "true" || "${VISER_MANUAL_HW_BACKEND}" == "bridge" ]]; then
  cmd+=(
    --simulator.config.bridge.enabled True
    --simulator.config.bridge.use_joystick True
    --simulator.config.bridge.joystick_device "${JOYSTICK_DEVICE}"
    --simulator.config.bridge.joystick_type "${JOYSTICK_TYPE}"
  )
fi

"${cmd[@]}"
