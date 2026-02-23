#!/usr/bin/env bash
set -euo pipefail

# Distill an object-generalist teacher into an object-aware student with reduced actor inputs:
# - torso_xy_rel + torso_yaw_rel
# - obj_pos_b + obj_target_pose_size_b (target goal pose + size)
#
# Default behavior (DISTILL_TWO_STAGE=1):
# - Stage 1: true DAgger data collection (teacher steps env) + pure BC
# - Stage 2: resume Stage 1 student, student steps env, mix RL+BC then switch to RL

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"/home/ubuntu/FAR/holosoma/logs/WholeBodyTracking/20260216_214200-g1_29dof_wbt_w_object_generalist-locomotion/model_30000.pt"}

if [[ $# -gt 0 ]]; then
  TEACHER_CHECKPOINT="$1"
  shift
else
  TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Usage: $0 <teacher_checkpoint.pt> [extra train_agent.py args...]" >&2
  echo "Default teacher checkpoint: ${DEFAULT_TEACHER_CHECKPOINT}" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
EXP=${EXP:-g1-29dof-wbt-w-object-distill-torso-box}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
OBJECT_URDF=${OBJECT_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}

DISTILL_TWO_STAGE=${DISTILL_TWO_STAGE:-1}

if [[ -z "${NPROC:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    if [[ "${CUDA_VISIBLE_DEVICES}" == "all" || "${CUDA_VISIBLE_DEVICES}" == "ALL" ]]; then
      if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC=$(nvidia-smi -L | wc -l | tr -d ' ')
      else
        NPROC=1
      fi
    else
      IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
      NPROC=${#_visible_gpus[@]}
    fi
  else
    NPROC=1
  fi
fi
if [[ "${NPROC}" -lt 1 ]]; then
  echo "NPROC must be >= 1. Got: ${NPROC}" >&2
  exit 1
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MAX_RESTARTS=${MAX_RESTARTS:-0}

MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
NUM_ENVS=${NUM_ENVS:-24576}
ACTOR_LR=${ACTOR_LR:-7e-5}
CRITIC_LR=${CRITIC_LR:-7e-5}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
SWITCH_TO_RL_AFTER=${SWITCH_TO_RL_AFTER:-}
CLIP_TEACHER_ACTIONS=${CLIP_TEACHER_ACTIONS:-True}
CLIP_ACTIONS_THRESHOLD=${CLIP_ACTIONS_THRESHOLD:-8.0}
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-1.0}
TAKE_TEACHER_ACTIONS=${TAKE_TEACHER_ACTIONS:-False}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}
LOGGER=${LOGGER:-logger:wandb}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_torso_box}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_torso_box}
TRAINING_PROJECT=${TRAINING_PROJECT:-WholeBodyTracking}
LOGGER_BASE_DIR=${LOGGER_BASE_DIR:-logs}

STAGE1_BC_LOSS_COEF=${STAGE1_BC_LOSS_COEF:-1.0}
STAGE2_BC_LOSS_COEF=${STAGE2_BC_LOSS_COEF:-0.3}
STAGE2_SWITCH_TO_RL_AFTER=${STAGE2_SWITCH_TO_RL_AFTER:-20000}
STAGE1_TAKE_TEACHER_ACTIONS=${STAGE1_TAKE_TEACHER_ACTIONS:-True}
STAGE2_TAKE_TEACHER_ACTIONS=${STAGE2_TAKE_TEACHER_ACTIONS:-False}
STAGE1_START_AT_TIMESTEP_ZERO_PROB=${STAGE1_START_AT_TIMESTEP_ZERO_PROB:-1.0}
STAGE2_START_AT_TIMESTEP_ZERO_PROB=${STAGE2_START_AT_TIMESTEP_ZERO_PROB:-0.2}
STAGE1_RESET_NOISE_SCALE=${STAGE1_RESET_NOISE_SCALE:-0.0}
STAGE2_RESET_NOISE_SCALE=${STAGE2_RESET_NOISE_SCALE:-0.1}
STAGE1_RUN_NAME=${STAGE1_RUN_NAME:-${RUN_NAME}_stage1}
STAGE2_RUN_NAME=${STAGE2_RUN_NAME:-${RUN_NAME}_stage2}
STAGE1_TRAINING_NAME=${STAGE1_TRAINING_NAME:-${TRAINING_NAME}_stage1}
STAGE2_TRAINING_NAME=${STAGE2_TRAINING_NAME:-${TRAINING_NAME}_stage2}
STAGE1_MASTER_PORT=${STAGE1_MASTER_PORT:-$((29500 + RANDOM % 1000))}
STAGE2_MASTER_PORT=${STAGE2_MASTER_PORT:-$((29500 + RANDOM % 1000))}
STAGE1_STUDENT_CHECKPOINT=${STAGE1_STUDENT_CHECKPOINT:-}

if [[ "${TEACHER_CHECKPOINT}" != wandb://* ]] && [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "Teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
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

EXTRA_ARGS=("$@")

run_distill_stage() {
  local stage_label="$1"
  local stage_bc_loss_coef="$2"
  local stage_run_name="$3"
  local stage_training_name="$4"
  local stage_master_port="$5"
  local stage_resume_checkpoint="${6:-}"
  local stage_switch_to_rl_after="${7:-}"
  local stage_take_teacher_actions="${8:-}"
  local stage_start_at_timestep_zero_prob="${9:-}"
  local stage_reset_noise_scale="${10:-}"

  echo "[INFO] Starting ${stage_label}"
  echo "[INFO]   run_name=${stage_run_name}"
  echo "[INFO]   training_name=${stage_training_name}"
  echo "[INFO]   bc_loss_coef=${stage_bc_loss_coef}"
  echo "[INFO]   take_teacher_actions=${stage_take_teacher_actions}"
  echo "[INFO]   start_at_timestep_zero_prob=${stage_start_at_timestep_zero_prob}"
  echo "[INFO]   reset_noise_scale=${stage_reset_noise_scale}"
  echo "[INFO]   distributed: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC}"

  local cmd=(
    torchrun
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --nproc_per_node="${NPROC}"
    --max_restarts="${MAX_RESTARTS}"
    --master_port="${stage_master_port}"
    src/holosoma/holosoma/train_agent.py
    "exp:${EXP}"
    --algo.config.distill.mode=dagger
    --algo.config.distill.policy_to_clone="${TEACHER_CHECKPOINT}"
    --algo.config.distill.bc_loss_coef="${stage_bc_loss_coef}"
    --algo.config.distill.clip_teacher_actions="${CLIP_TEACHER_ACTIONS}"
    --algo.config.distill.clip_actions_threshold="${CLIP_ACTIONS_THRESHOLD}"
    --algo.config.distill.teacher_obs_keys="${TEACHER_OBS_KEYS}"
    --algo.config.distill.take_teacher_actions="${stage_take_teacher_actions}"
    --training.num_envs="${NUM_ENVS}"
    --training.project="${TRAINING_PROJECT}"
    --training.name="${stage_training_name}"
    --training.multigpu=$([[ "${NPROC}" -gt 1 || "${NNODES}" -gt 1 ]] && echo True || echo False)
    --algo.config.actor_learning_rate="${ACTOR_LR}"
    --algo.config.critic_learning_rate="${CRITIC_LR}"
    --algo.config.normalize_actor_obs=False
    --algo.config.normalize_critic_obs=False
    --algo.config.save_interval="${SAVE_INTERVAL}"
    --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
    --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=False
    --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob="${stage_start_at_timestep_zero_prob}"
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale="${stage_reset_noise_scale}"
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False
    --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False
    --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0
    --robot.object.enabled=True
    --robot.object.object_urdf_path "${OBJECT_URDF}"
    "${LOGGER}"
    --logger.video.interval=1000
    --logger.name="${stage_run_name}"
  )

  if [[ -n "${stage_resume_checkpoint}" ]]; then
    cmd+=(--training.checkpoint "${stage_resume_checkpoint}")
  fi
  if [[ -n "${stage_switch_to_rl_after}" ]]; then
    cmd+=(--algo.config.distill.switch_to_rl_after="${stage_switch_to_rl_after}")
  fi
  cmd+=("${EXTRA_ARGS[@]}")

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
}

find_latest_stage_checkpoint() {
  local target_training_name="$1"
  local search_root="${LOGGER_BASE_DIR}/${TRAINING_PROJECT}"
  local latest_run_dir
  local latest_ckpt

  latest_run_dir=$(ls -dt "${search_root}"/*-"${target_training_name}"-* 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_run_dir}" ]]; then
    echo "" && return 0
  fi

  latest_ckpt=$(ls -1 "${latest_run_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  echo "${latest_ckpt}"
}

if [[ "${DISTILL_TWO_STAGE}" != "0" ]]; then
  # Stage 1: DAgger BC with teacher-driven rollout
  run_distill_stage \
    "Stage 1 (DAgger BC, teacher rollout)" \
    "${STAGE1_BC_LOSS_COEF}" \
    "${STAGE1_RUN_NAME}" \
    "${STAGE1_TRAINING_NAME}" \
    "${STAGE1_MASTER_PORT}" \
    "" \
    "" \
    "${STAGE1_TAKE_TEACHER_ACTIONS}" \
    "${STAGE1_START_AT_TIMESTEP_ZERO_PROB}" \
    "${STAGE1_RESET_NOISE_SCALE}"

  # Stage 2: resume student, mix RL/BC then switch to RL.
  if [[ -z "${STAGE1_STUDENT_CHECKPOINT}" ]]; then
    STAGE1_STUDENT_CHECKPOINT=$(find_latest_stage_checkpoint "${STAGE1_TRAINING_NAME}")
  fi
  if [[ -z "${STAGE1_STUDENT_CHECKPOINT}" || ! -f "${STAGE1_STUDENT_CHECKPOINT}" ]]; then
    echo "Unable to resolve Stage 1 student checkpoint. Set STAGE1_STUDENT_CHECKPOINT explicitly." >&2
    exit 1
  fi
  echo "[INFO] Using Stage 1 student checkpoint: ${STAGE1_STUDENT_CHECKPOINT}"

  run_distill_stage \
    "Stage 2 (resume + switch to RL)" \
    "${STAGE2_BC_LOSS_COEF}" \
    "${STAGE2_RUN_NAME}" \
    "${STAGE2_TRAINING_NAME}" \
    "${STAGE2_MASTER_PORT}" \
    "${STAGE1_STUDENT_CHECKPOINT}" \
    "${STAGE2_SWITCH_TO_RL_AFTER}" \
    "${STAGE2_TAKE_TEACHER_ACTIONS}" \
    "${STAGE2_START_AT_TIMESTEP_ZERO_PROB}" \
    "${STAGE2_RESET_NOISE_SCALE}"
else
  run_distill_stage \
    "Single Stage" \
    "${BC_LOSS_COEF}" \
    "${RUN_NAME}" \
    "${TRAINING_NAME}" \
    "${MASTER_PORT}" \
    "" \
    "${SWITCH_TO_RL_AFTER}" \
    "${TAKE_TEACHER_ACTIONS}" \
    "${START_AT_TIMESTEP_ZERO_PROB}" \
    "${RESET_NOISE_SCALE}"
fi
