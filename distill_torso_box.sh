#!/usr/bin/env bash
set -euo pipefail

# Distill an object-generalist teacher into an object-aware student with reduced actor inputs:
# - actor_obs_torso: sparse target root trajectory command
#   [rel_xy(2), rel_yaw(1), target_vxy(2), target_wz(1)]
# - actor_obs_proprio: base_lin_vel, base_ang_vel, dof_pos, dof_vel, actions
# - actor_obs_box: obj_target_pose_size_b = [obj_pos(3), obj_rot6d(6), obj_scale(3)]
#
# Single-stage run:
# - Pure DAgger by default
# - Optional DAgger/PPO scheduled mixing via ppo_start_epoch and dagger_end_epoch
# - Optional 0.5 teacher-action rollout mixing via teacher_action_mix_ratio

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/5vlz6pj8/model_10000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"

# Optional positional arg:
#   1) If first arg is a checkpoint path / wandb URI, use it as teacher checkpoint.
#   2) If first arg looks like an option (starts with '-'), keep default/env checkpoint and
#      forward all args to train_agent.py unchanged.
if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    shift
  elif [[ "$1" == -* ]]; then
    :
  else
    TEACHER_CHECKPOINT="$1"
    shift
  fi
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "Usage: $0 <teacher_checkpoint.pt> [extra train_agent.py args...]" >&2
  echo "Default teacher checkpoint: ${DEFAULT_TEACHER_CHECKPOINT}" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FORCE_EIGHT_GPU_CONFIG=${FORCE_EIGHT_GPU_CONFIG:-0}
if [[ "${FORCE_EIGHT_GPU_CONFIG}" != "0" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NPROC=8
  PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
  NUM_ENVS=${PER_GPU_ENVS}
  if command -v nvidia-smi >/dev/null 2>&1; then
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
    if [[ "${AVAILABLE_GPUS}" -lt 8 ]]; then
      echo "FORCE_EIGHT_GPU_CONFIG=1 requires >=8 visible GPUs, found ${AVAILABLE_GPUS}." >&2
      exit 1
    fi
  fi
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
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
  PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
  NUM_ENVS=${NUM_ENVS:-${PER_GPU_ENVS}}
fi

if [[ "${FORCE_EIGHT_GPU_CONFIG}" != "0" ]]; then
  if [[ "${NPROC}" -ne 8 ]]; then
    echo "Expected NPROC=8, got ${NPROC}." >&2
    exit 1
  fi
  if [[ "${NUM_ENVS}" -ne "${PER_GPU_ENVS}" ]]; then
    echo "Expected NUM_ENVS(per-GPU) == PER_GPU_ENVS, got NUM_ENVS=${NUM_ENVS}, PER_GPU_ENVS=${PER_GPU_ENVS}." >&2
    exit 1
  fi
fi

EXP=${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
OBJECT_URDF=${OBJECT_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}

if [[ "${NPROC}" -lt 1 ]]; then
  echo "NPROC must be >= 1. Got: ${NPROC}" >&2
  exit 1
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MAX_RESTARTS=${MAX_RESTARTS:-0}
TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-1800}

MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-10000}
ACTOR_LR=${ACTOR_LR:-7e-5}
CRITIC_LR=${CRITIC_LR:-7e-5}
# Distillation is sensitive to exploration noise; keep student near-deterministic by default.
ACTOR_MIN_NOISE_STD=${ACTOR_MIN_NOISE_STD:-0.01}
INIT_NOISE_STD=${INIT_NOISE_STD:-0.01}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
SWITCH_TO_RL_AFTER=${SWITCH_TO_RL_AFTER:-}
CLIP_TEACHER_ACTIONS=${CLIP_TEACHER_ACTIONS:-True}
CLIP_ACTIONS_THRESHOLD=${CLIP_ACTIONS_THRESHOLD:-8.0}
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs_legacy}
STRICT_TEACHER_LOAD=${STRICT_TEACHER_LOAD:-True}
PERCEPTION_INTO_POLICY_MODULES=${PERCEPTION_INTO_POLICY_MODULES:-True}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
PPO_START_EPOCH=${PPO_START_EPOCH:--1}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:--1}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-10.0}
DISTILL_LOSS_TYPE=${DISTILL_LOSS_TYPE:-mse}
DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS:-True}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}
LOGGER=${LOGGER:-logger:wandb}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_sparse_root_cmd}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_sparse_root_cmd}
TRAINING_PROJECT=${TRAINING_PROJECT:-WholeBodyTracking}

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

echo "[INFO] Distill teacher checkpoint: ${TEACHER_CHECKPOINT}"
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS} strict_teacher_load=${STRICT_TEACHER_LOAD}"
echo "[INFO] perception.inject_into_policy_modules=${PERCEPTION_INTO_POLICY_MODULES}"
echo "[INFO] Teacher observation mismatch will fail fast (no fallback)."
echo "[INFO] ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH} dagger_loss_coef=${DAGGER_LOSS_COEF}"
echo "[INFO] init_noise_std=${INIT_NOISE_STD} actor_min_noise_std=${ACTOR_MIN_NOISE_STD}"
echo "[INFO] per_gpu_envs=${NUM_ENVS} world_size=${NPROC} total_envs=$((NUM_ENVS * NPROC))"
echo "[INFO] torch_dist_timeout_sec=${TORCH_DIST_TIMEOUT_SEC}"

run_distill_stage() {
  local stage_label="$1"
  local stage_bc_loss_coef="$2"
  local stage_run_name="$3"
  local stage_training_name="$4"
  local stage_master_port="$5"
  local stage_resume_checkpoint="${6:-}"
  local stage_switch_to_rl_after="${7:-}"
  local stage_start_at_timestep_zero_prob="${8:-}"
  local stage_reset_noise_scale="${9:-}"

  echo "[INFO] Starting ${stage_label}"
  echo "[INFO]   run_name=${stage_run_name}"
  echo "[INFO]   training_name=${stage_training_name}"
  echo "[INFO]   bc_loss_coef=${stage_bc_loss_coef}"
  echo "[INFO]   teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
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
    --algo.config.distill.enabled=True
    --algo.config.distill.mode=dagger
    --algo.config.distill.policy-to-clone="${TEACHER_CHECKPOINT}"
    --algo.config.distill.bc-loss-coef="${stage_bc_loss_coef}"
    --algo.config.distill.clip-teacher-actions="${CLIP_TEACHER_ACTIONS}"
    --algo.config.distill.clip-actions-threshold="${CLIP_ACTIONS_THRESHOLD}"
    --algo.config.distill.teacher-obs-keys="${TEACHER_OBS_KEYS}"
    --algo.config.distill.strict-teacher-load="${STRICT_TEACHER_LOAD}"
    --perception.inject-into-policy-modules="${PERCEPTION_INTO_POLICY_MODULES}"
    --algo.config.distill.teacher-action-mix-ratio="${TEACHER_ACTION_MIX_RATIO}"
    --algo.config.distill.ppo-start-epoch="${PPO_START_EPOCH}"
    --algo.config.distill.dagger-end-epoch="${DAGGER_END_EPOCH}"
    --algo.config.distill.dagger-loss-coef="${DAGGER_LOSS_COEF}"
    --algo.config.distill.distill-loss-type="${DISTILL_LOSS_TYPE}"
    --algo.config.distill.dagger-ignore-zero-teacher-actions="${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS}"
    --training.num-envs="${NUM_ENVS}"
    --training.project="${TRAINING_PROJECT}"
    --training.name="${stage_training_name}"
    --training.multigpu=$([[ "${NPROC}" -gt 1 || "${NNODES}" -gt 1 ]] && echo True || echo False)
    --algo.config.num-learning-iterations="${NUM_LEARNING_ITERATIONS}"
    --algo.config.actor-learning-rate="${ACTOR_LR}"
    --algo.config.critic-learning-rate="${CRITIC_LR}"
    --algo.config.init-noise-std="${INIT_NOISE_STD}"
    --algo.config.module-dict.actor.min-noise-std="${ACTOR_MIN_NOISE_STD}"
    --algo.config.normalize-actor-obs=False
    --algo.config.normalize-critic-obs=False
    --algo.config.save-interval="${SAVE_INTERVAL}"
    --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
    --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
    --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion="${PAIR_TERRAIN_WITH_MOTION}"
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob="${stage_start_at_timestep_zero_prob}"
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale="${stage_reset_noise_scale}"
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=False
    --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=0
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=False
    --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0
    --robot.object.enabled=True
    --robot.object.object-urdf-path "${OBJECT_URDF}"
    "${LOGGER}"
    --logger.video.interval=1000
    --logger.name="${stage_run_name}"
  )

  if [[ -n "${stage_resume_checkpoint}" ]]; then
    cmd+=(--training.checkpoint "${stage_resume_checkpoint}")
  fi
  if [[ -n "${stage_switch_to_rl_after}" ]]; then
    cmd+=(--algo.config.distill.switch-to-rl-after="${stage_switch_to_rl_after}")
  fi
  cmd+=("${EXTRA_ARGS[@]}")

  TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
}

run_distill_stage \
  "Single Stage (VIRAL style)" \
  "${BC_LOSS_COEF}" \
  "${RUN_NAME}" \
  "${TRAINING_NAME}" \
  "${MASTER_PORT}" \
  "" \
  "${SWITCH_TO_RL_AFTER}" \
  "${START_AT_TIMESTEP_ZERO_PROB}" \
  "${RESET_NOISE_SCALE}"
