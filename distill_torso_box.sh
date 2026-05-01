#!/usr/bin/env bash
set -euo pipefail

# Base launcher for non-goal object distillation.
# Preferred entrypoint is `distill_root_box.sh`; this file is kept as a compatibility wrapper target.
# - actor_obs_root: sparse root command [rel_xy(2), rel_yaw(1)]
# - actor_obs_proprio: base_lin_vel, base_ang_vel, dof_pos, dof_vel; actor_obs_actions carries single-step action
# - actor_obs_box: optional wrapper-specific object state (for mocap variants)
#
# Single-stage run:
# - Pure DAgger by default
# - Optional DAgger/PPO scheduled mixing via ppo_start_epoch and dagger_end_epoch
# - Optional 0.5 teacher-action rollout mixing via teacher_action_mix_ratio

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/kge4jozt/model_12000.pt"}
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
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

# Resolve wandb:// teacher checkpoint once per launcher (node), then pass a local absolute path to all ranks.
if [[ "${TEACHER_CHECKPOINT}" == wandb://* ]]; then
  TEACHER_CACHE_ROOT=${TEACHER_CACHE_ROOT:-"${SCRIPT_DIR}/.teacher_checkpoints"}
  mkdir -p "${TEACHER_CACHE_ROOT}"
  TEACHER_CHECKPOINT=$(
    TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}" TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT}" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import wandb

uri = os.environ["TEACHER_CHECKPOINT"]
cache_root = Path(os.environ["TEACHER_CACHE_ROOT"]).expanduser()
prefix = "wandb://"
if not uri.startswith(prefix):
    raise ValueError(f"Expected wandb uri, got: {uri}")

parts = uri[len(prefix):].split("/")
if len(parts) < 4:
    raise ValueError(
        f"Invalid wandb checkpoint path: {uri}. "
        "Expected wandb://<entity>/<project>/<run_id>/<checkpoint_name>"
    )
entity, project, run_id = parts[:3]
file_name = "/".join(parts[3:])
cache_root.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")
downloaded = run.file(file_name).download(root=str(cache_root), replace=True)
print(str(Path(downloaded.name).resolve()))
PY
  )
fi

FORCE_EIGHT_GPU_CONFIG=${FORCE_EIGHT_GPU_CONFIG:-0}
if [[ "${FORCE_EIGHT_GPU_CONFIG}" != "0" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NPROC=8
  if command -v nvidia-smi >/dev/null 2>&1; then
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
    if [[ "${AVAILABLE_GPUS}" -lt 8 ]]; then
      echo "FORCE_EIGHT_GPU_CONFIG=1 requires >=8 visible GPUs, found ${AVAILABLE_GPUS}." >&2
      exit 1
    fi
  fi
else
  CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
  if [[ -z "${NPROC:-}" ]]; then
    NPROC="$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")"
  fi
fi

if [[ "${FORCE_EIGHT_GPU_CONFIG}" != "0" && "${NPROC}" -ne 8 ]]; then
  echo "Expected NPROC=8, got ${NPROC}." >&2
  exit 1
fi
if ! [[ "${NPROC}" =~ ^[0-9]+$ ]] || (( NPROC < 1 )); then
  echo "NPROC must be a positive integer. Got: ${NPROC}" >&2
  exit 1
fi

# In distill launchers, NUM_ENVS/PER_GPU_ENVS means envs per GPU. train_agent.py
# expects a global all-rank total and divides by WORLD_SIZE internally.
if [[ -n "${TOTAL_NUM_ENVS:-}" ]]; then
  if ! [[ "${TOTAL_NUM_ENVS}" =~ ^[0-9]+$ ]] || (( TOTAL_NUM_ENVS < NPROC )); then
    echo "TOTAL_NUM_ENVS must be an integer >= NPROC. Got TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}, NPROC=${NPROC}." >&2
    exit 1
  fi
  if (( TOTAL_NUM_ENVS % NPROC != 0 )); then
    echo "TOTAL_NUM_ENVS must be divisible by NPROC. Got TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}, NPROC=${NPROC}." >&2
    exit 1
  fi
  PER_GPU_ENVS=$((TOTAL_NUM_ENVS / NPROC))
  NUM_ENVS="${TOTAL_NUM_ENVS}"
else
  PER_GPU_ENVS=${PER_GPU_ENVS:-${NUM_ENVS:-4096}}
  if ! [[ "${PER_GPU_ENVS}" =~ ^[0-9]+$ ]] || (( PER_GPU_ENVS < 1 )); then
    echo "NUM_ENVS/PER_GPU_ENVS must be a positive per-GPU env count. Got: ${PER_GPU_ENVS:-<empty>}." >&2
    exit 1
  fi
  NUM_ENVS=$((PER_GPU_ENVS * NPROC))
fi

# Sim2real default: sparse root-command distill without clip_phase in student torso observation.
# Legacy option (old behavior with clip_phase):
#   EXP=g1-29dof-wbt-w-object-distill-sparse-root-cmd-legacy
EXP=${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}
DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml"
MOTION_DIR=${MOTION_DIR:-"${DEFAULT_MOTION_DIR}"}
DEFAULT_OBJECT_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
DEFAULT_OBJECT_MAP="${MOTION_DIR}/_clip_object_urdf_map.json"
if [[ -f "${DEFAULT_OBJECT_MAP}" ]]; then
  OBJECT_URDF=${OBJECT_URDF:-"${DEFAULT_OBJECT_MAP}"}
else
  OBJECT_URDF=${OBJECT_URDF:-"${DEFAULT_OBJECT_URDF}"}
fi

if [[ "${NPROC}" -lt 1 ]]; then
  echo "NPROC must be >= 1. Got: ${NPROC}" >&2
  exit 1
fi
if [[ "${NUM_ENVS}" -lt "${NPROC}" ]]; then
  echo "NUM_ENVS must be >= NPROC. Got NUM_ENVS=${NUM_ENVS}, NPROC=${NPROC}." >&2
  exit 1
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MAX_RESTARTS=${MAX_RESTARTS:-0}
TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-1800}

MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
ACTOR_LR=${ACTOR_LR:-7e-5}
CRITIC_LR=${CRITIC_LR:-7e-5}
# Distillation is sensitive to exploration noise; keep student near-deterministic by default.
ACTOR_MIN_NOISE_STD=${ACTOR_MIN_NOISE_STD:-0.01}
INIT_NOISE_STD=${INIT_NOISE_STD:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.005}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-268435456}
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-268435456}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-33554432}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
DISTILL_ENABLED=${DISTILL_ENABLED:-True}
DISTILL_MODE=${DISTILL_MODE:-dagger}
SWITCH_TO_RL_AFTER=${SWITCH_TO_RL_AFTER:-}
CLIP_TEACHER_ACTIONS=${CLIP_TEACHER_ACTIONS:-True}
CLIP_ACTIONS_THRESHOLD=${CLIP_ACTIONS_THRESHOLD:-8.0}
# Default teacher (kge4jozt/model_12000.pt) is trained with actor_obs-only input (181-dim).
# Legacy teachers may require actor_obs_legacy and/or perception_obs; override TEACHER_OBS_KEYS explicitly in that case.
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
STRICT_TEACHER_LOAD=${STRICT_TEACHER_LOAD:-True}
PERCEPTION_INTO_POLICY_MODULES=${PERCEPTION_INTO_POLICY_MODULES:-True}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
TEACHER_ACTION_MIX_RATIO_START=${TEACHER_ACTION_MIX_RATIO_START:-}
TEACHER_ACTION_MIX_RATIO_END=${TEACHER_ACTION_MIX_RATIO_END:-}
TEACHER_ACTION_MIX_RATIO_END_ITERATION=${TEACHER_ACTION_MIX_RATIO_END_ITERATION:-}
PPO_START_EPOCH=${PPO_START_EPOCH:-1000}
DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-10000}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-10.0}
DAGGER_MATCH_STD=${DAGGER_MATCH_STD:-False}
DISTILL_LOSS_TYPE=${DISTILL_LOSS_TYPE:-mse}
DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS:-True}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.0}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LOGGER=${LOGGER:-logger:wandb}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_sparse_root_cmd}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_sparse_root_cmd}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}

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
GLOBAL_EXTRA_ARGS=()
LOGGER_EXTRA_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == --logger.* ]]; then
    LOGGER_EXTRA_ARGS+=("${arg}")
  else
    GLOBAL_EXTRA_ARGS+=("${arg}")
  fi
done

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS} strict_teacher_load=${STRICT_TEACHER_LOAD}"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} dagger_loss_coef=${DAGGER_LOSS_COEF} teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
if [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" || -n "${TEACHER_ACTION_MIX_RATIO_END}" || -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
  echo "[INFO] teacher_action_mix_schedule=${TEACHER_ACTION_MIX_RATIO_START}->${TEACHER_ACTION_MIX_RATIO_END} end_iter=${TEACHER_ACTION_MIX_RATIO_END_ITERATION}"
fi
echo "[INFO] ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH}"
echo "[INFO] total_envs=${NUM_ENVS} world_size=${NPROC} per_gpu_envs=${PER_GPU_ENVS}"
echo "[INFO] init_noise_std=${INIT_NOISE_STD} actor_min_noise_std=${ACTOR_MIN_NOISE_STD} entropy_coef=${ENTROPY_COEF}"
echo "[INFO] physx_gpu_buffers found_lost_pairs=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY} found_lost_aggregate_pairs=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY} total_aggregate_pairs=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY} collision_stack=${PHYSX_GPU_COLLISION_STACK_SIZE}"
echo "[INFO] dagger_match_std=${DAGGER_MATCH_STD}"
echo "[INFO] default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S} default_pose_append=${ENABLE_DEFAULT_POSE_APPEND} append_duration_s=${DEFAULT_POSE_APPEND_DURATION_S}"

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
    --algo.config.distill.enabled="${DISTILL_ENABLED}"
    --algo.config.distill.mode="${DISTILL_MODE}"
    --algo.config.distill.policy-to-clone="${TEACHER_CHECKPOINT}"
    --algo.config.distill.bc-loss-coef="${stage_bc_loss_coef}"
    --algo.config.distill.clip-teacher-actions="${CLIP_TEACHER_ACTIONS}"
    --algo.config.distill.clip-actions-threshold="${CLIP_ACTIONS_THRESHOLD}"
    --algo.config.distill.teacher-obs-keys="${TEACHER_OBS_KEYS}"
    --algo.config.distill.strict-teacher-load="${STRICT_TEACHER_LOAD}"
    --algo.config.distill.teacher-action-mix-ratio="${TEACHER_ACTION_MIX_RATIO}"
    --algo.config.distill.ppo-start-epoch="${PPO_START_EPOCH}"
    --algo.config.distill.dagger-end-epoch="${DAGGER_END_EPOCH}"
    --algo.config.distill.dagger-loss-coef="${DAGGER_LOSS_COEF}"
    --algo.config.distill.dagger-match-std="${DAGGER_MATCH_STD}"
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
    --algo.config.entropy-coef="${ENTROPY_COEF}"
    --algo.config.module-dict.actor.min-noise-std="${ACTOR_MIN_NOISE_STD}"
    --algo.config.normalize-actor-obs=False
    --algo.config.normalize-critic-obs=False
    --algo.config.save-interval="${SAVE_INTERVAL}"
    --simulator.config.sim.physx.gpu-found-lost-pairs-capacity="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
    --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}"
    --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}"
    --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
    --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
    --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion="${PAIR_TERRAIN_WITH_MOTION}"
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob="${stage_start_at_timestep_zero_prob}"
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale="${stage_reset_noise_scale}"
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append="${ENABLE_DEFAULT_POSE_APPEND}"
    --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s="${DEFAULT_POSE_APPEND_DURATION_S}"
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend="${ENABLE_DEFAULT_POSE_PREPEND}"
    --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s="${DEFAULT_POSE_PREPEND_DURATION_S}"
    --robot.object.enabled=True
    --robot.object.object-urdf-path "${OBJECT_URDF}"
  )

  if [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" ]]; then
    cmd+=(--algo.config.distill.teacher-action-mix-ratio-start="${TEACHER_ACTION_MIX_RATIO_START}")
  fi
  if [[ -n "${TEACHER_ACTION_MIX_RATIO_END}" ]]; then
    cmd+=(--algo.config.distill.teacher-action-mix-ratio-end="${TEACHER_ACTION_MIX_RATIO_END}")
  fi
  if [[ -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
    cmd+=(--algo.config.distill.teacher-action-mix-ratio-end-iteration="${TEACHER_ACTION_MIX_RATIO_END_ITERATION}")
  fi

  if [[ -n "${stage_resume_checkpoint}" ]]; then
    cmd+=(--training.checkpoint "${stage_resume_checkpoint}")
  fi
  if [[ -n "${stage_switch_to_rl_after}" ]]; then
    cmd+=(--algo.config.distill.switch-to-rl-after="${stage_switch_to_rl_after}")
  fi
  cmd+=("${GLOBAL_EXTRA_ARGS[@]}")
  cmd+=("${LOGGER}")

  # logger:disabled does not accept logger sub-options such as --logger.name.
  # Keep legacy behavior for all other logger backends, but keep video logging disabled.
  if [[ "${LOGGER}" != "logger:disabled" ]]; then
    cmd+=(
      --logger.name="${stage_run_name}"
      --logger.video.enabled=False
      --logger.headless_recording=False
      --logger.video.upload_to_wandb=False
    )
  fi
  cmd+=("${LOGGER_EXTRA_ARGS[@]}")

  HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES="${PERCEPTION_INTO_POLICY_MODULES}" \
  TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${cmd[@]}"
}

run_distill_stage \
  "distill training" \
  "${BC_LOSS_COEF}" \
  "${RUN_NAME}" \
  "${TRAINING_NAME}" \
  "${MASTER_PORT}" \
  "" \
  "${SWITCH_TO_RL_AFTER}" \
  "${START_AT_TIMESTEP_ZERO_PROB}" \
  "${RESET_NOISE_SCALE}"
