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

# Validate the objective label before sourcing helpers, probing GPUs, resolving
# checkpoints, or touching motion/object assets. In particular, legacy
# ``mode=mse`` always optimizes an actual MSE objective; accepting Huber here
# would make the experiment label and optimized loss disagree.
DISTILL_MODE=${DISTILL_MODE:-dagger}
DISTILL_LOSS_TYPE=${DISTILL_LOSS_TYPE:-mse}
case "${DISTILL_MODE}" in
  mse)
    if [[ "${DISTILL_LOSS_TYPE}" != "mse" ]]; then
      echo "DISTILL_MODE=mse requires DISTILL_LOSS_TYPE=mse. Got: ${DISTILL_LOSS_TYPE}" >&2
      exit 1
    fi
    DISTILL_COEF_CLI_FIELD=loss-coef
    DISTILL_COEF_LOG_NAME=loss_coef
    ;;
  dagger)
    case "${DISTILL_LOSS_TYPE}" in
      mse|huber)
        ;;
      *)
        echo "DISTILL_LOSS_TYPE must be exactly 'mse' or 'huber' in DAgger mode. Got: ${DISTILL_LOSS_TYPE}" >&2
        exit 1
        ;;
    esac
    DISTILL_COEF_CLI_FIELD=bc-loss-coef
    DISTILL_COEF_LOG_NAME=bc_loss_coef
    ;;
  *)
    echo "DISTILL_MODE must be exactly 'mse' or 'dagger'. Got: ${DISTILL_MODE}" >&2
    exit 1
    ;;
esac

# This string-valued controller knob can be rejected before sourcing Python/GPU
# helpers. Numeric LR/KL validation follows once the pinned interpreter exists.
PPO_LR_SCHEDULE=${PPO_LR_SCHEDULE:-adaptive}
case "${PPO_LR_SCHEDULE}" in
  adaptive|fixed)
    ;;
  *)
    echo "PPO_LR_SCHEDULE must be exactly adaptive or fixed. Got: ${PPO_LR_SCHEDULE}" >&2
    exit 1
    ;;
esac

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/kge4jozt/model_12000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"

# Optional positional arg:
#   1) If first arg is a checkpoint path / wandb URI, use it as teacher checkpoint.
#   2) If first arg looks like an option (starts with '-'), keep default/env checkpoint and
#      forward non-launcher-owned args to train_agent.py unchanged.
if [[ $# -gt 0 ]]; then
  # An option value may itself end in ``.pt`` (for example
  # ``--training.checkpoint=/tmp/model.pt``).  Classify option-shaped tokens
  # before checkpoint suffixes so the launcher-owned duplicate guard below
  # always sees them instead of silently consuming them as the teacher.
  if [[ "$1" == -* ]]; then
    :
  elif [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    shift
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

# A second copy of an objective field in the forwarded train CLI could override
# the already-validated launcher value. Reject those duplicates before any
# helper/GPU/checkpoint I/O; every field below has an environment/positional
# launcher input and is emitted in one mode-aware location later in this file.
for forwarded_arg in "$@"; do
  normalized_forwarded_arg=${forwarded_arg//_/-}
  forwarded_option=${normalized_forwarded_arg%%=*}
  case "${normalized_forwarded_arg}" in
    --algo.config.distill.*)
      forwarded_distill_field=${normalized_forwarded_arg#--algo.config.distill.}
      forwarded_distill_field=${forwarded_distill_field%%=*}
      case "${forwarded_distill_field}" in
        enabled|mode|policy-to-clone|teacher-checkpoint|loss-coef|bc-loss-coef|\
        clip-teacher-actions|clip-actions-threshold|teacher-obs-keys|strict-teacher-load|\
        distill-loss-type|teacher-action-mix-ratio|teacher-action-mix-ratio-start|\
        teacher-action-mix-ratio-end|teacher-action-mix-ratio-end-iteration|\
        ppo-start-epoch|dagger-end-epoch|dagger-loss-coef|dagger-match-std|\
        dagger-replay-enabled|dagger-replay-capacity|dagger-replay-batch-size|\
        dagger-replay-fraction|dagger-replay-seed|\
        dagger-ignore-zero-teacher-actions|switch-to-rl-after)
          echo "Do not override launcher-owned distillation field via forwarded CLI: ${forwarded_arg}" >&2
          exit 1
          ;;
      esac
      ;;
  esac
  case "${forwarded_option}" in
    --algo.config.schedule|\
    --algo.config.desired-kl|\
    --algo.config.actor-learning-rate|\
    --algo.config.critic-learning-rate|\
    --algo.config.min-actor-learning-rate|\
    --algo.config.max-actor-learning-rate|\
    --algo.config.min-critic-learning-rate|\
    --algo.config.max-critic-learning-rate)
      echo "Do not override launcher-owned PPO LR controller field via forwarded CLI: ${forwarded_arg}" >&2
      exit 1
      ;;
    --algo.config.num-learning-iterations|\
    --training.checkpoint|\
    --training.policy-init-checkpoint|\
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob|\
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end|\
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter|\
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter|\
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob|\
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end|\
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter|\
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter)
      echo "Do not override launcher-owned training/reset field via forwarded CLI: ${forwarded_arg}" >&2
      exit 1
      ;;
  esac
done
unset forwarded_arg normalized_forwarded_arg forwarded_option forwarded_distill_field

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/reset_curriculum_contract.sh"
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

# Kit/Carbonite settings are consumed by IsaacLab's AppLauncher, not by the
# strict Tyro training CLI.  Keep the optional worker-count override in the
# supported AppLauncher ``kit_args`` channel so enabling it cannot become an
# unknown training option.  The scientific batch launcher clears both ambient
# Kit aliases before reaching this wrapper; direct callers must choose one
# authority instead of silently combining independently-authored strings.
HOLOSOMA_CARB_TASKING_THREAD_COUNT=${HOLOSOMA_CARB_TASKING_THREAD_COUNT:-}
if [[ -n "${HOLOSOMA_CARB_TASKING_THREAD_COUNT}" ]]; then
  holosoma_canonicalize_positive_int32 \
    HOLOSOMA_CARB_TASKING_THREAD_COUNT || exit
  if [[ -n "${HOLOSOMA_ISAACSIM_KIT_ARGS:-}" \
        || -n "${ISAACSIM_KIT_ARGS:-}" ]]; then
    echo "HOLOSOMA_CARB_TASKING_THREAD_COUNT cannot be combined with HOLOSOMA_ISAACSIM_KIT_ARGS or ISAACSIM_KIT_ARGS." >&2
    exit 1
  fi
  HOLOSOMA_ISAACSIM_KIT_ARGS="--/plugins/carb.tasking.plugin/threadCount=${HOLOSOMA_CARB_TASKING_THREAD_COUNT}"
  export HOLOSOMA_ISAACSIM_KIT_ARGS
fi
export HOLOSOMA_CARB_TASKING_THREAD_COUNT

flag_enabled() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_zero_number() {
  [[ "${1:-}" =~ ^[+-]?((0+([.]0*)?)|([.]0+))([eE][+-]?[0-9]+)?$ ]]
}

validate_canonical_uint32() {
  local name="$1" value="$2" allow_zero="$3" maximum=2147483647
  local LC_ALL=C
  if [[ "${allow_zero}" == 1 ]]; then
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || {
      echo "${name} must be a canonical integer in [0, ${maximum}]. Got: ${value}" >&2
      return 1
    }
  else
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
      echo "${name} must be a canonical integer in [1, ${maximum}]. Got: ${value}" >&2
      return 1
    }
  fi
  if (( ${#value} > ${#maximum} )) \
      || { (( ${#value} == ${#maximum} )) && [[ "${value}" > "${maximum}" ]]; }; then
    echo "${name} must be a canonical integer in [$((1 - allow_zero)), ${maximum}]. Got: ${value}" >&2
    return 1
  fi
}

case "$(echo "${HOLOSOMA_RANK_VISIBLE_DEVICES:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    HOLOSOMA_RANK_VISIBLE_DEVICES=1
    ;;
  0|false|no|off|"")
    HOLOSOMA_RANK_VISIBLE_DEVICES=0
    ;;
  *)
    echo "HOLOSOMA_RANK_VISIBLE_DEVICES must be a boolean. Got: ${HOLOSOMA_RANK_VISIBLE_DEVICES}" >&2
    exit 1
    ;;
esac
export HOLOSOMA_RANK_VISIBLE_DEVICES
case "$(echo "${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
    ;;
  0|false|no|off|"")
    HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=0
    ;;
  *)
    echo "HOLOSOMA_RANK_LOCAL_CPU_AFFINITY must be a boolean. Got: ${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY}" >&2
    exit 1
    ;;
esac
export HOLOSOMA_RANK_LOCAL_CPU_AFFINITY
# These aliases are authored by train_agent_rank_visible.py after torchrun has
# assigned this child.  Stale values from an earlier launch would otherwise be
# consumed even by the ordinary train entry and can select the wrong GPU.
unset HOLOSOMA_ORIGINAL_LOCAL_RANK
unset HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE
unset HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES

FORCE_EIGHT_GPU_CONFIG=${FORCE_EIGHT_GPU_CONFIG:-0}
case "$(echo "${FORCE_EIGHT_GPU_CONFIG}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    FORCE_EIGHT_GPU_CONFIG=1
    ;;
  0|false|no|off|"")
    FORCE_EIGHT_GPU_CONFIG=0
    ;;
  *)
    echo "FORCE_EIGHT_GPU_CONFIG must be a boolean. Got: ${FORCE_EIGHT_GPU_CONFIG}" >&2
    exit 1
    ;;
esac
if [[ "${FORCE_EIGHT_GPU_CONFIG}" == "1" ]]; then
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

if [[ "${FORCE_EIGHT_GPU_CONFIG}" == "1" && "${NPROC}" -ne 8 ]]; then
  echo "Expected NPROC=8, got ${NPROC}." >&2
  exit 1
fi
if ! [[ "${NPROC}" =~ ^[0-9]+$ ]] || (( NPROC < 1 )); then
  echo "NPROC must be a positive integer. Got: ${NPROC}" >&2
  exit 1
fi
VISIBLE_DEVICE_COUNT=$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")
if (( NPROC > VISIBLE_DEVICE_COUNT )); then
  echo "NPROC=${NPROC} exceeds CUDA_VISIBLE_DEVICES count=${VISIBLE_DEVICE_COUNT}: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR_EXPLICIT=0
MASTER_PORT_EXPLICIT=0
[[ -n "${MASTER_ADDR:-}" ]] && MASTER_ADDR_EXPLICIT=1
[[ -n "${MASTER_PORT:-}" ]] && MASTER_PORT_EXPLICIT=1
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
MAX_RESTARTS=${MAX_RESTARTS:-0}
TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-1800}
if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || (( NNODES < 1 )); then
  echo "NNODES must be a positive integer. Got: ${NNODES}" >&2
  exit 1
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be an integer in [0, NNODES). Got NODE_RANK=${NODE_RANK}, NNODES=${NNODES}." >&2
  exit 1
fi
if (( NNODES > 1 )) && (( MASTER_ADDR_EXPLICIT == 0 || MASTER_PORT_EXPLICIT == 0 )); then
  echo "Multi-node launch requires explicit shared MASTER_ADDR and MASTER_PORT on every node." >&2
  echo "Per-node loopback/random defaults cannot form one torchrun rendezvous." >&2
  exit 1
fi
if [[ ! "${MASTER_ADDR}" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}$ ]]; then
  echo "MASTER_ADDR must be a safe host/IP identifier. Got: ${MASTER_ADDR}" >&2
  exit 1
fi
if ! [[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "MASTER_PORT must be an integer in [1, 65535]. Got: ${MASTER_PORT}" >&2
  exit 1
fi
if ! [[ "${MAX_RESTARTS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "MAX_RESTARTS must be a non-negative integer. Got: ${MAX_RESTARTS}" >&2
  exit 1
fi
if (( MAX_RESTARTS != 0 )); then
  echo "Scientific distillation requires MAX_RESTARTS=0; torchrun restart would replay from the original checkpoint/fresh state, not the latest exact distributed state." >&2
  exit 1
fi
GLOBAL_WORLD_SIZE=$((NPROC * NNODES))
TRAINING_SEED=${TRAINING_SEED:-${SEED:-}}
if [[ -n "${TRAINING_SEED}" ]]; then
  if [[ ! "${TRAINING_SEED}" =~ ^[0-9]+$ ]] \
      || (( ${#TRAINING_SEED} > 10 )) \
      || (( 10#${TRAINING_SEED} > 4294967295 )); then
    echo "TRAINING_SEED must be an integer in [0, 4294967295]. Got: ${TRAINING_SEED}" >&2
    exit 1
  fi
  MAX_TRAINING_BASE_SEED=$((4294967295 - GLOBAL_WORLD_SIZE + 1))
  if (( 10#${TRAINING_SEED} > MAX_TRAINING_BASE_SEED )); then
    echo "TRAINING_SEED plus rank offsets must stay <= 4294967295. Got seed=${TRAINING_SEED}, world_size=${GLOBAL_WORLD_SIZE}, max_base=${MAX_TRAINING_BASE_SEED}" >&2
    exit 1
  fi
  unset MAX_TRAINING_BASE_SEED
fi

# In distill launchers, NUM_ENVS/PER_GPU_ENVS means envs per GPU. train_agent.py
# expects a global all-rank total and divides by WORLD_SIZE internally.
if [[ -n "${TOTAL_NUM_ENVS:-}" ]]; then
  if ! [[ "${TOTAL_NUM_ENVS}" =~ ^[0-9]+$ ]] || (( TOTAL_NUM_ENVS < GLOBAL_WORLD_SIZE )); then
    echo "TOTAL_NUM_ENVS must be an integer >= global world size. Got TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}, world_size=${GLOBAL_WORLD_SIZE}." >&2
    exit 1
  fi
  if (( TOTAL_NUM_ENVS % GLOBAL_WORLD_SIZE != 0 )); then
    echo "TOTAL_NUM_ENVS must be divisible by global world size. Got TOTAL_NUM_ENVS=${TOTAL_NUM_ENVS}, world_size=${GLOBAL_WORLD_SIZE}." >&2
    exit 1
  fi
  PER_GPU_ENVS=$((TOTAL_NUM_ENVS / GLOBAL_WORLD_SIZE))
  NUM_ENVS="${TOTAL_NUM_ENVS}"
else
  PER_GPU_ENVS=${PER_GPU_ENVS:-${NUM_ENVS:-4096}}
  if ! [[ "${PER_GPU_ENVS}" =~ ^[0-9]+$ ]] || (( PER_GPU_ENVS < 1 )); then
    echo "NUM_ENVS/PER_GPU_ENVS must be a positive per-GPU env count. Got: ${PER_GPU_ENVS:-<empty>}." >&2
    exit 1
  fi
  NUM_ENVS=$((PER_GPU_ENVS * GLOBAL_WORLD_SIZE))
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

NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
holosoma_canonicalize_positive_int32 NUM_LEARNING_ITERATIONS || exit
HOLOSOMA_VALIDATED_RESET_CURRICULUM=${HOLOSOMA_VALIDATED_RESET_CURRICULUM:-0}
case "${HOLOSOMA_VALIDATED_RESET_CURRICULUM}" in
  0)
    ;;
  1)
    for reset_contract_name in \
      START_AT_TIMESTEP_ZERO_PROB \
      START_AT_TIMESTEP_ZERO_PROB_END \
      START_AT_TIMESTEP_ZERO_PROB_START_ITER \
      START_AT_TIMESTEP_ZERO_PROB_END_ITER \
      FREEZE_AT_TIMESTEP_ZERO_PROB \
      FREEZE_AT_TIMESTEP_ZERO_PROB_END \
      FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER \
      FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER; do
      if ! [[ -v "${reset_contract_name}" ]] || [[ -z "${!reset_contract_name}" ]]; then
        echo "Validated reset curriculum is missing ${reset_contract_name}." >&2
        exit 1
      fi
    done
    unset reset_contract_name
    holosoma_configure_all_reset_curricula NUM_LEARNING_ITERATIONS || exit
    ;;
  *)
    echo "HOLOSOMA_VALIDATED_RESET_CURRICULUM must be exactly 0 or 1. Got: ${HOLOSOMA_VALIDATED_RESET_CURRICULUM}" >&2
    exit 1
    ;;
esac
ACTOR_LR=${ACTOR_LR:-1e-3}
CRITIC_LR=${CRITIC_LR:-1e-3}
PPO_DESIRED_KL=${PPO_DESIRED_KL:-0.01}

# Make PPO's former implicit LR bounds explicit while preserving the exact
# historical resolver: min(initial, 1e-5), max(initial, 1e-2). The batch
# controller normally supplies all four; direct invocations get the same
# materialized contract instead of falling back to an invisible config value.
materialize_default_lr_bound() {
  local name="$1"
  local initial="$2"
  local bound_kind="$3"
  "${PYTHON_BIN}" - "${name}" "${initial}" "${bound_kind}" <<'PY'
from __future__ import annotations

import math
import sys

name, raw_initial, bound_kind = sys.argv[1:]
try:
    initial = float(raw_initial)
except (TypeError, ValueError, OverflowError) as exc:
    raise SystemExit(f"{name} must be numeric. Got: {raw_initial}") from exc
if not math.isfinite(initial):
    raise SystemExit(f"{name} must be finite. Got: {raw_initial}")
if initial <= 0.0:
    raise SystemExit(f"{name} must be finite and > 0. Got: {raw_initial}")
if bound_kind == "min":
    value = min(initial, 1.0e-5)
elif bound_kind == "max":
    value = max(initial, 1.0e-2)
else:
    raise SystemExit(f"Unsupported LR bound kind: {bound_kind}")
print(repr(value))
PY
}
ACTOR_MIN_LR=${ACTOR_MIN_LR:-$(materialize_default_lr_bound ACTOR_LR "${ACTOR_LR}" min)}
ACTOR_MAX_LR=${ACTOR_MAX_LR:-$(materialize_default_lr_bound ACTOR_LR "${ACTOR_LR}" max)}
CRITIC_MIN_LR=${CRITIC_MIN_LR:-$(materialize_default_lr_bound CRITIC_LR "${CRITIC_LR}" min)}
CRITIC_MAX_LR=${CRITIC_MAX_LR:-$(materialize_default_lr_bound CRITIC_LR "${CRITIC_LR}" max)}
unset -f materialize_default_lr_bound

"${PYTHON_BIN}" - \
  "${PPO_DESIRED_KL}" \
  "${ACTOR_LR}" \
  "${ACTOR_MIN_LR}" \
  "${ACTOR_MAX_LR}" \
  "${CRITIC_LR}" \
  "${CRITIC_MIN_LR}" \
  "${CRITIC_MAX_LR}" <<'PY'
from __future__ import annotations

import math
import sys

names = (
    "PPO_DESIRED_KL",
    "ACTOR_LR",
    "ACTOR_MIN_LR",
    "ACTOR_MAX_LR",
    "CRITIC_LR",
    "CRITIC_MIN_LR",
    "CRITIC_MAX_LR",
)
raw_values = dict(zip(names, sys.argv[1:], strict=True))
values: dict[str, float] = {}
for name, raw in raw_values.items():
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"{name} must be numeric. Got: {raw}") from exc
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be finite. Got: {raw}")
    if value <= 0.0:
        raise SystemExit(f"{name} must be finite and > 0. Got: {raw}")
    values[name] = value

for optimizer_name in ("ACTOR", "CRITIC"):
    initial_name = f"{optimizer_name}_LR"
    minimum_name = f"{optimizer_name}_MIN_LR"
    maximum_name = f"{optimizer_name}_MAX_LR"
    if values[minimum_name] > values[initial_name]:
        raise SystemExit(
            f"{minimum_name} must be <= {initial_name}; "
            f"got {raw_values[minimum_name]}>{raw_values[initial_name]}."
        )
    if values[initial_name] > values[maximum_name]:
        raise SystemExit(
            f"{initial_name} must be <= {maximum_name}; "
            f"got {raw_values[initial_name]}>{raw_values[maximum_name]}."
        )
PY
# Distillation is sensitive to exploration noise; keep student near-deterministic by default.
ACTOR_MIN_NOISE_STD=${ACTOR_MIN_NOISE_STD:-0.01}
INIT_NOISE_STD=${INIT_NOISE_STD:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.005}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}
PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-268435456}
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-268435456}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY_MIN=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY_MIN:-67108864}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY_MIN}}
for positive_capacity_name in \
  PHYSX_GPU_COLLISION_STACK_SIZE \
  PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY \
  PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY \
  PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY_MIN \
  PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY; do
  positive_capacity_value=${!positive_capacity_name}
  if ! [[ "${positive_capacity_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${positive_capacity_name} must be a positive integer. Got: ${positive_capacity_value}" >&2
    exit 1
  fi
done
unset positive_capacity_name positive_capacity_value
if (( PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY < PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY_MIN )); then
  echo "[WARN] Raising PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY from ${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY} to ${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY_MIN}; PhysX reported AS real-mesh runs need >47M aggregate pairs." >&2
  PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY_MIN}"
fi
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
DISTILL_ENABLED=${DISTILL_ENABLED:-True}
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
if [[ "${DISTILL_MODE}" == "dagger" ]]; then
  PPO_START_EPOCH=${PPO_START_EPOCH:-1000}
  DAGGER_END_EPOCH=${DAGGER_END_EPOCH:-10000}
  DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-10.0}
  DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS:-True}
else
  PPO_START_EPOCH=${PPO_START_EPOCH:--1}
  DAGGER_END_EPOCH=${DAGGER_END_EPOCH:--1}
  DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-0.0}
  DAGGER_IGNORE_ZERO_TEACHER_ACTIONS=${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS:-False}
fi
DAGGER_MATCH_STD=${DAGGER_MATCH_STD:-False}
DAGGER_REPLAY_ENABLED=${DAGGER_REPLAY_ENABLED:-False}
DAGGER_REPLAY_CAPACITY=${DAGGER_REPLAY_CAPACITY:-512}
DAGGER_REPLAY_BATCH_SIZE=${DAGGER_REPLAY_BATCH_SIZE:-512}
DAGGER_REPLAY_FRACTION=${DAGGER_REPLAY_FRACTION:-0.5}
DAGGER_REPLAY_SEED=${DAGGER_REPLAY_SEED:-0}
case "$(echo "${DAGGER_REPLAY_ENABLED}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    DAGGER_REPLAY_ENABLED=True
    ;;
  0|false|no|off)
    DAGGER_REPLAY_ENABLED=False
    ;;
  *)
    echo "DAGGER_REPLAY_ENABLED must be a boolean. Got: ${DAGGER_REPLAY_ENABLED}" >&2
    exit 1
    ;;
esac
validate_canonical_uint32 DAGGER_REPLAY_CAPACITY "${DAGGER_REPLAY_CAPACITY}" 0 || exit
validate_canonical_uint32 DAGGER_REPLAY_BATCH_SIZE "${DAGGER_REPLAY_BATCH_SIZE}" 0 || exit
validate_canonical_uint32 DAGGER_REPLAY_SEED "${DAGGER_REPLAY_SEED}" 1 || exit
"${PYTHON_BIN}" - "${DAGGER_REPLAY_FRACTION}" <<'PY'
from __future__ import annotations

import math
import sys

raw = sys.argv[1]
try:
    value = float(raw)
except (TypeError, ValueError, OverflowError) as exc:
    raise SystemExit(f"DAGGER_REPLAY_FRACTION must be numeric. Got: {raw}") from exc
if not math.isfinite(value) or not 0.0 < value < 1.0:
    raise SystemExit(
        "DAGGER_REPLAY_FRACTION must be finite and strictly between 0 and 1. "
        f"Got: {raw}"
    )
PY
if [[ "${DAGGER_REPLAY_ENABLED}" == True ]]; then
  if [[ "${DISTILL_MODE}" != dagger ]]; then
    echo "DAGGER_REPLAY_ENABLED=True requires DISTILL_MODE=dagger." >&2
    exit 1
  fi
  case "$(echo "${DISTILL_ENABLED}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      ;;
    *)
      echo "DAGGER_REPLAY_ENABLED=True requires DISTILL_ENABLED=True." >&2
      exit 1
      ;;
  esac
  case "$(echo "${DAGGER_MATCH_STD}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      ;;
    *)
      echo "DAGGER_REPLAY_ENABLED=True requires DAGGER_MATCH_STD=False because replay stores no teacher std." >&2
      exit 1
      ;;
  esac
fi

if [[ "${DISTILL_MODE}" == "mse" ]]; then
  if [[ "${PPO_START_EPOCH}" != "-1" || "${DAGGER_END_EPOCH}" != "-1" ]]; then
    echo "DISTILL_MODE=mse requires PPO_START_EPOCH=-1 and DAGGER_END_EPOCH=-1; the PPO/DAgger schedule is not an MSE objective." >&2
    exit 1
  fi
  if ! is_zero_number "${TEACHER_ACTION_MIX_RATIO}"; then
    echo "DISTILL_MODE=mse requires TEACHER_ACTION_MIX_RATIO=0; teacher-action rollout mixing is DAgger-only." >&2
    exit 1
  fi
  if [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" || -n "${TEACHER_ACTION_MIX_RATIO_END}" || -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
    echo "DISTILL_MODE=mse does not accept TEACHER_ACTION_MIX_RATIO_START/END/END_ITERATION; rollout mix scheduling is DAgger-only." >&2
    exit 1
  fi
  case "${SWITCH_TO_RL_AFTER}" in
    ""|-1|0)
      ;;
    *)
      echo "DISTILL_MODE=mse requires SWITCH_TO_RL_AFTER to be empty, -1, or 0; BC-to-RL switching is DAgger-only." >&2
      exit 1
      ;;
  esac
  if ! is_zero_number "${DAGGER_LOSS_COEF}"; then
    echo "DISTILL_MODE=mse requires DAGGER_LOSS_COEF=0; dagger_loss_coef is not consumed by the MSE objective." >&2
    exit 1
  fi
  for dagger_boolean_name in DAGGER_MATCH_STD DAGGER_IGNORE_ZERO_TEACHER_ACTIONS; do
    dagger_boolean_value=${!dagger_boolean_name}
    case "$(echo "${dagger_boolean_value}" | tr '[:upper:]' '[:lower:]')" in
      0|false|no|off|"")
        ;;
      1|true|yes|on)
        echo "DISTILL_MODE=mse requires ${dagger_boolean_name}=False; this option is DAgger-only." >&2
        exit 1
        ;;
      *)
        echo "${dagger_boolean_name} must be a boolean. Got: ${dagger_boolean_value}" >&2
        exit 1
        ;;
    esac
  done
  unset dagger_boolean_name dagger_boolean_value
fi
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0.0}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LOGGER=${LOGGER:-logger:wandb}
LOGGER_BASE_DIR=${LOGGER_BASE_DIR:-}
RUN_NAME=${RUN_NAME:-g1_w_object_distill_sparse_root_cmd}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_sparse_root_cmd}
TRAINING_PROJECT=${TRAINING_PROJECT:-boxer}
RESUME_CKPT=${RESUME_CKPT:-${RESUME_CHECKPOINT:-}}
POLICY_INIT_CKPT=${POLICY_INIT_CKPT:-${POLICY_INIT_CHECKPOINT:-}}

# Resolve and validate the exact teacher once in the node-level launcher, then
# pass one stable local path to every torchrun child.  This shares the same
# weights-only/no-symlink checkpoint reader used by training instead of the
# former unchecked W&B replace-in-place download.
TEACHER_CACHE_ROOT=${TEACHER_CACHE_ROOT:-"${HOME}/.cache/holosoma/teacher"}
TEACHER_CHECKPOINT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/resolve_exact_checkpoint.py" \
  --ref "${TEACHER_CHECKPOINT}" \
  --cache-root "${TEACHER_CACHE_ROOT}")
export TEACHER_CHECKPOINT
if [[ -n "${RESUME_CKPT}" && -n "${POLICY_INIT_CKPT}" ]]; then
  echo "RESUME_CKPT and POLICY_INIT_CKPT are mutually exclusive." >&2
  exit 1
fi
if [[ -n "${RESUME_CKPT}" && "${RESUME_CKPT}" != wandb://* && ! -f "${RESUME_CKPT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 1
fi
if [[ -n "${POLICY_INIT_CKPT}" && "${POLICY_INIT_CKPT}" != wandb://* && ! -f "${POLICY_INIT_CKPT}" ]]; then
  echo "Policy init checkpoint not found: ${POLICY_INIT_CKPT}" >&2
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
if [[ -n "${LOGGER_BASE_DIR}" && "${LOGGER_BASE_DIR}" != /* ]]; then
  echo "LOGGER_BASE_DIR must be an absolute path. Got: ${LOGGER_BASE_DIR}" >&2
  exit 1
fi

EXTRA_ARGS=("$@")
GLOBAL_EXTRA_SUBCOMMAND_ARGS=()
GLOBAL_EXTRA_ARGS=()
LOGGER_EXTRA_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    algo:*|simulator:*|terrain:*|perception:*|observation:*|action:*|reward:*|termination:*|randomization:*|command:*|curriculum:*|robot:*|nightly:*)
      # Tyro CascadeSubcommandArgs requires a namespace subcommand to be
      # selected before any --<namespace>.* overrides. Keep all forwarded
      # component selectors adjacent to the top-level exp selector.
      GLOBAL_EXTRA_SUBCOMMAND_ARGS+=("${arg}")
      ;;
    --logger.*)
      LOGGER_EXTRA_ARGS+=("${arg}")
      ;;
    *)
      GLOBAL_EXTRA_ARGS+=("${arg}")
      ;;
  esac
done

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[INFO] resume_checkpoint=${RESUME_CKPT}"
fi
if [[ -n "${POLICY_INIT_CKPT}" ]]; then
  echo "[INFO] policy_init_checkpoint=${POLICY_INIT_CKPT}"
fi
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS} strict_teacher_load=${STRICT_TEACHER_LOAD}"
if [[ "${DISTILL_MODE}" == "dagger" ]]; then
  echo "[INFO] distill_mode=${DISTILL_MODE} ${DISTILL_COEF_LOG_NAME}=${BC_LOSS_COEF} dagger_loss_coef=${DAGGER_LOSS_COEF} teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
else
  echo "[INFO] distill_mode=${DISTILL_MODE} ${DISTILL_COEF_LOG_NAME}=${BC_LOSS_COEF}"
fi
if [[ "${DISTILL_MODE}" == "dagger" ]] && [[ -n "${TEACHER_ACTION_MIX_RATIO_START}" || -n "${TEACHER_ACTION_MIX_RATIO_END}" || -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
  echo "[INFO] teacher_action_mix_schedule=${TEACHER_ACTION_MIX_RATIO_START}->${TEACHER_ACTION_MIX_RATIO_END} end_iter=${TEACHER_ACTION_MIX_RATIO_END_ITERATION}"
fi
if [[ "${DISTILL_MODE}" == "dagger" ]]; then
  echo "[INFO] ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH}"
fi
echo "[INFO] total_envs=${NUM_ENVS} nnodes=${NNODES} nproc_per_node=${NPROC} global_world_size=${GLOBAL_WORLD_SIZE} per_gpu_envs=${PER_GPU_ENVS}"
echo "[INFO] ppo_lr_controller schedule=${PPO_LR_SCHEDULE} desired_kl=${PPO_DESIRED_KL} actor_lr=${ACTOR_LR} actor_bounds=[${ACTOR_MIN_LR},${ACTOR_MAX_LR}] critic_lr=${CRITIC_LR} critic_bounds=[${CRITIC_MIN_LR},${CRITIC_MAX_LR}]"
effective_supervised_actor_microbatch=0
effective_supervised_actor_stream_backward=0
if flag_enabled "${HOLOSOMA_DAGGER_SUPERVISED_ONLY:-0}"; then
  effective_supervised_actor_microbatch=${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH:-0}
  effective_supervised_actor_stream_backward=${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD:-0}
fi
echo "[INFO] hybrid_flags supervised_only=${HOLOSOMA_DAGGER_SUPERVISED_ONLY:-<unset>} actor_only_step=${HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP:-<unset>} supervised_actor_microbatch_requested=${HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH:-<unset>} supervised_actor_microbatch_effective=${effective_supervised_actor_microbatch} stream_backward_requested=${HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD:-<unset>} stream_backward_effective=${effective_supervised_actor_stream_backward} skip_critic_weight_sync=${HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC:-<unset>} skip_loss_dict_accumulation=${HOLOSOMA_SKIP_LOSS_DICT_ACCUMULATION:-<unset>}"
unset effective_supervised_actor_microbatch effective_supervised_actor_stream_backward
echo "[INFO] init_noise_std=${INIT_NOISE_STD} actor_min_noise_std=${ACTOR_MIN_NOISE_STD} entropy_coef=${ENTROPY_COEF}"
echo "[INFO] physx_gpu_buffers found_lost_pairs=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY} found_lost_aggregate_pairs=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY} total_aggregate_pairs=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY} collision_stack=${PHYSX_GPU_COLLISION_STACK_SIZE}"
if [[ "${DISTILL_MODE}" == "dagger" ]]; then
  echo "[INFO] dagger_match_std=${DAGGER_MATCH_STD}"
  echo "[INFO] dagger_replay enabled=${DAGGER_REPLAY_ENABLED} capacity_per_rank=${DAGGER_REPLAY_CAPACITY} batch_per_update=${DAGGER_REPLAY_BATCH_SIZE} fraction=${DAGGER_REPLAY_FRACTION} seed=${DAGGER_REPLAY_SEED}"
fi
echo "[INFO] default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND} duration_s=${DEFAULT_POSE_PREPEND_DURATION_S} default_pose_append=${ENABLE_DEFAULT_POSE_APPEND} append_duration_s=${DEFAULT_POSE_APPEND_DURATION_S}"
echo "[INFO] logger_base_dir=${LOGGER_BASE_DIR:-<config-default>}"

run_distill_stage() {
  local stage_label="$1"
  local stage_distill_loss_coef="$2"
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

  local train_entry="src/holosoma/holosoma/train_agent.py"
  if flag_enabled "${HOLOSOMA_RANK_VISIBLE_DEVICES:-0}" \
      || flag_enabled "${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}"; then
    train_entry="src/holosoma/holosoma/train_agent_rank_visible.py"
  fi
  echo "[INFO]   train_entry=${train_entry} rank_visible_devices=${HOLOSOMA_RANK_VISIBLE_DEVICES:-0} rank_local_cpu_affinity=${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}"

  local train_args=(
    "exp:${EXP}"
    "${GLOBAL_EXTRA_SUBCOMMAND_ARGS[@]}"
    "${LOGGER}"
    --algo.config.distill.enabled="${DISTILL_ENABLED}"
    --algo.config.distill.mode="${DISTILL_MODE}"
    --algo.config.distill.policy-to-clone="${TEACHER_CHECKPOINT}"
    "--algo.config.distill.${DISTILL_COEF_CLI_FIELD}=${stage_distill_loss_coef}"
    --algo.config.distill.clip-teacher-actions="${CLIP_TEACHER_ACTIONS}"
    --algo.config.distill.clip-actions-threshold="${CLIP_ACTIONS_THRESHOLD}"
    --algo.config.distill.teacher-obs-keys="${TEACHER_OBS_KEYS}"
    --algo.config.distill.strict-teacher-load="${STRICT_TEACHER_LOAD}"
    --algo.config.distill.distill-loss-type="${DISTILL_LOSS_TYPE}"
  )

  if [[ "${DISTILL_MODE}" == "dagger" ]]; then
    train_args+=(
      --algo.config.distill.teacher-action-mix-ratio="${TEACHER_ACTION_MIX_RATIO}"
      --algo.config.distill.ppo-start-epoch="${PPO_START_EPOCH}"
      --algo.config.distill.dagger-end-epoch="${DAGGER_END_EPOCH}"
      --algo.config.distill.dagger-loss-coef="${DAGGER_LOSS_COEF}"
      --algo.config.distill.dagger-match-std="${DAGGER_MATCH_STD}"
      --algo.config.distill.dagger-replay-enabled="${DAGGER_REPLAY_ENABLED}"
      --algo.config.distill.dagger-replay-capacity="${DAGGER_REPLAY_CAPACITY}"
      --algo.config.distill.dagger-replay-batch-size="${DAGGER_REPLAY_BATCH_SIZE}"
      --algo.config.distill.dagger-replay-fraction="${DAGGER_REPLAY_FRACTION}"
      --algo.config.distill.dagger-replay-seed="${DAGGER_REPLAY_SEED}"
      --algo.config.distill.dagger-ignore-zero-teacher-actions="${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS}"
    )
  fi

  train_args+=(
    --training.num-envs="${NUM_ENVS}"
    --training.project="${TRAINING_PROJECT}"
    --training.name="${stage_training_name}"
    --training.multigpu=$([[ "${NPROC}" -gt 1 || "${NNODES}" -gt 1 ]] && echo True || echo False)
    --algo.config.num-learning-iterations="${NUM_LEARNING_ITERATIONS}"
    --algo.config.schedule="${PPO_LR_SCHEDULE}"
    --algo.config.desired-kl="${PPO_DESIRED_KL}"
    --algo.config.actor-learning-rate="${ACTOR_LR}"
    --algo.config.critic-learning-rate="${CRITIC_LR}"
    --algo.config.min-actor-learning-rate="${ACTOR_MIN_LR}"
    --algo.config.max-actor-learning-rate="${ACTOR_MAX_LR}"
    --algo.config.min-critic-learning-rate="${CRITIC_MIN_LR}"
    --algo.config.max-critic-learning-rate="${CRITIC_MAX_LR}"
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

  if [[ "${HOLOSOMA_VALIDATED_RESET_CURRICULUM}" == 1 ]]; then
    train_args+=(
      --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end="${START_AT_TIMESTEP_ZERO_PROB_END}"
      --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter="${START_AT_TIMESTEP_ZERO_PROB_START_ITER}"
      --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter="${START_AT_TIMESTEP_ZERO_PROB_END_ITER}"
      --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob="${FREEZE_AT_TIMESTEP_ZERO_PROB}"
      --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end="${FREEZE_AT_TIMESTEP_ZERO_PROB_END}"
      --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter="${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}"
      --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter="${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}"
    )
  fi

  if [[ -n "${TRAINING_SEED}" ]]; then
    train_args+=(--training.seed="${TRAINING_SEED}")
  fi

  if [[ "${DISTILL_MODE}" == "dagger" && -n "${TEACHER_ACTION_MIX_RATIO_START}" ]]; then
    train_args+=(--algo.config.distill.teacher-action-mix-ratio-start="${TEACHER_ACTION_MIX_RATIO_START}")
  fi
  if [[ "${DISTILL_MODE}" == "dagger" && -n "${TEACHER_ACTION_MIX_RATIO_END}" ]]; then
    train_args+=(--algo.config.distill.teacher-action-mix-ratio-end="${TEACHER_ACTION_MIX_RATIO_END}")
  fi
  if [[ "${DISTILL_MODE}" == "dagger" && -n "${TEACHER_ACTION_MIX_RATIO_END_ITERATION}" ]]; then
    train_args+=(--algo.config.distill.teacher-action-mix-ratio-end-iteration="${TEACHER_ACTION_MIX_RATIO_END_ITERATION}")
  fi

  if [[ -n "${stage_resume_checkpoint}" ]]; then
    train_args+=(--training.checkpoint "${stage_resume_checkpoint}")
  fi
  if [[ -n "${POLICY_INIT_CKPT}" ]]; then
    train_args+=(--training.policy-init-checkpoint "${POLICY_INIT_CKPT}")
  fi
  if [[ "${DISTILL_MODE}" == "dagger" && -n "${stage_switch_to_rl_after}" ]]; then
    train_args+=(--algo.config.distill.switch-to-rl-after="${stage_switch_to_rl_after}")
  fi
  if [[ -n "${LOGGER_BASE_DIR}" ]]; then
    train_args+=(--logger.base-dir="${LOGGER_BASE_DIR}")
  fi
  train_args+=("${GLOBAL_EXTRA_ARGS[@]}")

  # logger:disabled does not accept logger sub-options such as --logger.name.
  # Keep legacy behavior for all other logger backends, but keep video logging disabled.
  if [[ "${LOGGER}" != "logger:disabled" ]]; then
    train_args+=(
      --logger.name="${stage_run_name}"
      --logger.video.enabled=False
      --logger.headless_recording=False
      --logger.video.upload_to_wandb=False
    )
  fi
  train_args+=("${LOGGER_EXTRA_ARGS[@]}")

  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/validate_train_cli.py" \
    --expected-motion-end-mode "${STUDENT_MOTION_END_MODE:-}" \
    -- "${train_args[@]}"

  local cmd=(
    "${PYTHON_BIN}"
    -m
    torch.distributed.run
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --nproc_per_node="${NPROC}"
    --max_restarts="${MAX_RESTARTS}"
    --master_port="${stage_master_port}"
    "${train_entry}"
    "${train_args[@]}"
  )

  if [[ "${PRINT_TRAIN_CMD:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
    printf '[INFO] final_train_command:'
    printf ' %q' \
      HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES="${PERCEPTION_INTO_POLICY_MODULES}" \
      HOLOSOMA_RANK_VISIBLE_DEVICES="${HOLOSOMA_RANK_VISIBLE_DEVICES:-0}" \
      HOLOSOMA_RANK_LOCAL_CPU_AFFINITY="${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}" \
      TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      "${cmd[@]}"
    printf '\n'
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      return 0
    fi
  fi

  HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES="${PERCEPTION_INTO_POLICY_MODULES}" \
  HOLOSOMA_RANK_VISIBLE_DEVICES="${HOLOSOMA_RANK_VISIBLE_DEVICES:-0}" \
  HOLOSOMA_RANK_LOCAL_CPU_AFFINITY="${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}" \
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
  "${RESUME_CKPT}" \
  "${SWITCH_TO_RL_AFTER}" \
  "${START_AT_TIMESTEP_ZERO_PROB}" \
  "${RESET_NOISE_SCALE}"
