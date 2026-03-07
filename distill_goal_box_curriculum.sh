#!/usr/bin/env bash
set -euo pipefail

# Single-launch curriculum driver for sparse-goal distill -> goal-reaching finetune.
#
# Stage A (goal-consistent distill):
#   - external goal prob fixed at 0
#   - runs distillation-focused training
#
# Stage B (goal-reaching finetune):
#   - resumes from Stage A checkpoint
#   - disables distillation (pure RL update)
#   - enables external-goal probability ramp
#
# Usage:
#   bash distill_goal_box_curriculum.sh [teacher_checkpoint.pt|wandb://...] [extra train args...]
#
# Mode:
#   DISTILL_GOAL_MODE=mocap|perception   (default: mocap)
#
# Stage A knobs:
#   STAGE_A_ITERS                          (default: 3000)
#   STAGE_A_GOAL_EXTERNAL_PROB_START       (default: 0.0)
#   STAGE_A_GOAL_EXTERNAL_PROB_END         (default: 0.0)
#   STAGE_A_GOAL_EXTERNAL_PROB_RAMP_RESETS (default: 500000)
#   STAGE_A_BC_LOSS_COEF                   (default: 1.0)
#   STAGE_A_PPO_START_EPOCH                (default: 0)
#   STAGE_A_DAGGER_END_EPOCH               (default: 10000)
#
# Stage B knobs:
#   STAGE_B_ITERS                          (default: 7000)
#   STAGE_B_GOAL_EXTERNAL_PROB_START       (default: 0.05)
#   STAGE_B_GOAL_EXTERNAL_PROB_END         (default: 0.35)
#   STAGE_B_GOAL_EXTERNAL_PROB_RAMP_RESETS (default: 1500000)
#   STAGE_B_BC_LOSS_COEF                   (default: 0.0)
#   STAGE_B_PPO_START_EPOCH                (default: -1)
#   STAGE_B_DAGGER_END_EPOCH               (default: -1)
#
# Optional:
#   CURRICULUM_DRY_RUN=1                   (print commands only; do not start training)
#   LOGS_NEW_ROOT=/data/logs_new           (override log root for stage-A checkpoint lookup)
#
# Shared defaults forwarded to both stages (can override by env):
#   TEACHER_OBS_KEYS=actor_obs
#   GOAL_CLIP_DELTA_MIN_STEPS=30
#   GOAL_CLIP_DELTA_MAX_STEPS=180
#   LOGGER=logger:wandb
#   TEACHER_ACTION_MIX_RATIO=0.0
#   DAGGER_LOSS_COEF=1.0
#   PAIR_TERRAIN_WITH_MOTION=False
#   START_AT_TIMESTEP_ZERO_PROB=0.05
#   RESET_NOISE_SCALE=1.0
#   SAVE_INTERVAL=200
#   ACTOR_LR / CRITIC_LR:
#     - mocap defaults to 5e-5 / 5e-5
#     - perception defaults to 7e-5 / 7e-5 (same as distill_torso_box.sh)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

DISTILL_GOAL_MODE=${DISTILL_GOAL_MODE:-mocap}
case "${DISTILL_GOAL_MODE}" in
  mocap)
    MODE_SCRIPT="${SCRIPT_DIR}/distill_goal_box_mocap.sh"
    ;;
  perception)
    MODE_SCRIPT="${SCRIPT_DIR}/distill_goal_box_perception.sh"
    ;;
  *)
    echo "[ERROR] DISTILL_GOAL_MODE must be one of: mocap, perception. Got: ${DISTILL_GOAL_MODE}" >&2
    exit 2
    ;;
esac

DEFAULT_TEACHER_CHECKPOINT=${DEFAULT_TEACHER_CHECKPOINT:-"wandb://zihanw22/boxer/kge4jozt/model_12000.pt"}
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT}}"

EXTRA_ARGS=("$@")
if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    EXTRA_ARGS=("${@:2}")
  fi
fi

# Stage B controls resume checkpoint automatically; strip user-provided --training.checkpoint from passthrough args.
FILTERED_EXTRA_ARGS=()
SKIP_NEXT_ARG=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${SKIP_NEXT_ARG}" -eq 1 ]]; then
    SKIP_NEXT_ARG=0
    continue
  fi
  case "${arg}" in
    --training.checkpoint)
      echo "[WARN] Ignoring passthrough arg '--training.checkpoint <path>' in curriculum wrapper."
      SKIP_NEXT_ARG=1
      ;;
    --training.checkpoint=*)
      echo "[WARN] Ignoring passthrough arg '${arg}' in curriculum wrapper."
      ;;
    *)
      FILTERED_EXTRA_ARGS+=("${arg}")
      ;;
  esac
done
EXTRA_ARGS=("${FILTERED_EXTRA_ARGS[@]}")

STAGE_A_ITERS=${STAGE_A_ITERS:-3000}
STAGE_A_GOAL_EXTERNAL_PROB_START=${STAGE_A_GOAL_EXTERNAL_PROB_START:-0.0}
STAGE_A_GOAL_EXTERNAL_PROB_END=${STAGE_A_GOAL_EXTERNAL_PROB_END:-0.0}
STAGE_A_GOAL_EXTERNAL_PROB_RAMP_RESETS=${STAGE_A_GOAL_EXTERNAL_PROB_RAMP_RESETS:-500000}
STAGE_A_BC_LOSS_COEF=${STAGE_A_BC_LOSS_COEF:-1.0}
STAGE_A_PPO_START_EPOCH=${STAGE_A_PPO_START_EPOCH:-0}
STAGE_A_DAGGER_END_EPOCH=${STAGE_A_DAGGER_END_EPOCH:-10000}

STAGE_B_ITERS=${STAGE_B_ITERS:-7000}
STAGE_B_GOAL_EXTERNAL_PROB_START=${STAGE_B_GOAL_EXTERNAL_PROB_START:-0.05}
STAGE_B_GOAL_EXTERNAL_PROB_END=${STAGE_B_GOAL_EXTERNAL_PROB_END:-0.35}
STAGE_B_GOAL_EXTERNAL_PROB_RAMP_RESETS=${STAGE_B_GOAL_EXTERNAL_PROB_RAMP_RESETS:-1500000}
STAGE_B_BC_LOSS_COEF=${STAGE_B_BC_LOSS_COEF:-0.0}
STAGE_B_PPO_START_EPOCH=${STAGE_B_PPO_START_EPOCH:--1}
STAGE_B_DAGGER_END_EPOCH=${STAGE_B_DAGGER_END_EPOCH:--1}
CURRICULUM_DRY_RUN=${CURRICULUM_DRY_RUN:-0}
LOGS_NEW_ROOT=${LOGS_NEW_ROOT:-/data/logs_new}

# Shared knobs are made explicit here so defaults are visible at curriculum entrypoint.
TEACHER_OBS_KEYS=${TEACHER_OBS_KEYS:-actor_obs}
GOAL_CLIP_DELTA_MIN_STEPS=${GOAL_CLIP_DELTA_MIN_STEPS:-30}
GOAL_CLIP_DELTA_MAX_STEPS=${GOAL_CLIP_DELTA_MAX_STEPS:-180}
LOGGER=${LOGGER:-logger:wandb}
TEACHER_ACTION_MIX_RATIO=${TEACHER_ACTION_MIX_RATIO:-0.0}
DAGGER_LOSS_COEF=${DAGGER_LOSS_COEF:-1.0}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-200}

if [[ "${DISTILL_GOAL_MODE}" == "mocap" ]]; then
  ACTOR_LR=${ACTOR_LR:-5e-5}
  CRITIC_LR=${CRITIC_LR:-5e-5}
else
  ACTOR_LR=${ACTOR_LR:-7e-5}
  CRITIC_LR=${CRITIC_LR:-7e-5}
fi

if [[ "${STAGE_A_ITERS}" -lt 1 || "${STAGE_B_ITERS}" -lt 1 ]]; then
  echo "[ERROR] STAGE_A_ITERS and STAGE_B_ITERS must be >= 1." >&2
  exit 2
fi

CURRICULUM_TAG=${CURRICULUM_TAG:-$(date +%Y%m%d_%H%M%S)}
BASE_RUN_NAME=${RUN_NAME:-g1_w_object_distill_goal_box_${DISTILL_GOAL_MODE}_curriculum}
BASE_TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_goal_box_${DISTILL_GOAL_MODE}_curriculum}
STAGE_A_RUN_NAME="${BASE_RUN_NAME}_stageA_${CURRICULUM_TAG}"
STAGE_B_RUN_NAME="${BASE_RUN_NAME}_stageB_${CURRICULUM_TAG}"
STAGE_A_TRAINING_NAME="${BASE_TRAINING_NAME}_stageA_${CURRICULUM_TAG}"
STAGE_B_TRAINING_NAME="${BASE_TRAINING_NAME}_stageB_${CURRICULUM_TAG}"

echo "[INFO] Curriculum single-launch mode"
echo "[INFO] mode_script=${MODE_SCRIPT}"
echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] stageA: iters=${STAGE_A_ITERS}, ext_prob=${STAGE_A_GOAL_EXTERNAL_PROB_START}->${STAGE_A_GOAL_EXTERNAL_PROB_END}, bc=${STAGE_A_BC_LOSS_COEF}, ppo_start=${STAGE_A_PPO_START_EPOCH}, dagger_end=${STAGE_A_DAGGER_END_EPOCH}"
echo "[INFO] stageB: iters=${STAGE_B_ITERS}, ext_prob=${STAGE_B_GOAL_EXTERNAL_PROB_START}->${STAGE_B_GOAL_EXTERNAL_PROB_END}, bc=${STAGE_B_BC_LOSS_COEF}, ppo_start=${STAGE_B_PPO_START_EPOCH}, dagger_end=${STAGE_B_DAGGER_END_EPOCH}"
echo "[INFO] shared: teacher_obs_keys=${TEACHER_OBS_KEYS}, goal_delta=[${GOAL_CLIP_DELTA_MIN_STEPS},${GOAL_CLIP_DELTA_MAX_STEPS}], logger=${LOGGER}"
echo "[INFO] shared(opt): mix=${TEACHER_ACTION_MIX_RATIO}, dagger_loss=${DAGGER_LOSS_COEF}, pair_terrain_with_motion=${PAIR_TERRAIN_WITH_MOTION}"
echo "[INFO] shared(train): actor_lr=${ACTOR_LR}, critic_lr=${CRITIC_LR}, start_t0_prob=${START_AT_TIMESTEP_ZERO_PROB}, reset_noise_scale=${RESET_NOISE_SCALE}, save_interval=${SAVE_INTERVAL}"
echo "[INFO] logs_root=${LOGS_NEW_ROOT} dry_run=${CURRICULUM_DRY_RUN}"

print_cmd() {
  printf '%q ' "$@"
  echo
}

echo "[INFO] === Stage A: goal-consistent distill ==="
STAGE_A_CMD=(
  env
  RUN_NAME="${STAGE_A_RUN_NAME}"
  TRAINING_NAME="${STAGE_A_TRAINING_NAME}"
  NUM_LEARNING_ITERATIONS="${STAGE_A_ITERS}"
  GOAL_EXTERNAL_PROB_START="${STAGE_A_GOAL_EXTERNAL_PROB_START}"
  GOAL_EXTERNAL_PROB_END="${STAGE_A_GOAL_EXTERNAL_PROB_END}"
  GOAL_EXTERNAL_PROB_RAMP_RESETS="${STAGE_A_GOAL_EXTERNAL_PROB_RAMP_RESETS}"
  GOAL_CLIP_DELTA_MIN_STEPS="${GOAL_CLIP_DELTA_MIN_STEPS}"
  GOAL_CLIP_DELTA_MAX_STEPS="${GOAL_CLIP_DELTA_MAX_STEPS}"
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}"
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}"
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}"
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}"
  ACTOR_LR="${ACTOR_LR}"
  CRITIC_LR="${CRITIC_LR}"
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}"
  RESET_NOISE_SCALE="${RESET_NOISE_SCALE}"
  SAVE_INTERVAL="${SAVE_INTERVAL}"
  BC_LOSS_COEF="${STAGE_A_BC_LOSS_COEF}"
  PPO_START_EPOCH="${STAGE_A_PPO_START_EPOCH}"
  DAGGER_END_EPOCH="${STAGE_A_DAGGER_END_EPOCH}"
  LOGGER="${LOGGER}"
  bash "${MODE_SCRIPT}" "${TEACHER_CHECKPOINT}"
)
STAGE_A_CMD+=("${EXTRA_ARGS[@]}")
if [[ "${CURRICULUM_DRY_RUN}" != "0" ]]; then
  echo "[DRYRUN] Stage A command:"
  print_cmd "${STAGE_A_CMD[@]}"
  echo "[DRYRUN] Stage B command is generated after Stage A checkpoint is produced."
  exit 0
fi
"${STAGE_A_CMD[@]}"

STAGE_A_DIR=$(find "${LOGS_NEW_ROOT}" -maxdepth 3 -type d -name "*-${STAGE_A_TRAINING_NAME}-locomotion" | sort | tail -n 1)
if [[ -z "${STAGE_A_DIR}" ]]; then
  echo "[ERROR] Failed to locate Stage A log directory for training name: ${STAGE_A_TRAINING_NAME}" >&2
  exit 2
fi

EXPECTED_STAGE_A_CKPT="${STAGE_A_DIR}/model_$(printf "%05d" $((STAGE_A_ITERS - 1))).pt"
if [[ -f "${EXPECTED_STAGE_A_CKPT}" ]]; then
  STAGE_A_CKPT="${EXPECTED_STAGE_A_CKPT}"
else
  STAGE_A_CKPT=$(find "${STAGE_A_DIR}" -maxdepth 1 -type f -name "model_*.pt" | sort | tail -n 1)
fi
if [[ -z "${STAGE_A_CKPT}" || ! -f "${STAGE_A_CKPT}" ]]; then
  echo "[ERROR] Failed to locate Stage A checkpoint in: ${STAGE_A_DIR}" >&2
  exit 2
fi
echo "[INFO] Stage A checkpoint: ${STAGE_A_CKPT}"

echo "[INFO] === Stage B: goal-reaching finetune (pure RL + external goal ramp) ==="
STAGE_B_CMD=(
  env
  RUN_NAME="${STAGE_B_RUN_NAME}"
  TRAINING_NAME="${STAGE_B_TRAINING_NAME}"
  NUM_LEARNING_ITERATIONS="${STAGE_B_ITERS}"
  GOAL_EXTERNAL_PROB_START="${STAGE_B_GOAL_EXTERNAL_PROB_START}"
  GOAL_EXTERNAL_PROB_END="${STAGE_B_GOAL_EXTERNAL_PROB_END}"
  GOAL_EXTERNAL_PROB_RAMP_RESETS="${STAGE_B_GOAL_EXTERNAL_PROB_RAMP_RESETS}"
  GOAL_CLIP_DELTA_MIN_STEPS="${GOAL_CLIP_DELTA_MIN_STEPS}"
  GOAL_CLIP_DELTA_MAX_STEPS="${GOAL_CLIP_DELTA_MAX_STEPS}"
  TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS}"
  TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO}"
  DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF}"
  PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}"
  ACTOR_LR="${ACTOR_LR}"
  CRITIC_LR="${CRITIC_LR}"
  START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB}"
  RESET_NOISE_SCALE="${RESET_NOISE_SCALE}"
  SAVE_INTERVAL="${SAVE_INTERVAL}"
  BC_LOSS_COEF="${STAGE_B_BC_LOSS_COEF}"
  PPO_START_EPOCH="${STAGE_B_PPO_START_EPOCH}"
  DAGGER_END_EPOCH="${STAGE_B_DAGGER_END_EPOCH}"
  LOGGER="${LOGGER}"
  bash "${MODE_SCRIPT}" "${TEACHER_CHECKPOINT}"
  --training.checkpoint "${STAGE_A_CKPT}"
)
STAGE_B_CMD+=("${EXTRA_ARGS[@]}")
"${STAGE_B_CMD[@]}"

echo "[INFO] Curriculum completed."
echo "[INFO] stageA_dir=${STAGE_A_DIR}"
