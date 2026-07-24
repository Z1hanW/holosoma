#!/usr/bin/env bash
set -euo pipefail

# Single-node 8-GPU pure-RL stability/contact ablation.
#
# PURE_RL_CONTACT_PROFILE controls only offline contact reward strength:
#   none   - no offline contact reward
#   weak   - 25% of the previous wrist/contact guidance strength
#   strong - previous 3.0/5.0 wrist/contact guidance strength
#
# Every profile shares the same PPO stability settings so their curves remain
# directly comparable. The std and action-rate bounds are opt-in here and do
# not change defaults in any other launcher.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

PROFILE_RAW=${PURE_RL_CONTACT_PROFILE:-none}
PROFILE=$(printf '%s' "${PROFILE_RAW}" | tr '[:upper:]' '[:lower:]')
case "${PROFILE}" in
  none|off|no_contact)
    PROFILE=none
    CONTACT_TERM_WEIGHT=0.0
    WRIST_WEIGHT=0.0
    CONTACT_WEIGHT=0.0
    ;;
  weak)
    CONTACT_TERM_WEIGHT=1.0
    WRIST_WEIGHT=0.75
    CONTACT_WEIGHT=1.25
    ;;
  strong|full)
    PROFILE=strong
    CONTACT_TERM_WEIGHT=1.0
    WRIST_WEIGHT=3.0
    CONTACT_WEIGHT=5.0
    ;;
  *)
    echo "[ERROR] PURE_RL_CONTACT_PROFILE must be none, weak, or strong. Got: ${PROFILE_RAW}" >&2
    exit 2
    ;;
esac

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_STAMP
export NPROC=${NPROC:-8}
export NNODES=1
export PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
export NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-8}
export NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-2}
export NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}

export ACTOR_LR=${ACTOR_LR:-2e-4}
export CRITIC_LR=${CRITIC_LR:-1e-4}
export INIT_NOISE_STD=${INIT_NOISE_STD:-0.25}
export ACTOR_MIN_NOISE_STD=${ACTOR_MIN_NOISE_STD:-0.05}
export ENTROPY_COEF=${ENTROPY_COEF:-0.0}
export CLIP_PARAM=${CLIP_PARAM:-0.1}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.5}
export PPO_SCHEDULE=adaptive

export PURE_RL_WRIST_WEIGHT=${WRIST_WEIGHT}
export PURE_RL_CONTACT_WEIGHT=${CONTACT_WEIGHT}
export RUN_NAME=${RUN_NAME:-debug_pure_rl_stable_${PROFILE}_contact_debug30_bigmlp_${RUN_STAMP}}
export TRAINING_NAME=${TRAINING_NAME:-debug_pure_rl_stable_${PROFILE}_contact_debug30_bigmlp_${RUN_STAMP}}

ACTOR_MAX_NOISE_STD=${ACTOR_MAX_NOISE_STD:-0.35}
ACTION_RATE_MAX_PENALTY=${ACTION_RATE_MAX_PENALTY:-25.0}
DESIRED_KL=${DESIRED_KL:-0.01}
MIN_ACTOR_LR=${MIN_ACTOR_LR:-2e-5}
MIN_CRITIC_LR=${MIN_CRITIC_LR:-1e-5}
TRAINING_SEED=${TRAINING_SEED:-42}

stability_args=(
  --training.seed="${TRAINING_SEED}"
  --algo.config.schedule=adaptive
  --algo.config.desired-kl="${DESIRED_KL}"
  --algo.config.max-actor-learning-rate="${ACTOR_LR}"
  --algo.config.min-actor-learning-rate="${MIN_ACTOR_LR}"
  --algo.config.max-critic-learning-rate="${CRITIC_LR}"
  --algo.config.min-critic-learning-rate="${MIN_CRITIC_LR}"
  --algo.config.module-dict.actor.max-noise-std="${ACTOR_MAX_NOISE_STD}"
  --reward.terms.action-rate-l2.params.max-penalty="${ACTION_RATE_MAX_PENALTY}"
  --reward.terms.offline-contact-guidance.weight="${CONTACT_TERM_WEIGHT}"
)

echo "[INFO] pure_rl_stability_sweep profile=${PROFILE} seed=${TRAINING_SEED}"
echo "[INFO] contact outer_weight=${CONTACT_TERM_WEIGHT} wrist=${WRIST_WEIGHT} contact=${CONTACT_WEIGHT}"
echo "[INFO] adaptive_ppo actor_lr=${ACTOR_LR} critic_lr=${CRITIC_LR} desired_kl=${DESIRED_KL}"
echo "[INFO] safeguards max_noise_std=${ACTOR_MAX_NOISE_STD} action_rate_max_penalty=${ACTION_RATE_MAX_PENALTY}"

exec bash "${SCRIPT_DIR}/debug_pure_rl_contact_bootstrap.sh" "${stability_args[@]}" "$@"
