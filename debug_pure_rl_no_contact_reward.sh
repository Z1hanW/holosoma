#!/usr/bin/env bash
set -euo pipefail

# Same as debug_pure_rl_contact_bootstrap.sh, except offline contact guidance
# contributes zero reward. Contact-window sampling and rollout-reference rewards
# remain unchanged so this isolates the reward term itself.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_STAMP

export RUN_NAME=${RUN_NAME:-debug_pure_rl_no_contact_reward_window_bootstrap_debug30_bigmlp_${RUN_STAMP}}
export TRAINING_NAME=${TRAINING_NAME:-debug_pure_rl_no_contact_reward_window_bootstrap_debug30_bigmlp_${RUN_STAMP}}

export PURE_RL_WRIST_WEIGHT=0.0
export PURE_RL_CONTACT_WEIGHT=0.0

no_contact_reward_args=(
  --reward.terms.offline-contact-guidance.weight=0.0
)

echo "[INFO] pure_rl_no_contact_reward=1"
echo "[INFO] offline_contact_guidance reward weight=0.0; contact-window sampling is unchanged"

exec bash "${SCRIPT_DIR}/debug_pure_rl_contact_bootstrap.sh" "${no_contact_reward_args[@]}" "$@"
