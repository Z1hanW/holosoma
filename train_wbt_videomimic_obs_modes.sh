#!/usr/bin/env bash
set -euo pipefail

# Mode 1: actor no-history + remove actor target group
#   DISABLE_ACTOR_HISTORY=true DISABLE_ACTOR_TARGET=true DISABLE_CRITIC_HISTORY=false
# Mode 2: actor + critic no-history (targets kept)
#   DISABLE_ACTOR_HISTORY=true DISABLE_ACTOR_TARGET=false DISABLE_CRITIC_HISTORY=true
DISABLE_ACTOR_TARGET="${DISABLE_ACTOR_TARGET:-false}"
DISABLE_ACTOR_HISTORY="${DISABLE_ACTOR_HISTORY:-false}"
DISABLE_CRITIC_HISTORY="${DISABLE_CRITIC_HISTORY:-false}"

python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  --observation_overrides.disable_actor_target="${DISABLE_ACTOR_TARGET}" \
  --observation_overrides.disable_actor_history="${DISABLE_ACTOR_HISTORY}" \
  --observation_overrides.disable_critic_history="${DISABLE_CRITIC_HISTORY}" \
  "$@"
