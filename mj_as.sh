#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}"

run_id="${1:-${HOLOSOMA_MJ_AS_RUN:-qihvpyqg}}"
checkpoint="${2:-${HOLOSOMA_MJ_AS_CHECKPOINT:-latest}}"

case "$run_id" in
  gjiefd3c|swl41n4x|qihvpyqg)
    ;;
  *)
    echo "mj_as.sh: unsupported run id '$run_id' (expected gjiefd3c, swl41n4x, or qihvpyqg)" >&2
    exit 2
    ;;
esac

exec bash ./mj_ro.sh \
  scale__any_monitor_43 \
  "wandb://zihanw22/carry-any/${run_id}/${checkpoint}" \
  ppo_first_contact_aware_h1
