#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}"

exec bash ./mj_ro.sh \
  scale__any_monitor_43 \
  wandb://zihanw22/boxer/d9m3z369/latest \
  ppo_first_contact_aware_h1
