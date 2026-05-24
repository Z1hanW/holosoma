#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HOLOSOMA_POLICY_PICKUP_BUTTON="${HOLOSOMA_POLICY_PICKUP_BUTTON:-1}"
export HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}"

exec bash ./mj_ro.sh \
  scale__any_monitor_43 \
  wandb://zihanw22/carry-any/36k1vwdf/latest \
  ppo_first_contact_aware_h1
# zzv6vtkk 36k1vwdf
# 