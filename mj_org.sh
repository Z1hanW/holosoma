#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

checkpoint="${HOLOSOMA_MJ_ORG_CHECKPOINT:-wandb://zihanw22/carry-any/1j98x3g1/latest}"

exec bash ./mj_ro.sh \
  scale__any_monitor_43 \
  "$checkpoint" \
  zihanw22/carry-any/1j98x3g1
