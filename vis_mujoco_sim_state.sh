#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$ROOT_DIR/src/holosoma/holosoma/viser_mujoco_sim_state.py" --launch-rollout "$@"
