#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/realsense_depth_publisher.py" \
  --port "$PERCEPTION_OBS_PORT" \
  "$@"
