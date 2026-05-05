#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PERCEPTION_OBS_SHM_NAME="${PERCEPTION_OBS_SHM_NAME:-depth_img_shm}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/realsense_depth_publisher.py" \
  --shm-name "$PERCEPTION_OBS_SHM_NAME" \
  "$@"
