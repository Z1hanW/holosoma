#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_dir="${ROOT_DIR}/logs/real_depth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/depth.log") 2>&1

echo "[real_depth] log_dir=${log_dir}"
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/sensors/image_server.py real_d435i \
  --image-saver-config.image-root-dir "${log_dir}/depth_images"
