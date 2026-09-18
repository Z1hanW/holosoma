#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_dir="${ROOT_DIR}/logs/real_depth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
if [[ "${HOLOSOMA_DEPLOYMENT_AUDIT:-1}" == "1" ]]; then
  export HOLOSOMA_DEPLOYMENT_AUDIT_DIR="${log_dir}/evidence"
else
  unset HOLOSOMA_DEPLOYMENT_AUDIT_DIR
fi
exec > >(tee -a "${log_dir}/depth.log") 2>&1

echo "[real_depth] log_dir=${log_dir}"
source scripts/source_inference_setup.sh
PYTHONPATH=src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python src/holosoma/holosoma/sensors/image_server.py real_d435i \
  --image-saver-config.image-root-dir "${log_dir}/depth_images"
