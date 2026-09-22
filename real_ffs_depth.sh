#!/usr/bin/env bash
# Laptop side of the FFS depth path. Receives the D435i IR stereo pair from the
# robot (stereo_relay_pub.py on the Jetson), runs Fast-FoundationStereo on the
# local GPU, and writes the policy's depth_img_shm exactly as real_depth.sh
# would from a local camera. Start this before real_ffs_run.sh.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HOLOSOMA_DEPTH_PREDICTOR=ffs
export HOLOSOMA_FFS_REPO="${HOLOSOMA_FFS_REPO:-$HOME/FAR/Fast-FoundationStereo}"
export HOLOSOMA_FFS_MODEL="${HOLOSOMA_FFS_MODEL:-$HOLOSOMA_FFS_REPO/weights/c-ffs/model_best_bp2_serialize.pth}"
export HOLOSOMA_REMOTE_STEREO_CONNECT="${HOLOSOMA_REMOTE_STEREO_CONNECT:-tcp://192.168.123.164:5602}"

for f in "$HOLOSOMA_FFS_MODEL" "$(dirname "$HOLOSOMA_FFS_MODEL")/cfg.yaml"; do
  if [[ ! -f "$f" ]]; then
    echo "[real_ffs_depth] ERROR: missing $f" >&2
    exit 1
  fi
done

log_dir="${ROOT_DIR}/logs/real_ffs_depth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/depth.log") 2>&1

echo "[real_ffs_depth] log_dir=${log_dir}"
echo "[real_ffs_depth] stereo source=${HOLOSOMA_REMOTE_STEREO_CONNECT}  predictor=ffs  model=${HOLOSOMA_FFS_MODEL}"
source scripts/source_inference_setup.sh
PYTHONPATH=src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python src/holosoma/holosoma/sensors/image_server.py remote_ffs_d435i \
  --image-saver-config.image-root-dir "${log_dir}/depth_images"
