#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_dir="${ROOT_DIR}/logs/real_depth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/depth.log") 2>&1

echo "[real_depth] log_dir=${log_dir}"
image_server_config="${HOLOSOMA_REAL_IMAGE_SERVER_CONFIG:-real_d435i}"

# lsvla-vision opens the D435i directly, which makes librealsense report the
# misleading "No device connected" error here. Temporarily yield the camera to
# this server and restore the service when this script exits.
lsvla_service="${HOLOSOMA_REAL_DEPTH_CONFLICTING_SERVICE:-lsvla-vision.service}"
restore_lsvla_service=0
image_server_pid=""
restore_conflicting_service() {
  if [[ -n "$image_server_pid" ]] && kill -0 "$image_server_pid" 2>/dev/null; then
    kill "$image_server_pid" 2>/dev/null || true
    wait "$image_server_pid" 2>/dev/null || true
  fi
  if (( restore_lsvla_service )); then
    echo "[real_depth] restarting ${lsvla_service}"
    systemctl --user start "$lsvla_service" || \
      echo "[real_depth] warning: failed to restart ${lsvla_service}" >&2
  fi
}
trap restore_conflicting_service EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${HOLOSOMA_REAL_DEPTH_STOP_CONFLICTING_SERVICE:-1}" == "1" ]] && \
    command -v systemctl >/dev/null 2>&1 && \
    systemctl --user is-active --quiet "$lsvla_service"; then
  echo "[real_depth] stopping ${lsvla_service}; it occupies the D435i"
  systemctl --user stop "$lsvla_service"
  restore_lsvla_service=1
fi

source scripts/source_inference_setup.sh
export LD_LIBRARY_PATH="/home/unitree/.local/librealsense-hsinference/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PYTHONPATH=src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma/holosoma/sensors/image_server.py "$image_server_config" \
  --image-saver-config.image-root-dir "${log_dir}/depth_images" &
image_server_pid=$!
wait "$image_server_pid"
