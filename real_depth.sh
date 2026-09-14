#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_dir="${ROOT_DIR}/logs/real_depth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/depth.log") 2>&1

echo "[real_depth] log_dir=${log_dir}"

# lsvla-vision opens the D435i directly, which makes librealsense report the
# misleading "No device connected" error here. Temporarily yield the camera to
# this server and restore the service when this script exits.
lsvla_service="${HOLOSOMA_REAL_DEPTH_CONFLICTING_SERVICE:-lsvla-vision.service}"
restore_lsvla_service=0
image_server_pid=""
viser_pid=""
restore_conflicting_service() {
  if [[ -n "$viser_pid" ]] && kill -0 "$viser_pid" 2>/dev/null; then
    kill "$viser_pid" 2>/dev/null || true
    wait "$viser_pid" 2>/dev/null || true
  fi
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
default_model_path="${ROOT_DIR}/_ckps/n94vaeq7_model_40000.onnx"
viewer_model_path="${HOLOSOMA_REAL_DEPTH_MODEL_PATH:-${HOLOSOMA_REAL_MODEL_PATH:-$default_model_path}}"
camera_profile_path="${log_dir}/camera_profile.json"
resolved_model_path_file="${log_dir}/resolved_model_path.txt"
image_server_config_file="${log_dir}/image_server_config.txt"
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 scripts/checkpoint_camera_profile.py \
  --model-path "$viewer_model_path" \
  --output "$camera_profile_path" \
  --resolved-model-path-output "$resolved_model_path_file" \
  --image-server-config-output "$image_server_config_file" \
  --download-dir "${log_dir}/checkpoint"
viewer_model_path="$(<"$resolved_model_path_file")"
recommended_image_server_config="$(<"$image_server_config_file")"
image_server_config="${HOLOSOMA_REAL_IMAGE_SERVER_CONFIG:-$recommended_image_server_config}"
echo "[real_depth] camera_model_path=${viewer_model_path}"
echo "[real_depth] image_server_config=${image_server_config}"

export LD_LIBRARY_PATH="/home/unitree/.local/librealsense-hsinference/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
image_save_args=(--no-save-images)
if [[ "${HOLOSOMA_REAL_DEPTH_SAVE_IMAGES:-0}" == "1" ]]; then
  image_save_args=(--save-images --image-saver-config.image-root-dir "${log_dir}/depth_images")
fi
PYTHONPATH=src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma/holosoma/sensors/image_server.py "$image_server_config" \
  "${image_save_args[@]}" &
image_server_pid=$!

if [[ "${HOLOSOMA_REAL_DEPTH_VISER:-1}" != "0" ]]; then
  viser_browser_args=()
  if [[ "${HOLOSOMA_REAL_VISER_OPEN_BROWSER:-1}" != "0" ]] \
      && { [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; }; then
    viser_browser_args+=(--open-browser)
  fi
  PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
  python3 scripts/real_viser.py \
    --state-path "${log_dir}/latest_command.json" \
    --camera-profile-path "$camera_profile_path" \
    --host "${HOLOSOMA_REAL_VISER_HOST:-127.0.0.1}" \
    --port "${HOLOSOMA_REAL_DEPTH_VISER_PORT:-8081}" \
    "${viser_browser_args[@]}" &
  viser_pid=$!
  echo "[real_depth] realtime viewer started pid=${viser_pid} port=${HOLOSOMA_REAL_DEPTH_VISER_PORT:-8081}"
else
  echo "[real_depth] realtime viewer disabled by HOLOSOMA_REAL_DEPTH_VISER=0"
fi
wait "$image_server_pid"
