#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

motion_file="${HOLOSOMA_REAL_DEBUG_SIM_GT_MOTION:-${ROOT_DIR}/data_demo/box_75.npz}"
motion_frame="${HOLOSOMA_REAL_DEBUG_SIM_GT_FRAME:-0}"
shm_name="${HOLOSOMA_REAL_DEBUG_SIM_GT_SHM_NAME:-sim_gt_depth_raw_shm}"

if [[ -n "${HOLOSOMA_MUJOCO_PYTHON:-}" ]]; then
  python_bin="$HOLOSOMA_MUJOCO_PYTHON"
elif [[ -x "/home/unitree/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python" ]]; then
  python_bin="/home/unitree/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"
elif [[ -x "/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python" ]]; then
  python_bin="/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"
else
  python_bin="python3"
fi

echo "[sim_gt_depth] python=${python_bin}"
echo "[sim_gt_depth] motion=${motion_file} frame=${motion_frame}"
echo "[sim_gt_depth] bridge=disabled (render-only, no DDS)"

MUJOCO_GL="${HOLOSOMA_SIM_GT_MUJOCO_GL:-egl}" \
"$python_bin" scripts/sim_gt_depth_server.py \
  --motion-file "$motion_file" \
  --motion-frame "$motion_frame" \
  --shm-name "$shm_name"
