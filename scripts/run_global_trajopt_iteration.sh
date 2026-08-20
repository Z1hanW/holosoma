#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/FAR/worktrees/holosoma-global-trajopt-gpu-20260820}"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/gmr/bin/python}"
LOG_ROOT="${LOG_ROOT:-/data/far_offload/CRISP-Real2Sim-Obj/_logs/global_trajopt_gpu_20260820}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${LOG_ROOT}/${STAMP}"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/holosoma_retargeting${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

{
  printf 'timestamp=%s\n' "${STAMP}"
  printf 'repo=%s\n' "${REPO_ROOT}"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'branch=%s\n' "$(git branch --show-current)"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader
} | tee "${RUN_DIR}/manifest.log"

"${PYTHON}" -m pytest -q \
  src/holosoma_retargeting/tests/test_trajectory_optimization.py \
  2>&1 | tee "${RUN_DIR}/tests.log"

"${PYTHON}" scripts/canary_mujoco_dynamics.py \
  --model \
  src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml \
  --output "${RUN_DIR}/g1_dynamics_canary.json" \
  2>&1 | tee "${RUN_DIR}/g1_dynamics_canary.log"

"${PYTHON}" scripts/benchmark_global_trajopt.py \
  --frames 1024 \
  --state-dim 64 \
  --scale-dim 8 \
  --scale-knots 32 \
  --tracking-rows 96 \
  --backends osqp torch-cuda \
  --output "${RUN_DIR}/benchmark_large.json" \
  2>&1 | tee "${RUN_DIR}/benchmark_large.log"

"${PYTHON}" scripts/benchmark_global_trajopt.py \
  --active-bounds \
  --backends osqp torch-cuda \
  --output "${RUN_DIR}/benchmark_active_bounds.json" \
  2>&1 | tee "${RUN_DIR}/benchmark_active_bounds.log"

printf 'complete=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" \
  | tee "${RUN_DIR}/complete.log"
