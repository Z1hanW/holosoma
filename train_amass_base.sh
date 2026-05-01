#!/usr/bin/env bash
set -euo pipefail

# 29-DoF robot-only AMASS generalist motion tracking (no object, no perception, no terrain pairing).
# Default experiment: exp:g1-29dof-wbt

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

SIM_ENV_BIN=/home/ubuntu/miniconda3/envs/sim/bin
if ! command -v torchrun >/dev/null 2>&1 && [[ -x "${SIM_ENV_BIN}/torchrun" ]]; then
  export PATH="${SIM_ENV_BIN}:${PATH}"
fi

CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
NPROC=${NPROC:-$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

EXP=${EXP:-g1-29dof-wbt}
WANDB_PROJECT=${WANDB_PROJECT:-boxer}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_amass_base}

MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/robot_only/amass_all_trainready"}
FLATTEN_IF_NESTED=${FLATTEN_IF_NESTED:-1}
FLAT_MOTION_DIR=${FLAT_MOTION_DIR:-"${SCRIPT_DIR}/.cache/amass_all_trainready_flat"}

PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
NUM_ITERS=${NUM_ITERS:-1000000}
ACTOR_LR=${ACTOR_LR:-7e-5}
CRITIC_LR=${CRITIC_LR:-7e-5}
NORM_ACTOR_OBS=${NORM_ACTOR_OBS:-False}
NORM_CRITIC_OBS=${NORM_CRITIC_OBS:-False}

START_AT_ZERO_PROB=${START_AT_ZERO_PROB:-0.05}
ADAPTIVE_SAMPLER=${ADAPTIVE_SAMPLER:-True}

LOGGER_NAME=${LOGGER_NAME:-amass_base_29dof_wbt}

EXTRA_ARGS=("$@")

if [[ ! -d "${MOTION_DIR}" && ! -f "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi

MOTION_SOURCE="${MOTION_DIR}"
if [[ -d "${MOTION_DIR}" ]]; then
  ROOT_NPZ_COUNT=$(find "${MOTION_DIR}" -maxdepth 1 -type f \( -name '*.npz' -o -name '*.NPZ' \) | wc -l | tr -d ' ')
  if [[ "${ROOT_NPZ_COUNT}" == "0" && "${FLATTEN_IF_NESTED}" == "1" ]]; then
    mkdir -p "${FLAT_MOTION_DIR}"
    echo "[INFO] MOTION_DIR has nested folders; flattening recursively into: ${FLAT_MOTION_DIR}"
    python - <<'PY' "${MOTION_DIR}" "${FLAT_MOTION_DIR}"
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

src_dir = Path(sys.argv[1]).resolve()
view_dir = Path(sys.argv[2]).resolve()
cfg_path = view_dir / "_train_view_config.json"

files = sorted(list(src_dir.rglob("*.npz")) + list(src_dir.rglob("*.NPZ")))
if not files:
    raise SystemExit(f"[ERROR] No .npz files found recursively in {src_dir}")

cfg = {"source_dir": str(src_dir), "count": len(files), "version": 1}
existing_links = sorted(view_dir.glob("*.npz"))
reuse = False
if cfg_path.exists():
    try:
        prev = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        prev = None
    if prev == cfg and len(existing_links) == len(files):
        reuse = True

if reuse:
    print(f"[INFO] Reusing flattened cache: {view_dir} ({len(existing_links)} files)")
    raise SystemExit(0)

for p in existing_links:
    p.unlink()

seen: set[str] = set()
for src in files:
    rel = src.relative_to(src_dir).as_posix()
    base = rel[:-4] if rel.lower().endswith(".npz") else rel
    base = base.replace("/", "__")
    name = f"{base}.npz"
    if name in seen:
        idx = 1
        while f"{base}__{idx}.npz" in seen:
            idx += 1
        name = f"{base}__{idx}.npz"
    seen.add(name)
    dst = view_dir / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"[INFO] Flattened cache ready: {view_dir} ({len(files)} files)")
PY
    MOTION_SOURCE="${FLAT_MOTION_DIR}"
  fi
fi

if [[ -d "${MOTION_SOURCE}" ]]; then
  CLIP_COUNT=$(find "${MOTION_SOURCE}" -maxdepth 1 \( -type f -o -type l \) \( -name '*.npz' -o -name '*.NPZ' \) | wc -l | tr -d ' ')
  if [[ "${CLIP_COUNT}" == "0" ]]; then
    echo "[ERROR] No .npz clips found in MOTION_SOURCE: ${MOTION_SOURCE}" >&2
    exit 1
  fi
  echo "[INFO] Motion source (dir): ${MOTION_SOURCE}  clips=${CLIP_COUNT}"
else
  echo "[INFO] Motion source (file): ${MOTION_SOURCE}"
fi

echo "[INFO] EXP=${EXP}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NPROC=${NPROC}"
echo "[INFO] NUM_ENVS=${NUM_ENVS}  NUM_ITERS=${NUM_ITERS}"
echo "[INFO] no perception, no object info, no terrain pairing"

train_cmd=(
  src/holosoma/holosoma/train_agent.py
  "exp:${EXP}"
  perception:none
  --training.project="${WANDB_PROJECT}"
  --training.name="${TRAINING_NAME}"
  --training.num_envs="${NUM_ENVS}"
  --algo.config.actor_learning_rate="${ACTOR_LR}"
  --algo.config.critic_learning_rate="${CRITIC_LR}"
  --algo.config.normalize_actor_obs="${NORM_ACTOR_OBS}"
  --algo.config.normalize_critic_obs="${NORM_CRITIC_OBS}"
  --algo.config.save_interval="${SAVE_INTERVAL}"
  --algo.config.num_learning_iterations="${NUM_ITERS}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_SOURCE}"
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler="${ADAPTIVE_SAMPLER}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob="${START_AT_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=False
)

train_cmd+=(logger:wandb)
train_cmd+=(--logger.name="${LOGGER_NAME}")
train_cmd+=(--logger.video.enabled=False)
train_cmd+=(--logger.headless_recording=False)
train_cmd+=(--logger.video.upload_to_wandb=False)

train_cmd+=("${EXTRA_ARGS[@]}")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  "${train_cmd[@]}"
