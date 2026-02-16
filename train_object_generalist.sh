#!/usr/bin/env bash
set -euo pipefail

# Generalist W-Object training (multi-clip directory input).
#
# Example:
#   MOTION_DIR=/ABS/PATH/to/omomo_carry ./train_object_generalist.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
EXP=${EXP:-g1-29dof-wbt-w-object-generalist}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry"}
NUM_ENVS=${NUM_ENVS:-12288}
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
STRICT_VALIDATE=${STRICT_VALIDATE:-1}
REQUIRE_OBJECT_SIZE=${REQUIRE_OBJECT_SIZE:-1}

if [[ -z "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR is required and must point to a directory of .npz clips."
  echo "        Example: MOTION_DIR=/ABS/PATH/to/omomo_carry ./train_object_generalist.sh"
  exit 1
fi

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR does not exist or is not a directory: ${MOTION_DIR}"
  exit 1
fi

if [[ "${STRICT_VALIDATE}" != "0" ]]; then
  echo "[INFO] Validating motion bank consistency..."
  MOTION_DIR="${MOTION_DIR}" REQUIRE_OBJECT_SIZE="${REQUIRE_OBJECT_SIZE}" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

motion_dir = Path(os.environ["MOTION_DIR"]).expanduser().resolve()
require_object_size = str(os.environ.get("REQUIRE_OBJECT_SIZE", "1")) != "0"
files = sorted(motion_dir.glob("*.npz"))
if len(files) < 2:
    raise SystemExit(
        f"[ERROR] {motion_dir} has {len(files)} npz file(s). "
        "Generalist multi-file training requires at least 2 clips."
    )

required_keys = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
    "fps",
)
required_object_keys = ("object_pos_w", "object_quat_w", "object_lin_vel_w")
size_keys = ("object_size", "box_size", "object_scale", "box_scale")

joint_names_ref = None
body_names_ref = None
fps_ref = None
missing_size_count = 0
frame_total = 0

for file_path in files:
    with np.load(file_path, allow_pickle=True) as data:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise SystemExit(f"[ERROR] {file_path.name} missing keys: {missing}")

        missing_obj = [k for k in required_object_keys if k not in data]
        if missing_obj:
            raise SystemExit(f"[ERROR] {file_path.name} missing object keys: {missing_obj}")

        if not any(k in data for k in size_keys):
            missing_size_count += 1

        joint_names = tuple(str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x) for x in data["joint_names"].tolist())
        body_names = tuple(str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x) for x in data["body_names"].tolist())
        if joint_names_ref is None:
            joint_names_ref = joint_names
            body_names_ref = body_names
        else:
            if joint_names != joint_names_ref:
                raise SystemExit(f"[ERROR] joint_names mismatch in {file_path.name}")
            if body_names != body_names_ref:
                raise SystemExit(f"[ERROR] body_names mismatch in {file_path.name}")

        fps_arr = np.asarray(data["fps"]).reshape(-1)
        fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0
        if fps_ref is None:
            fps_ref = fps
        elif abs(fps - fps_ref) > 1e-6:
            raise SystemExit(f"[ERROR] fps mismatch in {file_path.name}: {fps} != {fps_ref}")

        frame_total += int(np.asarray(data["joint_pos"]).shape[0])

print(f"[OK] clips={len(files)}, total_frames={frame_total}, fps={fps_ref}")
if missing_size_count > 0:
    msg = (
        f"{missing_size_count}/{len(files)} clips have no explicit object size key "
        "(object_size/box_size/object_scale/box_scale). Loader would fall back to [1,1,1]."
    )
    if require_object_size:
        raise SystemExit(f"[ERROR] {msg} Set REQUIRE_OBJECT_SIZE=0 to allow this fallback.")
    print(f"[WARN] {msg}")
PY
fi

echo "[INFO] Training exp=${EXP}"
echo "[INFO] Motion bank dir=${MOTION_DIR}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent.py \
  "exp:${EXP}" \
  --training.num_envs="${NUM_ENVS}" \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}" \
  --command.setup_terms.motion_command.params.motion_config.clip_weighting_strategy uniform_step \
  --algo.config.save_interval="${SAVE_INTERVAL}" \
  logger:wandb \
  "$@"
