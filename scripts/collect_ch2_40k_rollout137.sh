#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="${1:?exact teacher source checkout}"
readonly audit_root="${2:?fresh audit directory}"
readonly checkpoint_sha="${3:?frozen PT SHA256}"
readonly checkpoint="${audit_root}/checkpoint/model_40000.pt"
readonly source_bank="${repo_root}/data/prism137_eval"
readonly shard_bank="${source_bank}/_rank_shards/by-source/d5709e087f7fcb58ec2ec5cedec221e8817a3be97014e78b1de5683669ac669b/ws8"
if [[ "${DRY_RUN:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  readonly output_root="${audit_root}/output_dry_run"
else
  readonly output_root="${audit_root}/output"
fi
readonly runtime_cache="${audit_root}/runtime_cache"
readonly python_bin="/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python"

test "$(git -C "${repo_root}" rev-parse HEAD)" = "dc1a1d8a1c5cee77eb62bf6b7a7b0901384d9e73"
test "$(git -C "${repo_root}" rev-parse HEAD^{tree})" = "a699f6ec37024b0017dd08a5b2723f59b8ac8005"
git -C "${repo_root}" merge-base --is-ancestor \
  dc1a1d8a1c5cee77eb62bf6b7a7b0901384d9e73 origin/main
test -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)"
test "$(sha256sum "${checkpoint}" | awk '{print $1}')" = \
  "${checkpoint_sha}"
test "$(sha256sum "${source_bank}/manifest.json" | awk '{print $1}')" = \
  "190288351fa3a92b3608c0d0b1647fda3daee51969f526b18468bcbe9016183f"
test "$(sha256sum "${source_bank}/_clip_object_urdf_map.json" | awk '{print $1}')" = \
  "867522fd61c63e6fcf37e0a041792f438e821f34ff482e26b07b47de6bfb7b59"
test "$(sha256sum "${shard_bank}/manifest.json" | awk '{print $1}')" = \
  "6c8c3acfcb7aa84ec5982854ee7f18b827c0672e240b9ce0341414deb8298099"
test "$(find "${source_bank}" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 137
test ! -e "${output_root}"

"${python_bin}" - "${shard_bank}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
all_ids: list[str] = []
expected_counts = [32, 32, 16, 16, 16, 16, 8, 1]
for index, expected_count in enumerate(expected_counts):
    shard = root / f"rank_{index}"
    payload = json.loads((shard / "_clip_object_urdf_map.json").read_text())
    ids = sorted(payload["clips"])
    listed = (shard / "clip_ids.txt").read_text().splitlines()
    if len(ids) != expected_count or set(ids) != set(listed):
        raise SystemExit(f"rank_{index} clip set/count mismatch")
    for clip_id in ids:
        path = shard / f"{clip_id}.npz"
        if not path.is_file():
            raise SystemExit(f"missing shard motion: {path}")
    all_ids.extend(ids)
if len(all_ids) != 137 or len(set(all_ids)) != 137:
    raise SystemExit("8-way shard coverage is not exactly one copy of all 137 clips")
PY

test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
mkdir -p "${output_root}" "${audit_root}/logs" "${runtime_cache}"

declare -a owned_pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  for pid in "${owned_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for pid in "${owned_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

for gpu in 0 1 2 3 4 5 6 7; do
  shard_dir="${shard_bank}/rank_${gpu}"
  shard_output="${output_root}/shard_${gpu}"
  shard_log="${audit_root}/logs/shard_${gpu}.log"
  shard_cache="${runtime_cache}/shard_${gpu}"
  expected_count="$(wc -l < "${shard_dir}/clip_ids.txt" | tr -d ' ')"
  mkdir -p \
    "${shard_cache}/tmp" \
    "${shard_cache}/xdg_cache" \
    "${shard_cache}/xdg_config" \
    "${shard_cache}/xdg_data" \
    "${shard_cache}/robot_usd" \
    "${shard_cache}/object_usd"
  (
    set -euo pipefail
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export HOLOSOMA_DEVICE="cuda:0"
    unset WORLD_SIZE RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    export LOCAL_RANK="0"
    export OMNI_KIT_ACCEPT_EULA="YES"
    export ACCEPT_EULA="Y"
    export TMPDIR="${shard_cache}/tmp"
    export XDG_CACHE_HOME="${shard_cache}/xdg_cache"
    export XDG_CONFIG_HOME="${shard_cache}/xdg_config"
    export XDG_DATA_HOME="${shard_cache}/xdg_data"
    export HOLOSOMA_ROBOT_USD_CACHE_DIR="${shard_cache}/robot_usd"
    export HOLOSOMA_OBJECT_USD_CACHE_DIR="${shard_cache}/object_usd"
    export HOLOSOMA_EVAL_POLICY="checkpoint_actor"
    export HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS="1"
    export HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE="1"
    export PYTHONDONTWRITEBYTECODE="1"
    export PYTHON_BIN="${python_bin}"
    export AS_DATA_DIR="${shard_dir}"
    export AS_OBJECT_MAP="${shard_dir}/_clip_object_urdf_map.json"
    export AS_EXPECTED_TOTAL="${expected_count}"
    export AS_SINGLE_SLOT_MOTION_DIR="${shard_dir}"
    export NUM_ENVS="${expected_count}"
    export HEADLESS="True"
    export OUTPUT_DIR="${shard_output}"
    export SUCCESS_POSITION_THRESHOLD="0.5"
    export MIN_CONTACT_FRAMES="10"
    export CONTACT_FORCE_THRESHOLD="1.0"
    export CONTACT_VOXEL_SIZE="0.01"
    export PHYSX_GPU_COLLISION_STACK_SIZE="268435456"
    export PUBLISH_FOR_INFER_BOX="0"
    export LAUNCH_VISER="0"
    export VALIDATE_OUTPUT_FORMAT="1"
    export DISABLE_RANDOMIZATION="True"
    export START_AT_TIMESTEP_ZERO_PROB="1.0"
    export FREEZE_AT_TIMESTEP_ZERO_PROB="0.0"
    export RESET_NOISE_SCALE="0.0"
    export USE_ADAPTIVE_TIMESTEPS_SAMPLER="False"
    export MAX_EPISODE_LENGTH_S="1000000"
    export REAL_MESH_OBJECT_SPAWN="1"
    export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="0"
    export HOLOSOMA_OBJECT_COLLIDER_TYPE="convex_decomposition"
    cd "${repo_root}"
    exec setsid bash ./infer_teacher_as_contacts.sh "${checkpoint}" \
      --require-final-position-success-for-success \
      --require-no-middle-foot-object-contact-for-success \
      --middle-foot-contact-start-frac 0.20 \
      --middle-foot-contact-end-frac 0.80 \
      --foot-object-contact-force-threshold 1.0 \
      --no-save-glb \
      --no-save-preview-png \
      --no-save-face-heatmap-png
  ) >"${shard_log}" 2>&1 &
  owned_pids+=("$!")
done

status=0
for pid in "${owned_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[ERROR] At least one batch-evaluation shard failed." >&2
  exit "${status}"
fi

if [[ "${DRY_RUN:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  owned_pids=()
  trap - EXIT INT TERM HUP
  echo "[INFO] All 8 batch-evaluation shard commands passed dry-run validation."
  exit 0
fi

for gpu in 0 1 2 3 4 5 6 7; do
  test -s "${output_root}/shard_${gpu}/summary.csv"
  test -s "${output_root}/shard_${gpu}/summary.json"
done

touch "${audit_root}/all_shards_completed.marker"
owned_pids=()
trap - EXIT INT TERM HUP
echo "[INFO] All 8 batch-evaluation shards completed."
