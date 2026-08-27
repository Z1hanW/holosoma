#!/usr/bin/env bash
set -euo pipefail

# Bounded, non-formal, single-node HMI depth canary.  The caller must launch
# this from a clean exact-commit clone fetched from origin.  It never creates
# or resumes a W&B run.

SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXPECTED_COMMIT=${1:-}
RUN_LABEL=${2:-hmi_depth_stage1_ws8_canary}
HMI_STAGE=${3:-stage1}
POLICY_INIT_CHECKPOINT=${4:-}
POLICY_INIT_SHA256=${5:-}
TOTAL_ENVS=${6:-512}
EXTERNAL_MOTION_DIR=${HMI_CANARY_MOTION_DIR:-}
EXTERNAL_OBJECT_MAP=${HMI_CANARY_OBJECT_MAP:-}
EXTERNAL_SHARD_ROOT=${HMI_CANARY_SHARD_ROOT:-}
EXTERNAL_SHARD_MANIFEST_SHA256=${HMI_CANARY_SHARD_MANIFEST_SHA256:-}
EXTERNAL_EXPECTED_CLIP_COUNT=${HMI_CANARY_EXPECTED_CLIP_COUNT:-}

if [[ -z ${EXPECTED_COMMIT} || ! ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <full-commit-sha> [run-label] [stage1|stage2] [policy-init-pt] [policy-init-sha256] [total-envs]" >&2
  exit 2
fi
if [[ ! ${TOTAL_ENVS} =~ ^[1-9][0-9]*$ ]] || (( TOTAL_ENVS % 8 != 0 )); then
  echo "[ERROR] total-envs must be a positive integer divisible by 8" >&2
  exit 2
fi
ENVS_PER_RANK=$((TOTAL_ENVS / 8))

EXTERNAL_INPUT_VALUES=(
  "${EXTERNAL_MOTION_DIR}"
  "${EXTERNAL_OBJECT_MAP}"
  "${EXTERNAL_SHARD_ROOT}"
  "${EXTERNAL_SHARD_MANIFEST_SHA256}"
  "${EXTERNAL_EXPECTED_CLIP_COUNT}"
)
EXTERNAL_INPUT_COUNT=0
for VALUE in "${EXTERNAL_INPUT_VALUES[@]}"; do
  [[ -n ${VALUE} ]] && EXTERNAL_INPUT_COUNT=$((EXTERNAL_INPUT_COUNT + 1))
done
if (( EXTERNAL_INPUT_COUNT != 0 && EXTERNAL_INPUT_COUNT != ${#EXTERNAL_INPUT_VALUES[@]} )); then
  echo "[ERROR] external HMI canary inputs must be provided as one complete set" >&2
  exit 2
fi
if (( EXTERNAL_INPUT_COUNT > 0 )); then
  [[ ${EXTERNAL_SHARD_MANIFEST_SHA256} =~ ^[0-9a-f]{64}$ ]] || {
    echo "[ERROR] HMI_CANARY_SHARD_MANIFEST_SHA256 must be lowercase SHA256" >&2
    exit 2
  }
  [[ ${EXTERNAL_EXPECTED_CLIP_COUNT} =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] HMI_CANARY_EXPECTED_CLIP_COUNT must be positive" >&2
    exit 2
  }
fi
case "${HMI_STAGE}" in
  stage1)
    EXPERIMENT_PRESET=exp:g1-29dof-wbt-w-object-hmi-depth-stage1
    if [[ -n ${POLICY_INIT_CHECKPOINT} || -n ${POLICY_INIT_SHA256} ]]; then
      echo "[ERROR] Stage 1 canary must be fresh and cannot accept policy-init arguments" >&2
      exit 2
    fi
    ;;
  stage2)
    EXPERIMENT_PRESET=exp:g1-29dof-wbt-w-object-hmi-depth-stage2
    if [[ -z ${POLICY_INIT_CHECKPOINT} || ! -f ${POLICY_INIT_CHECKPOINT} || -L ${POLICY_INIT_CHECKPOINT} ]]; then
      echo "[ERROR] Stage 2 requires a non-symlink regular policy-init checkpoint" >&2
      exit 2
    fi
    if [[ ! ${POLICY_INIT_SHA256} =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] Stage 2 requires the exact lowercase policy-init SHA256" >&2
      exit 2
    fi
    ACTUAL_POLICY_INIT_SHA256=$(sha256sum "${POLICY_INIT_CHECKPOINT}" | awk '{print $1}')
    if [[ ${ACTUAL_POLICY_INIT_SHA256} != "${POLICY_INIT_SHA256}" ]]; then
      echo "[ERROR] policy-init SHA256 mismatch" >&2
      exit 2
    fi
    ;;
  *)
    echo "[ERROR] HMI stage must be stage1 or stage2, got '${HMI_STAGE}'" >&2
    exit 2
    ;;
esac

cd "${SOURCE_ROOT}"
ACTUAL_COMMIT=$(git rev-parse HEAD)
[[ ${ACTUAL_COMMIT} == "${EXPECTED_COMMIT}" ]] || {
  echo "[ERROR] HEAD=${ACTUAL_COMMIT}, expected=${EXPECTED_COMMIT}" >&2
  exit 2
}
git merge-base --is-ancestor "${EXPECTED_COMMIT}" origin/experiment/hmi-depth-interface || {
  echo "[ERROR] commit is not reachable from origin/experiment/hmi-depth-interface" >&2
  exit 2
}
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
if [[ -n $(git status --porcelain --untracked-files=all -- \
  '*.py' '*.sh' '*.yaml' '*.yml' '*.toml' '*.json' '*.urdf' '*.xml') ]]; then
  echo "[ERROR] clean exact-commit canary clone contains untracked executable/config source" >&2
  exit 2
fi

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total \
    --format=csv,noheader,nounits
)
[[ ${#GPU_ROWS[@]} -eq 8 ]] || {
  echo "[ERROR] expected exactly 8 GPUs, found ${#GPU_ROWS[@]}" >&2
  exit 2
}
for ROW in "${GPU_ROWS[@]}"; do
  [[ ${ROW##*, } == 0 ]] || {
    echo "[ERROR] unhealthy GPU row: ${ROW}" >&2
    exit 2
  }
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] node is not GPU-idle" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${SOURCE_ROOT}/scripts/source_isaacsim_setup.sh"
source "${SOURCE_ROOT}/scripts/gpu_launch_defaults.sh"

RUN_ROOT=/data/holosoma_canaries/${RUN_LABEL}_${EXPECTED_COMMIT:0:12}
mkdir -p "${RUN_ROOT}"
write_exit_code() {
  local rc=$?
  printf '%s\n' "${rc}" > "${RUN_ROOT}/exit_code.txt"
}
trap write_exit_code EXIT

export PYTHONPATH="${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=31459
export TORCH_DIST_BACKEND=gloo TORCH_DIST_TIMEOUT_SEC=1800 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1 HOLOSOMA_GLOO_GRAD_REDUCE=1 HOLOSOMA_GLOO_BARRIER=1
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0
export HOLOSOMA_RANK_VISIBLE_DEVICES=1 HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1 HOLOSOMA_CONTIGUOUS_MINIBATCHES=1
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off HOLOSOMA_REQUIRE_WANDB_RUN=0
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HEADLESS=1 OMP_NUM_THREADS=1
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0

TRAINING_ARGS=(
  "${EXPERIMENT_PRESET}"
  --training.multigpu=True
  --training.num-envs="${TOTAL_ENVS}"
  --training.seed=42
  --training.export-onnx=True
  --algo.config.num-learning-iterations=2
  --algo.config.save-interval=1
  --algo.config.num-steps-per-env=8
  --algo.config.num-learning-epochs=1
  --algo.config.num-mini-batches=1
  # The single-slot multi-URDF scene briefly overlaps all 4096 environments
  # during PhysX startup.  These are the e4096 scaling values of the accepted
  # e2048 formal object-generalist contract; smaller buffers silently drop
  # broadphase interactions before the environments are reset apart.
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=1073741824
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=1073741824
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=268435456
  --simulator.config.sim.physx.gpu-collision-stack-size=536870912
  --simulator.config.sim.physx.gpu-heap-capacity=67108864
  --simulator.config.sim.physx.gpu-temp-buffer-capacity=16777216
  --logger.base-dir="${RUN_ROOT}"
)
if (( EXTERNAL_INPUT_COUNT > 0 )); then
  for PATH_TO_CHECK in \
    "${EXTERNAL_MOTION_DIR}" \
    "${EXTERNAL_SHARD_ROOT}"; do
    [[ -d ${PATH_TO_CHECK} && ! -L ${PATH_TO_CHECK} ]] || {
      echo "[ERROR] expected non-symlink external directory: ${PATH_TO_CHECK}" >&2
      exit 2
    }
  done
  [[ -f ${EXTERNAL_OBJECT_MAP} && ! -L ${EXTERNAL_OBJECT_MAP} ]] || {
    echo "[ERROR] expected non-symlink external object map: ${EXTERNAL_OBJECT_MAP}" >&2
    exit 2
  }
  SHARD_MANIFEST=${EXTERNAL_SHARD_ROOT}/manifest.json
  [[ -f ${SHARD_MANIFEST} && ! -L ${SHARD_MANIFEST} ]] || {
    echo "[ERROR] missing rank-shard manifest: ${SHARD_MANIFEST}" >&2
    exit 2
  }
  ACTUAL_SHARD_MANIFEST_SHA256=$(sha256sum "${SHARD_MANIFEST}" | awk '{print $1}')
  [[ ${ACTUAL_SHARD_MANIFEST_SHA256} == "${EXTERNAL_SHARD_MANIFEST_SHA256}" ]] || {
    echo "[ERROR] rank-shard manifest SHA256 mismatch" >&2
    exit 2
  }

  readarray -t EXTERNAL_METADATA < <(
    "${PYTHON_BIN}" - \
      "${EXTERNAL_MOTION_DIR}/manifest.json" \
      "${SHARD_MANIFEST}" \
      "${EXTERNAL_EXPECTED_CLIP_COUNT}" \
      "${ENVS_PER_RANK}" <<'PY'
import json
import sys

view_path, shard_path, expected_clips_raw, expected_envs_raw = sys.argv[1:]
expected_clips = int(expected_clips_raw)
expected_envs = int(expected_envs_raw)
with open(view_path, "r", encoding="utf-8") as stream:
    view = json.load(stream)
with open(shard_path, "r", encoding="utf-8") as stream:
    shards = json.load(stream)
if view.get("clip_count") != expected_clips:
    raise SystemExit("view clip_count mismatch")
if shards.get("world_size") != 8 or shards.get("clip_count") != expected_clips:
    raise SystemExit("rank-shard topology mismatch")
if shards.get("environments_per_rank") != expected_envs:
    raise SystemExit("rank-shard environment count mismatch")
if shards.get("exact_clip_partition") is not True:
    raise SystemExit("rank shards do not exactly partition the motion bank")
if shards.get("rank_clip_counts_divide_environments_per_rank") is not True:
    raise SystemExit("rank-shard clip counts do not divide environments-per-rank")
if len(shards.get("shards", [])) != 8:
    raise SystemExit("rank-shard manifest does not contain eight shards")
if set(shards.get("clip_cover_counts", {}).values()) != {1}:
    raise SystemExit("rank shards do not cover every clip exactly once")
for role, value in (
    ("single-slot source digest", view.get("source_digest")),
    ("single-slot view digest", view.get("view_digest")),
    ("rank-shard source digest", shards.get("source_digest")),
):
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"invalid {role}")
    print(value)
PY
  )
  [[ ${#EXTERNAL_METADATA[@]} -eq 3 ]] || {
    echo "[ERROR] failed to bind external HMI provenance" >&2
    exit 2
  }
  export MOTION_DIR="${EXTERNAL_MOTION_DIR}"
  export OBJECT_SPEC_PATH="${EXTERNAL_OBJECT_MAP}"
  export OBJECT_URDF="${EXTERNAL_OBJECT_MAP}"
  export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST="${EXTERNAL_METADATA[0]}"
  export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST="${EXTERNAL_METADATA[1]}"
  export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${EXTERNAL_MOTION_DIR}"
  export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST="${EXTERNAL_METADATA[2]}"
  export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=8
  export HOLOSOMA_RANK_LOCAL_MOTION_ROOT="${EXTERNAL_SHARD_ROOT}"
  export HOLOSOMA_MOTION_SHARD_MANIFEST="${SHARD_MANIFEST}"
  export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1
  export HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
  export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
  export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
  export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
  export HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1
  export HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK=0
  export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh
  TRAINING_ARGS+=(
    --command.setup-terms.motion-command.params.motion-config.motion-file="${EXTERNAL_MOTION_DIR}"
    --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
    --robot.object.object-urdf-path="${EXTERNAL_OBJECT_MAP}"
  )
fi
if [[ ${HMI_STAGE} == stage2 ]]; then
  # This is an explicitly nonformal canary with a locally SHA-pinned Stage-1
  # artifact, not a scientific lineage. Formal Stage 2 must attach finalized
  # training provenance instead of using the legacy policy-load hatch.
  export HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD=1
  TRAINING_ARGS+=(--training.policy-init-checkpoint="${POLICY_INIT_CHECKPOINT}")
fi

echo "[INFO] validating the exact deployment graph with ONNX checker and ORT parity"
"${PYTHON_BIN}" -m pytest -q \
  src/holosoma/holosoma/managers/command/tests/test_hmi_depth_contract.py::test_hmi_depth_actor_real_onnx_checker_and_ort_parity

echo "[INFO] nonformal_hmi_canary commit=${EXPECTED_COMMIT} stage=${HMI_STAGE} world_size=8 total_envs=${TOTAL_ENVS} envs_per_rank=${ENVS_PER_RANK} iterations=2 collider=convex_decomposition contact_sensors=0 export_onnx=true external_clip_count=${EXTERNAL_EXPECTED_CLIP_COUNT:-default}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent_rank_visible.py \
  "${TRAINING_ARGS[@]}" \
  2>&1 | tee "${RUN_ROOT}/train.log"

if grep -Eq \
  'increase PxGpuDynamicsMemoryConfig::(foundLostPairsCapacity|foundLostAggregatePairsCapacity)|CUDA out of memory' \
  "${RUN_ROOT}/train.log"; then
  echo "[ERROR] e4096 canary observed an undersized PhysX GPU buffer or CUDA OOM" >&2
  exit 2
fi
