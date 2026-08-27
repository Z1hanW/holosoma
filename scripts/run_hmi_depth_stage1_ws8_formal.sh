#!/usr/bin/env bash
set -euo pipefail

# Single-node, eight-rank formal Stage-1 launcher.  Code always comes from the
# caller's clean exact-commit Git clone.  Scientific assets and audit receipts
# are immutable, SHA-bound inputs and never carry executable source.
if [[ $# -ne 13 ]]; then
  echo "usage: $0 <commit> <tree> <run-id> <run-name> <run-root> <run-contract> <contract-sha256> <rule90-manifest> <rule90-manifest-sha256> <source-id> <git-archive-sha256> <onnx-receipt> <onnx-receipt-sha256>" >&2
  exit 2
fi

SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXPECTED_COMMIT=$1
EXPECTED_TREE=$2
WANDB_RUN_ID=$3
RUN_NAME=$4
RUN_ROOT=$5
RUN_CONTRACT=$6
RUN_CONTRACT_SHA256=$7
RULE90_MANIFEST=$8
RULE90_MANIFEST_SHA256=$9
SOURCE_ID=${10}
GIT_ARCHIVE_SHA256=${11}
ONNX_RECEIPT=${12}
ONNX_RECEIPT_SHA256=${13}

readonly ORIGIN_URL=https://github.com/Z1hanW/holosoma
readonly REMOTE_REF=origin/experiment/hmi-depth-interface
readonly ENTITY=zihanw22
readonly PROJECT=carry-any
readonly WORLD_SIZE=8
readonly ENVS_PER_RANK=4096
readonly TOTAL_ENVS=32768
readonly TARGET_ITERATIONS=15000
readonly MASTER_PORT=31491
readonly PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=1073741824
readonly PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=1073741824
readonly PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=268435456
readonly PHYSX_GPU_COLLISION_STACK_SIZE=536870912
readonly PHYSX_GPU_HEAP_CAPACITY=67108864
readonly PHYSX_GPU_TEMP_BUFFER_CAPACITY=16777216

readonly BANK=/home/ubuntu/FAR/holosoma/data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball
readonly MOTION_VIEW="${BANK}/_scientific_corl79_single_slot/by-source/6209b4742cce3b2989c7ea1f96a55a27d57bcf91eeb90699d409747187ca2cca"
readonly OBJECT_MAP="${MOTION_VIEW}/_clip_object_urdf_map.json"
readonly SHARD_ROOT=/data/holosoma_inputs/hmi_depth_corl79_rank_shards_ws8_e4096/by-source/d0caf5664810488f84eeaf4cb5a8c3f10db465d5b9ce4f16ce093dec240f3800
readonly SHARD_MANIFEST="${SHARD_ROOT}/manifest.json"

readonly BANK_PACKAGE_SHA256=a818815e52e137be70a8fbbd4fcfbf644e449f4d7c1bda98399ca201ca922c11
readonly BANK_OBJECT_MAP_SHA256=f632eb303034f9b9840758df385d99ea0019beae790628775265366f3a127dc8
readonly VIEW_MANIFEST_SHA256=910399359c1bf8d236ec446667b27902de0037c24c7dfcb40aa70a1bf6d0522d
readonly OBJECT_MAP_SHA256=7926ea58ad4c13d4d0bdc7f02b03a6e6a65dedbaf01609f2e962f04f485069a0
readonly SHARD_MANIFEST_SHA256=5abba6c9aa00336f00d3273c77a3d24e1cbaf0e0ce9baead65902b91ef816043
readonly SINGLE_SLOT_SOURCE_DIGEST=531b535d01995643c2a3d591a3b8c6ca2dddb9ae427366ae554898e0d592a483
readonly SINGLE_SLOT_VIEW_DIGEST=6209b4742cce3b2989c7ea1f96a55a27d57bcf91eeb90699d409747187ca2cca
readonly RANK_SHARD_SOURCE_DIGEST=d0caf5664810488f84eeaf4cb5a8c3f10db465d5b9ce4f16ce093dec240f3800
readonly CANONICAL_CLIP_ID=box_10
readonly CANONICAL_NPZ_SHA256=61b7a9b47a2bd2f3eadb9fc37d94ac682b8923d4a7e0cc9121769d8b2c33c45a
readonly CANONICAL_URDF_SHA256=a7db8f4e0ee64d89af83d610c8e668a0f26030efcf5fa0ae25ff26b936958de6

die() { echo "[ERROR] $*" >&2; exit 2; }
check_sha() {
  local expected=$1 path=$2 actual
  [[ -f ${path} && ! -L ${path} ]] || die "missing non-symlink integrity input: ${path}"
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ ${actual} == "${expected}" ]] || die "SHA256 mismatch: ${path} expected=${expected} actual=${actual}"
}

[[ ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]] || die "invalid expected commit"
[[ ${EXPECTED_TREE} =~ ^[0-9a-f]{40}$ ]] || die "invalid expected tree"
[[ ${RUN_CONTRACT_SHA256} =~ ^[0-9a-f]{64}$ ]] || die "invalid contract SHA256"
[[ ${RULE90_MANIFEST_SHA256} =~ ^[0-9a-f]{64}$ ]] || die "invalid Rule-90 manifest SHA256"
[[ ${GIT_ARCHIVE_SHA256} =~ ^[0-9a-f]{64}$ ]] || die "invalid Git archive SHA256"
[[ ${ONNX_RECEIPT_SHA256} =~ ^[0-9a-f]{64}$ ]] || die "invalid ONNX receipt SHA256"
[[ ${SOURCE_ID} =~ ^src-[0-9a-f]{64}$ ]] || die "invalid immutable source id"
[[ ${WANDB_RUN_ID} =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "invalid W&B run id"
[[ ${RUN_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "invalid formal run name"
[[ ${RUN_ROOT} == /data/holosoma_training/* && ! -L ${RUN_ROOT} ]] || die "run root must be a non-symlink child of /data/holosoma_training"

[[ $(git -C "${SOURCE_ROOT}" rev-parse HEAD) == "${EXPECTED_COMMIT}" ]] || die "Git HEAD mismatch"
[[ $(git -C "${SOURCE_ROOT}" rev-parse HEAD^{tree}) == "${EXPECTED_TREE}" ]] || die "Git tree mismatch"
[[ $(git -C "${SOURCE_ROOT}" remote get-url origin) == "${ORIGIN_URL}" ]] || die "Git origin URL mismatch"
git -C "${SOURCE_ROOT}" fetch --quiet origin experiment/hmi-depth-interface
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${EXPECTED_COMMIT}" "${REMOTE_REF}" || die "commit is not reachable from required remote ref"
git -C "${SOURCE_ROOT}" diff --quiet --ignore-submodules -- || die "tracked worktree is dirty"
git -C "${SOURCE_ROOT}" diff --cached --quiet --ignore-submodules -- || die "Git index is dirty"
if [[ -n $(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=all -- \
  '*.py' '*.sh' '*.yaml' '*.yml' '*.toml' '*.json' '*.urdf' '*.xml') ]]; then
  die "clean formal clone contains untracked executable/config source"
fi
[[ -z $(git -C "${SOURCE_ROOT}" submodule status 2>/dev/null) ]] || die "unexpected submodule state"

check_sha "${BANK_PACKAGE_SHA256}" "${BANK}/nfs_package_manifest.json"
check_sha "${BANK_OBJECT_MAP_SHA256}" "${BANK}/_clip_object_urdf_map.json"
check_sha "${VIEW_MANIFEST_SHA256}" "${MOTION_VIEW}/manifest.json"
check_sha "${OBJECT_MAP_SHA256}" "${OBJECT_MAP}"
check_sha "${SHARD_MANIFEST_SHA256}" "${SHARD_MANIFEST}"
check_sha "${CANONICAL_NPZ_SHA256}" "${MOTION_VIEW}/${CANONICAL_CLIP_ID}.npz"
check_sha "${CANONICAL_URDF_SHA256}" "${MOTION_VIEW}/_single_slot_urdfs/${CANONICAL_CLIP_ID}.urdf"
check_sha "${RUN_CONTRACT_SHA256}" "${RUN_CONTRACT}"
check_sha "${RULE90_MANIFEST_SHA256}" "${RULE90_MANIFEST}"
check_sha "${ONNX_RECEIPT_SHA256}" "${ONNX_RECEIPT}"

# shellcheck disable=SC1091
source "${SOURCE_ROOT}/scripts/source_isaacsim_setup.sh"
source "${SOURCE_ROOT}/scripts/gpu_launch_defaults.sh"
export PYTHONPATH="${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" - \
  "${MOTION_VIEW}/manifest.json" "${SHARD_MANIFEST}" \
  "${RUN_CONTRACT}" "${ONNX_RECEIPT}" \
  "${EXPECTED_COMMIT}" "${EXPECTED_TREE}" "${WANDB_RUN_ID}" \
  "${RUN_NAME}" "${SOURCE_ID}" "${RUN_CONTRACT_SHA256}" <<'PY'
import json
import sys

(view_path, shards_path, contract_path, onnx_path, commit, tree, run_id,
 run_name, source_id, contract_sha) = sys.argv[1:]
view = json.load(open(view_path, encoding="utf-8"))
shards = json.load(open(shards_path, encoding="utf-8"))
contract = json.load(open(contract_path, encoding="utf-8"))
onnx = json.load(open(onnx_path, encoding="utf-8"))
assert view["clip_count"] == 79
assert view["source_digest"] == "531b535d01995643c2a3d591a3b8c6ca2dddb9ae427366ae554898e0d592a483"
assert view["view_digest"] == "6209b4742cce3b2989c7ea1f96a55a27d57bcf91eeb90699d409747187ca2cca"
assert shards["source_digest"] == "d0caf5664810488f84eeaf4cb5a8c3f10db465d5b9ce4f16ce093dec240f3800"
assert shards["world_size"] == 8 and shards["environments_per_rank"] == 4096
assert shards["clip_count"] == 79 and shards["exact_clip_partition"] is True
assert shards["rank_clip_counts_divide_environments_per_rank"] is True
assert [row["clip_count"] for row in shards["shards"]] == [16, 16, 16, 16, 8, 4, 2, 1]
assert set(shards["clip_cover_counts"].values()) == {1}
assert contract["version"] == 1 and contract["semantics"] == "formal_hmi_depth_stage1"
assert contract["git"]["commit"] == commit and contract["git"]["tree"] == tree
assert contract["git"]["source_id"] == source_id
assert contract["wandb"]["run_id"] == run_id and contract["wandb"]["name"] == run_name
assert contract["training"]["world_size"] == 8
assert contract["training"]["environments_per_rank"] == 4096
assert contract["training"]["total_environments"] == 32768
assert contract["training"]["target_iterations"] == 15000
assert contract["training"]["export_onnx"] is True
assert contract["training"]["fresh"] is True
assert contract["training"]["resume_checkpoint"] is None
assert contract["training"]["policy_init_checkpoint"] is None
assert contract["training"]["physx_gpu_buffers"] == {
    "found_lost_pairs_capacity": 1073741824,
    "found_lost_aggregate_pairs_capacity": 1073741824,
    "total_aggregate_pairs_capacity": 268435456,
    "collision_stack_size": 536870912,
    "heap_capacity": 67108864,
    "temp_buffer_capacity": 16777216,
}
assert contract["data"]["rank_shard_manifest_sha256"] == "5abba6c9aa00336f00d3273c77a3d24e1cbaf0e0ce9baead65902b91ef816043"
assert onnx["accepted"] is True and onnx["source_commit"] == commit
assert onnx["world_size"] == 8 and onnx["environments_per_rank"] == 4096
assert onnx["motion_view_manifest_sha256"] == "910399359c1bf8d236ec446667b27902de0037c24c7dfcb40aa70a1bf6d0522d"
assert onnx["rank_shard_manifest_sha256"] == "5abba6c9aa00336f00d3273c77a3d24e1cbaf0e0ce9baead65902b91ef816043"
assert onnx["terminal_pair"]["onnx_checker"] is True
assert onnx["terminal_pair"]["onnxruntime_loaded"] is True
assert onnx["terminal_pair"]["pytorch_vs_ort"] is True
PY

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
  --manifest "${RULE90_MANIFEST}" \
  --expected-manifest-sha256 "${RULE90_MANIFEST_SHA256}" \
  --expected-source-snapshot-id "${SOURCE_ID}" \
  --expected-entity "${ENTITY}" \
  --expected-project "${PROJECT}" \
  --expected-run-id "${WANDB_RUN_ID}" \
  --expected-run-name "${RUN_NAME}" \
  --expected-world-size "${WORLD_SIZE}" >/dev/null

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#GPU_ROWS[@]} -eq 8 ]] || die "expected exactly eight GPUs"
for ROW in "${GPU_ROWS[@]}"; do
  [[ ${ROW} == *"NVIDIA L40S"* ]] || die "unexpected GPU model: ${ROW}"
  [[ ${ROW##*, } == 0 ]] || die "nonzero volatile UECC: ${ROW}"
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  die "formal node is not GPU-idle"
fi
if ss -lnt | awk '{print $4}' | grep -Eq "(^|:)${MASTER_PORT}$"; then
  die "formal master port is already occupied"
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/wandb" "${RUN_ROOT}/training_logs" "${RUN_ROOT}/provenance-cache"
EXIT_FILE=${RUN_ROOT}/formal.exit
[[ ! -e ${EXIT_FILE} ]] || die "formal exit file already exists"
trap 'rc=$?; printf "%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "${EXIT_FILE}"' EXIT

FORMAL_GIT_VERIFICATION=${RUN_ROOT}/formal_git_verification.json
"${PYTHON_BIN}" - \
  "${FORMAL_GIT_VERIFICATION}" "${ORIGIN_URL}" "${REMOTE_REF}" \
  "${EXPECTED_COMMIT}" "${EXPECTED_TREE}" <<'PY'
import json
import os
import sys
import tempfile

path, remote_url, remote_ref, commit, tree = sys.argv[1:]
payload = {
    "version": 1,
    "verified_at_utc": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat().replace("+00:00", "Z"),
    "hostname": os.uname().nodename,
    "remote_url": remote_url,
    "remote_ref": remote_ref,
    "commit_sha": commit,
    "tree_sha": tree,
    "fetched_ref_commit": commit,
    "tracked_diff_clean": True,
    "untracked_clean": True,
    "declared_submodules": {},
    "legacy_unmapped_gitlinks_inactive_and_empty": True,
}
directory = os.path.dirname(path)
fd, temporary = tempfile.mkstemp(prefix=".formal-git-", suffix=".json", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY

export HOME=/home/ubuntu
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HEADLESS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT
export TORCH_DIST_BACKEND=gloo TORCH_DIST_TIMEOUT_SEC=1800 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1 HOLOSOMA_GLOO_GRAD_REDUCE=1 HOLOSOMA_GLOO_BARRIER=1
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0
export HOLOSOMA_RANK_VISIBLE_DEVICES=1 HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1 HOLOSOMA_CONTIGUOUS_MINIBATCHES=1
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1

export HOLOSOMA_SOURCE_ROOT="${SOURCE_ROOT}"
export HOLOSOMA_SOURCE_SNAPSHOT_ID="${SOURCE_ID}"
export HOLOSOMA_SOURCE_MANIFEST_SHA256="${SOURCE_ID#src-}"
export HOLOSOMA_RUN_CONTRACT="${RUN_CONTRACT}"
export HOLOSOMA_RUN_CONTRACT_SHA256="${RUN_CONTRACT_SHA256}"
export HOLOSOMA_GIT_REMOTE_URL="${ORIGIN_URL}"
export HOLOSOMA_GIT_REMOTE_REF="${REMOTE_REF}"
export HOLOSOMA_GIT_COMMIT="${EXPECTED_COMMIT}"
export HOLOSOMA_GIT_TREE="${EXPECTED_TREE}"
export HOLOSOMA_GIT_ARCHIVE_SHA256="${GIT_ARCHIVE_SHA256}"
export HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH="${FORMAL_GIT_VERIFICATION}"
export HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT="${RUN_ROOT}/provenance-cache"

export MOTION_DIR="${MOTION_VIEW}"
export OBJECT_SPEC_PATH="${OBJECT_MAP}" OBJECT_URDF="${OBJECT_MAP}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST="${SINGLE_SLOT_SOURCE_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST="${SINGLE_SLOT_VIEW_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${MOTION_VIEW}"
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST="${RANK_SHARD_SOURCE_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE="${WORLD_SIZE}"
export HOLOSOMA_RANK_LOCAL_MOTION_ROOT="${SHARD_ROOT}"
export HOLOSOMA_MOTION_SHARD_MANIFEST="${SHARD_MANIFEST}"
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1 HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1 HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1
export HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK=0 HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0

unset RESUME_CKPT RESUME_CHECKPOINT RESUME_MODEL_FILE RESUME_STEP
unset POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT RESUME_FROM_BOX BOX_RESUME_CKPT
unset TEACHER_CHECKPOINT TEACHER_CHECKPOINT_EXPECTED_SHA256
export WANDB_ENTITY="${ENTITY}" WANDB_PROJECT="${PROJECT}" WANDB_RUN_ID
export WANDB_RESUME=must WANDB_RESUME_SAME_RUN=0 WANDB_CONSOLE=off WANDB_INIT_TIMEOUT=120
export WANDB_DIR="${RUN_ROOT}/wandb" LOGGER_BASE_DIR="${RUN_ROOT}/training_logs"
export HOLOSOMA_REQUIRE_WANDB_RUN=1

HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/compute_training_provenance.py" \
  --training-regime pure_rl \
  --motion-dir "${MOTION_VIEW}" \
  --object-map "${OBJECT_MAP}" \
  --motion-shard-manifest "${SHARD_MANIFEST}" \
  --source-root "${SOURCE_ROOT}")
export HOLOSOMA_TRAINING_PROVENANCE

TRAIN_ARGS=(
  exp:g1-29dof-wbt-w-object-hmi-depth-stage1
  logger:wandb
  --training.project="${PROJECT}"
  --training.name="${RUN_NAME}"
  --training.num-envs="${TOTAL_ENVS}"
  --training.seed=42
  --training.multigpu=True
  --training.export-onnx=True
  --algo.config.num-learning-iterations="${TARGET_ITERATIONS}"
  --algo.config.num-steps-per-env=24
  --algo.config.num-learning-epochs=5
  --algo.config.num-mini-batches=4
  --algo.config.clip-param=0.2
  --algo.config.gamma=0.99
  --algo.config.lam=0.95
  --algo.config.value-loss-coef=1.0
  --algo.config.entropy-coef=0.005
  --algo.config.max-grad-norm=1.0
  --algo.config.schedule=adaptive
  --algo.config.desired-kl=0.01
  --algo.config.actor-learning-rate=0.001
  --algo.config.critic-learning-rate=0.001
  --algo.config.min-actor-learning-rate=0.00001
  --algo.config.max-actor-learning-rate=0.01
  --algo.config.min-critic-learning-rate=0.00001
  --algo.config.max-critic-learning-rate=0.01
  --algo.config.init-noise-std=1.0
  --algo.config.module-dict.actor.min-noise-std=0.01
  --algo.config.normalize-actor-obs=False
  --algo.config.normalize-critic-obs=False
  --algo.config.save-interval=1000
  --algo.config.reset-rollout-at-checkpoint=False
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --simulator.config.sim.physx.gpu-heap-capacity="${PHYSX_GPU_HEAP_CAPACITY}"
  --simulator.config.sim.physx.gpu-temp-buffer-capacity="${PHYSX_GPU_TEMP_BUFFER_CAPACITY}"
  --command.setup-terms.motion-command.params.motion-config.motion-file="${MOTION_VIEW}"
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
  --robot.object.object-urdf-path="${OBJECT_MAP}"
  --logger.base-dir="${RUN_ROOT}/training_logs"
)

echo "[INFO] formal_hmi_depth_stage1 commit=${EXPECTED_COMMIT} tree=${EXPECTED_TREE} run=${WANDB_RUN_ID} world_size=${WORLD_SIZE} envs_per_rank=${ENVS_PER_RANK} total_envs=${TOTAL_ENVS} target=${TARGET_ITERATIONS} export_onnx=true"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${WORLD_SIZE}" \
  --master_port="${MASTER_PORT}" \
  "${SOURCE_ROOT}/src/holosoma/holosoma/train_agent_rank_visible.py" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${RUN_ROOT}/logs/formal_console.log"

if grep -Eq \
  'increase PxGpuDynamicsMemoryConfig::(foundLostPairsCapacity|foundLostAggregatePairsCapacity)|CUDA out of memory' \
  "${RUN_ROOT}/logs/formal_console.log"; then
  echo "[ERROR] formal e4096 run observed an undersized PhysX GPU buffer or CUDA OOM" >&2
  exit 2
fi
