#!/usr/bin/env bash
set -euo pipefail

# Formal single-node HMI Stage 2 initialized from one authenticated m8 actor.
# The node must execute this file from a clean exact-commit clone fetched from
# origin. Scientific assets are separately SHA-bound and contain no source.
if [[ $# -ne 17 ]]; then
  echo "usage: $0 <commit> <tree> <run-id> <run-name> <run-root> <run-contract> <contract-sha256> <rule90-manifest> <rule90-sha256> <source-id> <git-archive-sha256> <onnx-receipt> <onnx-receipt-sha256> <policy-init-pt> <policy-init-sha256> <policy-pair-json> <policy-pair-sha256>" >&2
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
POLICY_INIT_CHECKPOINT=${14}
POLICY_INIT_SHA256=${15}
POLICY_PAIR_JSON=${16}
POLICY_PAIR_SHA256=${17}

readonly ORIGIN_URL=https://github.com/Z1hanW/holosoma
readonly REMOTE_REF=origin/experiment/hmi-depth-interface
readonly ENTITY=zihanw22
readonly PROJECT=carry-any
readonly WORLD_SIZE=8
readonly ENVS_PER_RANK=2048
readonly TOTAL_ENVS=16384
readonly TARGET_ITERATIONS=20000
readonly MASTER_PORT=31527
readonly POLICY_INIT_COMPLETED_ITERATION=38999
readonly POLICY_INIT_NEXT_ITERATION=39000
readonly POLICY_INIT_MIGRATION=precomputed_turn_then_forward_to_hmi_terminal_goal_unfreeze_native_depth_v1
readonly POLICY_INIT_RESET_NOISE_STD=0.45
readonly DEFM_SUBMODULE_SHA=63ec5e1c1a9b280dcde9910b845f57e9224ebab5
readonly POINT_TRANSFORMER_SUBMODULE_SHA=3229e9b7de1770c8ad17c316f8e349982de509f8

readonly MOTION_VIEW=/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/by-source/307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef
readonly OBJECT_MAP=${MOTION_VIEW}/_clip_object_urdf_map.json
readonly SHARD_ROOT=/data/holosoma_inputs/corl79_plus_debug30_hmi_stage2_rank_shards_ws8_e2048
readonly SHARD_MANIFEST=${SHARD_ROOT}/manifest.json
readonly VIEW_MANIFEST_SHA256=2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb
readonly OBJECT_MAP_SHA256=70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c
readonly SHARD_MANIFEST_SHA256=2586feeb880afc7176937b83293084edab0bf2f107fbc3a9ba03f3c01eac9de6
readonly SINGLE_SLOT_SOURCE_DIGEST=0d1ae14db44e06fd9806e8757d3a26051697ee2f60ce446deed5b25ac9bfe6c5
readonly SINGLE_SLOT_VIEW_DIGEST=307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef
readonly RANK_SHARD_SOURCE_DIGEST=cdb52e87be4582a03535f30e5c820221bca66a1a3b9444e133851e96a937080c
readonly CANONICAL_CLIP_ID=box_10
readonly CANONICAL_NPZ_SHA256=48e9f7a95facda057193a6507bb11a8360dae52a04a1ec165a3da2bae23aee01
readonly CANONICAL_URDF_SHA256=bb67d9630cbf35b16f6f55831d746c16aa60f76e0721b88996b1adc7b4818821

die() { echo "[ERROR] $*" >&2; exit 2; }
check_sha() {
  local expected=$1 path=$2 actual
  [[ -f ${path} && ! -L ${path} ]] || die "missing non-symlink integrity input: ${path}"
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ ${actual} == "${expected}" ]] || die "SHA256 mismatch: ${path} expected=${expected} actual=${actual}"
}

[[ ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]] || die "invalid expected commit"
[[ ${EXPECTED_TREE} =~ ^[0-9a-f]{40}$ ]] || die "invalid expected tree"
for digest in "${RUN_CONTRACT_SHA256}" "${RULE90_MANIFEST_SHA256}" "${GIT_ARCHIVE_SHA256}" \
  "${ONNX_RECEIPT_SHA256}" "${POLICY_INIT_SHA256}" "${POLICY_PAIR_SHA256}"; do
  [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || die "invalid SHA256 argument"
done
[[ ${SOURCE_ID} =~ ^src-[0-9a-f]{64}$ ]] || die "invalid immutable source id"
[[ ${WANDB_RUN_ID} =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "invalid W&B run id"
[[ ${RUN_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "invalid formal run name"
[[ ${RUN_ROOT} == /mnt/holosoma_training/* && ! -L ${RUN_ROOT} ]] || die "run root must be a non-symlink child of /mnt/holosoma_training"

[[ $(git -C "${SOURCE_ROOT}" rev-parse HEAD) == "${EXPECTED_COMMIT}" ]] || die "Git HEAD mismatch"
[[ $(git -C "${SOURCE_ROOT}" rev-parse HEAD^{tree}) == "${EXPECTED_TREE}" ]] || die "Git tree mismatch"
[[ $(git -C "${SOURCE_ROOT}" remote get-url origin) == "${ORIGIN_URL}" ]] || die "Git origin URL mismatch"
git -C "${SOURCE_ROOT}" fetch --quiet origin experiment/hmi-depth-interface
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${EXPECTED_COMMIT}" "${REMOTE_REF}" || die "commit is not reachable from required remote ref"
git -C "${SOURCE_ROOT}" diff --quiet --ignore-submodules -- || die "tracked worktree is dirty"
git -C "${SOURCE_ROOT}" diff --cached --quiet --ignore-submodules -- || die "Git index is dirty"
if [[ -n $(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=all -- \
  '*.py' '*.sh' '*.yaml' '*.yml' '*.toml' '*.json' '*.urdf' '*.xml') ]]; then
  die "formal clone contains untracked executable/config source"
fi
for submodule_spec in \
  "submodules/defm:${DEFM_SUBMODULE_SHA}" \
  "submodules/PointTransformerV3:${POINT_TRANSFORMER_SUBMODULE_SHA}"; do
  submodule_path=${submodule_spec%%:*}
  submodule_sha=${submodule_spec##*:}
  [[ $(git -C "${SOURCE_ROOT}/${submodule_path}" rev-parse HEAD) == "${submodule_sha}" ]] \
    || die "declared submodule SHA mismatch: ${submodule_path}"
  git -C "${SOURCE_ROOT}/${submodule_path}" diff --quiet || die "dirty declared submodule: ${submodule_path}"
  git -C "${SOURCE_ROOT}/${submodule_path}" diff --cached --quiet || die "dirty declared submodule index: ${submodule_path}"
  [[ -z $(git -C "${SOURCE_ROOT}/${submodule_path}" status --porcelain --untracked-files=all) ]] \
    || die "declared submodule contains untracked files: ${submodule_path}"
done
while read -r _mode _sha _stage legacy_path; do
  case "${legacy_path}" in
    submodules/defm|submodules/PointTransformerV3) continue ;;
  esac
  [[ -d ${SOURCE_ROOT}/${legacy_path} && -z $(find "${SOURCE_ROOT}/${legacy_path}" -mindepth 1 -print -quit 2>/dev/null) ]] \
    || die "legacy unmapped gitlink must remain inactive and empty: ${legacy_path}"
done < <(git -C "${SOURCE_ROOT}" ls-files --stage | awk '$1 == 160000')

check_sha "${VIEW_MANIFEST_SHA256}" "${MOTION_VIEW}/manifest.json"
check_sha "${OBJECT_MAP_SHA256}" "${OBJECT_MAP}"
check_sha "${SHARD_MANIFEST_SHA256}" "${SHARD_MANIFEST}"
check_sha "${CANONICAL_NPZ_SHA256}" "${MOTION_VIEW}/${CANONICAL_CLIP_ID}.npz"
check_sha "${CANONICAL_URDF_SHA256}" "${MOTION_VIEW}/_single_slot_urdfs/${CANONICAL_CLIP_ID}.urdf"
check_sha "${RUN_CONTRACT_SHA256}" "${RUN_CONTRACT}"
check_sha "${RULE90_MANIFEST_SHA256}" "${RULE90_MANIFEST}"
check_sha "${ONNX_RECEIPT_SHA256}" "${ONNX_RECEIPT}"
check_sha "${POLICY_INIT_SHA256}" "${POLICY_INIT_CHECKPOINT}"
check_sha "${POLICY_PAIR_SHA256}" "${POLICY_PAIR_JSON}"

# shellcheck disable=SC1091
source "${SOURCE_ROOT}/scripts/source_isaacsim_setup.sh"
source "${SOURCE_ROOT}/scripts/gpu_launch_defaults.sh"
export PYTHONPATH="${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" - \
  "${MOTION_VIEW}/manifest.json" "${SHARD_MANIFEST}" "${RUN_CONTRACT}" \
  "${ONNX_RECEIPT}" "${POLICY_PAIR_JSON}" "${EXPECTED_COMMIT}" "${EXPECTED_TREE}" \
  "${WANDB_RUN_ID}" "${RUN_NAME}" "${SOURCE_ID}" "${POLICY_INIT_SHA256}" <<'PY'
import json, sys
(view_path, shard_path, contract_path, receipt_path, pair_path, commit, tree,
 run_id, run_name, source_id, policy_sha) = sys.argv[1:]
view=json.load(open(view_path, encoding="utf-8"))
shards=json.load(open(shard_path, encoding="utf-8"))
contract=json.load(open(contract_path, encoding="utf-8"))
receipt=json.load(open(receipt_path, encoding="utf-8"))
pair=json.load(open(pair_path, encoding="utf-8"))
assert view["clip_count"] == 109
assert view["derived_payload_digest"] == "307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef"
assert view["source_view_digest"] == "0d1ae14db44e06fd9806e8757d3a26051697ee2f60ce446deed5b25ac9bfe6c5"
assert shards["world_size"] == 8 and shards["environments_per_rank"] == 2048
assert shards["clip_count"] == 109 and shards["exact_clip_partition"] is True
assert shards["rank_clip_counts_divide_environments_per_rank"] is True
assert [row["clip_count"] for row in shards["shards"]] == [32,16,16,16,16,8,4,1]
assert set(shards["clip_cover_counts"].values()) == {1}
assert pair["semantics"] == "atomic_same_iteration_pt_onnx_policy_pair"
assert pair["completed_iteration"] == 38999 and pair["next_iteration"] == 39000
assert pair["pt"]["sha256"] == policy_sha
assert pair["onnx"]["checker"] == "onnx.checker.check_model"
assert pair["onnx"]["pytorch_vs_ort"] is True
assert contract["semantics"] == "formal_hmi_depth_stage2_m8_policy_init"
assert contract["git"]["commit"] == commit and contract["git"]["tree"] == tree
assert contract["git"]["source_id"] == source_id
assert contract["wandb"]["run_id"] == run_id and contract["wandb"]["name"] == run_name
training=contract["training"]
assert training["world_size"] == 8 and training["environments_per_rank"] == 2048
assert training["total_environments"] == 16384 and training["target_iterations"] == 20000
assert training["fresh"] is True and training["resume_checkpoint"] is None
assert training["policy_init_checkpoint_sha256"] == policy_sha
assert training["policy_init_next_iteration"] == 39000
assert training["policy_init_actor_contract_migration"] == "precomputed_turn_then_forward_to_hmi_terminal_goal_unfreeze_native_depth_v1"
assert training["policy_init_reset_noise_std"] == 0.45
assert training["export_onnx"] is True and training["contact_reward"] is False
assert contract["hmi"]["stage"] == 2 and contract["hmi"]["track_ratio"] == 0.5
assert contract["hmi"]["upstream_commit"] == "c353731999b3578c41ad5a00f896415b45e6a9f5"
assert receipt["accepted"] is True and receipt["source_commit"] == commit
assert receipt["world_size"] == 8 and receipt["environments_per_rank"] == 2048
assert receipt["policy_init_sha256"] == policy_sha
assert receipt["actor_contract_migration_verified"] is True
assert receipt["terminal_pair"]["onnx_checker"] is True
assert receipt["terminal_pair"]["onnxruntime_loaded"] is True
assert receipt["terminal_pair"]["pytorch_vs_ort"] is True
PY

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
  --manifest "${RULE90_MANIFEST}" \
  --expected-manifest-sha256 "${RULE90_MANIFEST_SHA256}" \
  --expected-source-snapshot-id "${SOURCE_ID}" \
  --expected-entity "${ENTITY}" --expected-project "${PROJECT}" \
  --expected-run-id "${WANDB_RUN_ID}" --expected-run-name "${RUN_NAME}" \
  --expected-world-size "${WORLD_SIZE}" >/dev/null

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#GPU_ROWS[@]} -eq 8 ]] || die "expected exactly eight GPUs"
for row in "${GPU_ROWS[@]}"; do
  [[ ${row} == *"NVIDIA L40S"* && ${row##*, } == 0 ]] || die "unhealthy GPU: ${row}"
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  die "formal node is not GPU-idle"
fi
ss -lnt | awk '{print $4}' | grep -Eq "(^|:)${MASTER_PORT}$" && die "master port is occupied"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/wandb" "${RUN_ROOT}/training_logs" "${RUN_ROOT}/provenance-cache"
export TMPDIR="${RUN_ROOT}/tmp" TMP="${RUN_ROOT}/tmp" TEMP="${RUN_ROOT}/tmp"
mkdir -p "${TMPDIR}"
EXIT_FILE=${RUN_ROOT}/formal.exit
[[ ! -e ${EXIT_FILE} ]] || die "formal exit file already exists"
trap 'rc=$?; printf "%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "${EXIT_FILE}"' EXIT

FORMAL_GIT_VERIFICATION=${RUN_ROOT}/formal_git_verification.json
"${PYTHON_BIN}" - "${FORMAL_GIT_VERIFICATION}" "${ORIGIN_URL}" "${REMOTE_REF}" \
  "${EXPECTED_COMMIT}" "${EXPECTED_TREE}" "${SOURCE_ROOT}" <<'PY'
import datetime, json, os, sys, tempfile
path, remote_url, remote_ref, commit, tree, source_root = sys.argv[1:]
submodules={
    "submodules/defm": "63ec5e1c1a9b280dcde9910b845f57e9224ebab5",
    "submodules/PointTransformerV3": "3229e9b7de1770c8ad17c316f8e349982de509f8",
}
payload={
    "version": 1,
    "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "hostname": os.uname().nodename,
    "remote_url": remote_url,
    "remote_ref": remote_ref,
    "commit_sha": commit,
    "tree_sha": tree,
    "fetched_ref_commit": commit,
    "tracked_diff_clean": True,
    "untracked_clean": True,
    "declared_submodules": submodules,
    "legacy_unmapped_gitlinks_inactive_and_empty": True,
}
fd,tmp=tempfile.mkstemp(prefix=".git-verification-",suffix=".json",dir=os.path.dirname(path))
try:
    with os.fdopen(fd,"w",encoding="utf-8") as stream:
        json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,path)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
PY

export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HEADLESS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT
export TORCH_DIST_BACKEND=gloo TORCH_DIST_TIMEOUT_SEC=1800 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export HOLOSOMA_GLOO_GRAD_REDUCE=1 HOLOSOMA_GLOO_BARRIER=1 HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=0 HOLOSOMA_RANK_VISIBLE_DEVICES=1
export HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1 HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1
export HOLOSOMA_CONTIGUOUS_MINIBATCHES=1 HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True

export HOLOSOMA_SOURCE_ROOT="${SOURCE_ROOT}" HOLOSOMA_SOURCE_SNAPSHOT_ID="${SOURCE_ID}"
export HOLOSOMA_SOURCE_MANIFEST_SHA256="${SOURCE_ID#src-}"
export HOLOSOMA_RUN_CONTRACT="${RUN_CONTRACT}" HOLOSOMA_RUN_CONTRACT_SHA256="${RUN_CONTRACT_SHA256}"
export HOLOSOMA_GIT_REMOTE_URL="${ORIGIN_URL}" HOLOSOMA_GIT_REMOTE_REF="${REMOTE_REF}"
export HOLOSOMA_GIT_COMMIT="${EXPECTED_COMMIT}" HOLOSOMA_GIT_TREE="${EXPECTED_TREE}"
export HOLOSOMA_GIT_ARCHIVE_SHA256="${GIT_ARCHIVE_SHA256}"
export HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH="${FORMAL_GIT_VERIFICATION}"
export HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT="${RUN_ROOT}/provenance-cache"

export MOTION_DIR="${MOTION_VIEW}" OBJECT_SPEC_PATH="${OBJECT_MAP}" OBJECT_URDF="${OBJECT_MAP}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST="${SINGLE_SLOT_SOURCE_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST="${SINGLE_SLOT_VIEW_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${MOTION_VIEW}"
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST="${RANK_SHARD_SOURCE_DIGEST}"
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE="${WORLD_SIZE}" HOLOSOMA_RANK_LOCAL_MOTION_ROOT="${SHARD_ROOT}"
export HOLOSOMA_MOTION_SHARD_MANIFEST="${SHARD_MANIFEST}"
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1 HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1 HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1
export HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK=0 HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0
unset HOLOSOMA_ALLOW_LEGACY_UNVERIFIED_POLICY_LOAD
unset RESUME_CKPT RESUME_CHECKPOINT RESUME_MODEL_FILE RESUME_STEP TEACHER_CHECKPOINT TEACHER_CHECKPOINT_EXPECTED_SHA256

export WANDB_ENTITY="${ENTITY}" WANDB_PROJECT="${PROJECT}" WANDB_RUN_ID WANDB_RESUME=must WANDB_RESUME_SAME_RUN=0
export WANDB_CONSOLE=off WANDB_INIT_TIMEOUT=120 WANDB_DIR="${RUN_ROOT}/wandb" LOGGER_BASE_DIR="${RUN_ROOT}/training_logs"
export HOLOSOMA_REQUIRE_WANDB_RUN=1

HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/compute_training_provenance.py" \
  --training-regime pure_rl --motion-dir "${MOTION_VIEW}" --object-map "${OBJECT_MAP}" \
  --motion-shard-manifest "${SHARD_MANIFEST}" --source-root "${SOURCE_ROOT}" \
  --policy-init-checkpoint "${POLICY_INIT_CHECKPOINT}")
export HOLOSOMA_TRAINING_PROVENANCE

TRAIN_ARGS=(
  exp:g1-29dof-wbt-w-object-hmi-depth-stage2 logger:wandb
  --training.project="${PROJECT}" --training.name="${RUN_NAME}"
  --training.num-envs="${TOTAL_ENVS}" --training.seed=42 --training.multigpu=True
  --training.export-onnx=True --training.policy-init-checkpoint="${POLICY_INIT_CHECKPOINT}"
  --training.policy-init-actor-contract-migration="${POLICY_INIT_MIGRATION}"
  --training.policy-init-reset-noise-std="${POLICY_INIT_RESET_NOISE_STD}"
  --algo.config.num-learning-iterations="${TARGET_ITERATIONS}" --algo.config.num-steps-per-env=24
  --algo.config.num-learning-epochs=5 --algo.config.num-mini-batches=4
  --algo.config.clip-param=0.2 --algo.config.gamma=0.99 --algo.config.lam=0.95
  --algo.config.value-loss-coef=1.0 --algo.config.entropy-coef=0.005 --algo.config.max-grad-norm=1.0
  --algo.config.schedule=adaptive --algo.config.desired-kl=0.01
  --algo.config.actor-learning-rate=0.001 --algo.config.critic-learning-rate=0.001
  --algo.config.min-actor-learning-rate=0.00001 --algo.config.max-actor-learning-rate=0.01
  --algo.config.min-critic-learning-rate=0.00001 --algo.config.max-critic-learning-rate=0.01
  --algo.config.init-noise-std=0.45 --algo.config.module-dict.actor.min-noise-std=0.01
  --algo.config.module-dict.actor.layer-config.hidden-dims="[2048,1024,512,256,128]"
  --algo.config.normalize-actor-obs=False --algo.config.normalize-critic-obs=False
  --algo.config.save-interval=1000 --algo.config.reset-rollout-at-checkpoint=False
  --simulator.config.scene.env-spacing=5.0
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=335544320
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=469762048
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=83886080
  --simulator.config.sim.physx.gpu-collision-stack-size=268435456
  --simulator.config.sim.physx.gpu-heap-capacity=67108864
  --simulator.config.sim.physx.gpu-temp-buffer-capacity=16777216
  --command.setup-terms.motion-command.params.motion-config.motion-file="${MOTION_VIEW}"
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
  --perception.camera-apply-sensor-noise=True --perception.camera-warp-edge-noise=True
  --perception.camera-warp-enable-holes=True --perception.camera-warp-hole-prob=0.2
  --perception.camera-warp-additive-noise-std=0.03 --perception.camera-warp-depth-offset-std=0.03
  --perception.object-geometry-mode=mesh --perception.encoder-pretrained=False
  --perception.encoder-freeze-backbone=False
  --robot.object.object-urdf-path="${OBJECT_MAP}"
  --reward.terms.offline-contact-guidance.weight=0.0
  --logger.base-dir="${RUN_ROOT}/training_logs"
)

echo "[INFO] formal_hmi_stage2_m8 commit=${EXPECTED_COMMIT} run=${WANDB_RUN_ID} policy_init=${POLICY_INIT_SHA256} world=8 e2048 total=16384 target=20000 track_gen=0.5/0.5 contact_reward=false export_onnx=true"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${WORLD_SIZE}" --master_port="${MASTER_PORT}" \
  "${SOURCE_ROOT}/src/holosoma/holosoma/train_agent_rank_visible.py" "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${RUN_ROOT}/logs/formal_console.log"

if grep -Eq 'increase PxGpuDynamicsMemoryConfig::|CUDA out of memory|Policy-init actor semantic contract mismatch' "${RUN_ROOT}/logs/formal_console.log"; then
  die "formal run observed a fatal acceptance signature"
fi
