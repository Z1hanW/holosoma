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

if [[ -z ${EXPECTED_COMMIT} || ! ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <full-commit-sha> [run-label] [stage1|stage2] [policy-init-pt] [policy-init-sha256]" >&2
  exit 2
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
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_hull

TRAINING_ARGS=(
  "${EXPERIMENT_PRESET}"
  --training.multigpu=True
  --training.num-envs=512
  --training.seed=42
  --training.export-onnx=True
  --algo.config.num-learning-iterations=2
  --algo.config.save-interval=1
  --algo.config.num-steps-per-env=8
  --algo.config.num-learning-epochs=1
  --algo.config.num-mini-batches=1
  --logger.base-dir="${RUN_ROOT}"
)
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

echo "[INFO] nonformal_hmi_canary commit=${EXPECTED_COMMIT} stage=${HMI_STAGE} world_size=8 total_envs=512 iterations=2 export_onnx=true"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  src/holosoma/holosoma/train_agent_rank_visible.py \
  "${TRAINING_ARGS[@]}" \
  2>&1 | tee "${RUN_ROOT}/train.log"
