#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-3]$ ]]; then
  echo "usage: $0 NODE_RANK(0..3)" >&2
  exit 2
fi

export NODE_RANK="$1"
readonly EXPECTED_NODE_IPS=(
  10.99.0.141
  10.99.0.186
  10.99.1.154
  10.99.0.167
)
readonly EXPECTED_NODE_IP="${EXPECTED_NODE_IPS[$NODE_RANK]}"
readonly SOURCE_ID=src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_DIGEST=a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_ROOT="/home/ubuntu/FAR/holosoma_runs/${SOURCE_ID}"
readonly LAUNCH_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_student_pure_rl_depth_corl79_ws32_e1020_sparse30_no_contact_reward_20260723_215131
readonly RUN_ROOT="${LAUNCH_ROOT}/run"
readonly RUN_NAME=student_pure_rl_depth_corl79_ws32_e1020_sparse30_no_contact_reward_20260723_215131
readonly RUN_CONTRACT="${LAUNCH_ROOT}/run_contract.json"
readonly RUN_CONTRACT_SHA=804b56ad8fd740e6282c8fc96159d814f4aa26706ccb18238cfbd578c71f3cec
readonly REPLAY_MANIFEST="${LAUNCH_ROOT}/replay_preflight_manifest.json"
readonly REPLAY_MANIFEST_SHA=2765b91f3bd4c07883581af4ff1ca01475ebc062f00bf920510636d441b8b1dc
readonly PYTHON_RUNTIME=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f/site-packages
readonly PYTHON_RUNTIME_MANIFEST_SHA=2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f
readonly NCCL_ROOT=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly NCCL_SHA=e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly BANK_REL=data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball
readonly BANK="${SOURCE_ROOT}/${BANK_REL}"
readonly SINGLE_DIR="${BANK}/_scientific_corl79_single_slot/by-source/6209b4742cce3b2989c7ea1f96a55a27d57bcf91eeb90699d409747187ca2cca"
readonly SHARD_ROOT="${LAUNCH_ROOT}/rank_shards/ws32"
readonly CONTACT_ROOT="${BANK}/contact_export_from_teacher_success133_final0p5"
readonly CANONICAL_FIRST_OBJECT_MESH="${BANK}/objects/motion_bank_box_10/box_10.obj"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/wandb" "${RUN_ROOT}/training_logs" "${RUN_ROOT}/provenance-cache/node_${NODE_RANK}"
status_file="${RUN_ROOT}/node_rank_${NODE_RANK}.exit"
rm -f "${status_file}"
trap 'rc=$?; printf "%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$NODE_RANK" > "$status_file"' EXIT

if ! hostname -I | tr ' ' '\n' | grep -Fxq "${EXPECTED_NODE_IP}"; then
  echo "[ERROR] node-rank/IP mismatch: rank=${NODE_RANK} expected=${EXPECTED_NODE_IP} actual=$(hostname -I)" >&2
  exit 2
fi

check_sha() {
  local expected="$1" path="$2" actual
  [[ -f "${path}" ]] || { echo "[ERROR] missing integrity input: ${path}" >&2; exit 2; }
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "[ERROR] SHA mismatch: ${path} expected=${expected} actual=${actual}" >&2
    exit 2
  }
}

[[ -f "${SOURCE_ROOT}/.holosoma_snapshot/id" ]]
grep -Fxq "${SOURCE_ID}" "${SOURCE_ROOT}/.holosoma_snapshot/id"
(cd "${SOURCE_ROOT}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
check_sha "${RUN_CONTRACT_SHA}" "${RUN_CONTRACT}"
check_sha "${REPLAY_MANIFEST_SHA}" "${REPLAY_MANIFEST}"
check_sha "${PYTHON_RUNTIME_MANIFEST_SHA}" "${PYTHON_RUNTIME}/.holosoma-runtime-manifest.sha256"
check_sha "${NCCL_SHA}" "${NCCL_ROOT}/libnccl.so.2"
check_sha a818815e52e137be70a8fbbd4fcfbf644e449f4d7c1bda98399ca201ca922c11 "${BANK}/nfs_package_manifest.json"
check_sha f632eb303034f9b9840758df385d99ea0019beae790628775265366f3a127dc8 "${BANK}/_clip_object_urdf_map.json"
check_sha 910399359c1bf8d236ec446667b27902de0037c24c7dfcb40aa70a1bf6d0522d "${SINGLE_DIR}/manifest.json"
check_sha 7926ea58ad4c13d4d0bdc7f02b03a6e6a65dedbaf01609f2e962f04f485069a0 "${SINGLE_DIR}/_clip_object_urdf_map.json"
check_sha 6861cb9b62547c8d16f68d7759344805b9684a6335fe32923f90f8acd54d799c "${SHARD_ROOT}/manifest.json"
check_sha 61b7a9b47a2bd2f3eadb9fc37d94ac682b8923d4a7e0cc9121769d8b2c33c45a "${SINGLE_DIR}/box_10.npz"
check_sha a7db8f4e0ee64d89af83d610c8e668a0f26030efcf5fa0ae25ff26b936958de6 "${SINGLE_DIR}/_single_slot_urdfs/box_10.urdf"
check_sha 8cf2901790889babfeef397d63ad890751f9d6dd4f444f9095d718a1148e7b32 "${CANONICAL_FIRST_OBJECT_MESH}"

[[ "${HOME:-}" == /home/ubuntu ]] || { echo "[ERROR] unexpected HOME=${HOME:-unset}" >&2; exit 2; }
export PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export OMP_NUM_THREADS=1

"${PYTHON_BIN}" - "${SHARD_ROOT}/manifest.json" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
environments_per_rank = 1020
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
shards = manifest.get("shards", [])
if manifest.get("world_size") != 32 or len(shards) != 32:
    raise SystemExit(
        f"[ERROR] expected exactly 32 shards, got world_size={manifest.get('world_size')} count={len(shards)}"
    )
bad = [
    (int(shard["rank"]), int(shard["clip_count"]))
    for shard in shards
    if int(shard["clip_count"]) <= 0
    or environments_per_rank % int(shard["clip_count"]) != 0
]
if bad:
    raise SystemExit(
        "[ERROR] exact fixed scientific env-to-clip assignment is impossible "
        f"for environments_per_rank={environments_per_rank}: {bad}"
    )
if sum(int(shard["clip_count"]) for shard in shards) != 79:
    raise SystemExit("[ERROR] rank shards do not cover exactly 79 clips")
print(
    "[INFO] exact_fixed_env_assignment_ok "
    f"environments_per_rank={environments_per_rank} clip_counts={sorted({int(s['clip_count']) for s in shards})}"
)
PY

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/validate_contact_sidecars.py" \
  --motion-dir "${SINGLE_DIR}" \
  --contact-root "${CONTACT_ROOT}" \
  --expected-total 79 \
  --motion-end-mode episodic \
  --runtime-prepend-compensation \
  --runtime-prepend-duration-s 0.2 \
  --offline-contact-region-names '["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]' \
  --offline-wrist-region-names '["left_wrist","right_wrist"]' >/dev/null

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits)
[[ ${#gpu_rows[@]} -eq 8 ]] || { echo "[ERROR] expected 8 GPUs, found ${#gpu_rows[@]}" >&2; exit 2; }
for row in "${gpu_rows[@]}"; do
  [[ "${row}" == *"NVIDIA L40S"* ]] || { echo "[ERROR] unexpected GPU: ${row}" >&2; exit 2; }
  [[ "${row##*, }" == 0 ]] || { echo "[ERROR] nonzero volatile UECC: ${row}" >&2; exit 2; }
done
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d') ]]; then
  echo "[ERROR] selected node is no longer GPU-idle; refusing launch" >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader >&2 || true
  exit 2
fi

export HOLOSOMA_SOURCE_ROOT="${SOURCE_ROOT}"
export HOLOSOMA_SOURCE_SNAPSHOT_ID="${SOURCE_ID}"
export HOLOSOMA_SOURCE_MANIFEST_SHA256="${SOURCE_DIGEST}"
export HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256="${PYTHON_RUNTIME_MANIFEST_SHA}"
export PYTHON_RUNTIME_SITEPACKAGES="${PYTHON_RUNTIME}"
export HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT="${RUN_ROOT}/provenance-cache/node_${NODE_RANK}"
export HOLOSOMA_LAUNCH_TOKEN=f7cab8562dc011eff383e20a7a050c2b58d384b2f1ddbaced2bbb17a10574b68
export HOLOSOMA_LAUNCH_EPOCH=1784843492726076980

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
export NNODES=4
export MASTER_ADDR=10.99.0.141
export MASTER_PORT=31782
export TRAINING_SEED=42

export NCCL_LIB_DIR="${NCCL_ROOT}"
export NCCL_LIB_SHA256="${NCCL_SHA}"
export LD_LIBRARY_PATH="${NCCL_ROOT}"
export LD_PRELOAD="${NCCL_ROOT}/libnccl.so.2"
export NCCL_SOCKET_IFNAME=enp135s0
export GLOO_SOCKET_IFNAME=enp135s0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_RETRY_CNT=34
export NCCL_SOCKET_RETRY_SLEEP_MSEC=100
export NCCL_SOCKET_NTHREADS=2
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_DIST_BACKEND=gloo
export TORCH_DIST_TIMEOUT_SEC=3600
export HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC=300
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=65536
export TORCH_NCCL_PROPAGATE_ERROR=1
export TORCH_NCCL_DESYNC_DEBUG=0
export TORCH_NCCL_ENABLE_TIMING=0
export TORCH_NCCL_BLOCKING_WAIT=0
export HOLOSOMA_GLOO_GRAD_REDUCE=0
export HOLOSOMA_GLOO_BARRIER=1
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=1
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1
export HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=0
export HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER=1
export HOLOSOMA_RANK_VISIBLE_DEVICES=1
export HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE=1
export HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=0
export HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP=0
export HOLOSOMA_SYNC_AFTER_MICROBATCH_FORWARD=0
export HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=1
export HOLOSOMA_CONTIGUOUS_MINIBATCHES=1
export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1

export MOTION_DIR="${SINGLE_DIR}"
export OBJECT_SPEC_PATH="${SINGLE_DIR}/_clip_object_urdf_map.json"
export OBJECT_URDF="${OBJECT_SPEC_PATH}"
export CONTACT_EXPORT_ROOT="${CONTACT_ROOT}"
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST=531b535d01995643c2a3d591a3b8c6ca2dddb9ae427366ae554898e0d592a483
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST=6209b4742cce3b2989c7ea1f96a55a27d57bcf91eeb90699d409747187ca2cca
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${SINGLE_DIR}"
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST=e98a6f9e66c07e9e552388593f98f2c7eeff4bff7bc7f3112612d0fff7c5c4f8
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=32
export HOLOSOMA_RANK_LOCAL_MOTION_ROOT="${SHARD_ROOT}"
export HOLOSOMA_MOTION_SHARD_MANIFEST="${SHARD_ROOT}/manifest.json"
export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1
export HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE=1
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0
export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=1
export HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=1
export HOLOSOMA_MOTION_METRICS_INTERVAL=16
export HOLOSOMA_DISABLE_AUTO_RESET=0
export HOLOSOMA_DISABLE_CLIP_END_RESET=0
export HOLOSOMA_DISABLE_MOTION_END_RESET=0
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True
export HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH=1
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=1

unset RESUME_CKPT RESUME_CHECKPOINT RESUME_MODEL_FILE RESUME_STEP
unset POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT RESUME_FROM_BOX BOX_RESUME_CKPT
unset TEACHER_CHECKPOINT TEACHER_CHECKPOINT_EXPECTED_SHA256
unset RESUME_WANDB_ID WANDB_MODE WANDB_DISABLED
export WANDB_PROJECT=carry-any
export WANDB_ENTITY=zihanw22
export WANDB_RUN_ID=ptabkuyq
export WANDB_RESUME=must
export WANDB_RESUME_SAME_RUN=0
export WANDB_CONSOLE=off
export WANDB_INIT_TIMEOUT=120
export WANDB_DIR="${RUN_ROOT}/wandb"
export LOGGER_BASE_DIR="${RUN_ROOT}/training_logs"

HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/compute_training_provenance.py" \
  --training-regime pure_rl \
  --motion-dir "${MOTION_DIR}" \
  --object-map "${OBJECT_SPEC_PATH}" \
  --contact-root "${CONTACT_EXPORT_ROOT}" \
  --motion-shard-manifest "${HOLOSOMA_MOTION_SHARD_MANIFEST}" \
  --contact-interval-runtime-prepend-compensation true \
  --source-root "${SOURCE_ROOT}")
export HOLOSOMA_TRAINING_PROVENANCE

TRAIN_ARGS=(
  exp:g1-29dof-wbt-w-object-distill-sparse-root-cmd
  command:g1-29dof-wbt-w-object-generalist
  reward:g1-29dof-wbt-w-object-generalist-offline-contact-guidance
  perception:camera_depth_d435i
  termination:g1_29dof_wbt_generalist
  randomization:g1_29dof_wbt_w_object_with_action_delay
  logger:wandb
  --training.project=carry-any
  --training.name="${RUN_NAME}"
  --training.num-envs=32640
  --training.seed=42
  --training.multigpu=True
  --training.export-onnx=False
  --algo.config.distill.enabled=False
  --algo.config.distill.mode=mse
  --algo.config.num-learning-iterations=40000
  --algo.config.num-steps-per-env=24
  --algo.config.num-learning-epochs=7
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
  --algo.config.module-dict.actor.type=MLP
  --algo.config.module-dict.actor.input-dim="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
  --algo.config.module-dict.actor.layer-config.hidden-dims='[512,256,128]'
  --algo.config.module-dict.critic.layer-config.hidden-dims='[512,256,128]'
  --observation.groups.actor_obs_root_contact_aware.history-length=1
  --observation.groups.actor_obs_drop_button.history-length=1
  --observation.groups.actor_obs_proprio_with_actions_no_linvel.history-length=1
  --observation.groups.critic_proprio_history.history-length=1
  --command.setup-terms.motion-command.params.motion-config.motion-file="${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.clip-weighting-strategy=uniform_clip
  --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion=False
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=True
  --command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root="${CONTACT_ROOT}/clips"
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end=0.2
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter=0
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter=39999
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end=0.0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter=0
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter=39999
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale=1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0.2
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=True
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=2.0
  --command.setup-terms.motion-command.params.motion-config.contact-interval-runtime-prepend-compensation=True
  --command.setup-terms.motion-command.params.motion-config.contact-aware-button-window-mode=contact_interval
  --command.setup-terms.motion-command.params.motion-config.contact-aware-carry-window-mode=peak_height
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-alpha=0.91
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-smoothing-steps=5
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-command-mode=t1_aligned_segment
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-segment-steps=30
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-zero-yaw-threshold-deg=0.0
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled=True
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-half-width-steps=50
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-density-boost=7.0
  --reward.terms.offline-contact-guidance.params.contact-export-root="${CONTACT_ROOT}"
  --reward.terms.offline-contact-guidance.params.contact-region-names='["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]'
  --reward.terms.offline-contact-guidance.params.wrist-region-names='["left_wrist","right_wrist"]'
  --reward.terms.offline-contact-guidance.weight=0.0
  --randomization.setup_terms.push_randomizer_state.params.push_interval_s='[0.5,2.0]'
  --randomization.setup_terms.push_randomizer_state.params.max_push_vel='[0.7,0.7,0.25,0.7,0.7,1.0]'
  --perception.camera-apply-sensor-noise=True
  --perception.camera-warp-edge-noise=True
  --perception.camera-warp-enable-holes=True
  --perception.camera-warp-hole-prob=0.2
  --perception.camera-warp-additive-noise-std=0.03
  --perception.camera-warp-depth-offset-std=0.03
  --perception.object-geometry-mode=mesh
  --robot.object.enabled=True
  --robot.object.object-urdf-path="${OBJECT_SPEC_PATH}"
  --simulator.config.sim.max-episode-length-s=8.0
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=268435456
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=268435456
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=67108864
  --simulator.config.sim.physx.gpu-collision-stack-size=268435456
  --logger.entity=zihanw22
  --logger.id=ptabkuyq
  --logger.resume=must
  --logger.name="${RUN_NAME}"
  --logger.base-dir="${LOGGER_BASE_DIR}"
  --logger.video.enabled=False
  --logger.headless-recording=False
  --logger.video.upload-to-wandb=False
)

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/validate_train_cli.py" \
  --expected-motion-end-mode episodic \
  -- "${TRAIN_ARGS[@]}"

if [[ "${NODE_RANK}" == 0 ]]; then
  "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/wandb_replay_preflight.py" verify \
    --manifest "${REPLAY_MANIFEST}" \
    --expected-manifest-sha256 "${REPLAY_MANIFEST_SHA}" \
    --expected-source-snapshot-id "${SOURCE_ID}" \
    --expected-entity zihanw22 \
    --expected-project carry-any \
    --expected-run-id ptabkuyq \
    --expected-run-name "${RUN_NAME}" \
    --expected-world-size 32
fi

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "[INFO] formal_student_pure_rl_depth_sparse30_ws32_preflight_only_ok node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP}"
  exit 0
fi

cd "${SOURCE_ROOT}"
echo "[INFO] formal_student_pure_rl_depth_sparse30_ws32_preflight_ok node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP} source=${SOURCE_ID} wandb=zihanw22/carry-any/ptabkuyq pure_ppo=true distill=false teacher_loaded=false depth_student=true fresh_policy=true offline_contact_guidance_weight=0.0 sparse_root_mode=t1_aligned_segment sparse_root_segment_steps=30 environments_per_rank=1020"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=4 \
  --node_rank="${NODE_RANK}" \
  --master_addr=10.99.0.141 \
  --nproc_per_node=8 \
  --max_restarts=0 \
  --master_port=31782 \
  src/holosoma/holosoma/train_agent_rank_visible.py \
  "${TRAIN_ARGS[@]}"
