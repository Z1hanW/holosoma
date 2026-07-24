#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-7]$ ]]; then
  echo "usage: $0 NODE_RANK(0..7)" >&2
  exit 2
fi

export NODE_RANK="$1"
readonly EXPECTED_NODE_IPS=(
  10.99.0.24
  10.99.0.39
  10.99.0.54
  10.99.0.61
  10.99.0.180
  10.99.0.183
  10.99.0.201
  10.99.0.244
)
readonly EXPECTED_NODE_IP="${EXPECTED_NODE_IPS[$NODE_RANK]}"
readonly SOURCE_ID=src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_DIGEST=a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_ROOT="/home/ubuntu/FAR/holosoma_runs/${SOURCE_ID}"
readonly LAUNCH_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_student_pure_rl_depth_rollout3_ws64_fresh_20260720_235642
readonly RUN_ROOT="${LAUNCH_ROOT}/run"
readonly RUN_NAME=student_pure_rl_depth_rollout3_ws64_fresh_20260720_235642
readonly RUN_CONTRACT="${LAUNCH_ROOT}/run_contract.json"
readonly RUN_CONTRACT_SHA=a0dc0f91a5ccd729f1e6649c541203d2b343738cbe1e1bda98c5e88c483945a2
readonly REPLAY_MANIFEST="${LAUNCH_ROOT}/replay_preflight_manifest.json"
readonly REPLAY_MANIFEST_SHA=e2b7533166972c9650efeb647b1c47ab11000530f256e9af85897285a5d83e3e
readonly PYTHON_RUNTIME=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f/site-packages
readonly PYTHON_RUNTIME_MANIFEST_SHA=2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f
readonly NCCL_ROOT=/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly NCCL_SHA=e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899
readonly BANK_REL=data/ds_as_data/teacher_ab3_true_rollout_u8_solid_sharedphys__src_5ae3055eeff0098dad099ec26d721b76ac7675207e7d41a6ee3aaa2a15080b49
readonly BANK="${SOURCE_ROOT}/${BANK_REL}"
readonly SINGLE_DIR="${BANK}/_scientific_teacher_ab_single_slot/by-source/25d7623ec5d785ee12595322f5939afe4655ea8472ed14fc9b1af5cf2c915f83"
readonly SHARD_ROOT="${SINGLE_DIR}/_rank_shards/by-source/06ee3e38177b35bff7567c4f31a38261d9ac460da14f514eb41c27367f9a9e4a/ws64"
readonly CONTACT_ROOT="${BANK}/contact_export_from_teacher_realmesh_rollout"
readonly SOURCE_OBJECT_URDF_ROOT="${SOURCE_ROOT}/data/ds_as_data/teacher_ab3_original_raw_local_20260720/_single_slot_urdfs"
readonly SHARED_OBJECT_MESH_ROOT="${SOURCE_ROOT}/data/ds_as_data/prism_debug30_convexhull_allmesh/objects_convex_hull"

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
check_sha 8c9d666027e0f462c7414399c1555fd32b761654e292a1a3fb5e6352a4be4d6a "${BANK}/manifest.json"
check_sha 02acd5041a0dc9f5a60e0f3a09b9de7014a1d6fc4032113d8ead2682ac6c5afc "${BANK}/_clip_object_urdf_map.json"
check_sha c1e0c883be5b1062e5510a6406bc10ee056642fdba2d9d2d9c974e9cc65d7a65 "${SINGLE_DIR}/manifest.json"
check_sha 311cd1a9507b2ee8a218ad36ecf7284a774bd752bd5af04cb248528cc302c6be "${SHARD_ROOT}/manifest.json"
check_sha 382e0aaffc8a6e4dd4c1906eaed50c5ed3e244bdd3e769e5581f374e60f06126 "${BANK}/scaledown__any_barrel_25.npz"
check_sha 87644e984e4af1e7b75f3e3f83d822a0ee18e2b7604968d38f16d5b80cae46bb "${BANK}/unscale__any_ball_29.npz"
check_sha 9c981f20edb97a9d598fee7beb15c42278ad4bb0bc1725812540fd243f35adb4 "${BANK}/unscale__any_bin_29.npz"
check_sha 0a69875d5a2d4ed19a62040a1ee326e81f68b49c934f4ed659d21b23828843e3 "${SINGLE_DIR}/_single_slot_urdfs/scaledown_any_barrel_25.urdf"
check_sha 2e9d3d7c47f5915415e2aefa0b76b7294a829bfb5bb74dc637466a40dc38d556 "${SINGLE_DIR}/_single_slot_urdfs/unscale_any_ball_29.urdf"
check_sha c2009cb217f8157bfd581bceffe1ad422d98722b425af8341057674118e1c385 "${SINGLE_DIR}/_single_slot_urdfs/unscale_any_bin_29.urdf"
check_sha 5f52fbf224f05e9b3345bf49fabfeec4507a5507236fd002b66befca88f8d5e3 "${SOURCE_OBJECT_URDF_ROOT}/scaledown_any_barrel_25.urdf"
check_sha ae6a5e08a01e04c81dd10f4e79892cb394e1b24b4fa19f6c6b6579a9fe9d33de "${SOURCE_OBJECT_URDF_ROOT}/unscale_any_ball_29.urdf"
check_sha 5a0d2fadf8d3ce48d1d5b865c8c83164a51e6a7dc05f2dd69bdbe6de317d94ce "${SOURCE_OBJECT_URDF_ROOT}/unscale_any_bin_29.urdf"
check_sha 24d046ad6047fa8f33c63138c9be35975d5a0e078bc6df321b2068442d64f4c5 "${SHARED_OBJECT_MESH_ROOT}/object_mesh_yup_e7481f4304/object_mesh_yup_convex_hull.obj"
check_sha 9734a65b4cd1127c96fad2b499832cbe5f5c7608200c593127c45db31b92d5b9 "${SHARED_OBJECT_MESH_ROOT}/object_mesh_yup_e6603064bd/object_mesh_yup_convex_hull.obj"
check_sha daae95872696e55484f37a166978fca182303ce1bb73b26d851b0d085784890d "${SHARED_OBJECT_MESH_ROOT}/object_mesh_yup_266fbb26f5/object_mesh_yup_convex_hull.obj"

export HOME=/home/ubuntu
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

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/validate_contact_sidecars.py" \
  --motion-dir "${SINGLE_DIR}" \
  --contact-root "${CONTACT_ROOT}" \
  --expected-total 3 \
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
export HOLOSOMA_LAUNCH_TOKEN=1a3ce6fd526fe0d410d08b9dff55783d4ccf2fb6f5697b5e79a6ac9226f69a8c
export HOLOSOMA_LAUNCH_EPOCH=1784591802

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
export NNODES=8
export MASTER_ADDR=10.99.0.24
export MASTER_PORT=31751
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
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST=18aa7972ca802ca654420bb06475ce9731cead28c46010b28680341f9bdcaf25
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST=25d7623ec5d785ee12595322f5939afe4655ea8472ed14fc9b1af5cf2c915f83
export HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR="${SINGLE_DIR}"
export HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST=06ee3e38177b35bff7567c4f31a38261d9ac460da14f514eb41c27367f9a9e4a
export HOLOSOMA_EXTERNAL_AS_WORLD_SIZE=64
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
export WANDB_RUN_ID=rvxqm5y5
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
  --training.num-envs=4096
  --training.seed=42
  --training.multigpu=True
  --training.export-onnx=True
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
  --command.setup-terms.motion-command.params.motion-config.contact-aware-sparse-root-command-mode=tracking_error
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled=True
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-half-width-steps=50
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-density-boost=7.0
  --reward.terms.offline-contact-guidance.params.contact-export-root="${CONTACT_ROOT}"
  --reward.terms.offline-contact-guidance.params.contact-region-names='["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]'
  --reward.terms.offline-contact-guidance.params.wrist-region-names='["left_wrist","right_wrist"]'
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
  --logger.id=rvxqm5y5
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
    --expected-run-id rvxqm5y5 \
    --expected-run-name "${RUN_NAME}" \
    --expected-world-size 64
fi

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "[INFO] formal_student_pure_rl_depth_ws64_preflight_only_ok node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP}"
  exit 0
fi

cd "${SOURCE_ROOT}"
echo "[INFO] formal_student_pure_rl_depth_ws64_preflight_ok node_rank=${NODE_RANK} host=$(hostname) ip=${EXPECTED_NODE_IP} source=${SOURCE_ID} wandb=zihanw22/carry-any/rvxqm5y5 pure_ppo=true distill=false teacher_loaded=false depth_student=true fresh_policy=true"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr=10.99.0.24 \
  --nproc_per_node=8 \
  --max_restarts=0 \
  --master_port=31751 \
  src/holosoma/holosoma/train_agent_rank_visible.py \
  "${TRAIN_ARGS[@]}"
