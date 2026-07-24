#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! $1 =~ ^(ball|bin|barrel)$ || ! $2 =~ ^[0-7]$ ]]; then
  echo "usage: $0 ROLE GPU_INDEX MOTION_CLIP" >&2
  exit 2
fi

readonly ROLE="$1"
readonly GPU_INDEX="$2"
readonly MOTION_CLIP="$3"
readonly RUN_ROOT=/home/ubuntu/FAR/holosoma_runs/formal_teacher8k_eval3_20260720
readonly SOURCE_ID=src-6a871a6c74d045b8ff1686002f6dcc8eacae438022151013e6630b3227a28eca
readonly SOURCE_DIGEST=6a871a6c74d045b8ff1686002f6dcc8eacae438022151013e6630b3227a28eca
readonly SOURCE_ROOT="${RUN_ROOT}/src_${ROLE}"
readonly PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
readonly PYTHON_RUNTIME=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f/site-packages
readonly CHECKPOINT="${RUN_ROOT}/checkpoint/model_08000.pt"
readonly CHECKPOINT_SHA256=a6093a6fbfb84932517002323fab735aff4759214d3b56acd65e8db934929124
readonly RECORD_SCRIPT="${RECORD_SCRIPT:-scripts/record_checkpoint_inference.py}"
case "${ROLE}" in
  ball)
    readonly BANK=/home/ubuntu/FAR/holosoma/data/ds_as_data/prism_same_teacher_r30_ball29__src_7e46823ba0cf87a3d2b9bba20c717d589854a96dd0214dc0fa5fb71e8825ee7e/_single_slot_motion_bank/by-source/fa4df7d3163ba36db2634099afb2c11d4b70465c894be6b1f47e997376da4148
    readonly EXPECTED_MOTION_SHA256=87644e984e4af1e7b75f3e3f83d822a0ee18e2b7604968d38f16d5b80cae46bb
    readonly EXPECTED_MAP_SHA256=30fb9d3f81ceceef22fc6aa5df83778aac34ba65d2d653c13584fd7feb5aa92d
    ;;
  bin)
    readonly BANK=/home/ubuntu/FAR/holosoma/data/ds_as_data/prism_same_teacher_r30_bin29__src_3a47e17abfc6efb0609bd2ceae72c2439a7ef0bf2ddb70066a80f2e9e728f662/_single_slot_motion_bank/by-source/19a086fa493933ddd7faf75a5b1e5a3833db3ef64c9e356d434b0d6ae1fc553f
    readonly EXPECTED_MOTION_SHA256=9c981f20edb97a9d598fee7beb15c42278ad4bb0bc1725812540fd243f35adb4
    readonly EXPECTED_MAP_SHA256=b8cb4c033916af0b598f5966f026f54fb6ab54ec79005ddbef2338f1da2aec5f
    ;;
  barrel)
    readonly BANK=/home/ubuntu/FAR/holosoma/data/ds_as_data/prism_same_teacher_r30_barrel25__src_79ac8251e8e1f76b117348d993dcebef31630784a7e645fe321746e801e7bc2f/_single_slot_motion_bank/by-source/c15a17642a04cffd7a4c61683256358c363d27473cf6c22146743f86b4689eba
    readonly EXPECTED_MOTION_SHA256=382e0aaffc8a6e4dd4c1906eaed50c5ed3e244bdd3e769e5581f374e60f06126
    readonly EXPECTED_MAP_SHA256=9e7bf562367cc32802c251d843c6d79218be9dcd23626bbedc9df5482718cdd3
    ;;
esac

cd "${SOURCE_ROOT}"
sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]]
[[ -f "${BANK}/${MOTION_CLIP}.npz" ]]
[[ "$(find "${BANK}" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 1 ]]
[[ "$(sha256sum "${BANK}/${MOTION_CLIP}.npz" | awk '{print $1}')" == "${EXPECTED_MOTION_SHA256}" ]]
[[ "$(sha256sum "${BANK}/_clip_object_urdf_map.json" | awk '{print $1}')" == "${EXPECTED_MAP_SHA256}" ]]

mkdir -p \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/metrics" \
  "${RUN_ROOT}/videos/raw/${ROLE}" \
  "${RUN_ROOT}/eval_logs/${ROLE}" \
  "${RUN_ROOT}/cache/${ROLE}/torch_extensions" \
  "${RUN_ROOT}/cache/${ROLE}/warp" \
  "${RUN_ROOT}/cache/${ROLE}/ov"

export HOME=/home/ubuntu
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TORCH_EXTENSIONS_DIR="${RUN_ROOT}/cache/${ROLE}/torch_extensions"
export WARP_CACHE_PATH="${RUN_ROOT}/cache/${ROLE}/warp"
export OMNI_USER_CACHE_PATH="${RUN_ROOT}/cache/${ROLE}/ov"

export HOLOSOMA_SOURCE_ROOT="${SOURCE_ROOT}"
export HOLOSOMA_SOURCE_SNAPSHOT_ID="${SOURCE_ID}"
export HOLOSOMA_SOURCE_MANIFEST_SHA256="${SOURCE_DIGEST}"
export HOLOSOMA_EVAL_POLICY=checkpoint_actor
export HOLOSOMA_DISABLE_AUTO_RESET=1
export HOLOSOMA_DISABLE_CLIP_END_RESET=1
export HOLOSOMA_RESET_TO_DEFAULT_POSE=0
# Evaluation clones may be reused after a diagnostic launch.  Force IsaacLab to
# rebuild the clone-local URDF conversion output so a partial USD can never be
# mistaken for a valid cache entry on the next launch.
export HOLOSOMA_CLEAN_ROBOT_USD_CACHE=1
export HOLOSOMA_EVAL_DEBUG_PATH="${RUN_ROOT}/metrics/${ROLE}_policy_io.jsonl"
export HOLOSOMA_EVAL_DEBUG_LIMIT=12

echo "[INFO] teacher8k_eval_preflight_ok role=${ROLE} gpu=${GPU_INDEX} clip=${MOTION_CLIP} checkpoint_sha256=${CHECKPOINT_SHA256} source=${SOURCE_ID}"
"${PYTHON_BIN}" "${RECORD_SCRIPT}" \
  --checkpoint="${CHECKPOINT}" \
  --steps=501 \
  --video-episode-id=8000 \
  --metrics-path="${RUN_ROOT}/metrics/${ROLE}.jsonl" \
  --training.headless=False \
  --training.num-envs=1 \
  --training.seed=42 \
  --simulator.config.sim.max-episode-length-s=100000.0 \
  --command.setup-terms.motion-command.params.motion-config.motion-file="${BANK}" \
  --robot.object.object-urdf-path="${BANK}/_clip_object_urdf_map.json" \
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=False \
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled=False \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=1.0 \
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.0 \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale=0.0 \
  --logger.base-dir="${RUN_ROOT}/eval_logs/${ROLE}" \
  --logger.video.enabled=True \
  --logger.video.interval=1 \
  --logger.video.width=640 \
  --logger.video.height=360 \
  --logger.video.playback-rate=1.0 \
  --logger.video.output-format=h264 \
  --logger.video.save-dir="${RUN_ROOT}/videos/raw/${ROLE}" \
  --logger.video.upload-to-wandb=False \
  --logger.video.record-env-id=0 \
  randomization:disabled

cd "${SOURCE_ROOT}"
sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256
echo "[INFO] teacher8k_eval_complete role=${ROLE} clip=${MOTION_CLIP} source_manifest_postcheck=ok"
