#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU START_INDEX LIMIT SHARD_ID" >&2
  exit 2
fi

readonly GPU_INDEX=$1
readonly START_INDEX=$2
readonly CLIP_LIMIT=$3
readonly SHARD_ID=$4
readonly SOURCE_ROOT=${REFERENCE_SOURCE_ROOT:-/home/ubuntu/FAR/holosoma}
readonly RUN_ROOT=${REFERENCE_RUN_ROOT:-/home/ubuntu/FAR/holosoma/outputs/prism_debug30_reference_replays_wandb_20260722_182522}
readonly MOTION_VIEW=${REFERENCE_MOTION_VIEW:-/home/ubuntu/FAR/holosoma/data/ds_as_data/prism_debug30_convexhull_allmesh_solid_box_bin_barrel_ball/_scientific_teacher64_single_slot/by-source/77738917deb60e578dc695841b3a07b10ad4f50371d3c0500474f41c78f71f90}
readonly OBJECT_MAP=${REFERENCE_OBJECT_MAP:-${MOTION_VIEW}/_clip_object_urdf_map.json}
readonly EXPECTED_TOTAL=${REFERENCE_EXPECTED_TOTAL:-30}
readonly SHARD_ROOT="${RUN_ROOT}/shard_${SHARD_ID}"
readonly PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python
readonly FFPROBE_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/ffprobe
readonly CACHE_RANK_BASE=${REFERENCE_CACHE_RANK_BASE:-100}
readonly CACHE_RANK=$((CACHE_RANK_BASE + SHARD_ID))

mkdir -p "${SHARD_ROOT}/logs"

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/prepare_as_replay_pairs.py" \
  --motion-dir "${MOTION_VIEW}" \
  --object-map "${OBJECT_MAP}" \
  --work-root "${SHARD_ROOT}/single_pair_banks" \
  --video-root "${SHARD_ROOT}/videos" \
  --manifest "${SHARD_ROOT}/manifest.tsv" \
  --expected-total "${EXPECTED_TOTAL}" \
  --start-index "${START_INDEX}" \
  --limit "${CLIP_LIMIT}" \
  --single-slot \
  --force

validate_video() {
  local video_path=$1
  local probe
  [[ ! -L "${video_path}" && -s "${video_path}" ]] || return 1
  probe=$("${FFPROBE_BIN}" \
    -v error \
    -count_frames \
    -select_streams v:0 \
    -show_entries stream=codec_type,width,height,nb_read_frames \
    -of csv=p=0 \
    "${video_path}" 2>/dev/null) || return 1
  [[ "${probe}" =~ ^video,[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$ ]]
}

terminate_replay() {
  local replay_pid=$1
  if ! kill -0 "${replay_pid}" 2>/dev/null; then
    return 0
  fi
  kill -TERM -- "-${replay_pid}" 2>/dev/null || kill -TERM "${replay_pid}" 2>/dev/null || true
  for _ in $(seq 1 15); do
    kill -0 "${replay_pid}" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-${replay_pid}" 2>/dev/null || kill -KILL "${replay_pid}" 2>/dev/null || true
}

record_one() {
  local clip_id=$1
  local pair_dir=$2
  local pair_map=$3
  local video_dir=$4
  local safe_clip=${clip_id//[^A-Za-z0-9_.-]/_}
  local log_path="${SHARD_ROOT}/logs/${safe_clip}.log"
  local replay_pid deadline candidate video_path='' replay_status=0
  local -a existing=()

  mapfile -t existing < <(find -H "${video_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.mp4' -print | sort)
  if [[ ${#existing[@]} -eq 1 ]] && validate_video "${existing[0]}"; then
    echo "[SKIP] clip=${clip_id} valid_video=${existing[0]}"
    return 0
  fi
  if [[ ${#existing[@]} -ne 0 ]]; then
    echo "[ERROR] clip=${clip_id} has unexpected pre-existing MP4 count=${#existing[@]}" >&2
    return 1
  fi

  echo "[START] clip=${clip_id} gpu=${GPU_INDEX}"
  setsid env \
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" \
    PATH="/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin:${PATH}" \
    LOCAL_RANK=0 \
    LOCAL_WORLD_SIZE=1 \
    HOLOSOMA_ORIGINAL_LOCAL_RANK="${CACHE_RANK}" \
    HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE=1 \
    PYTHONHASHSEED=42 \
    OMNI_KIT_ACCEPT_EULA=YES \
    HEADLESS=1 \
    HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf \
    HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition \
    HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 \
    HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh \
    HOLOSOMA_REPLAY_KEEP_OPEN=0 \
    VISER_ENABLE_CLIP_GUI=0 \
    VISER_ENABLE_MANUAL_GUI=0 \
    VISER_MANUAL_USE_HW_JOYSTICK=0 \
    "${PYTHON_BIN}" "${SOURCE_ROOT}/src/holosoma/holosoma/replay.py" \
      exp:g1-29dof-wbt-w-object-generalist \
      command:g1-29dof-wbt-w-object-generalist \
      logger:disabled \
      --training.headless=True \
      --training.num-envs=1 \
      --command.setup-terms.motion-command.params.motion-config.motion-file "${pair_dir}" \
      --command.setup-terms.motion-command.params.motion-config.motion-clip-name "${clip_id}" \
      --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=False \
      --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0 \
      --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob 0.0 \
      --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale 0.0 \
      --robot.object.enabled=True \
      --robot.object.object-urdf-path "${pair_map}" \
      --logger.video.enabled=True \
      --logger.headless_recording=True \
      --logger.video.upload_to_wandb=False \
      --logger.video.save_dir "${video_dir}" \
      --logger.video.width 1280 \
      --logger.video.height 720 \
      --logger.video.output_format h264 \
      --logger.video.playback_rate 1.0 \
      --logger.video.camera_smoothing 0.90 \
      --logger.video.show_command_overlay=False \
      --simulator.config.sim.physx.gpu-max-rigid-contact-count=33554432 \
      --simulator.config.sim.physx.gpu-max-rigid-patch-count=4194304 \
      --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=134217728 \
      --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=134217728 \
      --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=16777216 \
      --simulator.config.sim.physx.gpu-collision-stack-size=67108864 \
      --simulator.config.sim.physx.gpu-heap-capacity=67108864 \
      --simulator.config.sim.physx.gpu-temp-buffer-capacity=16777216 \
      randomization:disabled \
      </dev/null \
      >"${log_path}" 2>&1 &
  replay_pid=$!
  deadline=$((SECONDS + 300))

  while (( SECONDS < deadline )); do
    while IFS= read -r candidate; do
      if grep -Fq 'Successfully saved video file:' "${log_path}" && validate_video "${candidate}"; then
        video_path=${candidate}
        break 2
      fi
    done < <(find -H "${video_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.mp4' -print | sort)
    replay_state=$(ps -o stat= -p "${replay_pid}" 2>/dev/null | tr -d '[:space:]' || true)
    if ! kill -0 "${replay_pid}" 2>/dev/null || [[ "${replay_state}" == Z* ]]; then
      break
    fi
    sleep 2
  done

  if [[ -z "${video_path}" ]]; then
    while IFS= read -r candidate; do
      if validate_video "${candidate}"; then
        video_path=${candidate}
        break
      fi
    done < <(find -H "${video_dir}" -mindepth 1 -maxdepth 1 -type f -name '*.mp4' -print | sort)
  fi

  terminate_replay "${replay_pid}"
  set +e
  wait "${replay_pid}"
  replay_status=$?
  set -e

  if [[ -z "${video_path}" ]] || ! validate_video "${video_path}"; then
    echo "[ERROR] clip=${clip_id} failed status=${replay_status} log=${log_path}" >&2
    tail -n 80 "${log_path}" >&2 || true
    return 1
  fi

  echo "[DONE] clip=${clip_id} gpu=${GPU_INDEX} video=${video_path}"
}

while IFS=$'\t' read -r clip_id pair_dir pair_map _object_urdf _source_npz video_dir; do
  source_clip=${_source_npz##*/}
  source_clip=${source_clip%.npz}
  if [[ "${clip_id}" != "${source_clip}" ]]; then
    echo "[ERROR] manifest stdin corruption detected: clip=${clip_id} source_clip=${source_clip}" >&2
    exit 1
  fi
  record_one "${clip_id}" "${pair_dir}" "${pair_map}" "${video_dir}"
done < "${SHARD_ROOT}/manifest.tsv"

touch "${SHARD_ROOT}/COMPLETE"
echo "[SHARD_DONE] shard=${SHARD_ID} start=${START_INDEX} count=${CLIP_LIMIT} gpu=${GPU_INDEX}"
