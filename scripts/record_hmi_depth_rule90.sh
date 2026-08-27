#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <expected-commit> <motion-view> <object-map> <output-root> <canonical-clip-id>" >&2
  exit 2
fi

SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXPECTED_COMMIT=$1
MOTION_VIEW=$2
OBJECT_MAP=$3
OUTPUT_ROOT=$4
CANONICAL_CLIP_ID=$5

[[ ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]] || { echo "[ERROR] invalid commit" >&2; exit 2; }
[[ $(git -C "${SOURCE_ROOT}" rev-parse HEAD) == "${EXPECTED_COMMIT}" ]] || {
  echo "[ERROR] Rule-90 source commit mismatch" >&2
  exit 2
}
git -C "${SOURCE_ROOT}" diff --quiet --ignore-submodules --
git -C "${SOURCE_ROOT}" diff --cached --quiet --ignore-submodules --
[[ -d ${MOTION_VIEW} && ! -L ${MOTION_VIEW} ]] || { echo "[ERROR] invalid motion view" >&2; exit 2; }
[[ -f ${OBJECT_MAP} && ! -L ${OBJECT_MAP} ]] || { echo "[ERROR] invalid object map" >&2; exit 2; }

# shellcheck disable=SC1091
source "${SOURCE_ROOT}/scripts/source_isaacsim_setup.sh"
export PYTHONPATH="${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HEADLESS=1 OMP_NUM_THREADS=1

PAIR_ROOT=${OUTPUT_ROOT}/single_pair_banks
VIDEO_ROOT=${OUTPUT_ROOT}/videos
LOG_ROOT=${OUTPUT_ROOT}/logs
PAIR_MANIFEST=${OUTPUT_ROOT}/pair_manifest.tsv
mkdir -p "${PAIR_ROOT}" "${VIDEO_ROOT}" "${LOG_ROOT}"

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/prepare_as_replay_pairs.py" \
  --motion-dir "${MOTION_VIEW}" \
  --object-map "${OBJECT_MAP}" \
  --work-root "${PAIR_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --manifest "${PAIR_MANIFEST}" \
  --expected-total 79 \
  --start-index 0 \
  --limit 1 \
  --single-slot \
  --force

IFS=$'\t' read -r CLIP_ID PAIR_DIR PAIR_MAP _OBJECT_URDF SOURCE_NPZ VIDEO_DIR < "${PAIR_MANIFEST}"
[[ ${CLIP_ID} == "${CANONICAL_CLIP_ID}" ]] || {
  echo "[ERROR] canonical first clip mismatch: expected=${CANONICAL_CLIP_ID} actual=${CLIP_ID}" >&2
  exit 2
}
[[ ${SOURCE_NPZ##*/} == "${CANONICAL_CLIP_ID}.npz" ]] || {
  echo "[ERROR] canonical source NPZ mismatch" >&2
  exit 2
}

mapfile -t BEFORE_VIDEOS < <(find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' -print | sort)
(( ${#BEFORE_VIDEOS[@]} == 0 )) || {
  echo "[ERROR] Rule-90 output directory is not fresh" >&2
  exit 2
}

env \
  CUDA_VISIBLE_DEVICES=0 \
  LOCAL_RANK=0 \
  LOCAL_WORLD_SIZE=1 \
  HOLOSOMA_ORIGINAL_LOCAL_RANK=90 \
  HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE=1 \
  PYTHONHASHSEED=42 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf \
  HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition \
  HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 \
  HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh \
  HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=1 \
  HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0 \
  HOLOSOMA_REPLAY_KEEP_OPEN=0 \
  VISER_ENABLE_CLIP_GUI=0 \
  VISER_ENABLE_MANUAL_GUI=0 \
  VISER_MANUAL_USE_HW_JOYSTICK=0 \
  timeout --signal=TERM --kill-after=15s 300s \
  "${PYTHON_BIN}" "${SOURCE_ROOT}/src/holosoma/holosoma/replay.py" \
    exp:g1-29dof-wbt-w-object-hmi-depth-stage1 \
    logger:disabled \
    --training.headless=True \
    --training.num-envs=1 \
    --command.setup-terms.motion-command.params.motion-config.motion-file="${PAIR_DIR}" \
    --command.setup-terms.motion-command.params.motion-config.motion-clip-name="${CLIP_ID}" \
    --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler=False \
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=1.0 \
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.0 \
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale=0.0 \
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=False \
    --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0.0 \
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=False \
    --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=0.0 \
    --robot.object.enabled=True \
    --robot.object.object-urdf-path="${PAIR_MAP}" \
    --logger.video.enabled=True \
    --logger.headless-recording=True \
    --logger.video.upload-to-wandb=False \
    --logger.video.save-dir="${VIDEO_DIR}" \
    --logger.video.width=1280 \
    --logger.video.height=720 \
    --logger.video.output-format=h264 \
    --logger.video.playback-rate=1.0 \
    --logger.video.camera-smoothing=0.90 \
    --logger.video.show-command-overlay=False \
    randomization:disabled \
  >"${LOG_ROOT}/rule90_${CLIP_ID}.log" 2>&1

mapfile -t VIDEOS < <(find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' -size +0c -print | sort)
(( ${#VIDEOS[@]} == 1 )) || {
  echo "[ERROR] expected exactly one fresh non-empty Rule-90 MP4, found ${#VIDEOS[@]}" >&2
  exit 2
}
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration \
  -of json "${VIDEOS[0]}" > "${OUTPUT_ROOT}/ffprobe.json"
printf '%s\n' "${VIDEOS[0]}" > "${OUTPUT_ROOT}/video_path.txt"
sha256sum "${VIDEOS[0]}" > "${OUTPUT_ROOT}/video.sha256"
echo "[INFO] Rule-90 recording ready: ${VIDEOS[0]}"
