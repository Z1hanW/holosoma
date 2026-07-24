#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ID=src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399
readonly SOURCE_ROOT="/home/ubuntu/FAR/holosoma_runs/${SOURCE_ID}"
readonly PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11
readonly PYTHON_RUNTIME=/home/ubuntu/FAR/holosoma_runs/.runtime/python/python-runtime-v2-2dcddba9d88fd6b5a3fe701e2b74c59eae988071e52d8bdb44a2474bc16b9e1f/site-packages
readonly BANK_REL=data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball
readonly BANK="${SOURCE_ROOT}/${BANK_REL}"
readonly MOTION_VIEW="${BANK}/_scientific_corl79_single_slot/by-source/6209b4742cce3b2989c7ea1f96a55a27d57bcf91eeb90699d409747187ca2cca"
readonly OUTPUT_ROOT=/home/ubuntu/FAR/holosoma/outputs/formal_student_pure_rl_depth_corl79_ws64_e1024_no_contact_reward_20260722_212804/replay/capture
readonly PAIR_ROOT="${OUTPUT_ROOT}/single_pair_banks"
readonly VIDEO_ROOT="${OUTPUT_ROOT}/videos"
readonly MANIFEST="${OUTPUT_ROOT}/manifest.tsv"
readonly LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

export HOME=/home/ubuntu
export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHON_RUNTIME}:${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/prepare_as_replay_pairs.py" \
  --motion-dir "${MOTION_VIEW}" \
  --object-map "${MOTION_VIEW}/_clip_object_urdf_map.json" \
  --work-root "${PAIR_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --manifest "${MANIFEST}" \
  --expected-total 79 \
  --start-index 0 \
  --limit 1 \
  --single-slot \
  --force

IFS=$'\t' read -r clip_id pair_dir pair_map _object_urdf source_npz video_dir < "${MANIFEST}"
[[ "${clip_id}" == box_10 ]]
[[ "${source_npz##*/}" == box_10.npz ]]

env \
  CUDA_VISIBLE_DEVICES=0 \
  LOCAL_RANK=0 \
  LOCAL_WORLD_SIZE=1 \
  HOLOSOMA_ORIGINAL_LOCAL_RANK=90 \
  HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE=1 \
  PYTHONHASHSEED=42 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  OMNI_KIT_ACCEPT_EULA=YES \
  ACCEPT_EULA=Y \
  HEADLESS=1 \
  HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf \
  HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition \
  HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0 \
  HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=mesh \
  HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0 \
  HOLOSOMA_REPLAY_KEEP_OPEN=0 \
  VISER_ENABLE_CLIP_GUI=0 \
  VISER_ENABLE_MANUAL_GUI=0 \
  VISER_MANUAL_USE_HW_JOYSTICK=0 \
  timeout --signal=TERM --kill-after=15s 300s \
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
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=True \
    --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0.2 \
    --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=True \
    --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=2.0 \
    --robot.object.enabled=True \
    --robot.object.object-urdf-path "${pair_map}" \
    --logger.video.enabled=True \
    --logger.headless-recording=True \
    --logger.video.upload-to-wandb=False \
    --logger.video.save_dir "${video_dir}" \
    --logger.video.width 1280 \
    --logger.video.height 720 \
    --logger.video.output_format h264 \
    --logger.video.playback_rate 1.0 \
    --logger.video.camera_smoothing 0.90 \
    --logger.video.show-command-overlay=False \
    --simulator.config.sim.physx.gpu-max-rigid-contact-count=33554432 \
    --simulator.config.sim.physx.gpu-max-rigid-patch-count=4194304 \
    --simulator.config.sim.physx.gpu-found-lost-pairs-capacity=134217728 \
    --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=134217728 \
    --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity=16777216 \
    --simulator.config.sim.physx.gpu-collision-stack-size=67108864 \
    --simulator.config.sim.physx.gpu-heap-capacity=67108864 \
    --simulator.config.sim.physx.gpu-temp-buffer-capacity=16777216 \
    randomization:disabled \
  >"${LOG_ROOT}/1_box_10.log" 2>&1

mapfile -t videos < <(find "${video_dir}" -maxdepth 1 -type f -name '*.mp4' -print)
[[ ${#videos[@]} -eq 1 ]]
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration \
  -of json "${videos[0]}"
