#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${ROOT_DIR}"

if [[ -x /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11 ]]; then
  PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11}
else
  PYTHON_BIN=${PYTHON_BIN:-python3}
fi

RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}
DATA_DIR=${DATA_DIR:-"${ROOT_DIR}/data/ds_as_data/prism_debug30_convexhull_allmesh_solid_box_bin_barrel_ball"}
OBJECT_MAP=${OBJECT_MAP:-"${DATA_DIR}/_clip_object_urdf_map.json"}
EXPECTED_TOTAL=${EXPECTED_TOTAL:-30}
OUT_ROOT=${OUT_ROOT:-"${ROOT_DIR}/outputs/as_prism_debug30_inference_mp4_${RUN_STAMP}"}
PAIR_WORK_ROOT=${PAIR_WORK_ROOT:-"${ROOT_DIR}/data/ds_as_data/_as_prism_debug30_infer_pairs/${RUN_STAMP}"}
MANIFEST=${MANIFEST:-"${OUT_ROOT}/pairs_manifest.tsv"}
CHECKPOINTS_TSV=${CHECKPOINTS_TSV:-"${OUT_ROOT}/checkpoints.tsv"}
RESOLVED_CHECKPOINTS_TSV=${RESOLVED_CHECKPOINTS_TSV:-"${OUT_ROOT}/resolved_checkpoints.tsv"}
TASKS_TSV=${TASKS_TSV:-"${OUT_ROOT}/tasks.tsv"}
STATUS_TSV=${STATUS_TSV:-"${OUT_ROOT}/status.tsv"}
LOG_DIR=${LOG_DIR:-"${OUT_ROOT}/logs"}
MP4_DIR=${MP4_DIR:-"${OUT_ROOT}/mp4"}
PARALLEL_JOBS=${PARALLEL_JOBS:-4}
GPU_IDS=${GPU_IDS:-0,1,2,3}
TASK_TIMEOUT_S=${TASK_TIMEOUT_S:-900}
MAX_STEPS_MARGIN=${MAX_STEPS_MARGIN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}
VIDEO_FORMAT=${VIDEO_FORMAT:-mp4}
VIDEO_WIDTH=${VIDEO_WIDTH:-640}
VIDEO_HEIGHT=${VIDEO_HEIGHT:-360}
MIN_VIDEO_FRAMES=${MIN_VIDEO_FRAMES:-30}
VIDEO_MONITOR_INTERVAL_S=${VIDEO_MONITOR_INTERVAL_S:-5}
POST_VIDEO_GRACE_S=${POST_VIDEO_GRACE_S:-5}

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${MP4_DIR}" "${PAIR_WORK_ROOT}"

is_truthy() {
  case "$(echo "${1:-0}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

safe_name() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip()
value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
value = re.sub(r"_+", "_", value).strip("_.")
print(value or "item")
PY
}

write_default_checkpoints() {
  cat > "${CHECKPOINTS_TSV}" <<'EOF'
run_slug	run_name	checkpoint	note
corl79_resume11700_no_contact_guidance_64gpu	g1_w_object_distill_as_button_solid_corl79_resume11700_no_contact_guidance_64gpu	wandb://zihanw22/carry-any/jjdjpkoh/model_22500.pt	run_id=jjdjpkoh state=running latest_pt=2026-07-07T00:00Z_refresh
solid_ch51_64gpu_hybridloss_lr1e3_p001_new	g1_w_object_distill_as_button_solid_ch51_64gpu_hybridloss_lr1e3_p001_new	wandb://zihanw22/carry-any/9ez2ivr4/model_11700.pt	run_id=9ez2ivr4 state=crashed latest_pt
debug29_4node_rv_ppo001_h2048_nocg_20260627_2230	debug29_distill_as_button_model05000_4node_rv_ppo001_h2048_nocg_20260627_2230	wandb://zihanw22/carry-any/588q4jzw/model_16500.pt	run_id=588q4jzw state=crashed latest_pt
debug29_1node8_rv_ppo001_h2048_20260627_0645	debug29_distill_as_button_model05000_1node8_rv_ppo001_h2048_20260627_0645	wandb://zihanw22/carry-any/e2ccwodf/model_39999.pt	run_id=e2ccwodf state=finished latest_pt
debug29_4node_rv_ppo001_20260626_2225	debug29_distill_as_button_model05000_4node_rv_ppo001_20260626_2225	__SKIP__	run_id=o1z2q5cv has no model_*.pt file on W&B; latest upload is model_39999.onnx
solid79_normal_geom_init_as22000_48gpu	g1_w_object_distill_as_solid79_normal_geom_init_as22000_48gpu	wandb://zihanw22/carry-any/u7gflh72/model_39999.pt	run_id=u7gflh72 state=finished latest_pt
EOF
}

download_checkpoint() {
  local slug="$1"
  local checkpoint="$2"
  local cache_dir="${OUT_ROOT}/checkpoints/${slug}"
  mkdir -p "${cache_dir}"

  case "${checkpoint}" in
    wandb://*)
      if is_truthy "${DRY_RUN}"; then
        printf '%s\n' "${checkpoint}"
        return 0
      fi
      "${PYTHON_BIN}" - "${checkpoint}" "${cache_dir}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

import wandb

uri = sys.argv[1]
root = Path(sys.argv[2]).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)
match = re.fullmatch(r"wandb://([^/]+)/([^/]+)/([^/]+)/(.+)", uri)
if not match:
    raise SystemExit(f"Unsupported W&B checkpoint URI: {uri}")
entity, project, run_id, file_name = match.groups()
target = root / Path(file_name).name
if not target.is_file() or target.stat().st_size == 0:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    run.file(file_name).download(root=str(root), replace=True)
if not target.is_file() or target.stat().st_size == 0:
    raise SystemExit(f"Downloaded checkpoint is missing or empty: {target}")
print(target)
PY
      ;;
    /*|./*|../*)
      if [[ ! -f "${checkpoint}" ]]; then
        echo "[ERROR] Local checkpoint not found: ${checkpoint}" >&2
        return 2
      fi
      realpath -m "${checkpoint}"
      ;;
    *)
      echo "[ERROR] Unsupported checkpoint reference: ${checkpoint}" >&2
      return 2
      ;;
  esac
}

npz_steps() {
  "${PYTHON_BIN}" - "$1" "${MAX_STEPS_MARGIN}" <<'PY'
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
margin = int(sys.argv[2])
data = np.load(path)
for key in ("joint_pos", "body_pos_w", "object_pos_w", "root_pos_w"):
    if key in data and getattr(data[key], "ndim", 0) >= 1:
        print(int(data[key].shape[0]) + margin)
        break
else:
    lengths = [int(value.shape[0]) for value in data.values() if getattr(value, "ndim", 0) >= 1 and value.shape[0] > 8]
    if not lengths:
        raise SystemExit(f"Could not infer clip length from {path}")
    print(max(lengths) + margin)
PY
}

select_video() {
  "${PYTHON_BIN}" - "$1" "${MIN_VIDEO_FRAMES}" <<'PY'
from pathlib import Path
import sys

video_dir = Path(sys.argv[1])
min_frames = int(sys.argv[2])
videos = sorted(video_dir.glob("*.mp4"))
if not videos:
    raise SystemExit(0)

rows: list[tuple[int, int, Path]] = []
try:
    import cv2
except Exception:
    cv2 = None

for path in videos:
    frames = -1
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    rows.append((frames, int(path.stat().st_size), path))

eligible = [row for row in rows if row[0] >= min_frames]
if not eligible:
    raise SystemExit(0)
best = max(eligible, key=lambda row: (row[0], row[1], row[2].stat().st_mtime))
print(best[2])
PY
}

append_status() {
  local line="$1"
  {
    flock 9
    printf '%s\n' "${line}" >> "${STATUS_TSV}"
  } 9>"${STATUS_TSV}.lock"
}

prepare_inputs() {
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] DATA_DIR not found: ${DATA_DIR}" >&2
    return 2
  fi
  if [[ ! -f "${OBJECT_MAP}" ]]; then
    echo "[ERROR] OBJECT_MAP not found: ${OBJECT_MAP}" >&2
    return 2
  fi

  if [[ ! -f "${CHECKPOINTS_TSV}" ]]; then
    write_default_checkpoints
  fi

  local prepare_args=(
    --motion-dir "${DATA_DIR}"
    --object-map "${OBJECT_MAP}"
    --work-root "${PAIR_WORK_ROOT}/pairs"
    --video-root "${OUT_ROOT}/videos/by_clip_template"
    --manifest "${MANIFEST}"
    --expected-total "${EXPECTED_TOTAL}"
    --single-slot
    --force
  )
  if [[ -n "${LIMIT_CLIPS:-}" ]]; then
    prepare_args+=(--limit "${LIMIT_CLIPS}")
  fi
  if [[ -n "${CLIP_REGEX:-}" ]]; then
    prepare_args+=(--clip-regex "${CLIP_REGEX}")
  fi
  if [[ -n "${CLIP_LIST:-}" ]]; then
    prepare_args+=(--clip-list "${CLIP_LIST}")
  fi
  if [[ -n "${START_INDEX:-}" ]]; then
    prepare_args+=(--start-index "${START_INDEX}")
  fi

  "${PYTHON_BIN}" scripts/prepare_as_replay_pairs.py "${prepare_args[@]}"

  printf 'run_slug\trun_name\tcheckpoint\tresolved_checkpoint\tnote\n' > "${RESOLVED_CHECKPOINTS_TSV}"
  local run_count=0
  while IFS=$'\t' read -r run_slug run_name checkpoint note; do
    [[ "${run_slug}" == "run_slug" ]] && continue
    [[ -z "${run_slug}" ]] && continue
    if [[ -n "${LIMIT_RUNS:-}" && "${run_count}" -ge "${LIMIT_RUNS}" ]]; then
      continue
    fi
    run_count=$((run_count + 1))
    if [[ -z "${checkpoint}" || "${checkpoint}" == "__SKIP__" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "${run_slug}" "${run_name}" "${checkpoint}" "__SKIP__" "${note}" >> "${RESOLVED_CHECKPOINTS_TSV}"
      continue
    fi
    echo "[INFO] Resolving checkpoint for ${run_slug}: ${checkpoint}"
    local resolved
    resolved=$(download_checkpoint "${run_slug}" "${checkpoint}")
    printf '%s\t%s\t%s\t%s\t%s\n' "${run_slug}" "${run_name}" "${checkpoint}" "${resolved}" "${note}" >> "${RESOLVED_CHECKPOINTS_TSV}"
  done < "${CHECKPOINTS_TSV}"

  printf 'task_id\trun_slug\trun_name\tcheckpoint\tclip_id\tpair_dir\tpair_map\tsource_npz\tmax_steps\tvideo_dir\tmp4_path\tlog_path\tauto_forward_log\n' > "${TASKS_TSV}"
  local task_id=0
  while IFS=$'\t' read -r run_slug run_name checkpoint resolved note; do
    [[ "${run_slug}" == "run_slug" ]] && continue
    [[ -z "${run_slug}" ]] && continue
    if [[ -z "${resolved}" || "${resolved}" == "__SKIP__" ]]; then
      append_status "$(printf 'skipped\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "${run_slug}" "${run_name}" "" "" "" "" "" "" "${note}")"
      continue
    fi
    while IFS=$'\t' read -r clip_id pair_dir pair_map active_urdf source_npz template_video_dir; do
      [[ -z "${clip_id}" ]] && continue
      local max_steps
      max_steps=$(npz_steps "${source_npz}")
      local clip_safe
      clip_safe=$(safe_name "${clip_id}")
      local video_dir="${OUT_ROOT}/videos/${run_slug}/${clip_safe}"
      local log_path="${LOG_DIR}/${run_slug}__${clip_safe}.log"
      local auto_log="${LOG_DIR}/${run_slug}__${clip_safe}.auto_forward.jsonl"
      local mp4_path="${MP4_DIR}/${run_slug}__${clip_safe}.mp4"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${task_id}" "${run_slug}" "${run_name}" "${resolved}" "${clip_id}" "${pair_dir}" "${pair_map}" \
        "${source_npz}" "${max_steps}" "${video_dir}" "${mp4_path}" "${log_path}" "${auto_log}" >> "${TASKS_TSV}"
      task_id=$((task_id + 1))
    done < "${MANIFEST}"
  done < "${RESOLVED_CHECKPOINTS_TSV}"
}

run_task() {
  local task_id="$1"
  local run_slug="$2"
  local run_name="$3"
  local checkpoint="$4"
  local clip_id="$5"
  local pair_dir="$6"
  local pair_map="$7"
  local source_npz="$8"
  local max_steps="$9"
  local video_dir="${10}"
  local mp4_path="${11}"
  local log_path="${12}"
  local auto_log="${13}"
  local gpu_csv="${GPU_IDS}"
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  local gpu_count="${#gpu_ids[@]}"
  local gpu="${gpu_ids[$((task_id % gpu_count))]}"

  if is_truthy "${SKIP_EXISTING}" && [[ -s "${mp4_path}" ]]; then
    append_status "$(printf 'exists\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "${run_slug}" "${run_name}" "${clip_id}" "${checkpoint}" "${gpu}" "${max_steps}" "${mp4_path}" "${log_path}" "already_present")"
    return 0
  fi

  mkdir -p "${video_dir}" "$(dirname "${mp4_path}")" "$(dirname "${log_path}")"
  rm -f "${video_dir}"/*.mp4 "${mp4_path}"

  local cmd=(
    bash ./infer_as_joystick.sh "${checkpoint}"
    --training.max-eval-steps "${max_steps}"
    --command.setup-terms.motion-command.params.motion-config.motion-clip-name "${clip_id}"
    --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob 0.0
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale 0.0
    logger:disabled
    --logger.video.enabled True
    --logger.headless-recording True
    --logger.video.upload-to-wandb False
    --logger.video.interval 1
    --logger.video.save-dir "${video_dir}"
    --logger.video.width "${VIDEO_WIDTH}"
    --logger.video.height "${VIDEO_HEIGHT}"
    --logger.video.output-format "${VIDEO_FORMAT}"
    --logger.video.playback-rate 1.0
    --logger.video.camera-smoothing 0.90
    --logger.video.show-command-overlay False
    --logger.video.record-env-id 0
  )

  if is_truthy "${DRY_RUN}"; then
    printf '[DRY_RUN] task=%s gpu=%s clip=%s run=%s ' "${task_id}" "${gpu}" "${clip_id}" "${run_slug}" | tee -a "${log_path}"
    printf 'timeout %q ' "${TASK_TIMEOUT_S}" | tee -a "${log_path}"
    printf '%q ' "${cmd[@]}" | tee -a "${log_path}"
    printf '\n' | tee -a "${log_path}"
    append_status "$(printf 'dry_run\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "${run_slug}" "${run_name}" "${clip_id}" "${checkpoint}" "${gpu}" "${max_steps}" "${mp4_path}" "${log_path}" "dry_run")"
    return 0
  fi

  set +e
  (
    set -euo pipefail
    export PYTHON_BIN
    export OMOMO_DATA_DIR="${pair_dir}"
    export OMOMO_OBJECT_MAP="${pair_map}"
    export OMOMO_EXPECTED_TOTAL=1
    export AS_AUTO_FORWARD_AFTER_LIFT="${AS_AUTO_FORWARD_AFTER_LIFT:-1}"
    export AS_AUTO_FORWARD_AFTER_LIFT_COMMAND="${AS_AUTO_FORWARD_AFTER_LIFT_COMMAND:-0.11,0,0}"
    export AS_AUTO_FORWARD_AFTER_LIFT_DURATION_S="${AS_AUTO_FORWARD_AFTER_LIFT_DURATION_S:-0}"
    export AS_AUTO_FORWARD_AFTER_LIFT_REL_Z_DELTA="${AS_AUTO_FORWARD_AFTER_LIFT_REL_Z_DELTA:-0.10}"
    export AS_AUTO_FORWARD_AFTER_LIFT_CONSECUTIVE_STEPS="${AS_AUTO_FORWARD_AFTER_LIFT_CONSECUTIVE_STEPS:-5}"
    export VISER_AUTO_FORWARD_AFTER_LIFT_LOG_PATH="${auto_log}"
    export VISER_ENABLE_CLIP_GUI=0
    export VISER_ENABLE_MANUAL_GUI=0
    export VISER_FORCE_MANUAL_CONTROL=1
    export VISER_MANUAL_CONTROL_DEFAULT=1
    export VISER_MANUAL_COMMAND_DEFAULT=0,0,0
    export DEPTH_PERCEPTION_PRESET=checkpoint
    export HOLOSOMA_RESET_TO_DEFAULT_POSE=0
    export HEADLESS=True
    export NUM_ENVS=1
    export OBJECT_SPAWN_MODE=urdf
    export OBJECT_GEOMETRY_MODE=mesh
    export HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition
    export HOLOSOMA_DISABLE_AUTO_RESET=1
    export HOLOSOMA_DISABLE_MOTION_END_RESET=1
    export HOLOSOMA_DISABLE_CLIP_END_RESET=1
    export HOLOSOMA_DEVICE="cuda:${gpu}"
    export OMNI_KIT_ACCEPT_EULA=YES
    export ACCEPT_EULA=Y
    exec setsid "${cmd[@]}"
  ) > "${log_path}" 2>&1 &
  local runner_pid=$!
  local rc=0
  local monitor_video=""
  local deadline=$((SECONDS + TASK_TIMEOUT_S))
  while kill -0 "${runner_pid}" 2>/dev/null; do
    monitor_video=$(select_video "${video_dir}" || true)
    if [[ -n "${monitor_video}" ]]; then
      sleep "${POST_VIDEO_GRACE_S}"
      if kill -0 "${runner_pid}" 2>/dev/null; then
        kill -TERM "-${runner_pid}" 2>/dev/null || kill -TERM "${runner_pid}" 2>/dev/null || true
        sleep 2
        kill -KILL "-${runner_pid}" 2>/dev/null || kill -KILL "${runner_pid}" 2>/dev/null || true
      fi
      wait "${runner_pid}"
      rc=$?
      break
    fi
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      kill -TERM "-${runner_pid}" 2>/dev/null || kill -TERM "${runner_pid}" 2>/dev/null || true
      sleep 2
      kill -KILL "-${runner_pid}" 2>/dev/null || kill -KILL "${runner_pid}" 2>/dev/null || true
      wait "${runner_pid}"
      rc=124
      break
    fi
    sleep "${VIDEO_MONITOR_INTERVAL_S}"
  done
  if ! kill -0 "${runner_pid}" 2>/dev/null && [[ "${rc}" -eq 0 ]]; then
    wait "${runner_pid}"
    rc=$?
  fi
  set -e

  local produced=""
  produced=$(select_video "${video_dir}" || true)
  if [[ -n "${produced}" ]]; then
    cp -f "${produced}" "${mp4_path}"
    local ok_status="ok"
    if [[ "${rc}" -ne 0 ]]; then
      ok_status="ok_after_exit_${rc}"
    fi
    append_status "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "${ok_status}" "${run_slug}" "${run_name}" "${clip_id}" "${checkpoint}" "${gpu}" "${max_steps}" "${mp4_path}" "${log_path}" "source=${produced}")"
    return 0
  fi

  local reason="exit=${rc}"
  if [[ "${rc}" -eq 0 && -z "${produced}" ]]; then
    reason="missing_video_min_frames_${MIN_VIDEO_FRAMES}"
  fi
  append_status "$(printf 'failed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "${run_slug}" "${run_name}" "${clip_id}" "${checkpoint}" "${gpu}" "${max_steps}" "${mp4_path}" "${log_path}" "${reason}")"
  return "${rc}"
}

run_all_tasks() {
  printf 'status\trun_slug\trun_name\tclip_id\tcheckpoint\tgpu\tmax_steps\tmp4_path\tlog_path\tnote\n' > "${STATUS_TSV}"
  prepare_inputs

  local total_tasks
  total_tasks=$(( $(wc -l < "${TASKS_TSV}") - 1 ))
  echo "[INFO] Output root: ${OUT_ROOT}"
  echo "[INFO] Pair work root: ${PAIR_WORK_ROOT}"
  echo "[INFO] Tasks: ${total_tasks}; parallel=${PARALLEL_JOBS}; gpu_ids=${GPU_IDS}; timeout_s=${TASK_TIMEOUT_S}"

  if [[ "${total_tasks}" -le 0 ]]; then
    echo "[WARN] No runnable tasks were generated."
    return 0
  fi

  local running=0
  local failures=0
  local task_id run_slug run_name checkpoint clip_id pair_dir pair_map source_npz max_steps video_dir mp4_path log_path auto_log
  while IFS=$'\t' read -r task_id run_slug run_name checkpoint clip_id pair_dir pair_map source_npz max_steps video_dir mp4_path log_path auto_log; do
    [[ "${task_id}" == "task_id" ]] && continue
    while [[ "${running}" -ge "${PARALLEL_JOBS}" ]]; do
      if ! wait -n; then
        failures=$((failures + 1))
      fi
      running=$((running - 1))
    done
    run_task "${task_id}" "${run_slug}" "${run_name}" "${checkpoint}" "${clip_id}" "${pair_dir}" "${pair_map}" "${source_npz}" "${max_steps}" "${video_dir}" "${mp4_path}" "${log_path}" "${auto_log}" &
    running=$((running + 1))
  done < "${TASKS_TSV}"

  while [[ "${running}" -gt 0 ]]; do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    running=$((running - 1))
  done

  echo "[INFO] Finished. Status: ${STATUS_TSV}"
  echo "[INFO] MP4 directory: ${MP4_DIR}"
  return "${failures}"
}

run_all_tasks
