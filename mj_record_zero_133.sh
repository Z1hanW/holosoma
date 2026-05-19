#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BANK_DIR="${HOLOSOMA_ZERO_RECORD_BANK_DIR:-/home/user/FAR/data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5}"
CLIP_MAP="${BANK_DIR}/_single_slot_motion_bank/_clip_object_urdf_map.json"
RUN_REF="${HOLOSOMA_ZERO_RECORD_RUN:-ppo_first_contact_aware_h1}"
WANDB_CHECKPOINT="wandb://zihanw22/carry-any/zxz3hd8h/model_07500.onnx"
LOCAL_CHECKPOINT="${HOLOSOMA_ZERO_RECORD_LOCAL_CHECKPOINT:-/tmp/zxz3hd8h_model_07500.onnx}"
CHECKPOINT="${HOLOSOMA_ZERO_RECORD_CHECKPOINT:-$WANDB_CHECKPOINT}"
DURATION="${HOLOSOMA_ZERO_RECORD_DURATION:-7s}"
OBJECT_MASS="${HOLOSOMA_MJ_OBJECT_MASS:-2.0}"
OUT_DIR="${HOLOSOMA_ZERO_RECORD_OUT_DIR:-${ROOT_DIR}/artifacts/zero_command_zxz3hd8h_07500_2kg_7s_$(date +%Y%m%d_%H%M%S)}"
MAX_CLIPS="${HOLOSOMA_ZERO_RECORD_MAX_CLIPS:-}"

if [[ ! -f "$CLIP_MAP" ]]; then
  echo "[mj_record_zero_133] missing clip map: $CLIP_MAP" >&2
  exit 1
fi

running="$(pgrep -af "${ROOT_DIR}/src/holosoma/holosoma/run_sim.py|${ROOT_DIR}/src/holosoma_inference/holosoma_inference/run_policy.py|${ROOT_DIR}/mj_env.sh|${ROOT_DIR}/mj_ro.sh" || true)"
if [[ -n "$running" ]]; then
  echo "[mj_record_zero_133] another Holosoma MuJoCo/policy process is already running:" >&2
  echo "$running" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/videos" "$OUT_DIR/logs" "$OUT_DIR/metrics"

mapfile -t clips < <(
  CLIP_MAP="$CLIP_MAP" python3 - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path(os.environ["CLIP_MAP"]).read_text())
for name in sorted(data["clips"]):
    print(name)
PY
)

if [[ -n "$MAX_CLIPS" ]]; then
  clips=("${clips[@]:0:$MAX_CLIPS}")
fi

printf '%s\n' "${clips[@]}" >"${OUT_DIR}/clips.txt"

echo "[mj_record_zero_133] clips=${#clips[@]}"
echo "[mj_record_zero_133] checkpoint=$CHECKPOINT"
echo "[mj_record_zero_133] duration=$DURATION mass=${OBJECT_MASS}kg command=zero"
echo "[mj_record_zero_133] out_dir=$OUT_DIR"

failed=()
for index in "${!clips[@]}"; do
  clip="${clips[$index]}"
  motion="${BANK_DIR}/${clip}.npz"
  log_path="${OUT_DIR}/logs/${clip}.log"
  video_path="${OUT_DIR}/videos/${clip}.mp4"
  metrics_path="${OUT_DIR}/metrics/${clip}.json"

  if [[ ! -f "$motion" ]]; then
    echo "[mj_record_zero_133] missing motion: $motion" | tee "$log_path"
    failed+=("$clip")
    continue
  fi
  if [[ -s "$video_path" ]]; then
    echo "[mj_record_zero_133] $((index + 1))/${#clips[@]} skip existing $video_path"
    continue
  fi

  checkpoint_arg="$CHECKPOINT"
  if [[ -f "$LOCAL_CHECKPOINT" ]]; then
    checkpoint_arg="$LOCAL_CHECKPOINT"
  fi

  echo "[mj_record_zero_133] $((index + 1))/${#clips[@]} recording $clip"
  set +e
  HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=1 \
  HOLOSOMA_MJ_OBJECT_MASS="$OBJECT_MASS" \
  bash "${ROOT_DIR}/mj_debug.sh" \
    --clip "$motion" \
    --run "$RUN_REF" \
    --checkpoint "$checkpoint_arg" \
    --duration "$DURATION" \
    --record \
    --no-sim-state \
    --no-auto-motion \
    >"$log_path" 2>&1
  status=$?
  set -e

  debug_dir="$(sed -n 's/^\[mj_debug\] log_dir=//p' "$log_path" | tail -n 1)"
  if [[ "$status" -eq 0 && -n "$debug_dir" && -s "${debug_dir}/viewer_capture.mp4" ]]; then
    cp -f "${debug_dir}/viewer_capture.mp4" "$video_path"
    if [[ -f "${debug_dir}/metrics.json" ]]; then
      cp -f "${debug_dir}/metrics.json" "$metrics_path"
    fi
    echo "[mj_record_zero_133] saved $video_path"
  else
    echo "[mj_record_zero_133] failed $clip status=$status debug_dir=${debug_dir:-unknown}" >&2
    tail -n 80 "$log_path" >&2 || true
    failed+=("$clip")
  fi
done

if (( ${#failed[@]} > 0 )); then
  printf '%s\n' "${failed[@]}" >"${OUT_DIR}/failed.txt"
  echo "[mj_record_zero_133] failed=${#failed[@]} list=${OUT_DIR}/failed.txt" >&2
  exit 1
fi

echo "[mj_record_zero_133] done videos=${OUT_DIR}/videos"
