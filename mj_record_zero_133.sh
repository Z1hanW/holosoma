#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAR_DIR="$(cd "$ROOT_DIR/.." && pwd)"
TRACKING_DIR="$ROOT_DIR/mujoco_tracking"
DATASET_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5"
if [[ -e "$ROOT_DIR/data/$DATASET_NAME" ]]; then
  DATASET="$ROOT_DIR/data/$DATASET_NAME"
else
  DATASET="$FAR_DIR/data/$DATASET_NAME"
fi
MUJOCO_PY="${MUJOCO_PY:-/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
LOCAL_MODEL="${HOLOSOMA_ZERO_RECORD_LOCAL_MODEL:-/tmp/zxz3hd8h_model_07500.onnx}"
MODEL="${HOLOSOMA_ZERO_RECORD_MODEL:-$LOCAL_MODEL}"
SECONDS_TO_RECORD="${HOLOSOMA_ZERO_RECORD_SECONDS:-20}"
OBJECT_MASS="${HOLOSOMA_MJ_OBJECT_MASS:-2.0}"
OUT_DIR="${HOLOSOMA_ZERO_RECORD_OUT_DIR:-${ROOT_DIR}/logs/zero_command_videos_133_zxz3hd8h_07500_2kg_20s_mjdebug_g1_collision_$(date +%Y%m%d_%H%M%S)}"
TMP_ROOT="${HOLOSOMA_ZERO_RECORD_TMP_ROOT:-$ROOT_DIR/logs/.zero_command_video_tmp_$(date +%Y%m%d_%H%M%S)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./mj_record_zero_133.sh"
  echo "Writes one trace-rendered MP4 per sequence into: $OUT_DIR"
  echo "Optional env: MAX_CLIPS=1, HOLOSOMA_ZERO_RECORD_SECONDS=20, HOLOSOMA_MJ_OBJECT_MASS=2.0"
  exit 0
fi

[[ -f "$DATASET/summary.csv" ]] || { echo "missing summary.csv in $DATASET" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$TMP_ROOT"

mapfile -t CLIPS < <("$MUJOCO_PY" - "$DATASET/summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        clip = row.get("clip_id", "").strip()
        if clip:
            print(clip)
PY
)

TOTAL="${#CLIPS[@]}"
if [[ "$TOTAL" -ne 133 ]]; then
  echo "expected 133 clips, got $TOTAL" >&2
  exit 1
fi
if [[ -n "${MAX_CLIPS:-${HOLOSOMA_ZERO_RECORD_MAX_CLIPS:-}}" ]]; then
  LIMIT="${MAX_CLIPS:-$HOLOSOMA_ZERO_RECORD_MAX_CLIPS}"
  CLIPS=("${CLIPS[@]:0:$LIMIT}")
  TOTAL="${#CLIPS[@]}"
fi

cleanup() {
  if [[ "${KEEP_TMP:-0}" != "1" ]]; then
    rm -rf "$TMP_ROOT"
  fi
}
trap cleanup EXIT

printf '%s\n' "${CLIPS[@]}" >"$OUT_DIR/clips.txt"

echo "output: $OUT_DIR"
echo "clips:  $TOTAL"
echo "model:  $MODEL"
echo "mass:   ${OBJECT_MASS}kg"
echo "seconds: $SECONDS_TO_RECORD"
echo "command: sparse root [0, 0, 0]"

FAIL_DIR="$OUT_DIR/failed"
FAILED=0
DONE=0

for CLIP in "${CLIPS[@]}"; do
  DONE=$((DONE + 1))
  MP4="$OUT_DIR/${CLIP}.mp4"
  if [[ -s "$MP4" ]]; then
    echo "[$DONE/$TOTAL] skip $CLIP"
    continue
  fi

  RUN_DIR="$TMP_ROOT/$CLIP"
  CONVEX_DIR="$RUN_DIR/coacd_outputs"
  LOG="$RUN_DIR/record.log"
  mkdir -p "$RUN_DIR"

  echo "[$DONE/$TOTAL] record $CLIP"
  if RUN_DIR="$RUN_DIR" \
    OUTPUT_MP4="$MP4" \
    CONVEX_DIR="$CONVEX_DIR" \
    ZERO_MODEL="$MODEL" \
    SECONDS_TO_RECORD="$SECONDS_TO_RECORD" \
    HOLOSOMA_MJ_OBJECT_MASS="$OBJECT_MASS" \
    "$TRACKING_DIR/record_zero_command.sh" "$CLIP" >"$LOG" 2>&1; then
    if [[ -s "$MP4" ]]; then
      if [[ "${KEEP_TMP:-0}" != "1" ]]; then
        rm -rf "$RUN_DIR"
      fi
    else
      mkdir -p "$FAIL_DIR"
      cp "$LOG" "$FAIL_DIR/${CLIP}.log" 2>/dev/null || true
      if [[ "${KEEP_TMP:-0}" != "1" ]]; then
        rm -rf "$RUN_DIR"
      fi
      echo "$CLIP" >> "$FAIL_DIR/failed.txt"
      echo "failed: $CLIP did not produce $MP4" >&2
      FAILED=$((FAILED + 1))
    fi
  else
    mkdir -p "$FAIL_DIR"
    cp "$LOG" "$FAIL_DIR/${CLIP}.log" 2>/dev/null || true
    if [[ "${KEEP_TMP:-0}" != "1" ]]; then
      rm -rf "$RUN_DIR"
    fi
    echo "$CLIP" >> "$FAIL_DIR/failed.txt"
    echo "failed: $CLIP" >&2
    FAILED=$((FAILED + 1))
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "finished with $FAILED failure(s). Logs: $FAIL_DIR" >&2
  exit 1
fi

MISSING=()
for CLIP in "${CLIPS[@]}"; do
  if [[ ! -s "$OUT_DIR/${CLIP}.mp4" ]]; then
    MISSING+=("$CLIP")
  fi
done
if (( ${#MISSING[@]} > 0 )); then
  mkdir -p "$FAIL_DIR"
  printf '%s\n' "${MISSING[@]}" > "$FAIL_DIR/missing.txt"
  echo "finished with ${#MISSING[@]} missing MP4(s). See $FAIL_DIR/missing.txt" >&2
  exit 1
fi

rmdir "$FAIL_DIR" 2>/dev/null || true
echo "done: $OUT_DIR"
