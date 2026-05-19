#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOSOMA="$(cd "$HERE/.." && pwd)"
FAR="$(cd "$HOLOSOMA/.." && pwd)"

DATASET_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5"
if [[ -e "$HOLOSOMA/data/$DATASET_NAME" ]]; then
  DATASET="$HOLOSOMA/data/$DATASET_NAME"
else
  DATASET="$FAR/data/$DATASET_NAME"
fi

OUT_DIR="${OUT_DIR:-$HOLOSOMA/logs/cotracking_videos_133_$(date +%Y%m%d_%H%M%S)}"
TMP_ROOT="${TMP_ROOT:-$HOLOSOMA/logs/.cotracking_video_tmp_$(date +%Y%m%d_%H%M%S)}"
MUJOCO_PY="/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./mujoco_tracking/record_all_cotracking_videos.sh"
  echo "Writes one MP4 per sequence into: $OUT_DIR"
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
if [[ -n "${MAX_CLIPS:-}" ]]; then
  CLIPS=("${CLIPS[@]:0:$MAX_CLIPS}")
  TOTAL="${#CLIPS[@]}"
fi

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "output: $OUT_DIR"
echo "clips:  $TOTAL"

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
    "$HERE/record_cotracking.sh" "$CLIP" >"$LOG" 2>&1; then
    if [[ -s "$MP4" ]]; then
      rm -rf "$RUN_DIR"
    else
      mkdir -p "$FAIL_DIR"
      cp "$LOG" "$FAIL_DIR/${CLIP}.log" 2>/dev/null || true
      rm -rf "$RUN_DIR"
      echo "$CLIP" >> "$FAIL_DIR/failed.txt"
      echo "failed: $CLIP did not produce $MP4" >&2
      FAILED=$((FAILED + 1))
    fi
  else
    mkdir -p "$FAIL_DIR"
    cp "$LOG" "$FAIL_DIR/${CLIP}.log" 2>/dev/null || true
    rm -rf "$RUN_DIR"
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
