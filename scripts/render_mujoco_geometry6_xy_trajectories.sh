#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: render_mujoco_geometry6_xy_trajectories.sh AUDIT_ROOT" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
audit_root="$(realpath "$1")"
final_root="$audit_root/final"
individual_root="$final_root/individual"
mkdir -p "$individual_root"

clip_slugs=(
  "unscale_any_ball_29"
  "scaledown_any_ball_26"
  "scaledown_any_bin_25"
  "unscale_any_bin_27"
  "unscale_any_bin_22"
  "scaledown_any_bin_21"
)

videos=()
for index in "${!clip_slugs[@]}"; do
  number="$(printf '%02d' "$((index + 1))")"
  slug="${clip_slugs[$index]}"
  run_dir="$audit_root/runs/${number}_${slug}"
  video="$individual_root/${number}_${slug}__0mcqao8k_model40000__mujoco_forward015__zoomout_xy_trajectory.mp4"
  videos+=("$video")
  valid_frames="$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$video" 2>/dev/null || true)"
  if [[ "$valid_frames" == "501" ]]; then
    echo "[render $number/06] $slug: reusing 501-frame video"
    continue
  fi
  echo "[render $number/06] $slug"
  MUJOCO_GL=egl \
  HOLOSOMA_RECORD_HIDE_WRIST_YAW_CYLINDERS=0 \
  PYTHONPATH="src/holosoma${PYTHONPATH:+:$PYTHONPATH}" \
    /data/ubuntu/conda-envs/dexjoco/bin/python \
      scripts/render_mujoco_xy_trajectory_video.py \
      --run-dir "$run_dir" \
      --clip-slug "$slug" \
      --output "$video"
done

master="$final_root/00_master__0mcqao8k_model40000__debug30_geometry_diverse6__mujoco_post_lift_persistent_forward015__zoomout_xy_trajectories_6x1.mp4"
ffmpeg -hide_banner -loglevel error -y \
  -i "${videos[0]}" -i "${videos[1]}" -i "${videos[2]}" \
  -i "${videos[3]}" -i "${videos[4]}" -i "${videos[5]}" \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
  -map "[v]" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -movflags +faststart -an "$master"

echo "$master"
