#!/usr/bin/env bash
set -euo pipefail
SRC=/home/ubuntu/FAR/holosoma/src/holosoma_retargeting_my/converted_res/robot_only/amass_all_trainready
DST=/nfs/zzzihanw/crisp/home/ubuntu/FAR/holosoma/src/holosoma_retargeting_my/converted_res/robot_only/amass_all_trainready
mkdir -p "$DST/LAFAN1_npz" "$DST/TWIST1_motion_data_npz" "$DST/TWIST2_motion_data_npz"
rsync -aL --human-readable --info=stats2,progress2 --ignore-existing "$SRC/TWIST1_motion_data_npz/" "$DST/TWIST1_motion_data_npz/"
rsync -aL --human-readable --info=stats2,progress2 --ignore-existing "$SRC/TWIST2_motion_data_npz/" "$DST/TWIST2_motion_data_npz/"
echo "[DONE] rsync_amass_split complete"
