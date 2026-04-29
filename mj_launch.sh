#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/source_mujoco_setup.sh "${1:-box_75}"
[[ -f "$HOLOSOMA_MJ_MOTION" && -f "$HOLOSOMA_MJ_OBJECT_URDF" && -f "${HOLOSOMA_MJ_OBJECT_URDF%.urdf}.obj" ]] || { echo "missing data_demo files for $HOLOSOMA_MJ_CLIP" >&2; exit 1; }
python -u src/holosoma/holosoma/run_sim.py simulator:mujoco-split robot:g1-29dof-w-object-mujoco terrain:terrain-locomotion-plane perception:camera-depth-d435i-mujoco --motion-init.enabled=True --motion-init.motion-file "$HOLOSOMA_MJ_MOTION"
