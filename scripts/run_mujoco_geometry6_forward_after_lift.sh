#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_mujoco_geometry6_forward_after_lift.sh --output-root DIR [options]

Options:
  --model-onnx FILE    Exact policy ONNX (defaults to 0mcqao8k/model_40000)
  --staged-root DIR    Six staged motion/object pairs from the latest _check_vis set
  --port-base PORT     First MuJoCo split-runtime port (default: 7955)
EOF
}

OUTPUT_ROOT=""
MODEL_ONNX="/data/holosoma_eval_audits/sx_0mc_checkpoint_progression_native_20260808_185459/runs/0mcqao8k/wandb_files/model_40000/model_40000.onnx"
STAGED_ROOT="/data/holosoma_eval_audits/sx_0mc_checkpoint_progression_debug30_geometry6_postlift_forward015_20260809_201421/staged_pairs"
PORT_BASE=7955

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model-onnx) MODEL_ONNX="$2"; shift 2 ;;
    --staged-root) STAGED_ROOT="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "Missing required --output-root" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$MODEL_ONNX" ]]; then
  echo "Missing model ONNX: $MODEL_ONNX" >&2
  exit 2
fi
if [[ ! -d "$STAGED_ROOT" ]]; then
  echo "Missing staged geometry root: $STAGED_ROOT" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p "$OUTPUT_ROOT/runs"
OUTPUT_ROOT="$(realpath "$OUTPUT_ROOT")"
MODEL_ONNX="$(realpath "$MODEL_ONNX")"
STAGED_ROOT="$(realpath "$STAGED_ROOT")"

# One frozen MuJoCo physics mapping is used for every geometry.  The training
# URDF collision meshes preserve the real rubber-hand surfaces; non-carry robot
# bodies are filtered out of object contacts.  MuJoCo's noslip post-solve and
# friction impedance ratio compensate for the PhysX velocity-iteration contact
# solve without introducing a weld, attachment, gantry, or state assist.
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML=0
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML=0
export MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES=1
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION=1.6
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION=0.02
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION=0.005
export HOLOSOMA_MUJOCO_NOSLIP_ITERATIONS=10
export HOLOSOMA_MUJOCO_IMPRATIO=10

pair_dirs=(
  "0000_unscale_any_ball_29"
  "0001_scaledown_any_ball_26"
  "0002_scaledown_any_bin_25"
  "0003_unscale_any_bin_27"
  "0004_unscale_any_bin_22"
  "0005_scaledown_any_bin_21"
)
clip_files=(
  "unscale__any_ball_29.npz"
  "scaledown__any_ball_26.npz"
  "scaledown__any_bin_25.npz"
  "unscale__any_bin_27.npz"
  "unscale__any_bin_22.npz"
  "scaledown__any_bin_21.npz"
)
clip_slugs=(
  "unscale_any_ball_29"
  "scaledown_any_ball_26"
  "scaledown_any_bin_25"
  "unscale_any_bin_27"
  "unscale_any_bin_22"
  "scaledown_any_bin_21"
)

for index in "${!pair_dirs[@]}"; do
  pair_dir="$STAGED_ROOT/${pair_dirs[$index]}"
  motion_file="$pair_dir/${clip_files[$index]}"
  object_urdf="$pair_dir/_single_slot_urdfs/${clip_slugs[$index]}.urdf"
  run_dir="$OUTPUT_ROOT/runs/$(printf '%02d' "$((index + 1))")_${clip_slugs[$index]}"
  if [[ ! -f "$motion_file" || ! -f "$object_urdf" ]]; then
    echo "Incomplete staged pair: $pair_dir" >&2
    exit 2
  fi
  if [[ -f "$run_dir/audit/command_audit.json" ]] && \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python -c \
      'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' \
      "$run_dir/audit/command_audit.json"; then
    echo "[geometry $((index + 1))/6] ${clip_slugs[$index]}: reusing existing passed audit"
    continue
  fi
  if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty unaudited run directory: $run_dir" >&2
    exit 2
  fi
  echo "[geometry $((index + 1))/6] ${clip_slugs[$index]}"
  scripts/run_mujoco_forward_after_lift_rollout.sh \
    --motion-file "$motion_file" \
    --object-urdf "$object_urdf" \
    --model-onnx "$MODEL_ONNX" \
    --output-dir "$run_dir" \
    --port-base "$PORT_BASE" \
    --forward-command-m 0.15 \
    --lift-rel-z-delta-m 0.30 \
    --actor-steps 501
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    scripts/audit_mujoco_forward_after_lift_rollout.py \
    --run-dir "$run_dir" \
    >"$run_dir/audit/audit_stdout.json"
done

echo "Completed six audited MuJoCo rollouts: $OUTPUT_ROOT"
