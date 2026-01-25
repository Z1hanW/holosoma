#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

POST_SCENE_ROOT=${1:-"/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene"}
ROBOT_URDF=${ROBOT_URDF:-"models/g1/g1_29dof.urdf"}
OUT_ROOT=${OUT_ROOT:-"$SCRIPT_DIR/demo_results/g1/climbing/mocap_crisp"}

DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/crisp/vmm_data"}
MOTION_ROOT="$DATA_ROOT/motion"
GEO_ROOT="$DATA_ROOT/geo"

OBJ_ROOT="$GEO_ROOT/obj"
URDF_ROOT="$GEO_ROOT/urdf"
XML_ROOT="$GEO_ROOT/xml"
PIECES_ROOT="$GEO_ROOT/pieces"
OBJECT_ROOT="$GEO_ROOT/scene_mesh_sqs"

OBJECT_NAME="scene_mesh_sqs"
TASK_NAME="human_motion"
TEMPLATE_XML="$SCRIPT_DIR/demo_data/far_robot/far_robot/g1_29dof_w_stairs.xml"
MESH_DIR="$SCRIPT_DIR/models/g1/meshes"

if [ ! -f "$TEMPLATE_XML" ]; then
    echo "[ERROR] missing template scene xml: $TEMPLATE_XML" >&2
    exit 1
fi

for seq_dir in "$POST_SCENE_ROOT"/*; do
    [ -d "$seq_dir" ] || continue

    seq_name=$(basename "$seq_dir")
    hmr_dir="$seq_dir/gv/hmr"
    hmr_npz="$hmr_dir/human_motion.npz"
    scene_dir="$seq_dir/gv/scene_mesh_sqs"
    scene_obj="$scene_dir/scene_mesh_sqs.obj"
    scene_urdf="$scene_dir/scene_mesh_sqs.urdf"
    pieces_dir="$scene_dir/pieces"

    if [ ! -f "$hmr_npz" ] || [ ! -f "$scene_obj" ] || [ ! -f "$scene_urdf" ]; then
        echo "[WARN] skip $seq_name: missing hmr or scene files" >&2
        continue
    fi

    mkdir -p "$MOTION_ROOT"
    motion_name="${TASK_NAME}_${seq_name}"
    motion_file="$MOTION_ROOT/$motion_name.npz"
    ln -sf "$hmr_npz" "$motion_file"

    stage_obj_dir="$OBJECT_ROOT/$seq_name"
    mkdir -p "$stage_obj_dir"

    ln -sf "$scene_obj" "$stage_obj_dir/scene_mesh_sqs.obj"
    ln -sf "$scene_urdf" "$stage_obj_dir/scene_mesh_sqs.urdf"
    if [ -d "$pieces_dir" ]; then
        ln -sfn "$pieces_dir" "$stage_obj_dir/pieces"
    fi

    cat > "$stage_obj_dir/box_assets.xml" <<'EOF'
<mujocoinclude>
    <mesh name="scene_mesh_sqs" file="scene_mesh_sqs.obj" scale="1.0 1.0 1.0"/>
</mujocoinclude>
EOF

    cp -f "$TEMPLATE_XML" "$stage_obj_dir/g1_29dof_w_scene_mesh_sqs.xml"
    python - <<PY
import re
from pathlib import Path

path = Path("$stage_obj_dir/g1_29dof_w_scene_mesh_sqs.xml")
text = path.read_text()
text = re.sub(r'meshdir="[^"]*"', f'meshdir="{Path("$MESH_DIR").as_posix()}"', text, count=1)
path.write_text(text)
PY

    mkdir -p "$OBJ_ROOT/$seq_name" "$URDF_ROOT/$seq_name" "$XML_ROOT/$seq_name"
    ln -sf "$stage_obj_dir/scene_mesh_sqs.obj" "$OBJ_ROOT/$seq_name/scene_mesh_sqs.obj"
    ln -sf "$stage_obj_dir/scene_mesh_sqs.urdf" "$URDF_ROOT/$seq_name/scene_mesh_sqs.urdf"
    ln -sf "$stage_obj_dir/box_assets.xml" "$XML_ROOT/$seq_name/box_assets.xml"
    ln -sf "$stage_obj_dir/g1_29dof_w_scene_mesh_sqs.xml" "$XML_ROOT/$seq_name/g1_29dof_w_scene_mesh_sqs.xml"
    if [ -d "$stage_obj_dir/pieces" ]; then
        mkdir -p "$PIECES_ROOT"
        ln -sfn "$stage_obj_dir/pieces" "$PIECES_ROOT/$seq_name"
    fi

    mkdir -p "$OUT_ROOT"
    "$SCRIPT_DIR/retgt_smplx.sh" "$MOTION_ROOT" "$motion_name" "$OBJECT_NAME" "$stage_obj_dir" "$ROBOT_URDF" "smplx" "$OUT_ROOT"
done
