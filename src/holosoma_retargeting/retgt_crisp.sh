#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

POST_SCENE_ROOT=${1:-"/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene"}
HMR_TYPE=${2:-${HMR_TYPE:-"gv"}}
SEQ_NAME=${3:-${SEQ_NAME:-""}}
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
SCENE_XML_OVERRIDE=${SCENE_XML_OVERRIDE:-""}

OBJECT_NAME="scene_mesh_sqs"
TASK_NAME="human_motion"
TEMPLATE_XML="$SCRIPT_DIR/models/g1/g1_29dof_w_stairs.xml"
MESH_DIR="$SCRIPT_DIR/models/g1/assets"

if [ ! -f "$TEMPLATE_XML" ]; then
    echo "[ERROR] missing template scene xml: $TEMPLATE_XML" >&2
    exit 1
fi

seq_dirs=()
if [ -n "$SEQ_NAME" ]; then
    seq_dir="$POST_SCENE_ROOT/$SEQ_NAME"
    if [ -d "$SEQ_NAME" ]; then
        seq_dir="$SEQ_NAME"
    fi
    if [ ! -d "$seq_dir" ]; then
        echo "[ERROR] sequence not found: $seq_dir" >&2
        exit 1
    fi
    seq_dirs=("$seq_dir")
else
    for seq_dir in "$POST_SCENE_ROOT"/*; do
        [ -d "$seq_dir" ] || continue
        seq_dirs+=("$seq_dir")
    done
fi

for seq_dir in "${seq_dirs[@]}"; do

    seq_name=$(basename "$seq_dir")
    hmr_dir="$seq_dir/$HMR_TYPE/hmr"
    hmr_npz="$hmr_dir/$seq_name.npz"
    scene_dir="$seq_dir/$HMR_TYPE/scene_mesh_sqs"
    scene_obj="$scene_dir/scene_mesh_sqs.obj"
    scene_urdf="$scene_dir/scene_mesh_sqs.urdf"
    pieces_dir="$scene_dir/pieces"

    if [ ! -f "$hmr_npz" ] || [ ! -f "$scene_obj" ] || [ ! -f "$scene_urdf" ]; then
        echo "[WARN] skip $seq_name: missing hmr or scene files" >&2
        continue
    fi

    mkdir -p "$MOTION_ROOT"
    motion_file="$MOTION_ROOT/$seq_name.npz"
    ln -sf "$hmr_npz" "$motion_file"

    stage_obj_dir="$OBJECT_ROOT/$seq_name"
    mkdir -p "$stage_obj_dir"

    ln -sf "$scene_obj" "$stage_obj_dir/scene_mesh_sqs.obj"
    ln -sf "$scene_urdf" "$stage_obj_dir/scene_mesh_sqs.urdf"
    if [ -d "$pieces_dir" ]; then
        ln -sfn "$pieces_dir" "$stage_obj_dir/pieces"
    fi

    python - <<PY
from pathlib import Path
import re

pieces_dir = Path("$stage_obj_dir/pieces")
assets_path = Path("$stage_obj_dir/box_assets.xml")
body_path = Path("$stage_obj_dir/box_body.xml")
fallback_mesh = Path("$stage_obj_dir/scene_mesh_sqs.obj")
object_prefix = "$OBJECT_NAME"

def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"piece_{name}"
    return name

meshes = []
if pieces_dir.exists():
    for piece in sorted(pieces_dir.glob("*.obj")):
        mesh_name = f"piece_{sanitize(piece.stem)}"
        meshes.append((mesh_name, piece))

if not meshes and fallback_mesh.exists():
    meshes.append(("piece_scene_mesh_sqs", fallback_mesh))

if not meshes:
    raise SystemExit("No mesh pieces found for box_assets.xml.")

asset_lines = ["<mujocoinclude>"]
for mesh_name, mesh_path in meshes:
    asset_lines.append(
        f'    <mesh name="{mesh_name}" file="{mesh_path.as_posix()}" scale="1.0 1.0 1.0"/>'
    )
asset_lines.append('    <material name="scene_piece_material" rgba="0.6 0.6 0.6 1"/>')
asset_lines.append("</mujocoinclude>")
assets_path.write_text("\\n".join(asset_lines) + "\\n")

body_lines = ["<mujocoinclude>"]
for idx, (mesh_name, _mesh_path) in enumerate(meshes, start=1):
    body_lines.append(f'    <body name="{object_prefix}_piece_{idx}" pos="0 0 0" quat="1 0 0 0">')
    body_lines.append(
        f'        <geom name="{object_prefix}_piece_{idx}_geom" type="mesh" mesh="{mesh_name}" '
        f'pos="0 0 0" quat="1 0 0 0" material="scene_piece_material" contype="1" conaffinity="1"/>'
    )
    body_lines.append("    </body>")
body_lines.append("</mujocoinclude>")
body_path.write_text("\\n".join(body_lines) + "\\n")
PY

    cp -f "$TEMPLATE_XML" "$stage_obj_dir/g1_29dof_w_scene_mesh_sqs.xml"
    python - <<PY
import re
from pathlib import Path

path = Path("$stage_obj_dir/g1_29dof_w_scene_mesh_sqs.xml")
text = path.read_text()
text = re.sub(r'meshdir="[^"]*"', f'meshdir="{Path("$MESH_DIR").as_posix()}"', text, count=1)
# Drop any template piece assets/geoms so we can inject our own pieces safely.
text = re.sub(r"\n\\s*<mesh name=\"part_[^\"]+\"[^>]*>", "", text)
text = re.sub(r"\n\\s*<geom name=\"part_[^\"]+\"[^>]*>", "", text)
if "box_assets.xml" not in text:
    text = text.replace("</asset>", '  <include file="box_assets.xml"/>\n  </asset>', 1)
if "box_body.xml" not in text:
    text = text.replace("</worldbody>", '  <include file="box_body.xml"/>\n  </worldbody>', 1)
path.write_text(text)
PY

    mkdir -p "$OBJ_ROOT/$seq_name" "$URDF_ROOT/$seq_name" "$XML_ROOT/$seq_name"
    ln -sf "$stage_obj_dir/scene_mesh_sqs.obj" "$OBJ_ROOT/$seq_name/scene_mesh_sqs.obj"
    ln -sf "$stage_obj_dir/scene_mesh_sqs.urdf" "$URDF_ROOT/$seq_name/scene_mesh_sqs.urdf"
    ln -sf "$stage_obj_dir/box_assets.xml" "$XML_ROOT/$seq_name/box_assets.xml"
    ln -sf "$stage_obj_dir/g1_29dof_w_scene_mesh_sqs.xml" "$XML_ROOT/$seq_name/g1_29dof_w_scene_mesh_sqs.xml"
    ln -sf "$stage_obj_dir/scene_mesh_sqs.obj" "$XML_ROOT/$seq_name/scene_mesh_sqs.obj"
    ln -sf "$stage_obj_dir/box_body.xml" "$XML_ROOT/$seq_name/box_body.xml"
    if [ -d "$stage_obj_dir/pieces" ]; then
        mkdir -p "$PIECES_ROOT"
        ln -sfn "$stage_obj_dir/pieces" "$PIECES_ROOT/$seq_name"
        ln -sfn "$stage_obj_dir/pieces" "$XML_ROOT/$seq_name/pieces"
    fi

    mkdir -p "$OUT_ROOT"
    if [ -n "$SCENE_XML_OVERRIDE" ]; then
        scene_xml_file=${SCENE_XML_OVERRIDE//\{seq\}/$seq_name}
    else
        scene_xml_file="$XML_ROOT/$seq_name/g1_29dof_w_scene_mesh_sqs.xml"
    fi
    bash "$SCRIPT_DIR/retgt_smplx.sh" "$MOTION_ROOT" "$seq_name" "$OBJECT_NAME" "$stage_obj_dir" "$ROBOT_URDF" "smplx" "$OUT_ROOT" "$scene_xml_file"
done
