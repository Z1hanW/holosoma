#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

POST_SCENE_ROOT=${1:-"/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene"}
HMR_TYPE=${2:-${HMR_TYPE:-"gv"}}
SEQ_NAME=${3:-${SEQ_NAME:-""}}
ROBOT_URDF=${ROBOT_URDF:-"models/g1/g1_29dof.urdf"}
OUT_ROOT=${OUT_ROOT:-"$SCRIPT_DIR/demo_results/g1/climbing/mocap_crisp"}
ROBOT_HEIGHT=${ROBOT_HEIGHT:-1.32}
HUMAN_HEIGHT=${HUMAN_HEIGHT:-1.78}

DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/crisp/vmm_data"}
SCENE_XML_OVERRIDE=${SCENE_XML_OVERRIDE:-""}

OBJECT_NAME="scene_mesh_sqs"
# Motion file name expected by downstream code (matches retargeting_gt behavior)
TASK_NAME=${TASK_NAME:-"human_motion"}
TEMPLATE_XML="$SCRIPT_DIR/models/g1/g1_29dof_w_terrain.xml"
ROBOT_SRC_DIR="$SCRIPT_DIR/models/g1"
ROBOT_URDF_SRC="$ROBOT_SRC_DIR/g1_29dof.urdf"

if [ ! -f "$TEMPLATE_XML" ]; then
    echo "[ERROR] missing template scene xml: $TEMPLATE_XML" >&2
    exit 1
fi
if [ ! -f "$ROBOT_URDF_SRC" ]; then
    echo "[ERROR] missing robot urdf: $ROBOT_URDF_SRC" >&2
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
        echo "[ERROR] missing hmr or scene files for $seq_name" >&2
        exit 1
    fi
    if [ ! -d "$pieces_dir" ]; then
        echo "[ERROR] missing pieces dir for $seq_name: $pieces_dir" >&2
        exit 1
    fi

    stage_obj_dir="$DATA_ROOT/$seq_name"
    mkdir -p "$stage_obj_dir"

    # Put motion and geometry into the same sequence folder (real files, no symlinks).
    cp -f "$hmr_npz" "$stage_obj_dir/$TASK_NAME.npz"

    # Copy all robot assets into the sequence folder.
    robot_dir="$stage_obj_dir/g1"
    mkdir -p "$robot_dir"
    cp -R "$ROBOT_SRC_DIR/." "$robot_dir/"

    cp -f "$scene_obj" "$stage_obj_dir/scene_mesh_sqs.obj"
    scene_urdf_local="$stage_obj_dir/scene_mesh_sqs.urdf"
    cp -f "$scene_urdf" "$scene_urdf_local"
    rm -rf "$stage_obj_dir/pieces"
    cp -R "$pieces_dir" "$stage_obj_dir/pieces"
    python - <<PY
from pathlib import Path
import xml.etree.ElementTree as ET

urdf_path = Path("$scene_urdf_local")
pieces_dir = Path("$stage_obj_dir/pieces")

try:
    root = ET.parse(urdf_path).getroot()
except ET.ParseError as exc:
    raise SystemExit(f"[ERROR] Failed to parse URDF: {urdf_path}: {exc}") from exc

for mesh in root.findall(".//mesh"):
    filename = mesh.get("filename")
    if not filename:
        continue
    base = Path(filename).name
    if (pieces_dir / base).exists():
        mesh.set("filename", f"pieces/{base}")

# Always overwrite the local URDF so paths are normalized.
urdf_path.write_text(ET.tostring(root, encoding="unicode"))
PY

    python - <<PY
from pathlib import Path
import re

pieces_dir = Path("$stage_obj_dir/pieces")
assets_path = Path("$stage_obj_dir/box_assets.xml")
body_path = Path("$stage_obj_dir/box_body.xml")
object_prefix = "$OBJECT_NAME"
robot_height = float("$ROBOT_HEIGHT")
human_height = float("$HUMAN_HEIGHT")

def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"piece_{name}"
    return name

scale = robot_height / human_height
scale_str = f"{scale} {scale} {scale}"

meshes = [(f"piece_{sanitize(piece.stem)}", piece) for piece in sorted(pieces_dir.glob("*.obj"))]
if not meshes:
    raise SystemExit("No mesh pieces found in pieces/ for box_assets.xml.")

asset_lines = ["<mujocoinclude>"]
for mesh_name, mesh_path in meshes:
    asset_lines.append(
        f'    <mesh name="{mesh_name}" file="{mesh_path.as_posix()}" scale="{scale_str}"/>'
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

    scene_xml_local="$stage_obj_dir/g1_29dof_w_terrain.xml"
    if [ -n "$SCENE_XML_OVERRIDE" ]; then
        scene_xml_src=${SCENE_XML_OVERRIDE//\{seq\}/$seq_name}
        cp -f "$scene_xml_src" "$scene_xml_local"
    else
        cp -f "$TEMPLATE_XML" "$scene_xml_local"
    fi
    python - <<PY
import re
from pathlib import Path

path = Path("$scene_xml_local")
text = path.read_text()
text = re.sub(r'meshdir="[^"]*"', f'meshdir="{Path("$robot_dir/assets").as_posix()}"', text, count=1)
# Drop any template piece assets/geoms so we can inject our own pieces safely.
text = re.sub(r"\n\\s*<mesh name=\"part_[^\"]+\"[^>]*>", "", text)
text = re.sub(r"\n\\s*<geom name=\"part_[^\"]+\"[^>]*>", "", text)
if "box_assets.xml" not in text:
    text = text.replace("</asset>", '  <include file="box_assets.xml"/>\n  </asset>', 1)
if "box_body.xml" not in text:
    text = text.replace("</worldbody>", '  <include file="box_body.xml"/>\n  </worldbody>', 1)
path.write_text(text)
PY

    # Keep box assets alongside the robot package too (for the in-robot scene XML).
    cp -f "$stage_obj_dir/box_assets.xml" "$robot_dir/box_assets.xml"
    cp -f "$stage_obj_dir/box_body.xml" "$robot_dir/box_body.xml"

    # Provide scene XMLs next to the robot URDF for MuJoCo loading.
    cp -f "$scene_xml_local" "$robot_dir/g1_29dof_w_scene_mesh_sqs.xml"
    cp -f "$scene_xml_local" "$robot_dir/g1_29dof_w_terrain.xml"

    mkdir -p "$OUT_ROOT"
    scene_xml_file="$scene_xml_local"
    robot_urdf_local="$robot_dir/g1_29dof.urdf"
    echo "[retgt_crisp] seq=$seq_name"
    echo "  data_path=$stage_obj_dir"
    echo "  motion=$stage_obj_dir/$TASK_NAME.npz"
    echo "  object_dir=$stage_obj_dir"
    echo "  scene_xml=$scene_xml_local"
    echo "  robot_urdf=$robot_urdf_local"
    echo "  object_urdf=$scene_urdf_local"
    echo "  object_obj=$stage_obj_dir/scene_mesh_sqs.obj"
    echo "  pieces=$stage_obj_dir/pieces"
    echo "  box_assets=$stage_obj_dir/box_assets.xml"
    echo "  box_body=$stage_obj_dir/box_body.xml"
    SAVE_MODE=True bash "$SCRIPT_DIR/retgt_smplx.sh" "$stage_obj_dir" "$TASK_NAME" "$OBJECT_NAME" "$stage_obj_dir" "$robot_urdf_local" "smplx" "$OUT_ROOT" "$scene_xml_file"
done
