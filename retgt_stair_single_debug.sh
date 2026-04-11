#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RETARGET_ROOT="${RETARGET_ROOT:-${SCRIPT_DIR}/src/holosoma_retargeting_my}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/crisp_rl/bin/python}"
POST_SCENE_ROOT="${POST_SCENE_ROOT:-/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene}"
SEQ_NAME="${1:-${SEQ_NAME:-stair_16}}"
HMR_TYPE="${HMR_TYPE:-gv}"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/_tmp/retgt_debug_${SEQ_NAME}}"
OUT_DIR="${OUT_DIR:-${WORK_ROOT}/retarget_out}"
TASK_NAME="${TASK_NAME:-human_motion}"
OBJECT_NAME="${OBJECT_NAME:-scene_mesh_sqs}"
ROBOT_HEIGHT="${ROBOT_HEIGHT:-1.32}"
HUMAN_HEIGHT="${HUMAN_HEIGHT:-1.78}"
HUMAN_Z_OFFSET="${HUMAN_Z_OFFSET:-}"
Q_A_INIT_IDX="${Q_A_INIT_IDX:--7}"
STEP_SIZE="${STEP_SIZE:-0.05}"
PENETRATION_TOLERANCE="${PENETRATION_TOLERANCE:-0.0}"
ACTIVATE_OBJ_NON_PENETRATION="${ACTIVATE_OBJ_NON_PENETRATION:-1}"
ACTIVATE_FOOT_STICKING="${ACTIVATE_FOOT_STICKING:-1}"

DEBUG="${DEBUG:-0}"
VISUALIZE="${VISUALIZE:-0}"
COPY_TO_MOTION_DIR="${COPY_TO_MOTION_DIR:-0}"
TARGET_MOTION_DIR="${TARGET_MOTION_DIR:-${SCRIPT_DIR}/data/ds_crisp_data/___crisp_clean_motion_hmr_retargeted_g1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${RETARGET_ROOT}" ]]; then
  echo "[ERROR] RETARGET_ROOT not found: ${RETARGET_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${RETARGET_ROOT}/examples/robot_retarget.py" ]]; then
  echo "[ERROR] Missing retarget entry: ${RETARGET_ROOT}/examples/robot_retarget.py" >&2
  exit 1
fi

if [[ ! -f "${RETARGET_ROOT}/models/g1/g1_29dof_w_terrain.xml" ]]; then
  echo "[ERROR] Missing terrain template xml: ${RETARGET_ROOT}/models/g1/g1_29dof_w_terrain.xml" >&2
  exit 1
fi

resolve_seq_name() {
  local requested="$1"
  local direct_npz="${POST_SCENE_ROOT}/${requested}/${HMR_TYPE}/hmr/${requested}.npz"
  if [[ -f "${direct_npz}" ]]; then
    echo "${requested}"
    return 0
  fi

  if [[ "${requested}" == stairs_* ]]; then
    local singular="${requested/stairs_/stair_}"
    local singular_npz="${POST_SCENE_ROOT}/${singular}/${HMR_TYPE}/hmr/${singular}.npz"
    if [[ -f "${singular_npz}" ]]; then
      echo "${singular}"
      return 0
    fi
  fi

  return 1
}

if ! RESOLVED_SEQ_NAME="$(resolve_seq_name "${SEQ_NAME}")"; then
  echo "[ERROR] Missing source motion npz for sequence: ${SEQ_NAME}" >&2
  echo "[ERROR] Expected at: ${POST_SCENE_ROOT}/{seq}/${HMR_TYPE}/hmr/{seq}.npz" >&2
  exit 1
fi

SEQ_ROOT="${POST_SCENE_ROOT}/${RESOLVED_SEQ_NAME}/${HMR_TYPE}"
SRC_NPZ="${SEQ_ROOT}/hmr/${RESOLVED_SEQ_NAME}.npz"
SRC_OBJ="${SEQ_ROOT}/${OBJECT_NAME}/${OBJECT_NAME}.obj"
SRC_URDF="${SEQ_ROOT}/${OBJECT_NAME}/${OBJECT_NAME}.urdf"
SRC_PIECES_DIR="${SEQ_ROOT}/${OBJECT_NAME}/pieces"

if [[ ! -f "${SRC_NPZ}" ]]; then
  echo "[ERROR] Missing source motion: ${SRC_NPZ}" >&2
  exit 1
fi
if [[ ! -f "${SRC_OBJ}" ]]; then
  echo "[ERROR] Missing source terrain obj: ${SRC_OBJ}" >&2
  exit 1
fi
if [[ ! -f "${SRC_URDF}" ]]; then
  echo "[ERROR] Missing source terrain urdf: ${SRC_URDF}" >&2
  exit 1
fi
if [[ ! -d "${SRC_PIECES_DIR}" ]]; then
  echo "[ERROR] Missing source terrain pieces: ${SRC_PIECES_DIR}" >&2
  exit 1
fi

if [[ -z "${HUMAN_Z_OFFSET}" ]]; then
  HUMAN_Z_OFFSET="$(
    "${PYTHON_BIN}" - <<PY
import sys
import numpy as np
import trimesh

src_npz = "${SRC_NPZ}"
src_obj = "${SRC_OBJ}"
source = "ray_down"

try:
    with np.load(src_npz, allow_pickle=True) as data:
        joints = np.asarray(data["global_joint_positions"], dtype=np.float64)
except Exception as exc:
    print(f"[AUTO_Z][ERROR] failed loading joints: {exc}", file=sys.stderr)
    print("0.0")
    raise SystemExit(0)

if joints.ndim != 3 or joints.shape[0] == 0 or joints.shape[2] != 3:
    print(f"[AUTO_Z][ERROR] invalid global_joint_positions shape: {joints.shape}", file=sys.stderr)
    print("0.0")
    raise SystemExit(0)

frame0 = joints[0]
ray_joint_idx = int(np.argmin(frame0[:, 2]))
min_joint_z = float(frame0[ray_joint_idx, 2])
ray_xy = frame0[ray_joint_idx, :2]

try:
    mesh = trimesh.load(src_obj, process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
except Exception as exc:
    print(f"[AUTO_Z][ERROR] failed loading obj mesh: {exc}", file=sys.stderr)
    print("0.0")
    raise SystemExit(0)

if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
    print("[AUTO_Z][WARN] empty terrain vertices, fallback offset=0.0", file=sys.stderr)
    print("0.0")
    raise SystemExit(0)

terrain_ref_z = None
hit_count = 0

try:
    top_z = float(verts[:, 2].max()) + 10.0
    ray_origin = np.array([[float(ray_xy[0]), float(ray_xy[1]), top_z]], dtype=np.float64)
    ray_dir = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, _ray_id, _tri_id = intersector.intersects_location(ray_origin, ray_dir, multiple_hits=True)
    hit_count = int(len(locations))
    if hit_count > 0:
        # Vertical downward ray: first touched surface from above is max z among intersections.
        terrain_ref_z = float(np.max(locations[:, 2]))
except Exception as exc:
    print(f"[AUTO_Z][WARN] ray cast failed: {exc}", file=sys.stderr)

if terrain_ref_z is None:
    # Fallback: nearest vertex in XY if ray cast is unavailable.
    d2 = (verts[:, 0] - ray_xy[0]) ** 2 + (verts[:, 1] - ray_xy[1]) ** 2
    nearest = int(np.argmin(d2))
    terrain_ref_z = float(verts[nearest, 2])
    source = "nearest_vertex"

offset = max(0.0, terrain_ref_z - min_joint_z)
print(
    f"[AUTO_Z] min_joint_z={min_joint_z:.4f} terrain_ref_z={terrain_ref_z:.4f} "
    f"offset={offset:.4f} source={source} ray_joint_idx={ray_joint_idx} ray_xy=({ray_xy[0]:.4f},{ray_xy[1]:.4f}) hits={hit_count}",
    file=sys.stderr,
)
print(f"{offset:.6f}")
PY
  )"
  HUMAN_Z_OFFSET_SOURCE="auto"
else
  HUMAN_Z_OFFSET_SOURCE="manual"
fi

WORK_ROOT="${WORK_ROOT/${SEQ_NAME}/${RESOLVED_SEQ_NAME}}"
OUT_DIR="${OUT_DIR/${SEQ_NAME}/${RESOLVED_SEQ_NAME}}"
STAGE_DIR="${WORK_ROOT}/stage"
mkdir -p "${STAGE_DIR}" "${OUT_DIR}"
rm -rf "${STAGE_DIR}/pieces" "${STAGE_DIR}/g1"
mkdir -p "${STAGE_DIR}/g1"

cp -f "${SRC_NPZ}" "${STAGE_DIR}/${TASK_NAME}.npz"
cp -f "${SRC_OBJ}" "${STAGE_DIR}/${OBJECT_NAME}.obj"
cp -f "${SRC_URDF}" "${STAGE_DIR}/${OBJECT_NAME}.urdf"
cp -R "${SRC_PIECES_DIR}" "${STAGE_DIR}/pieces"
cp -R "${RETARGET_ROOT}/models/g1/." "${STAGE_DIR}/g1/"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import xml.etree.ElementTree as ET

urdf_path = Path("${STAGE_DIR}/${OBJECT_NAME}.urdf")
pieces_dir = Path("${STAGE_DIR}/pieces")

root = ET.parse(urdf_path).getroot()
for mesh in root.findall(".//mesh"):
    filename = mesh.get("filename")
    if not filename:
        continue
    base = Path(filename).name
    if (pieces_dir / base).exists():
        mesh.set("filename", f"pieces/{base}")
urdf_path.write_text(ET.tostring(root, encoding="unicode"))
PY

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import re

pieces_dir = Path("${STAGE_DIR}/pieces")
assets_path = Path("${STAGE_DIR}/box_assets.xml")
body_path = Path("${STAGE_DIR}/box_body.xml")
object_prefix = "${OBJECT_NAME}"
robot_height = float("${ROBOT_HEIGHT}")
human_height = float("${HUMAN_HEIGHT}")

def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"piece_{name}"
    return name

scale = robot_height / human_height
scale_str = f"{scale} {scale} {scale}"
meshes = [(f"piece_{sanitize(p.stem)}", p) for p in sorted(pieces_dir.glob("*.obj"))]
if not meshes:
    raise SystemExit("No terrain piece obj files under pieces/.")

asset_lines = ["<mujocoinclude>"]
for mesh_name, mesh_path in meshes:
    asset_lines.append(f'    <mesh name="{mesh_name}" file="{mesh_path.as_posix()}" scale="{scale_str}"/>')
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

SCENE_XML_LOCAL="${STAGE_DIR}/g1_29dof_w_terrain.xml"
cp -f "${RETARGET_ROOT}/models/g1/g1_29dof_w_terrain.xml" "${SCENE_XML_LOCAL}"

"${PYTHON_BIN}" - <<PY
import re
from pathlib import Path

scene_xml = Path("${SCENE_XML_LOCAL}")
meshdir = Path("${STAGE_DIR}/g1/assets").as_posix()

text = scene_xml.read_text()
text = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshdir}"', text, count=1)
text = re.sub(r"\n\\s*<mesh name=\"part_[^\"]+\"[^>]*>", "", text)
text = re.sub(r"\n\\s*<geom name=\"part_[^\"]+\"[^>]*>", "", text)
if "box_assets.xml" not in text:
    text = text.replace("</asset>", '  <include file="box_assets.xml"/>\\n  </asset>', 1)
if "box_body.xml" not in text:
    text = text.replace("</worldbody>", '  <include file="box_body.xml"/>\\n  </worldbody>', 1)
scene_xml.write_text(text)
PY

cp -f "${SCENE_XML_LOCAL}" "${STAGE_DIR}/g1_29dof_w_${OBJECT_NAME}.xml"
cp -f "${SCENE_XML_LOCAL}" "${STAGE_DIR}/g1/g1_29dof_w_terrain.xml"
cp -f "${SCENE_XML_LOCAL}" "${STAGE_DIR}/g1/g1_29dof_w_${OBJECT_NAME}.xml"

OUT_NPZ="${OUT_DIR}/${TASK_NAME}_original.npz"
PY_ALIAS_ROOT="${WORK_ROOT}/py_alias"
mkdir -p "${PY_ALIAS_ROOT}"
ln -sfn "${RETARGET_ROOT}" "${PY_ALIAS_ROOT}/holosoma_retargeting"

cmd=(
  "${PYTHON_BIN}" examples/robot_retarget.py
  --task-type climbing
  --robot g1
  --data_format smplx
  --data_path "${STAGE_DIR}"
  --task-name "${TASK_NAME}"
  --task-config.object-name "${OBJECT_NAME}"
  --task-config.object-dir "${STAGE_DIR}"
  --task-config.scene-xml-file "${SCENE_XML_LOCAL}"
  --task-config.human-z-offset "${HUMAN_Z_OFFSET}"
  --robot-config.robot-urdf-file "${STAGE_DIR}/g1/g1_29dof.urdf"
  --save_dir "${OUT_DIR}"
  --retargeter.q-a-init-idx "${Q_A_INIT_IDX}"
  --retargeter.penetration-tolerance "${PENETRATION_TOLERANCE}"
  --retargeter.step-size "${STEP_SIZE}"
  --save-mode
)

if [[ "${ACTIVATE_OBJ_NON_PENETRATION}" == "1" ]]; then
  cmd+=(--retargeter.activate-obj-non-penetration)
else
  cmd+=(--retargeter.no-activate-obj-non-penetration)
fi

if [[ "${ACTIVATE_FOOT_STICKING}" == "1" ]]; then
  cmd+=(--retargeter.activate-foot-sticking)
else
  cmd+=(--retargeter.no-activate-foot-sticking)
fi

if [[ "${VISUALIZE}" == "1" ]]; then
  cmd+=(--retargeter.visualize)
else
  cmd+=(--retargeter.no-visualize)
fi

if [[ "${DEBUG}" == "1" ]]; then
  cmd+=(--retargeter.debug)
else
  cmd+=(--retargeter.no-debug)
fi

echo "[INFO] source_npz=${SRC_NPZ}"
echo "[INFO] source_obj=${SRC_OBJ}"
echo "[INFO] source_urdf=${SRC_URDF}"
echo "[INFO] source_pieces=${SRC_PIECES_DIR}"
echo "[INFO] stage_dir=${STAGE_DIR}"
echo "[INFO] scene_xml=${SCENE_XML_LOCAL}"
echo "[INFO] object_dir=${STAGE_DIR}"
echo "[INFO] py_alias=${PY_ALIAS_ROOT}/holosoma_retargeting -> ${RETARGET_ROOT}"
echo "[INFO] output_npz=${OUT_NPZ}"
echo "[INFO] debug=${DEBUG} visualize=${VISUALIZE}"
echo "[INFO] human_z_offset=${HUMAN_Z_OFFSET} source=${HUMAN_Z_OFFSET_SOURCE}"
echo "[INFO] q_a_init_idx=${Q_A_INIT_IDX} step_size=${STEP_SIZE} penetration_tol=${PENETRATION_TOLERANCE}"
echo "[INFO] activate_obj_non_penetration=${ACTIVATE_OBJ_NON_PENETRATION} activate_foot_sticking=${ACTIVATE_FOOT_STICKING}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1 command:"
  printf '  %q' "${cmd[@]}"
  echo
  exit 0
fi

(
  cd "${RETARGET_ROOT}"
  PYTHONPATH="${PY_ALIAS_ROOT}:/home/ubuntu/FAR/holosoma/src${PYTHONPATH:+:${PYTHONPATH}}" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "${cmd[@]}"
)

if [[ ! -f "${OUT_NPZ}" ]]; then
  echo "[ERROR] Retarget output not found: ${OUT_NPZ}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<PY
import numpy as np
from pathlib import Path
src = np.load("${SRC_NPZ}")
dst = np.load("${OUT_NPZ}")
src_frames = src["global_joint_positions"].shape[0]
print(f"[INFO] src_frames_raw={src_frames} src_frames_downsampled={src_frames // 4} out_frames={dst['qpos'].shape[0]}")

expected_scale = float("${ROBOT_HEIGHT}") / float("${HUMAN_HEIGHT}")
scale_arr = np.asarray(dst["object_mesh_scale"], dtype=np.float64).reshape(-1) if "object_mesh_scale" in dst else None
scene_xml = str(np.asarray(dst["scene_xml_file"]).reshape(-1)[0]) if "scene_xml_file" in dst else ""
object_urdf = str(np.asarray(dst["object_urdf_path"]).reshape(-1)[0]) if "object_urdf_path" in dst else ""

if scale_arr is None or scale_arr.size < 3:
    raise SystemExit("[ERROR] Missing object_mesh_scale in retarget output npz")
if not np.allclose(scale_arr[:3], expected_scale, rtol=1e-5, atol=1e-5):
    raise SystemExit(
        f"[ERROR] object_mesh_scale mismatch: got={scale_arr[:3].tolist()} expected~={expected_scale:.6f}"
    )
if "_scaled_" not in scene_xml or "_scaled_" not in object_urdf:
    raise SystemExit(
        f"[ERROR] Scaled asset path missing: scene_xml_file={scene_xml}, object_urdf_path={object_urdf}"
    )
if not Path(scene_xml).exists() or not Path(object_urdf).exists():
    raise SystemExit(
        f"[ERROR] Scaled asset file missing on disk: scene_xml_file={scene_xml}, object_urdf_path={object_urdf}"
    )

print(f"[INFO] verified_scale={expected_scale:.6f} object_mesh_scale={scale_arr[:3].tolist()}")
print(f"[INFO] verified_scene_xml={scene_xml}")
print(f"[INFO] verified_object_urdf={object_urdf}")
PY

if [[ "${COPY_TO_MOTION_DIR}" == "1" ]]; then
  mkdir -p "${TARGET_MOTION_DIR}"
  cp -f "${OUT_NPZ}" "${TARGET_MOTION_DIR}/${RESOLVED_SEQ_NAME}.npz"
  echo "[INFO] copied_to=${TARGET_MOTION_DIR}/${RESOLVED_SEQ_NAME}.npz"
fi

echo "[INFO] done"
