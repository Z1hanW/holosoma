#!/usr/bin/env bash
set -euo pipefail

# Copy the 51-clip convex-hull AS solid distillation bank from NFS into this
# repo's data/ds_as_data tree. The bank contains only final-position-success
# clips whose retained contact points are all within 1cm of the convex hull
# surface. It is self-contained and includes real-mesh rollout contact sidecars.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

CH_BANK_NAME=${CH_BANK_NAME:-as_realmesh67000_finalpos_convexsurface51_convexhull}
NFS_CH_BANK=${NFS_CH_BANK:-"/nfs/zzzihanw/ds_as_data/_distill/${CH_BANK_NAME}.tar"}
PULL_CODE=${PULL_CODE:-0}

if [[ "${PULL_CODE}" == "1" ]]; then
  GIT_REMOTE=${GIT_REMOTE:-origin}
  GIT_BRANCH=${GIT_BRANCH:-$(git branch --show-current)}
  if [[ -z "${GIT_BRANCH}" ]]; then
    echo "[ERROR] Could not infer current git branch. Set GIT_BRANCH." >&2
    exit 2
  fi
  echo "[INFO] Pulling ${GIT_REMOTE}/${GIT_BRANCH}"
  git pull --ff-only "${GIT_REMOTE}" "${GIT_BRANCH}"
fi

select_numpy_python3() {
  local candidates=()
  if [[ -n "${PYTHON3_BIN:-}" ]]; then
    candidates+=("${PYTHON3_BIN}")
  fi
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=(
    "/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3"
    "/home/ubuntu/miniconda3/envs/hssim/bin/python3"
    "/home/ubuntu/miniconda3/envs/sim/bin/python3"
    "/opt/conda/bin/python3"
    "$(command -v python3 || true)"
  )

  local py
  for py in "${candidates[@]}"; do
    if [[ -z "${py}" || ! -x "${py}" ]]; then
      continue
    fi
    if "${py}" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
PY
    then
      export PATH="$(dirname "${py}"):${PATH}"
      echo "[INFO] Using python3=${py}"
      return 0
    fi
  done

  echo "[ERROR] Could not find a python3 with numpy. Set PYTHON3_BIN to the training-env python." >&2
  exit 1
}

select_numpy_python3

export CORL_BANK_NAME="${CORL_BANK_NAME:-${CH_BANK_NAME}}"
export NFS_CORL_BANK="${NFS_CORL_BANK:-${NFS_CH_BANK}}"
export LOCAL_BANK_NAME="${LOCAL_BANK_NAME:-${CH_BANK_NAME}}"
export EXPECTED_CLIP_COUNT="${EXPECTED_CLIP_COUNT:-51}"

echo "[INFO] Copying convex-hull 51 bank"
echo "[INFO] NFS_CORL_BANK=${NFS_CORL_BANK}"
echo "[INFO] LOCAL_BANK_NAME=${LOCAL_BANK_NAME}"
echo "[INFO] EXPECTED_CLIP_COUNT=${EXPECTED_CLIP_COUNT}"

bash "${SCRIPT_DIR}/cp_corl.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

python3 - "${SCRIPT_DIR}/data/ds_as_data/${LOCAL_BANK_NAME}" "${EXPECTED_CLIP_COUNT}" <<'PY'
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

bank = Path(sys.argv[1]).expanduser().resolve()
expected = int(sys.argv[2])
map_path = bank / "_clip_object_urdf_map.json"
payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if len(clips) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} clips, found {len(clips)} in {map_path}")

missing: list[str] = []
for clip_id, entry in sorted(clips.items()):
    if not (bank / f"{clip_id}.npz").is_file():
        missing.append(f"{clip_id}: missing top-level npz")
    if not (bank / "_single_slot_motion_bank" / f"{clip_id}.npz").is_file():
        missing.append(f"{clip_id}: missing single-slot npz")
    if isinstance(entry, dict):
        urdf = Path(str(entry.get("object_urdf_path", "")).strip())
        if not urdf.is_file():
            missing.append(f"{clip_id}: missing object URDF {urdf}")
        else:
            root = ET.parse(urdf).getroot()
            visual_meshes = []
            collision_meshes = []
            for mesh_tag in root.findall(".//visual//mesh"):
                raw = str(mesh_tag.get("filename", "")).strip()
                mesh = Path(raw)
                if not mesh.is_absolute():
                    mesh = (urdf.parent / mesh).resolve()
                visual_meshes.append(mesh)
            for mesh_tag in root.findall(".//collision//mesh"):
                raw = str(mesh_tag.get("filename", "")).strip()
                mesh = Path(raw)
                if not mesh.is_absolute():
                    mesh = (urdf.parent / mesh).resolve()
                collision_meshes.append(mesh)
            if not visual_meshes:
                missing.append(f"{clip_id}: URDF has no visual mesh")
            if not collision_meshes:
                missing.append(f"{clip_id}: URDF has no collision mesh")
            for mesh in visual_meshes:
                if not mesh.is_file():
                    missing.append(f"{clip_id}: missing visual mesh {mesh}")
                if "objects_visual_real" not in str(mesh):
                    missing.append(f"{clip_id}: visual mesh is not real mesh copy {mesh}")
            for mesh in collision_meshes:
                if not mesh.is_file():
                    missing.append(f"{clip_id}: missing collision mesh {mesh}")
                if "objects_convex_hull" not in str(mesh):
                    missing.append(f"{clip_id}: collision mesh is not convex hull {mesh}")
            if {str(mesh) for mesh in visual_meshes} == {str(mesh) for mesh in collision_meshes}:
                missing.append(f"{clip_id}: visual and collision meshes should differ")

contact_root = bank / "contact_export_from_teacher_success133_final0p5"
clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root
contact_dirs = [path for path in clips_root.iterdir() if path.is_dir()] if clips_root.is_dir() else []
if len(contact_dirs) != expected:
    missing.append(f"contact dirs={len(contact_dirs)} expected={expected} under {clips_root}")

if missing:
    raise SystemExit("[ERROR] Convex-hull 51 bank validation failed:\n  " + "\n  ".join(missing[:30]))

print(f"[INFO] Convex-hull 51 bank ready: {bank}")
print(f"[INFO] contact_root={contact_root}")
PY
