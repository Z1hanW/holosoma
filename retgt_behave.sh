#!/usr/bin/env bash
set -euo pipefail

IN_ROOT="/data/behave/annotation_30fps"
OUT_ROOT="/data/behave/annotation_30fps_zup"

python - <<'PY'
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

IN_ROOT = Path("/data/behave/annotation_30fps")
OUT_ROOT = Path("/data/behave/annotation_30fps_zup")

ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def _load_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _save_npz(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _rotate_rotvec(rotvecs: np.ndarray) -> np.ndarray:
    rotmats = Rotation.from_rotvec(rotvecs).as_matrix()
    rotmats = ROT[None, :, :] @ rotmats
    return Rotation.from_matrix(rotmats).as_rotvec().astype(rotvecs.dtype, copy=False)


def _rotate_rotmat(rotmats: np.ndarray) -> np.ndarray:
    return (ROT[None, :, :] @ rotmats).astype(rotmats.dtype, copy=False)


def _apply_z_offset(smpl_trans: np.ndarray, obj_trans: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    min_z = float(np.min([smpl_trans[:, 2].min(), obj_trans[:, 2].min()]))
    smpl_trans = smpl_trans.copy()
    obj_trans = obj_trans.copy()
    smpl_trans[:, 2] -= min_z
    obj_trans[:, 2] -= min_z
    return smpl_trans, obj_trans, min_z


def _process_sequence(seq_dir: Path) -> None:
    out_dir = OUT_ROOT / seq_dir.name
    if out_dir.exists():
        return

    smpl_path = seq_dir / "smpl_fit_all.npz"
    obj_path = seq_dir / "object_fit_all.npz"
    info_path = seq_dir / "info.json"
    if not smpl_path.exists() or not obj_path.exists():
        return

    smpl = _load_npz(smpl_path)
    obj = _load_npz(obj_path)

    poses = smpl.get("poses")
    trans = smpl.get("trans")
    if poses is None or trans is None:
        raise ValueError(f"Missing poses/trans in {smpl_path}")
    poses = np.asarray(poses)
    trans = np.asarray(trans)
    if poses.ndim != 2 or poses.shape[1] not in (72, 156):
        raise ValueError(f"Unsupported poses shape {poses.shape} in {smpl_path}")

    obj_angles = None
    obj_rots = None
    obj_trans = obj.get("trans")
    if obj_trans is None:
        obj_trans = obj.get("obj_trans")
    if obj_trans is None:
        raise ValueError(f"Missing object translation in {obj_path}")
    obj_trans = np.asarray(obj_trans)

    if "angles" in obj:
        obj_angles = np.asarray(obj["angles"])
        obj_angles = _rotate_rotvec(obj_angles)
        obj["angles"] = obj_angles
    elif "angle" in obj:
        obj_angles = np.asarray(obj["angle"])
        obj_angles = _rotate_rotvec(obj_angles)
        obj["angle"] = obj_angles
    elif "obj_rot" in obj:
        obj_rots = np.asarray(obj["obj_rot"])
        if obj_rots.ndim == 2 and obj_rots.shape[1] == 3:
            obj_angles = _rotate_rotvec(obj_rots)
            obj["obj_rot"] = Rotation.from_rotvec(obj_angles).as_matrix().astype(obj_rots.dtype, copy=False)
        elif obj_rots.ndim == 3:
            obj["obj_rot"] = _rotate_rotmat(obj_rots)
        else:
            raise ValueError(f"Unsupported obj_rot shape {obj_rots.shape} in {obj_path}")
    elif "rot" in obj:
        obj_rots = np.asarray(obj["rot"])
        if obj_rots.ndim == 2 and obj_rots.shape[1] == 3:
            obj_angles = _rotate_rotvec(obj_rots)
            obj["rot"] = Rotation.from_rotvec(obj_angles).as_matrix().astype(obj_rots.dtype, copy=False)
        elif obj_rots.ndim == 3:
            obj["rot"] = _rotate_rotmat(obj_rots)
        else:
            raise ValueError(f"Unsupported rot shape {obj_rots.shape} in {obj_path}")
    else:
        raise ValueError(f"No object rotation found in {obj_path}")

    obj_trans = (obj_trans @ ROT.T).astype(obj_trans.dtype, copy=False)

    global_orient = poses[:, :3]
    global_orient = _rotate_rotvec(global_orient)
    poses = poses.copy()
    poses[:, :3] = global_orient

    trans = (trans @ ROT.T).astype(trans.dtype, copy=False)
    trans, obj_trans, _ = _apply_z_offset(trans, obj_trans)

    smpl["poses"] = poses
    smpl["trans"] = trans
    obj["trans"] = obj_trans
    if "obj_trans" in obj:
        obj["obj_trans"] = obj_trans

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(out_dir / "smpl_fit_all.npz", smpl)
    _save_npz(out_dir / "object_fit_all.npz", obj)
    if info_path.exists():
        shutil.copy2(info_path, out_dir / "info.json")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"Missing input root: {IN_ROOT}")
    for seq_dir in sorted(p for p in IN_ROOT.iterdir() if p.is_dir()):
        _process_sequence(seq_dir)


if __name__ == "__main__":
    main()
PY
