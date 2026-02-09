#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

IN_ROOT="/data/behave/annotation_30fps"
OUT_ROOT="/data/behave/annotation_30fps_zup"
OBJ_ROOT="/data/behave/objects"
SMPL_MODEL_ROOT="/data/behave/HMR"

python - <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import torch

REPO_ROOT = Path.cwd()
if (REPO_ROOT / "behave").exists():
    sys.path.insert(0, str(REPO_ROOT / "behave"))
    sys.path.insert(0, str(REPO_ROOT))

IN_ROOT = Path("/data/behave/annotation_30fps")
OUT_ROOT = Path("/data/behave/annotation_30fps_zup")
OBJ_ROOT = Path("/data/behave/objects")
SMPL_MODEL_ROOT = Path("/data/behave/HMR")

ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)

from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer  # noqa: E402


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


def _get_obj_name(seq_name: str) -> str:
    parts = seq_name.split("_")
    if len(parts) <= 2:
        raise ValueError(f"Cannot parse object name from sequence: {seq_name}")
    return parts[2]


def _load_centered_object_mesh(obj_name: str) -> trimesh.Trimesh:
    mesh_path = OBJ_ROOT / obj_name / f"{obj_name}_f1000.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    center = np.mean(mesh.vertices, axis=0)
    mesh.vertices = mesh.vertices - center
    return mesh


def _compute_min_z(
    smpl_poses: np.ndarray,
    smpl_betas: np.ndarray,
    smpl_trans: np.ndarray,
    obj_rotmats: np.ndarray,
    obj_trans: np.ndarray,
    obj_name: str,
    gender: str,
    *,
    already_rotated: bool,
) -> float:
    smpl = SMPL_Layer(
        center_idx=0,
        gender=gender,
        num_betas=int(smpl_betas.shape[1]),
        model_root=str(SMPL_MODEL_ROOT),
        hands=True,
    )
    smpl = smpl.to(torch.device("cpu"))

    with torch.no_grad():
        verts, _, _, _ = smpl(
            torch.from_numpy(smpl_poses),
            th_betas=torch.from_numpy(smpl_betas),
            th_trans=torch.from_numpy(smpl_trans),
        )
    human_verts = verts.detach().cpu().numpy()

    obj_mesh = _load_centered_object_mesh(obj_name)
    base = obj_mesh.vertices.astype(np.float32)
    obj_verts = (base[None, :, :] @ obj_rotmats.transpose(0, 2, 1)) + obj_trans[:, None, :]

    if not already_rotated:
        human_verts = human_verts @ ROT.T
        obj_verts = obj_verts @ ROT.T

    min_z = float(
        min(
            np.min(human_verts[..., 2]),
            np.min(obj_verts[..., 2]),
        )
    )
    return min_z


def _compute_pelvis_delta(
    smpl_poses: np.ndarray,
    smpl_betas: np.ndarray,
    smpl_trans: np.ndarray,
    smpl_poses_rot: np.ndarray,
    smpl_trans_rot: np.ndarray,
    gender: str,
) -> np.ndarray:
    smpl = SMPL_Layer(
        center_idx=0,
        gender=gender,
        num_betas=int(smpl_betas.shape[1]),
        model_root=str(SMPL_MODEL_ROOT),
        hands=True,
    )
    smpl = smpl.to(torch.device("cpu"))

    with torch.no_grad():
        verts, joints, _, _ = smpl(
            torch.from_numpy(smpl_poses),
            th_betas=torch.from_numpy(smpl_betas),
            th_trans=torch.from_numpy(smpl_trans),
        )
        _, joints_rot, _, _ = smpl(
            torch.from_numpy(smpl_poses_rot),
            th_betas=torch.from_numpy(smpl_betas),
            th_trans=torch.from_numpy(smpl_trans_rot),
        )
    pelvis = joints.detach().cpu().numpy()[:, 0, :]
    pelvis_expected = pelvis @ ROT.T
    pelvis_rot = joints_rot.detach().cpu().numpy()[:, 0, :]
    delta = pelvis_expected - pelvis_rot
    return np.mean(delta, axis=0)


def _process_sequence(seq_dir: Path) -> None:
    out_dir = OUT_ROOT / seq_dir.name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    obj_name = _get_obj_name(seq_dir.name).lower()
    if "box" not in obj_name or obj_name == "toolbox":
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
    betas = smpl.get("betas")
    gender = str(smpl.get("gender", "male"))
    if poses is None or trans is None:
        raise ValueError(f"Missing poses/trans in {smpl_path}")
    poses = np.asarray(poses)
    trans = np.asarray(trans)
    betas = np.asarray(betas) if betas is not None else np.zeros((poses.shape[0], 10), dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] not in (72, 156):
        raise ValueError(f"Unsupported poses shape {poses.shape} in {smpl_path}")

    if betas.ndim == 1:
        betas = betas[None, :]
    if betas.shape[0] == 1 and poses.shape[0] > 1:
        betas = np.repeat(betas, poses.shape[0], axis=0)

    obj_trans = obj.get("trans")
    if obj_trans is None:
        obj_trans = obj.get("obj_trans")
    if obj_trans is None:
        raise ValueError(f"Missing object translation in {obj_path}")
    obj_trans = np.asarray(obj_trans)

    obj_rotmats = None
    if "angles" in obj:
        obj_angles = np.asarray(obj["angles"])
        obj_rotmats = Rotation.from_rotvec(obj_angles).as_matrix().astype(np.float32)
    elif "angle" in obj:
        obj_angles = np.asarray(obj["angle"])
        obj_rotmats = Rotation.from_rotvec(obj_angles).as_matrix().astype(np.float32)
    elif "obj_rot" in obj:
        obj_rots = np.asarray(obj["obj_rot"])
        if obj_rots.ndim == 2 and obj_rots.shape[1] == 3:
            obj_rotmats = Rotation.from_rotvec(obj_rots).as_matrix().astype(np.float32)
        elif obj_rots.ndim == 3:
            obj_rotmats = obj_rots.astype(np.float32)
        else:
            raise ValueError(f"Unsupported obj_rot shape {obj_rots.shape} in {obj_path}")
    elif "rot" in obj:
        obj_rots = np.asarray(obj["rot"])
        if obj_rots.ndim == 2 and obj_rots.shape[1] == 3:
            obj_rotmats = Rotation.from_rotvec(obj_rots).as_matrix().astype(np.float32)
        elif obj_rots.ndim == 3:
            obj_rotmats = obj_rots.astype(np.float32)
        else:
            raise ValueError(f"Unsupported rot shape {obj_rots.shape} in {obj_path}")
    else:
        raise ValueError(f"No object rotation found in {obj_path}")

    obj_rotmats_rot = ROT[None, :, :] @ obj_rotmats
    obj_trans_rot = (obj_trans @ ROT.T).astype(np.float32, copy=False)

    global_orient = poses[:, :3]
    global_orient = _rotate_rotvec(global_orient)
    poses_rot = poses.copy()
    poses_rot[:, :3] = global_orient

    trans_rot = (trans @ ROT.T).astype(np.float32, copy=False)

    delta = _compute_pelvis_delta(
        poses.astype(np.float32),
        betas.astype(np.float32),
        trans.astype(np.float32),
        poses_rot.astype(np.float32),
        trans_rot.astype(np.float32),
        gender,
    )
    trans_rot = trans_rot + delta[None, :]
    obj_trans_rot = obj_trans_rot + delta[None, :]

    obj_name = _get_obj_name(seq_dir.name)
    min_z = _compute_min_z(
        poses_rot.astype(np.float32),
        betas.astype(np.float32),
        trans_rot.astype(np.float32),
        obj_rotmats_rot.astype(np.float32),
        obj_trans_rot.astype(np.float32),
        obj_name,
        gender,
        already_rotated=True,
    )

    trans_rot = trans_rot.copy()
    obj_trans_rot = obj_trans_rot.copy()
    trans_rot[:, 2] -= min_z
    obj_trans_rot[:, 2] -= min_z

    smpl["poses"] = poses_rot
    smpl["trans"] = trans_rot.astype(np.float32, copy=False)

    if "angles" in obj:
        obj["angles"] = Rotation.from_matrix(obj_rotmats_rot).as_rotvec().astype(np.float32, copy=False)
    elif "angle" in obj:
        obj["angle"] = Rotation.from_matrix(obj_rotmats_rot).as_rotvec().astype(np.float32, copy=False)
    elif "obj_rot" in obj:
        obj["obj_rot"] = obj_rotmats_rot.astype(np.float32, copy=False)
    elif "rot" in obj:
        obj["rot"] = obj_rotmats_rot.astype(np.float32, copy=False)

    obj["trans"] = obj_trans_rot
    if "obj_trans" in obj:
        obj["obj_trans"] = obj_trans_rot

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_npz(out_dir / "smpl_fit_all.npz", smpl)
    _save_npz(out_dir / "object_fit_all.npz", obj)
    if info_path.exists():
        shutil.copy2(info_path, out_dir / "info.json")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"Missing input root: {IN_ROOT}")
    if not OBJ_ROOT.exists():
        raise FileNotFoundError(f"Missing object root: {OBJ_ROOT}")
    if not SMPL_MODEL_ROOT.exists():
        raise FileNotFoundError(f"Missing SMPL model root: {SMPL_MODEL_ROOT}")
    for seq_dir in sorted(p for p in IN_ROOT.iterdir() if p.is_dir()):
        _process_sequence(seq_dir)


if __name__ == "__main__":
    main()
PY
