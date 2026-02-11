#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
BEHAVE_ROOT = Path(__file__).resolve().parent
for path in (BEHAVE_ROOT, REPO_ROOT / "viser" / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import viser  # type: ignore[import-not-found]

from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer  # noqa: E402

try:
    from data.const import OBJ_NAMES  # noqa: E402
except Exception:
    OBJ_NAMES = []


SIMPLIFIED_MESH = {
    "backpack": "backpack/backpack_f1000.ply",
    "basketball": "basketball/basketball_f1000.ply",
    "boxlarge": "boxlarge/boxlarge_f1000.ply",
    "boxtiny": "boxtiny/boxtiny_f1000.ply",
    "boxlong": "boxlong/boxlong_f1000.ply",
    "boxsmall": "boxsmall/boxsmall_f1000.ply",
    "boxmedium": "boxmedium/boxmedium_f1000.ply",
    "chairblack": "chairblack/chairblack_f2500.ply",
    "chairwood": "chairwood/chairwood_f2500.ply",
    "monitor": "monitor/monitor_closed_f1000.ply",
    "keyboard": "keyboard/keyboard_f1000.ply",
    "plasticcontainer": "plasticcontainer/plasticcontainer_f1000.ply",
    "stool": "stool/stool_f1000.ply",
    "tablesquare": "tablesquare/tablesquare_f2000.ply",
    "toolbox": "toolbox/toolbox_f1000.ply",
    "suitcase": "suitcase/suitcase_f1000.ply",
    "tablesmall": "tablesmall/tablesmall_f1000.ply",
    "yogamat": "yogamat/yogamat_f1000.ply",
    "yogaball": "yogaball/yogaball_f1000.ply",
    "trashbin": "trashbin/trashbin_f1000.ply",
}


def _decode_scalar(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _extract_key(data: Dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _guess_obj_name_from_stem(stem: str) -> str | None:
    parts = stem.split("_")
    for part in parts:
        if part in OBJ_NAMES:
            return part
    if len(parts) > 2:
        return parts[2]
    return None


def _normalize_pose_array(poses: np.ndarray) -> np.ndarray:
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses


def _smpl_to_smplh(poses: np.ndarray) -> np.ndarray:
    if poses.shape[-1] != 72:
        return poses
    batch = poses.shape[0]
    out = np.zeros((batch, 156), dtype=poses.dtype)
    out[:, :69] = poses[:, :69]
    out[:, 111:114] = poses[:, 69:]
    return out


def _extract_smpl_params(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses = _extract_key(
        data,
        (
            "poses",
            "pose",
            "smpl_poses",
            "smplh_poses",
            "smpl_pose",
            "smplh_pose",
        ),
    )
    if poses is None:
        global_orient = _extract_key(data, ("global_orient", "root_orient"))
        body_pose = _extract_key(data, ("body_pose", "pose_body"))
        if global_orient is None or body_pose is None:
            raise ValueError("SMPL pose not found in annotation file.")
        global_orient = _as_numpy(global_orient)
        body_pose = _as_numpy(body_pose)
        if global_orient.ndim == 1:
            global_orient = global_orient[None, :]
        if body_pose.ndim == 1:
            body_pose = body_pose[None, :]
        poses = np.concatenate([global_orient, body_pose], axis=-1)
    poses = _as_numpy(poses)
    poses = _normalize_pose_array(poses)
    if poses.shape[-1] == 72:
        poses = _smpl_to_smplh(poses)
    if poses.shape[-1] != 156:
        raise ValueError(f"Unsupported SMPL pose shape: {poses.shape}")

    betas = _extract_key(data, ("betas", "beta", "smpl_betas", "smplh_betas"))
    if betas is None:
        betas = np.zeros((1, 10), dtype=np.float32)
    betas = _as_numpy(betas).astype(np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]

    trans = _extract_key(data, ("trans", "transl", "smpl_trans", "smplh_trans", "root_trans"))
    if trans is None:
        trans = np.zeros((poses.shape[0], 3), dtype=np.float32)
    trans = _as_numpy(trans).astype(np.float32)
    if trans.ndim == 1:
        trans = trans[None, :]

    return poses.astype(np.float32), betas, trans


def _normalize_joint_positions(joints: np.ndarray) -> np.ndarray:
    joints = _as_numpy(joints)
    if joints.ndim == 2 and joints.shape[1] % 3 == 0:
        joint_count = joints.shape[1] // 3
        joints = joints.reshape(joints.shape[0], joint_count, 3)
    elif joints.ndim == 2 and joints.shape[0] == 3:
        joints = joints.T[None, :, :]
    elif joints.ndim == 2 and joints.shape[1] == 3:
        joints = joints[None, :, :]
    elif joints.ndim == 3 and joints.shape[-1] == 3:
        pass
    elif joints.ndim == 3 and joints.shape[1] == 3:
        joints = np.transpose(joints, (0, 2, 1))
    else:
        raise ValueError(f"Unsupported joints shape: {joints.shape}")
    return joints.astype(np.float32)


def _extract_joint_positions(data: Dict[str, Any]) -> np.ndarray | None:
    joints = _extract_key(
        data,
        (
            "global_joint_positions",
            "joint_positions",
            "joints",
            "j3d",
            "J",
            "J3D",
        ),
    )
    if joints is None:
        return None
    return _normalize_joint_positions(_as_numpy(joints))


def _extract_object_params(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    obj_rot = _extract_key(data, ("obj_rot", "obj_rots", "object_rot", "object_rots"))
    obj_trans = _extract_key(data, ("obj_trans", "obj_transl", "object_trans", "object_transl"))
    if obj_rot is None or obj_trans is None:
        raise ValueError("Object rotation/translation not found in annotation file.")
    obj_rot = _as_numpy(obj_rot)
    obj_trans = _as_numpy(obj_trans).astype(np.float32)

    if obj_rot.ndim == 2 and obj_rot.shape == (3, 3):
        obj_rot = obj_rot[None, :, :]
    elif obj_rot.ndim == 2 and obj_rot.shape[1] == 3:
        from scipy.spatial.transform import Rotation

        obj_rot = Rotation.from_rotvec(obj_rot).as_matrix().astype(np.float32)
    elif obj_rot.ndim == 1 and obj_rot.shape[0] == 3:
        from scipy.spatial.transform import Rotation

        obj_rot = Rotation.from_rotvec(obj_rot[None, :]).as_matrix().astype(np.float32)
    elif obj_rot.ndim == 3 and obj_rot.shape[-2:] == (3, 3):
        obj_rot = obj_rot.astype(np.float32)
    elif obj_rot.ndim == 2 and obj_rot.shape[1] == 9:
        obj_rot = obj_rot.reshape(-1, 3, 3).astype(np.float32)
    else:
        raise ValueError(f"Unsupported object rotation shape: {obj_rot.shape}")

    if obj_trans.ndim == 1:
        obj_trans = obj_trans[None, :]
    obj_trans = obj_trans.astype(np.float32)
    return obj_rot, obj_trans


def _load_object_mesh(
    objects_root: Path,
    obj_name: str,
    override_mesh: Path | None,
    *,
    use_simplified: bool,
) -> trimesh.Trimesh:
    if override_mesh is not None:
        mesh_path = override_mesh
    elif use_simplified:
        mesh_path = objects_root / obj_name / f"{obj_name}_f1000.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Simplified object mesh not found: {mesh_path}")
    else:
        mesh_path = objects_root / obj_name / f"{obj_name}.obj"
        if not mesh_path.exists():
            alt = objects_root / obj_name / f"{obj_name}.ply"
            if alt.exists():
                mesh_path = alt
    if not mesh_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    center = np.mean(mesh.vertices, axis=0)
    mesh.vertices = mesh.vertices - center
    return mesh


def _make_body_part_colors(joint_count: int) -> np.ndarray:
    if joint_count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    colors = np.zeros((joint_count, 3), dtype=np.uint8)

    torso_color = np.array([255, 215, 0], dtype=np.uint8)
    left_arm_color = np.array([80, 160, 255], dtype=np.uint8)
    right_arm_color = np.array([255, 90, 90], dtype=np.uint8)
    left_leg_color = np.array([80, 200, 120], dtype=np.uint8)
    right_leg_color = np.array([170, 100, 255], dtype=np.uint8)
    default_color = np.array([200, 200, 200], dtype=np.uint8)

    if joint_count >= 22:
        torso = [0, 3, 6, 9, 12, 15]
        left_leg = [1, 4, 7, 10]
        right_leg = [2, 5, 8, 11]
        left_arm = [13, 16, 18, 20]
        right_arm = [14, 17, 19, 21]
        if joint_count >= 24:
            left_arm.append(22)
            right_arm.append(23)
        for idx in torso:
            if idx < joint_count:
                colors[idx] = torso_color
        for idx in left_leg:
            if idx < joint_count:
                colors[idx] = left_leg_color
        for idx in right_leg:
            if idx < joint_count:
                colors[idx] = right_leg_color
        for idx in left_arm:
            if idx < joint_count:
                colors[idx] = left_arm_color
        for idx in right_arm:
            if idx < joint_count:
                colors[idx] = right_arm_color
        for idx in range(joint_count):
            if not colors[idx].any():
                colors[idx] = default_color
        return colors

    for idx in range(joint_count):
        hue = idx / max(1, joint_count - 1)
        colors[idx] = np.array(
            [int(255 * (1 - hue)), int(128 + 127 * hue), int(255 * hue)], dtype=np.uint8
        )
    return colors


def _collect_npz_paths(path: Path, recursive: bool) -> Tuple[list[str], Dict[str, Path]]:
    if path.is_file():
        return [path.stem], {path.stem: path}
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    paths = sorted(path.rglob("*.npz") if recursive else path.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in: {path}")
    labels: list[str] = []
    label_to_path: Dict[str, Path] = {}
    for p in paths:
        label = p.relative_to(path).as_posix() if path in p.parents else p.stem
        if label in label_to_path:
            label = p.stem
        labels.append(label)
        label_to_path[label] = p
    return labels, label_to_path


def _collect_seq_dirs(path: Path) -> Tuple[list[str], Dict[str, Path]]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation root not found: {path}")
    seq_dirs = sorted([p for p in path.iterdir() if p.is_dir()])
    labels: list[str] = []
    label_to_path: Dict[str, Path] = {}
    for p in seq_dirs:
        parts = p.name.split("_")
        if len(parts) <= 2:
            continue
        obj_name = parts[2].lower()
        if "box" not in obj_name or obj_name == "toolbox":
            continue
        if (p / "object_fit_all.npz").exists() and (p / "smpl_fit_all.npz").exists():
            label = p.name
            labels.append(label)
            label_to_path[label] = p
    if not labels:
        raise FileNotFoundError(f"No sequences with object_fit_all.npz and smpl_fit_all.npz in: {path}")
    return labels, label_to_path


def _load_annotation_pair(seq_dir: Path) -> Dict[str, Any]:
    smpl_path = seq_dir / "smpl_fit_all.npz"
    obj_path = seq_dir / "object_fit_all.npz"
    smpl = _load_npz(smpl_path)
    obj = _load_npz(obj_path)
    if "obj_trans" not in obj:
        for key in ("trans", "transl", "translation"):
            if key in obj:
                obj["obj_trans"] = obj.pop(key)
                break
    if "obj_rot" not in obj:
        for key in ("rot", "rots", "rotation", "angle", "angles"):
            if key in obj:
                obj["obj_rot"] = obj.pop(key)
                break
    merged = dict(smpl)
    merged.update(obj)
    merged["seq_name"] = seq_dir.name
    return merged


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _decode_frame_times(frame_times: Any | None) -> list[str] | None:
    if frame_times is None:
        return None
    arr = _as_numpy(frame_times)
    if arr.size == 0:
        return None
    if arr.dtype == object:
        return [str(_decode_scalar(x)) for x in arr.tolist()]
    if np.issubdtype(arr.dtype, np.bytes_):
        return [x.decode("utf-8") for x in arr.tolist()]
    if np.issubdtype(arr.dtype, np.floating):
        return [f"{float(x):.3f}" for x in arr.tolist()]
    return [str(x) for x in arr.tolist()]


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _safe_set_prop(handle: Any, name: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        impl = getattr(handle, "_impl", None)
        if impl is not None and hasattr(impl, "props") and hasattr(handle, "_queue_update"):
            current = getattr(impl.props, name, None)
            cast_value = value
            if isinstance(current, np.ndarray) and hasattr(current, "dtype"):
                if cast_value.dtype != current.dtype:
                    cast_value = cast_value.astype(current.dtype)
                if hasattr(current, "shape") and cast_value.shape == current.shape:
                    current[:] = cast_value
                else:
                    setattr(impl.props, name, cast_value.copy())
            else:
                setattr(impl.props, name, np.asarray(cast_value))
            handle._queue_update(name, cast_value)
            return
        current = getattr(handle, name, None)
        if isinstance(current, tuple):
            flat = value.reshape(-1).tolist()
            value = tuple(float(x) for x in flat[: len(current)])
        elif isinstance(current, (float, int, bool)) and value.size == 1:
            value = float(value.reshape(-1)[0])
    setattr(handle, name, value)


def _build_sequence(
    data: Dict[str, Any],
    objects_root: Path | None,
    smpl_model_root: Path | None,
    override_obj_name: str | None,
    override_mesh: Path | None,
    gender_override: str | None,
    device_name: str,
    stride: int,
    max_frames: int | None,
    *,
    use_simplified_mesh: bool,
) -> Dict[str, Any]:
    joints = _extract_joint_positions(data)
    obj_rot = None
    obj_trans = None
    try:
        obj_rot, obj_trans = _extract_object_params(data)
    except Exception:
        obj_rot = None
        obj_trans = None

    gender = _decode_scalar(_extract_key(data, ("gender",))) or gender_override or "male"
    gender = gender.lower()
    if gender not in ("male", "female"):
        gender = gender_override or "male"

    obj_name = override_obj_name
    if obj_name is None and use_simplified_mesh:
        seq_name = _decode_scalar(_extract_key(data, ("seq_name", "sequence", "name", "seq")))
        if seq_name is not None:
            parts = str(seq_name).split("_")
            if len(parts) > 2:
                obj_name = parts[2]
    if obj_name is None:
        meta = _extract_key(data, ("meta", "metadata"))
        if meta is not None:
            meta_list = _as_numpy(meta).tolist()
            if isinstance(meta_list, list) and len(meta_list) >= 3:
                obj_name = _decode_scalar(meta_list[2])
    obj_name = obj_name or _decode_scalar(_extract_key(data, ("obj_name", "object_name", "cat", "object")))
    obj_name = obj_name or _guess_obj_name_from_stem(str(_decode_scalar(_extract_key(data, ("seq_name", "sequence")))) or "")
    if obj_name is None:
        obj_name = _guess_obj_name_from_stem(str(_decode_scalar(_extract_key(data, ("name", "seq")))) or "")
    if obj_name is None:
        raise ValueError("Object name not found in annotation file. Pass --object-name to override.")

    has_object = obj_rot is not None and obj_trans is not None
    mesh = None
    if has_object:
        if objects_root is None:
            raise FileNotFoundError("Objects root not found. Pass --objects-root or --dataset-root.")
        mesh = _load_object_mesh(objects_root, obj_name, override_mesh, use_simplified=use_simplified_mesh)

    frame_times = _decode_frame_times(_extract_key(data, ("frame_times", "frames", "frame_ids")))

    if joints is None:
        if smpl_model_root is None:
            raise FileNotFoundError(
                "SMPL model root not found. Pass --smpl-model-root or set SMPL_MODEL_PATH/SMPLH_MODEL_PATH."
            )
        poses, betas, trans = _extract_smpl_params(data)
        num_frames = poses.shape[0]
        if betas.shape[0] == 1 and num_frames > 1:
            betas = np.repeat(betas, num_frames, axis=0)
        if obj_rot is not None and obj_trans is not None:
            num_frames = min(num_frames, betas.shape[0], trans.shape[0], obj_rot.shape[0], obj_trans.shape[0])
        else:
            num_frames = min(num_frames, betas.shape[0], trans.shape[0])
    else:
        num_frames = joints.shape[0]
        if obj_rot is not None and obj_trans is not None:
            num_frames = min(num_frames, obj_rot.shape[0], obj_trans.shape[0])
    if frame_times is not None:
        num_frames = min(num_frames, len(frame_times))

    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    if stride <= 0:
        stride = 1

    idx = np.arange(0, num_frames, stride)
    if joints is None:
        poses = poses[idx]
        betas = betas[idx]
        trans = trans[idx]
    else:
        joints = joints[idx]
    if obj_rot is not None and obj_trans is not None:
        obj_rot = obj_rot[idx]
        obj_trans = obj_trans[idx]
    if frame_times is not None:
        frame_times = [frame_times[i] for i in idx]

    if joints is None:
        device = _resolve_device(device_name)
        import torch

        smpl = SMPL_Layer(
            center_idx=0,
            gender=gender,
            num_betas=int(betas.shape[1]),
            model_root=str(smpl_model_root),
            hands=True,
        )
        smpl = smpl.to(device)

        with torch.no_grad():
            _, smpl_joints, _, _ = smpl(
                torch.from_numpy(poses).to(device=device),
                th_betas=torch.from_numpy(betas).to(device=device),
                th_trans=torch.from_numpy(trans).to(device=device),
            )
        joints = smpl_joints.detach().cpu().numpy().astype(np.float32)

    obj_faces = None
    obj_verts = None
    if has_object and mesh is not None:
        obj_faces = mesh.faces.astype(np.int32)
        base_verts = mesh.vertices.astype(np.float32)
        obj_verts = (base_verts[None, :, :] @ obj_rot.transpose(0, 2, 1)) + obj_trans[:, None, :]

    return {
        "joints": joints.astype(np.float32),
        "frame_times": frame_times,
        "n_frames": int(joints.shape[0]),
        "gender": gender,
        "obj_name": obj_name,
        "obj_verts": obj_verts.astype(np.float32) if obj_verts is not None else None,
        "obj_faces": obj_faces,
        "has_object": bool(has_object and obj_verts is not None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Viser viewer for BEHAVE 30fps annotation npz files.")
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to a 30fps annotation .npz file, a folder of .npz files, or the annotation_30fps root.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively for .npz files.")
    parser.add_argument("--annotation-root", action="store_true", help="Interpret input as annotation_30fps root.")
    parser.add_argument("--dataset-root", type=str, default=None, help="BEHAVE dataset root path.")
    parser.add_argument("--objects-root", type=str, default=None, help="Path to BEHAVE objects folder.")
    parser.add_argument("--object-name", type=str, default=None, help="Override object name.")
    parser.add_argument("--object-mesh", type=str, default=None, help="Override object mesh path.")
    parser.add_argument("--smpl-model-root", type=str, default=None, help="Path to SMPLH model root.")
    parser.add_argument("--gender", type=str, default=None, help="Override gender (male/female).")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (auto/cpu/cuda).")
    parser.add_argument("--stride", type=int, default=1, help="Stride for frame subsampling.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames.")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
    parser.add_argument("--show-object-frame", action="store_true", help="Show object frame axes.")
    parser.add_argument("--no-autoplay", dest="autoplay", action="store_false", help="Start paused.")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Disable looping.")
    parser.set_defaults(autoplay=True, loop=True)
    args = parser.parse_args()

    npz_root = Path(args.npz_path).expanduser().resolve()
    if args.annotation_root:
        labels, label_to_path = _collect_seq_dirs(npz_root)
    else:
        labels, label_to_path = _collect_npz_paths(npz_root, args.recursive)

    if args.dataset_root:
        objects_root = Path(args.dataset_root).expanduser().resolve() / "objects"
    elif args.objects_root:
        objects_root = Path(args.objects_root).expanduser().resolve()
    else:
        objects_root = None
    if objects_root is not None and not objects_root.exists():
        objects_root = None

    if args.smpl_model_root:
        smpl_model_root = Path(args.smpl_model_root).expanduser().resolve()
    else:
        env_path = (
            os.environ.get("SMPL_MODEL_PATH")
            or os.environ.get("SMPLH_MODEL_PATH")
            or os.environ.get("SMPLX_MODEL_PATH")
        )
        smpl_model_root = Path(env_path).expanduser().resolve() if env_path else None
    if smpl_model_root is not None and not smpl_model_root.exists():
        smpl_model_root = None
    if smpl_model_root is not None and smpl_model_root.is_file():
        smpl_model_root = smpl_model_root.parent

    override_mesh = Path(args.object_mesh).expanduser().resolve() if args.object_mesh else None

    cache: Dict[str, Dict[str, Any]] = {}

    def _load_label(label: str) -> Dict[str, Any]:
        if label in cache:
            return cache[label]
        if args.annotation_root:
            data = _load_annotation_pair(label_to_path[label])
        else:
            data = _load_npz(label_to_path[label])
        seq = _build_sequence(
            data,
            objects_root=objects_root,
            smpl_model_root=smpl_model_root,
            override_obj_name=args.object_name,
            override_mesh=override_mesh,
            gender_override=args.gender,
            device_name=args.device,
            stride=args.stride,
            max_frames=args.max_frames,
            use_simplified_mesh=args.annotation_root,
        )
        cache[label] = seq
        return seq

    active_label = labels[0]
    state = _load_label(active_label)

    server = viser.ViserServer()
    if not args.no_grid:
        server.scene.add_grid("/grid", width=8, height=8, position=(0.0, 0.0, 0.0))

    joint_colors = _make_body_part_colors(int(state["joints"].shape[1]))
    joint_handle = server.scene.add_point_cloud(
        "/joints",
        points=state["joints"][0],
        colors=joint_colors,
        point_size=0.035,
        point_shape="circle",
    )

    object_handle = None
    if state.get("has_object"):
        object_handle = server.scene.add_mesh_simple(
            "/object",
            vertices=state["obj_verts"][0],
            faces=state["obj_faces"],
            color=(120, 180, 220),
            flat_shading=False,
        )

    with server.gui.add_folder("Sequence"):
        seq_dropdown = server.gui.add_dropdown("Sequence", options=labels, initial_value=active_label)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, int(state["n_frames"]) - 1),
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_input = server.gui.add_number("FPS", initial_value=int(args.fps), min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=args.loop)

    with server.gui.add_folder("Display"):
        show_joints_cb = server.gui.add_checkbox("Show joints", initial_value=True)
        show_object_cb = server.gui.add_checkbox("Show object", initial_value=bool(state.get("has_object")))

    info_md = server.gui.add_markdown("")

    playing = {"flag": bool(args.autoplay)}
    updating_slider = {"flag": False}

    def _update_info(frame_idx: int) -> None:
        frame_times = state.get("frame_times")
        if frame_times is None:
            info_md.content = (
                f"Sequence: `{seq_dropdown.value}` | frame {frame_idx}/{state['n_frames'] - 1} "
                f"| obj={state['obj_name']} | gender={state['gender']}"
            )
        else:
            info_md.content = (
                f"Sequence: `{seq_dropdown.value}` | frame {frame_idx}/{state['n_frames'] - 1} "
                f"| time={frame_times[frame_idx]} | obj={state['obj_name']} | gender={state['gender']}"
            )

    def _apply_frame(frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, state["n_frames"] - 1))
        with server.atomic():
            _safe_set_prop(joint_handle, "points", state["joints"][frame_idx])
            _safe_set_prop(joint_handle, "visible", bool(show_joints_cb.value))
            if object_handle is not None and state.get("has_object"):
                _safe_set_prop(object_handle, "vertices", state["obj_verts"][frame_idx])
            _safe_set_prop(object_handle, "visible", bool(show_object_cb.value))
        _update_info(frame_idx)

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_slider["flag"]:
            return
        _apply_frame(int(frame_slider.value))

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]

    @seq_dropdown.on_update
    def _(_evt) -> None:
        nonlocal state, object_handle
        label = str(seq_dropdown.value)
        state = _load_label(label)
        new_colors = _make_body_part_colors(int(state["joints"].shape[1]))
        _safe_set_prop(joint_handle, "colors", new_colors)
        _safe_set_prop(joint_handle, "points", state["joints"][0])
        if state.get("has_object"):
            if object_handle is None:
                object_handle = server.scene.add_mesh_simple(
                    "/object",
                    vertices=state["obj_verts"][0],
                    faces=state["obj_faces"],
                    color=(120, 180, 220),
                    flat_shading=False,
                )
            else:
                _safe_set_prop(object_handle, "faces", state["obj_faces"])
                _safe_set_prop(object_handle, "vertices", state["obj_verts"][0])
        elif object_handle is not None:
            _safe_set_prop(object_handle, "visible", False)
        updating_slider["flag"] = True
        frame_slider.max = max(0, int(state["n_frames"]) - 1)
        frame_slider.value = 0
        updating_slider["flag"] = False
        _apply_frame(0)

    def _player_loop() -> None:
        while True:
            if playing["flag"]:
                fps_val = max(1.0, float(fps_input.value))
                next_frame = int(frame_slider.value) + 1
                last_frame = int(state["n_frames"]) - 1
                if next_frame > last_frame:
                    if loop_cb.value:
                        next_frame = 0
                    else:
                        next_frame = last_frame
                        playing["flag"] = False
                updating_slider["flag"] = True
                frame_slider.value = next_frame
                updating_slider["flag"] = False
                _apply_frame(next_frame)
                time.sleep(1.0 / fps_val)
            else:
                time.sleep(0.01)

    _apply_frame(0)
    threading.Thread(target=_player_loop, daemon=True).start()
    print("Open the Viser URL printed above. Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
