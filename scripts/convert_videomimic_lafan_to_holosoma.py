#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"
)


def _decode_strings(values: Iterable) -> list[str]:
    decoded: list[str] = []
    for item in values:
        if isinstance(item, (bytes, np.bytes_)):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _normalize_joint_name(name: str) -> str:
    if name.endswith("_joint"):
        return name[: -len("_joint")]
    return name


def _normalize_link_name(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".stl"):
        return name[: -len(".stl")]
    return name


def _load_template_names(path: Path) -> tuple[list[str], list[str]]:
    with np.load(path, allow_pickle=True) as npz:
        joint_names = _decode_strings(np.asarray(npz["joint_names"]))
        body_names = _decode_strings(np.asarray(npz["body_names"]))
    return joint_names, body_names


def _load_replay_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data


def _load_replay_h5(path: Path) -> dict:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("h5py is required to load .h5 replay files.") from exc

    with h5py.File(path, "r") as f:
        data = {
            "root_pos": f["root_pos"][:],
            "root_quat": f["root_quat"][:],
            "joints": f["joints"][:],
            "link_pos": f["link_pos"][:],
            "link_quat": f["link_quat"][:],
        }
        joint_names = f.attrs.get("/joint_names") or f.attrs.get("joint_names")
        link_names = f.attrs.get("/link_names") or f.attrs.get("link_names")
        fps = f.attrs.get("/fps") or f.attrs.get("fps")
        if joint_names is None and "joint_names" in f:
            joint_names = f["joint_names"][:]
        if link_names is None and "link_names" in f:
            link_names = f["link_names"][:]
        if fps is None and "fps" in f:
            fps = f["fps"][()]
        if joint_names is not None:
            data["joint_names"] = _decode_strings(np.asarray(joint_names))
        if link_names is not None:
            data["link_names"] = _decode_strings(np.asarray(link_names))
        if fps is not None:
            data["fps"] = float(np.asarray(fps).reshape(-1)[0])
    return data


def _load_replay(path: Path) -> dict:
    if path.suffix == ".pkl":
        return _load_replay_pkl(path)
    if path.suffix in {".h5", ".hdf5"}:
        return _load_replay_h5(path)
    raise ValueError(f"Unsupported replay file: {path}")


def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.split(a, 4, axis=-1)
    bx, by, bz, bw = np.split(b, 4, axis=-1)
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.concatenate([x, y, z, w], axis=-1)


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[..., :3]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., 3:4] * uv + uuv)


def _quat_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1)
    flip = dot < 0.0
    q1 = np.where(flip[..., None], -q1, q1)
    dot = np.abs(dot)

    near = dot > 0.9995
    t = np.asarray(t)
    while t.ndim < dot.ndim:
        t = t[..., None]

    result = np.empty_like(q0)
    if np.any(near):
        lerp = q0 + t[..., None] * (q1 - q0)
        lerp = lerp / np.linalg.norm(lerp, axis=-1, keepdims=True)
        result = np.where(near[..., None], lerp, result)

    if np.any(~near):
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        out = (s0[..., None] * q0) + (s1[..., None] * q1)
        out = out / np.linalg.norm(out, axis=-1, keepdims=True)
        result = np.where(near[..., None], result, out)
    return result


def _resample_linear(data: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if abs(src_fps - dst_fps) < 1e-6:
        return data
    n = data.shape[0]
    if n == 1:
        return data.copy()
    duration = (n - 1) / src_fps
    out_len = int(round(duration * dst_fps)) + 1
    t_src = np.linspace(0.0, duration, n)
    t_dst = np.linspace(0.0, duration, out_len)
    flat = data.reshape(n, -1)
    out = np.empty((out_len, flat.shape[1]), dtype=flat.dtype)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, flat[:, i])
    return out.reshape((out_len,) + data.shape[1:])


def _resample_quat_xyzw(quats: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if abs(src_fps - dst_fps) < 1e-6:
        return quats
    n = quats.shape[0]
    if n == 1:
        return quats.copy()
    duration = (n - 1) / src_fps
    out_len = int(round(duration * dst_fps)) + 1
    t_dst = np.linspace(0.0, duration, out_len)
    src_idx = t_dst * src_fps
    idx0 = np.floor(src_idx).astype(int)
    idx1 = np.clip(idx0 + 1, 0, n - 1)
    alpha = (src_idx - idx0).astype(quats.dtype)
    q0 = quats[idx0]
    q1 = quats[idx1]
    return _quat_slerp_xyzw(q0, q1, alpha)


def _finite_diff(data: np.ndarray, fps: float) -> np.ndarray:
    if data.shape[0] == 1:
        return np.zeros_like(data)
    vel = (data[1:] - data[:-1]) * fps
    return np.concatenate([vel, vel[-1:]], axis=0)


def _angular_velocity_xyzw(quats: np.ndarray, fps: float) -> np.ndarray:
    if quats.shape[0] == 1:
        return np.zeros(quats.shape[:-1] + (3,), dtype=quats.dtype)
    q0 = quats[:-1]
    q1 = quats[1:]
    dq = _quat_mul_xyzw(q1, _quat_conjugate_xyzw(q0))
    dq = dq / np.linalg.norm(dq, axis=-1, keepdims=True)
    w = np.clip(dq[..., 3], -1.0, 1.0)
    v = dq[..., :3]
    sin_half = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    small = sin_half < 1e-8
    axis = np.zeros_like(v)
    axis[~small] = v[~small] / sin_half[~small][..., None]
    omega = axis * (angle[..., None] * fps)
    omega[small] = 2.0 * v[small] * fps
    return np.concatenate([omega, omega[-1:]], axis=0)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _infer_link_frame(
    link_names: list[str],
    link_pos: np.ndarray,
    root_pos: np.ndarray,
    mode: str,
) -> str:
    if mode != "auto":
        return mode
    link_names_norm = [_normalize_link_name(name) for name in link_names]
    for pelvis_name in ("pelvis", "pelvis_link"):
        if pelvis_name in link_names_norm:
            idx = link_names_norm.index(pelvis_name)
            diff = np.linalg.norm(link_pos[:, idx] - root_pos, axis=-1)
            if np.median(diff) < 1e-3:
                return "world"
            return "local"
    return "world"


def _fill_body(
    name: str,
    link_names: list[str],
    link_name_map: dict[str, int],
    link_pos_w: np.ndarray,
    link_quat_w_xyzw: np.ndarray,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    missing_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    name_norm = _normalize_link_name(name)
    if name_norm in link_name_map:
        idx = link_name_map[name_norm]
        return link_pos_w[:, idx], link_quat_w_xyzw[:, idx]
    if name == "world":
        pos = np.zeros_like(root_pos)
        quat = np.zeros_like(root_quat_xyzw)
        quat[..., 3] = 1.0
        return pos, quat
    if name == "pelvis":
        return root_pos, root_quat_xyzw
    if name.endswith("_contour_link"):
        base = name.replace("_contour_link", "")
        base_norm = _normalize_link_name(base)
        if base_norm in link_name_map:
            idx = link_name_map[base_norm]
            return link_pos_w[:, idx], link_quat_w_xyzw[:, idx]
    if missing_policy == "root":
        return root_pos, root_quat_xyzw
    if missing_policy == "zero":
        pos = np.zeros_like(root_pos)
        quat = np.zeros_like(root_quat_xyzw)
        quat[..., 3] = 1.0
        return pos, quat
    raise KeyError(f"Missing body '{name}' in replay data.")


def _reorder_joints(
    joints: np.ndarray, joint_names: list[str], target_joint_names: list[str], missing_policy: str
) -> np.ndarray:
    out = np.zeros((joints.shape[0], len(target_joint_names)), dtype=joints.dtype)
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    missing: list[str] = []
    for i, name in enumerate(target_joint_names):
        name_norm = _normalize_joint_name(name)
        idx = name_to_idx.get(name_norm)
        if idx is not None:
            out[:, i] = joints[:, idx]
        elif missing_policy == "error":
            raise KeyError(f"Missing joint '{name}' in replay data.")
        else:
            missing.append(name)
    if missing and missing_policy != "error":
        missing_str = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        print(f"[warn] Missing joints in replay data: {missing_str}{suffix}")
    return out


def convert_clip(
    data: dict,
    target_joint_names: list[str],
    target_body_names: list[str],
    target_fps: float,
    link_frame: str,
    missing_policy: str,
) -> dict:
    for key in ("root_pos", "root_quat", "joints", "link_pos", "link_quat"):
        if key not in data:
            raise KeyError(f"Missing '{key}' in replay data.")

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_quat_xyzw = np.asarray(data["root_quat"], dtype=np.float32)
    joints = np.asarray(data["joints"], dtype=np.float32)
    link_pos = np.asarray(data["link_pos"], dtype=np.float32)
    link_quat_xyzw = np.asarray(data["link_quat"], dtype=np.float32)
    joint_names = _decode_strings(data.get("joint_names", []))
    link_names = _decode_strings(data.get("link_names", []))
    joint_names = [_normalize_joint_name(name) for name in joint_names]
    link_names = [_normalize_link_name(name) for name in link_names]
    link_name_map = {name: i for i, name in enumerate(link_names)}
    fps = float(data.get("fps", target_fps))

    if abs(fps - target_fps) > 1e-6:
        root_pos = _resample_linear(root_pos, fps, target_fps)
        joints = _resample_linear(joints, fps, target_fps)
        link_pos = _resample_linear(link_pos, fps, target_fps)
        root_quat_xyzw = _resample_quat_xyzw(root_quat_xyzw, fps, target_fps)
        link_quat_xyzw = _resample_quat_xyzw(link_quat_xyzw, fps, target_fps)
        fps = target_fps

    frame_mode = _infer_link_frame(link_names, link_pos, root_pos, link_frame)
    if frame_mode == "local":
        link_pos_w = root_pos[:, None, :] + _quat_rotate_xyzw(root_quat_xyzw[:, None, :], link_pos)
        link_quat_w_xyzw = _quat_mul_xyzw(root_quat_xyzw[:, None, :], link_quat_xyzw)
    else:
        link_pos_w = link_pos
        link_quat_w_xyzw = link_quat_xyzw

    dof_pos = _reorder_joints(joints, joint_names, target_joint_names, missing_policy)

    body_pos_w = np.zeros((root_pos.shape[0], len(target_body_names), 3), dtype=np.float32)
    body_quat_w_xyzw = np.zeros((root_pos.shape[0], len(target_body_names), 4), dtype=np.float32)

    for i, name in enumerate(target_body_names):
        pos, quat = _fill_body(
            name,
            link_names,
            link_name_map,
            link_pos_w,
            link_quat_w_xyzw,
            root_pos,
            root_quat_xyzw,
            missing_policy,
        )
        body_pos_w[:, i] = pos
        body_quat_w_xyzw[:, i] = quat

    body_lin_vel_w = _finite_diff(body_pos_w, fps)
    body_ang_vel_w = _angular_velocity_xyzw(body_quat_w_xyzw, fps)

    root_lin_vel = _finite_diff(root_pos, fps)
    root_ang_vel = _angular_velocity_xyzw(root_quat_xyzw, fps)
    dof_vel = _finite_diff(dof_pos, fps)

    joint_pos = np.concatenate([root_pos, _xyzw_to_wxyz(root_quat_xyzw), dof_pos], axis=-1)
    joint_vel = np.concatenate([root_lin_vel, root_ang_vel, dof_vel], axis=-1)

    return {
        "fps": np.array([int(round(fps))], dtype=np.int64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": _xyzw_to_wxyz(body_quat_w_xyzw),
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.array(target_joint_names, dtype=object),
        "body_names": np.array(target_body_names, dtype=object),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VideoMimic LAFAN replay data into Holosoma WBT motion files."
    )
    parser.add_argument("--input", required=True, help="Replay file or directory.")
    parser.add_argument("--pattern", default="*.pkl", help="Glob pattern when input is a directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for .npz files.")
    parser.add_argument("--template-motion", default=str(DEFAULT_TEMPLATE), help="Template .npz for names.")
    parser.add_argument("--target-fps", type=float, default=50.0, help="Target FPS for output motions.")
    parser.add_argument(
        "--link-frame",
        choices=["auto", "world", "local"],
        default="auto",
        help="Interpretation of link_pos/link_quat in replay data.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "root", "zero"],
        default="root",
        help="How to fill missing joints/bodies.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit number of clips to convert.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if input_path.is_dir():
        files = sorted(input_path.glob(args.pattern))
    else:
        files = [input_path]

    if args.max_clips is not None:
        files = files[: args.max_clips]

    if not files:
        raise FileNotFoundError(f"No files found for {input_path} with pattern {args.pattern}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = Path(args.template_motion).expanduser()
    if not template_path.exists():
        raise FileNotFoundError(f"Template motion not found: {template_path}")
    target_joint_names, target_body_names = _load_template_names(template_path)

    for path in files:
        out_path = output_dir / f"{path.stem}_holosoma.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue
        data = _load_replay(path)
        converted = convert_clip(
            data,
            target_joint_names=target_joint_names,
            target_body_names=target_body_names,
            target_fps=args.target_fps,
            link_frame=args.link_frame,
            missing_policy=args.missing_policy,
        )
        np.savez(out_path, **converted)
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
