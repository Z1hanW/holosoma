#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DiverseMotionConfig:
    motion_root: str = str(REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared")
    topk: int = 10
    resample_length: int = 32
    output: str = "csv"
    verbose: bool = False


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def _quat_continuous(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.ndim != 2:
        return q
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    return q


def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_len, axis=0)

    x_old = np.linspace(0.0, 1.0, arr.shape[0])
    x_new = np.linspace(0.0, 1.0, target_len)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(x_new, x_old, flat[:, j])
    return out.reshape((target_len,) + arr.shape[1:])


def _feature_vector(path: Path, target_len: int) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        if "joint_pos" not in data:
            raise ValueError(f"{path} does not contain joint_pos.")
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] < 8:
            raise ValueError(f"Invalid joint_pos in {path}: {joint_pos.shape}")

        object_pos = np.asarray(data["object_pos_w"], dtype=np.float64)
        object_quat = _quat_continuous(np.asarray(data["object_quat_w"], dtype=np.float64))
        object_size = np.asarray(data["object_size"], dtype=np.float64).reshape(-1)
        fps = float(np.asarray(data.get("fps", 30)).reshape(-1)[0])
        if fps <= 0:
            fps = 30.0

    base_pos = joint_pos[:, :3]
    base_quat = _quat_continuous(joint_pos[:, 3:7])
    joints = joint_pos[:, 7:]

    base_pos_rel = base_pos - base_pos[0]
    object_pos_rel = object_pos - object_pos[0]
    object_rel_to_base = object_pos - base_pos

    base_speed = np.linalg.norm(np.diff(base_pos, axis=0), axis=1)
    object_speed = np.linalg.norm(np.diff(object_pos, axis=0), axis=1)
    joint_speed = np.linalg.norm(np.diff(joints, axis=0), axis=1)

    feature_blocks = [
        _resample(base_pos_rel, target_len).reshape(-1),
        _resample(object_pos_rel, target_len).reshape(-1),
        _resample(object_rel_to_base, target_len).reshape(-1),
        _resample(base_quat, target_len).reshape(-1),
        _resample(object_quat, target_len).reshape(-1),
        _resample(joints, target_len).reshape(-1),
        np.asarray(
            [
                joint_pos.shape[0] / fps,
                np.linalg.norm(base_pos[-1, :2] - base_pos[0, :2]),
                np.linalg.norm(object_pos[-1, :2] - object_pos[0, :2]),
                np.ptp(base_pos_rel[:, 2]),
                np.ptp(object_pos_rel[:, 2]),
                float(base_speed.mean() * fps) if base_speed.size else 0.0,
                float(object_speed.mean() * fps) if object_speed.size else 0.0,
                float(joint_speed.mean() * fps) if joint_speed.size else 0.0,
            ],
            dtype=np.float64,
        ),
        object_size,
    ]

    stats = {
        "clip": path.stem,
        "frames": int(joint_pos.shape[0]),
        "seconds": float(joint_pos.shape[0] / fps),
        "root_xy_disp": float(np.linalg.norm(base_pos[-1, :2] - base_pos[0, :2])),
        "object_xy_disp": float(np.linalg.norm(object_pos[-1, :2] - object_pos[0, :2])),
        "object_size": [float(x) for x in object_size.tolist()],
    }
    return np.concatenate(feature_blocks, axis=0), stats


def _select_diverse_indices(x: np.ndarray, topk: int) -> list[int]:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    if x.shape[0] == 0:
        return []

    x = (x - x.mean(axis=0)) / np.clip(x.std(axis=0), 1.0e-6, None)
    distances = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2), 0.0))

    selected = [int(np.argmax(distances.mean(axis=1)))]
    while len(selected) < min(topk, x.shape[0]):
        min_dist = distances[:, selected].min(axis=1)
        min_dist[selected] = -math.inf
        selected.append(int(np.argmax(min_dist)))
    return selected


def main(cfg: DiverseMotionConfig) -> None:
    motion_root = Path(cfg.motion_root).expanduser().resolve()
    if not motion_root.is_dir():
        raise FileNotFoundError(f"Motion root not found: {motion_root}")

    paths = sorted((path for path in motion_root.glob("*.npz") if path.is_file()), key=_natural_sort_key)
    if not paths:
        raise ValueError(f"No .npz files found under {motion_root}")

    features: list[np.ndarray] = []
    stats: list[dict[str, object]] = []
    for path in paths:
        feature_vec, clip_stats = _feature_vector(path, cfg.resample_length)
        features.append(feature_vec)
        stats.append(clip_stats)

    x = np.stack(features, axis=0)
    selected = _select_diverse_indices(x, cfg.topk)
    selected_stats = [stats[idx] for idx in selected]
    selected_names = [str(item["clip"]) for item in selected_stats]

    output_mode = cfg.output.strip().lower()
    if output_mode == "csv":
        print(",".join(selected_names))
        return

    if output_mode == "lines":
        for name in selected_names:
            print(name)
        return

    if output_mode == "table":
        for item in selected_stats:
            size = ",".join(f"{value:.3f}" for value in item["object_size"])
            print(
                f"{item['clip']}\tframes={item['frames']}\tsec={item['seconds']:.2f}\t"
                f"root_xy={item['root_xy_disp']:.3f}\tobj_xy={item['object_xy_disp']:.3f}\tsize={size}"
            )
        return

    raise ValueError(f"Unsupported output mode: {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(DiverseMotionConfig))
