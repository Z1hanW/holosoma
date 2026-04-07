#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for trajectory heatmap plotting.") from exc

from select_diverse_motions import _feature_vector, _natural_sort_key, _select_diverse_indices

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RootHeatmapConfig:
    motion_root: str = str(REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared")
    output_path: str = str(REPO_ROOT / "logs/plots/retargeted_root_heatmap.png")
    clip_names_csv: str = ""
    diverse_topk: int = 0
    bounds_csv: str = ""
    resample_length: int = 32
    bins: int = 180
    blur_sigma_bins: float = 1.25
    align_start: bool = True
    align_heading: bool = False
    show_trajectories: bool = True
    trajectory_alpha: float = 0.22
    trajectory_linewidth: float = 1.2
    show_start_markers: bool = True
    title: str = "Top-Down Root Trajectory Heatmap"
    colormap: str = "magma"
    padding_m: float = 0.15


def _parse_csv(csv_value: str) -> list[str]:
    return [part.strip() for part in csv_value.split(",") if part.strip()]


def _parse_bounds_csv(bounds_csv: str) -> tuple[float, float, float, float] | None:
    if not bounds_csv.strip():
        return None
    parts = [part.strip() for part in bounds_csv.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("bounds_csv must contain exactly 4 comma-separated values: x_min,x_max,y_min,y_max")
    x_min, x_max, y_min, y_max = (float(part) for part in parts)
    if not x_min < x_max:
        raise ValueError(f"Invalid x bounds: {x_min}, {x_max}")
    if not y_min < y_max:
        raise ValueError(f"Invalid y bounds: {y_min}, {y_max}")
    return x_min, x_max, y_min, y_max


def _resolve_paths(cfg: RootHeatmapConfig, motion_root: Path) -> list[Path]:
    if not motion_root.is_dir():
        raise FileNotFoundError(f"Motion root not found: {motion_root}")

    paths = sorted((path.resolve() for path in motion_root.glob("*.npz") if path.is_file()), key=_natural_sort_key)
    if not paths:
        raise ValueError(f"No .npz files found under {motion_root}")

    selected_names = _parse_csv(cfg.clip_names_csv)
    if selected_names:
        by_stem = {path.stem: path for path in paths}
        missing = [name for name in selected_names if name not in by_stem]
        if missing:
            raise ValueError(f"Unknown clip names in clip_names_csv: {missing}")
        return [by_stem[name] for name in selected_names]

    if cfg.diverse_topk > 0:
        features: list[np.ndarray] = []
        for path in paths:
            feature_vec, _ = _feature_vector(path, cfg.resample_length)
            features.append(feature_vec)
        indices = _select_diverse_indices(np.stack(features, axis=0), cfg.diverse_topk)
        return [paths[idx] for idx in indices]

    return paths


def _load_root_xy(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "body_pos_w" in data and "body_names" in data:
            body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
            body_names = [str(name) for name in np.asarray(data["body_names"]).reshape(-1)]
            if "pelvis" in body_names:
                pelvis_idx = body_names.index("pelvis")
                return np.asarray(body_pos[:, pelvis_idx, :2], dtype=np.float64)

        if "joint_pos" in data:
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            if joint_pos.ndim == 2 and joint_pos.shape[1] >= 2:
                return np.asarray(joint_pos[:, :2], dtype=np.float64)

        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            if qpos.ndim == 2 and qpos.shape[1] >= 2:
                return np.asarray(qpos[:, :2], dtype=np.float64)

    raise ValueError(f"Could not resolve root XY trajectory from {path}")


def _yaw_from_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in np.asarray(quat_wxyz).reshape(4)]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _load_initial_root_yaw(path: Path) -> float:
    with np.load(path, allow_pickle=True) as data:
        if "joint_pos" in data:
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            if joint_pos.ndim == 2 and joint_pos.shape[1] >= 7:
                return _yaw_from_wxyz(joint_pos[0, 3:7])

        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            if qpos.ndim == 2 and qpos.shape[1] >= 7:
                return _yaw_from_wxyz(qpos[0, 3:7])

    raise ValueError(f"Could not resolve initial root yaw from {path}")


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray((rot @ np.asarray(xy, dtype=np.float64).T).T, dtype=np.float64)


def _gaussian_kernel1d(sigma_bins: float) -> np.ndarray:
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def _convolve_along_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = kernel.shape[0] // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis, padded)


def _smooth_histogram(hist: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma_bins)
    if kernel.shape[0] == 1:
        return hist
    smoothed = _convolve_along_axis(hist, kernel, axis=0)
    smoothed = _convolve_along_axis(smoothed, kernel, axis=1)
    return smoothed


def _trajectory_bounds(trajectories: list[np.ndarray], padding_m: float) -> tuple[float, float, float, float]:
    all_xy = np.concatenate(trajectories, axis=0)
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    return x_min - padding_m, x_max + padding_m, y_min - padding_m, y_max + padding_m


def main(cfg: RootHeatmapConfig) -> None:
    motion_root = Path(cfg.motion_root).expanduser().resolve()
    output_path = Path(cfg.output_path).expanduser().resolve()

    selected_paths = _resolve_paths(cfg, motion_root)
    trajectories: list[np.ndarray] = []
    clip_names: list[str] = []
    for path in selected_paths:
        xy = _load_root_xy(path)
        if cfg.align_start:
            xy = np.asarray(xy - xy[0], dtype=np.float64)
        if cfg.align_heading:
            xy = _rotate_xy(xy, -_load_initial_root_yaw(path))
        trajectories.append(xy)
        clip_names.append(path.stem)

    bounds = _parse_bounds_csv(cfg.bounds_csv)
    if bounds is None:
        x_min, x_max, y_min, y_max = _trajectory_bounds(trajectories, cfg.padding_m)
    else:
        x_min, x_max, y_min, y_max = bounds
    all_xy = np.concatenate(trajectories, axis=0)
    hist, x_edges, y_edges = np.histogram2d(
        all_xy[:, 0],
        all_xy[:, 1],
        bins=int(cfg.bins),
        range=[[x_min, x_max], [y_min, y_max]],
    )
    hist = _smooth_histogram(hist, float(cfg.blur_sigma_bins))

    total_mass = float(hist.sum())
    if total_mass <= 0.0:
        raise ValueError("Heatmap histogram is empty.")

    bin_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
    prob_density = hist / total_mass / max(bin_area, 1.0e-12)
    masked_density = np.ma.masked_where(prob_density.T <= 0.0, prob_density.T)

    fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=180, constrained_layout=True)
    ax.set_facecolor("#0b0d10")
    image = ax.imshow(
        masked_density,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        cmap=cfg.colormap,
        aspect="equal",
        interpolation="bilinear",
    )

    if cfg.show_trajectories:
        for xy in trajectories:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="#d7f9ff",
                alpha=float(cfg.trajectory_alpha),
                linewidth=float(cfg.trajectory_linewidth),
                solid_capstyle="round",
            )

    if cfg.show_start_markers:
        starts = np.stack([xy[0] for xy in trajectories], axis=0)
        ax.scatter(
            starts[:, 0],
            starts[:, 1],
            s=28,
            color="#80ffdb",
            edgecolors="black",
            linewidths=0.4,
            label="start",
            zorder=4,
        )

    ax.scatter(
        [0.0],
        [0.0],
        s=80,
        marker="+",
        color="#ffffff",
        linewidths=1.0,
        label="origin",
        zorder=5,
    )
    ax.set_title(cfg.title, fontsize=14, pad=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(color="white", alpha=0.08, linewidth=0.6)
    ax.legend(loc="upper right", framealpha=0.85)

    cbar = fig.colorbar(image, ax=ax, shrink=0.95)
    cbar.set_label("Occupancy probability density [1/m^2]")

    summary = (
        f"clips={len(trajectories)} | samples={all_xy.shape[0]} | "
        f"align_start={cfg.align_start} | bins={cfg.bins}"
    )
    ax.text(
        0.02,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 6},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[root_heatmap] Saved image to {output_path}")
    print(f"[root_heatmap] Selected clips: {','.join(clip_names)}")
    print(f"[root_heatmap] Total trajectory samples: {all_xy.shape[0]}")
    print(f"[root_heatmap] Bounds: {x_min:.6f},{x_max:.6f},{y_min:.6f},{y_max:.6f}")


if __name__ == "__main__":
    main(tyro.cli(RootHeatmapConfig))
