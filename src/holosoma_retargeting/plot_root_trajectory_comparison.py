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
    from matplotlib import colors
    import matplotlib.patheffects as pe
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for trajectory comparison plotting.") from exc

from plot_root_trajectory_heatmap import (
    _load_initial_root_yaw,
    _load_root_xy,
    _parse_bounds_csv,
    _rotate_xy,
    _resolve_paths,
    _smooth_histogram,
    _trajectory_bounds,
    RootHeatmapConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RootComparisonConfig:
    motion_root_a: str = str(REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared")
    motion_root_b: str = str(REPO_ROOT / "src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry")
    title_a: str = "pure-ds"
    title_b: str = "OMOMO-carry"
    output_path: str = str(REPO_ROOT / "logs/plots/root_heatmap_compare_pureds43_vs_omomo_carry.png")
    bounds_csv: str = ""
    bins: int = 180
    blur_sigma_bins: float = 1.25
    align_start: bool = True
    align_heading: bool = False
    show_trajectories: bool = True
    trajectory_alpha: float = 0.18
    trajectory_linewidth: float = 1.05
    show_start_markers: bool = True
    colormap: str = "magma"
    heatmap_gamma: float = 0.45
    vmax_percentile: float = 99.4
    mask_percentile: float = 0.4
    padding_m: float = 0.15
    suptitle: str = "Top-Down Root Trajectory Comparison"
    layout: str = "horizontal"
    show_panel_stats: bool = True


def _parse_root_list(csv_value: str) -> list[Path]:
    parts = [part.strip() for part in csv_value.split(",") if part.strip()]
    if not parts:
        raise ValueError("Dataset root list is empty.")
    roots: list[Path] = []
    seen: set[Path] = set()
    for part in parts:
        path = Path(part).expanduser().resolve()
        if path in seen:
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {path}")
        roots.append(path)
        seen.add(path)
    return roots


def _load_dataset(root_csv: str, *, align_start: bool, align_heading: bool) -> tuple[list[np.ndarray], list[str]]:
    trajectories: list[np.ndarray] = []
    names: list[str] = []
    for root in _parse_root_list(root_csv):
        cfg = RootHeatmapConfig(motion_root=str(root))
        paths = _resolve_paths(cfg, root)
        for path in paths:
            xy = _load_root_xy(path)
            if align_start:
                xy = np.asarray(xy - xy[0], dtype=np.float64)
            if align_heading:
                xy = _rotate_xy(xy, -_load_initial_root_yaw(path))
            trajectories.append(xy)
            names.append(path.stem)
    if not trajectories:
        raise ValueError(f"No trajectories loaded from {root_csv}")
    return trajectories, names


def _compute_density(
    trajectories: list[np.ndarray],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bins: int,
    blur_sigma_bins: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], int]:
    all_xy = np.concatenate(trajectories, axis=0)
    hist, x_edges, y_edges = np.histogram2d(
        all_xy[:, 0],
        all_xy[:, 1],
        bins=int(bins),
        range=[[x_min, x_max], [y_min, y_max]],
    )
    hist = _smooth_histogram(hist, float(blur_sigma_bins))
    total_mass = float(hist.sum())
    if total_mass <= 0.0:
        raise ValueError("Heatmap histogram is empty.")
    bin_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
    density = hist / total_mass / max(bin_area, 1.0e-12)
    return density.T, (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), int(all_xy.shape[0])


def main(cfg: RootComparisonConfig) -> None:
    output_path = Path(cfg.output_path).expanduser().resolve()
    layout = cfg.layout.strip().lower()
    if layout not in {"horizontal", "vertical"}:
        raise ValueError(f"Unsupported layout: {cfg.layout}")

    trajectories_a, names_a = _load_dataset(
        cfg.motion_root_a,
        align_start=cfg.align_start,
        align_heading=cfg.align_heading,
    )
    trajectories_b, names_b = _load_dataset(
        cfg.motion_root_b,
        align_start=cfg.align_start,
        align_heading=cfg.align_heading,
    )

    bounds = _parse_bounds_csv(cfg.bounds_csv)
    if bounds is None:
        all_traj = trajectories_a + trajectories_b
        x_min, x_max, y_min, y_max = _trajectory_bounds(all_traj, cfg.padding_m)
    else:
        x_min, x_max, y_min, y_max = bounds

    density_a, extent, samples_a = _compute_density(
        trajectories_a,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        bins=cfg.bins,
        blur_sigma_bins=cfg.blur_sigma_bins,
    )
    density_b, _, samples_b = _compute_density(
        trajectories_b,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        bins=cfg.bins,
        blur_sigma_bins=cfg.blur_sigma_bins,
    )

    positive = np.concatenate([density_a[density_a > 0.0], density_b[density_b > 0.0]])
    if positive.size:
        vmax = float(np.percentile(positive, cfg.vmax_percentile))
        vmax = max(vmax, float(positive.max()) * 1.0e-3)
        mask_floor = float(np.percentile(positive, cfg.mask_percentile))
    else:
        vmax = 1.0
        mask_floor = 0.0
    norm = colors.PowerNorm(gamma=float(cfg.heatmap_gamma), vmin=0.0, vmax=vmax)

    if layout == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 14.6), dpi=220, sharex=True, sharey=True, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.4), dpi=220, sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    datasets = [
        (axes[0], density_a, trajectories_a, cfg.title_a, len(names_a), samples_a),
        (axes[1], density_b, trajectories_b, cfg.title_b, len(names_b), samples_b),
    ]
    image = None
    for ax, density, trajectories, title, clip_count, sample_count in datasets:
        ax.set_facecolor("#07090d")
        masked = np.ma.masked_where(density <= mask_floor, density)
        image = ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            cmap=cfg.colormap,
            norm=norm,
            aspect="equal",
            interpolation="bilinear",
        )
        if cfg.show_trajectories:
            for xy in trajectories:
                line = ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    color="#cfe7ff",
                    alpha=float(cfg.trajectory_alpha),
                    linewidth=float(cfg.trajectory_linewidth),
                    solid_capstyle="round",
                )[0]
                line.set_path_effects(
                    [pe.Stroke(linewidth=float(cfg.trajectory_linewidth) + 0.35, foreground="#0b1220"), pe.Normal()]
                )
        if cfg.show_start_markers:
            starts = np.stack([xy[0] for xy in trajectories], axis=0)
            ax.scatter(
                starts[:, 0],
                starts[:, 1],
                s=22,
                color="#8fffe9",
                edgecolors="black",
                linewidths=0.4,
                zorder=4,
            )
        ax.scatter([0.0], [0.0], s=84, marker="+", color="#ffffff", linewidths=1.0, zorder=5)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("x [m]", fontsize=12)
        ax.grid(color="white", alpha=0.11, linewidth=0.65)
        ax.tick_params(labelsize=11)
        if cfg.show_panel_stats:
            ax.text(
                0.02,
                0.02,
                f"clips={clip_count} | samples={sample_count}",
                transform=ax.transAxes,
                fontsize=9.2,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 5},
            )

    for ax in axes:
        ax.set_ylabel("y [m]", fontsize=12)
    fig.suptitle(cfg.suptitle, fontsize=17)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.96, pad=0.02)
        cbar.set_label("Occupancy probability density [1/m^2]")
        cbar.ax.tick_params(labelsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[root_compare] Saved image to {output_path}")
    print(f"[root_compare] Dataset A clips={len(names_a)} samples={samples_a} roots={cfg.motion_root_a}")
    print(f"[root_compare] Dataset B clips={len(names_b)} samples={samples_b} roots={cfg.motion_root_b}")
    print(f"[root_compare] Bounds: {x_min:.6f},{x_max:.6f},{y_min:.6f},{y_max:.6f}")


if __name__ == "__main__":
    main(tyro.cli(RootComparisonConfig))
