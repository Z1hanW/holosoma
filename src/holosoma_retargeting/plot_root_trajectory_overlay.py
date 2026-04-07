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
    from matplotlib.patches import Patch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for trajectory overlay plotting.") from exc

from plot_root_trajectory_comparison import _compute_density, _load_dataset
from plot_root_trajectory_heatmap import _parse_bounds_csv, _trajectory_bounds

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RootOverlayConfig:
    motion_root_a: str = str(REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared")
    motion_root_b: str = str(REPO_ROOT / "src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry")
    label_a: str = "pure-ds (43)"
    label_b: str = "OMOMO-carry (62)"
    output_path: str = str(REPO_ROOT / "logs/plots/root_heatmap_overlay_pureds43_vs_omomo_carry.png")
    bounds_csv: str = ""
    bins: int = 180
    blur_sigma_bins: float = 1.25
    align_start: bool = True
    align_heading: bool = False
    show_start_markers: bool = True
    show_trajectories: bool = False
    trajectory_alpha: float = 0.04
    trajectory_linewidth: float = 0.5
    cmap_a: str = "Blues"
    cmap_b: str = "Oranges"
    alpha_a: float = 0.82
    alpha_b: float = 0.82
    alpha_gamma: float = 0.75
    padding_m: float = 0.15
    title: str = "Overlayed Top-Down Root Trajectory Heatmaps"


def _make_overlay_rgba(density: np.ndarray, cmap_name: str, alpha_max: float, alpha_gamma: float) -> np.ndarray:
    scaled = np.asarray(density, dtype=np.float64)
    vmax = float(scaled.max())
    if vmax > 0.0:
        scaled = scaled / vmax
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(scaled, 0.0, 1.0))
    rgba[..., 3] = (np.clip(scaled, 0.0, 1.0) ** float(alpha_gamma)) * float(alpha_max)
    rgba[density <= 0.0, 3] = 0.0
    return rgba


def main(cfg: RootOverlayConfig) -> None:
    output_path = Path(cfg.output_path).expanduser().resolve()

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
        x_min, x_max, y_min, y_max = _trajectory_bounds(trajectories_a + trajectories_b, cfg.padding_m)
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

    rgba_a = _make_overlay_rgba(density_a, cfg.cmap_a, cfg.alpha_a, cfg.alpha_gamma)
    rgba_b = _make_overlay_rgba(density_b, cfg.cmap_b, cfg.alpha_b, cfg.alpha_gamma)

    fig, ax = plt.subplots(figsize=(8.2, 7.1), dpi=180, constrained_layout=True)
    ax.set_facecolor("#0b0d10")
    ax.imshow(rgba_a, origin="lower", extent=extent, aspect="equal", interpolation="bilinear")
    ax.imshow(rgba_b, origin="lower", extent=extent, aspect="equal", interpolation="bilinear")

    if cfg.show_trajectories:
        for xy in trajectories_a:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="#8ecae6",
                alpha=float(cfg.trajectory_alpha),
                linewidth=float(cfg.trajectory_linewidth),
            )
        for xy in trajectories_b:
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="#ffb703",
                alpha=float(cfg.trajectory_alpha),
                linewidth=float(cfg.trajectory_linewidth),
            )

    if cfg.show_start_markers:
        starts_a = np.stack([xy[0] for xy in trajectories_a], axis=0)
        starts_b = np.stack([xy[0] for xy in trajectories_b], axis=0)
        ax.scatter(starts_a[:, 0], starts_a[:, 1], s=12, color="#56cfe1", edgecolors="none", zorder=4)
        ax.scatter(starts_b[:, 0], starts_b[:, 1], s=12, color="#fb8500", edgecolors="none", zorder=4)

    ax.scatter([0.0], [0.0], s=80, marker="+", color="#ffffff", linewidths=1.0, zorder=5)
    ax.set_title(cfg.title, fontsize=14, pad=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(color="white", alpha=0.08, linewidth=0.6)

    legend_handles = [
        Patch(facecolor=plt.get_cmap(cfg.cmap_a)(0.8), edgecolor="none", label=cfg.label_a),
        Patch(facecolor=plt.get_cmap(cfg.cmap_b)(0.8), edgecolor="none", label=cfg.label_b),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    ax.text(
        0.02,
        0.02,
        (
            f"{cfg.label_a}: clips={len(names_a)}, samples={samples_a}\n"
            f"{cfg.label_b}: clips={len(names_b)}, samples={samples_b}\n"
            "Per-dataset relative occupancy, overlaid at shared scale"
        ),
        transform=ax.transAxes,
        fontsize=8.8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 5},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[root_overlay] Saved image to {output_path}")
    print(f"[root_overlay] Dataset A clips={len(names_a)} samples={samples_a} roots={cfg.motion_root_a}")
    print(f"[root_overlay] Dataset B clips={len(names_b)} samples={samples_b} roots={cfg.motion_root_b}")
    print(f"[root_overlay] Bounds: {x_min:.6f},{x_max:.6f},{y_min:.6f},{y_max:.6f}")


if __name__ == "__main__":
    main(tyro.cli(RootOverlayConfig))
