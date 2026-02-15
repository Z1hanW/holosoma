from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tyro


def _print_depth_stats(depth_raw: np.ndarray) -> None:
    depth = depth_raw.astype(np.float32)
    flat = depth.reshape(-1)
    finite_mask = np.isfinite(flat)
    valid = flat[finite_mask]

    print(f"[Depth] shape: {depth_raw.shape}")
    print(f"[Depth] dtype: {depth_raw.dtype}")
    print(f"[Depth] total_pixels: {flat.size}")
    print(f"[Depth] finite_pixels: {valid.size}")
    print(f"[Depth] nan_or_inf_pixels: {flat.size - valid.size}")
    if valid.size == 0:
        print("[Depth] No finite values found.")
        return

    print(f"[Depth] min: {valid.min():.6f}")
    print(f"[Depth] max: {valid.max():.6f}")
    print(f"[Depth] mean: {valid.mean():.6f}")
    print(f"[Depth] std: {valid.std():.6f}")
    print(f"[Depth] median: {np.median(valid):.6f}")
    p1, p5, p95, p99 = np.percentile(valid, [1, 5, 95, 99])
    print(f"[Depth] p01: {p1:.6f}")
    print(f"[Depth] p05: {p5:.6f}")
    print(f"[Depth] p95: {p95:.6f}")
    print(f"[Depth] p99: {p99:.6f}")

def _reverse_depth_for_visualization(depth_vis: np.ndarray, near_clip: float, far_clip: float) -> np.ndarray:
    # forward: [0, 255] -> [0, 1]
    depth_float = depth_vis.astype(np.float32)/255.0 # [0, 1]
    return depth_float * (far_clip - near_clip) + near_clip


@dataclass(frozen=True)
class InspectDepthPngConfig:
    depth_image_path: str
    near_clip: float = 0.1
    far_clip: float = 2.0
    hist_bins: int = 120
    show_plot: bool = True
    save_plot_path: str | None = None


def main(cfg: InspectDepthPngConfig) -> None:
    depth_path = Path(cfg.depth_image_path)
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image does not exist: {depth_path}")
    if cfg.far_clip <= cfg.near_clip:
        raise ValueError("far_clip must be greater than near_clip")

    depth_vis = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_vis is None:
        raise RuntimeError(f"Failed to load depth image: {depth_path}")

    if depth_vis.ndim == 3:
        print(f"[Depth] Warning: image has {depth_vis.shape[2]} channels; using channel 0.")
        depth_vis = depth_vis[..., 0]


    depth_float = _reverse_depth_for_visualization(depth_vis, cfg.near_clip, cfg.far_clip) 
    print(f"[Depth] Depth float: {depth_float.min():.6f} - {depth_float.max():.6f}")
    valid = depth_float[np.isfinite(depth_float)]
    _print_depth_stats(depth_float)

    

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(valid.reshape(-1), bins=cfg.hist_bins, color="royalblue", alpha=0.9)
    axes[0].set_title("Depth Value Distribution")
    axes[0].set_xlabel("Depth value")
    axes[0].set_ylabel("Pixel count")
    axes[0].grid(alpha=0.3)

    axes[1].imshow(depth_vis, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Depth Visualization (clip: {cfg.near_clip}-{cfg.far_clip})")
    axes[1].axis("off")

    plt.tight_layout()

    if cfg.save_plot_path is not None:
        save_path = Path(cfg.save_plot_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[Depth] Saved figure to: {save_path}")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(InspectDepthPngConfig))
