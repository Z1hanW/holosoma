from dataclasses import dataclass
from pathlib import Path
import re
import sys

import imageio.v2 as imageio
from matplotlib import cm
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import tyro

try:
    from holosoma.models.gum.infer import GUM
    from holosoma.sensors.image_server import GUMConfig
    from holosoma.sensors.zed import BuildGUMIntrinsicsAndExtrinsics
except ModuleNotFoundError:
    # Allows running this script directly from repository root.
    repo_src = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_src))
    from holosoma.models.gum.infer import GUM
    from holosoma.sensors.image_server import GUMConfig
    from holosoma.sensors.zed import BuildGUMIntrinsicsAndExtrinsics


STEP_PATTERN = re.compile(r".*_(\d+)_\d{8}_\d{6}_\d+\.png$")


def _extract_step(path: Path) -> int:
    match = STEP_PATTERN.match(path.name)
    if not match:
        raise ValueError(
            f"Filename does not match expected pattern '<camera>_<step>_<timestamp>.png': {path.name}"
        )
    return int(match.group(1))


def _normalize_for_vis(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.uint8)
    vmin = np.nanpercentile(arr[valid], 1.0)
    vmax = np.nanpercentile(arr[valid], 99.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    vis = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (vis * 255.0).astype(np.uint8)


def _colorize(arr: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(arr.astype(np.float32) / 255.0)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def _discover_frame_pairs(session_dir: Path, camera_name: str) -> list[tuple[Path, Path, int]]:
    rgb_dir = session_dir / camera_name / "rgb"
    depth_dir = session_dir / camera_name / "depth"
    if not rgb_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError(
            f"Expected directories not found:\n"
            f"  rgb: {rgb_dir}\n"
            f"  depth: {depth_dir}\n"
            f"Try a different --camera_name or --session_dir."
        )

    rgb_paths = sorted(rgb_dir.glob("*.png"), key=_extract_step)
    depth_paths = sorted(depth_dir.glob("*.png"), key=_extract_step)
    if not rgb_paths or not depth_paths:
        raise FileNotFoundError(f"No PNG files found in {rgb_dir} or {depth_dir}")

    rgb_by_step = {_extract_step(p): p for p in rgb_paths}
    depth_by_step = {_extract_step(p): p for p in depth_paths}
    common_steps = sorted(set(rgb_by_step) & set(depth_by_step))
    if not common_steps:
        raise RuntimeError("No matching rgb/depth frame steps found.")

    return [(rgb_by_step[s], depth_by_step[s], s) for s in common_steps]


def _discover_rgb_frames(session_dir: Path, camera_name: str) -> list[tuple[Path, int]]:
    rgb_dir = session_dir / camera_name / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(
            f"Expected directory not found:\n"
            f"  rgb: {rgb_dir}\n"
            f"Try a different --camera_name or --session_dir."
        )

    rgb_paths = sorted(rgb_dir.glob("*.png"), key=_extract_step)
    if not rgb_paths:
        raise FileNotFoundError(f"No PNG files found in {rgb_dir}")

    return [(path, _extract_step(path)) for path in rgb_paths]


def _save_pred_depth_png(
    pred_depth: np.ndarray,
    output_path: Path,
    near_clip: float,
    far_clip: float,
) -> None:
    if far_clip <= near_clip:
        raise ValueError("far_clip must be greater than near_clip")

    # Follow _prepare_depth_for_visualization behavior:
    # clip to [near_clip, far_clip], normalize, then convert to uint8 PNG.
    pred_for_png = np.clip(pred_depth.astype(np.float32), near_clip, far_clip)
    pred_for_png = (pred_for_png - near_clip) / (far_clip - near_clip)
    pred_for_png = (pred_for_png * 255.0).astype(np.uint8)
    imageio.imwrite(str(output_path), pred_for_png)


def _load_gt_depth_for_compare(
    depth_path: Path,
    depth_scale: float,
    target_shape: tuple[int, int],
) -> np.ndarray:
    gt_depth = imageio.imread(str(depth_path))
    gt_depth = gt_depth.astype(np.float32) * depth_scale
    if gt_depth.shape[:2] != target_shape:
        gt_depth = np.asarray(
            Image.fromarray(gt_depth).resize(
                (target_shape[1], target_shape[0]),
                resample=Image.NEAREST,
            )
        )
    return gt_depth


def _compare_prediction_with_gt(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
) -> tuple[np.ndarray, float | None, float | None]:
    valid = np.isfinite(gt_depth)
    mae = None
    rmse = None
    if np.any(valid):
        diff = np.abs(pred_depth[valid] - gt_depth[valid])
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean((pred_depth[valid] - gt_depth[valid]) ** 2)))

    gt_vis = _colorize(_normalize_for_vis(gt_depth), "turbo")
    pred_vis = _colorize(_normalize_for_vis(pred_depth), "turbo")
    err_map = np.zeros_like(pred_depth, dtype=np.float32)
    err_map[valid] = np.abs(pred_depth[valid] - gt_depth[valid])
    err_vis = _colorize(_normalize_for_vis(err_map), "inferno")
    panel = np.concatenate([gt_vis, pred_vis, err_vis], axis=1)
    return panel, mae, rmse


@dataclass
class VisualizationConfig:
    session_dir: Path
    """Path to image_server_images session folder."""

    camera_name: str = "back"
    """Camera folder name (e.g. back/front)."""

    camera_serial: int = 33082869
    """Camera serial number used to compute calibration."""

    zed_settings_dir: str = "/usr/local/zed/settings"
    """Directory containing ZED camera settings files (`SN<serial>.conf`)."""

    zed_resolution: str = "HD"
    """ZED resolution key used when loading calibration (2K/FHD/HD/VGA)."""

    gum_config: GUMConfig = GUMConfig()
    """GUM model configuration."""

    depth_scale: float = 1.0
    """Multiply loaded depth image by this scale."""

    pred_near_clip: float = 0.1
    """Near clip used when saving predicted depth PNG."""

    pred_far_clip: float = 2.0
    """Far clip used when saving predicted depth PNG."""

    max_frames: int = 200
    """Max number of frames to process."""

    output_dir: Path = Path("./gum_real_world_vis")
    """Directory where visualization images/metrics are saved."""

    depth_gum_dirname: str = "depth_gum"
    """Folder name under <session_dir>/<camera_name>/ used to save GUM depth PNGs."""

    compare_with_gt_depth: bool = True
    """If True, load GT depth and generate compare visualizations/metrics."""


def main() -> None:
    cfg = tyro.cli(VisualizationConfig)
    if cfg.compare_with_gt_depth:
        pairs = _discover_frame_pairs(cfg.session_dir, cfg.camera_name)
        if cfg.max_frames > 0:
            pairs = pairs[: cfg.max_frames]
        rgb_frames = [(rgb_path, step) for rgb_path, _depth_path, step in pairs]
        gt_depth_by_step = {step: depth_path for _rgb_path, depth_path, step in pairs}
    else:
        rgb_frames = _discover_rgb_frames(cfg.session_dir, cfg.camera_name)
        if cfg.max_frames > 0:
            rgb_frames = rgb_frames[: cfg.max_frames]
        gt_depth_by_step = {}

    gum = GUM(cfg=cfg.gum_config)

    intrinsics, extrinsics = BuildGUMIntrinsicsAndExtrinsics.compute_from_serial(
        serial_number=cfg.camera_serial,
        zed_settings_dir=cfg.zed_settings_dir,
        resolution=cfg.zed_resolution,
    )

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_gum_dir = cfg.session_dir / cfg.camera_name / cfg.depth_gum_dirname
    depth_gum_dir.mkdir(parents=True, exist_ok=True)

    mae_values: list[float] = []
    rmse_values: list[float] = []
    for rgb_path, step in tqdm(rgb_frames, desc="Running GUM"):
        rgb = imageio.imread(str(rgb_path))
        if rgb is None:
            continue

        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        pred_depth = gum.predict(
            rgb,
            intrinsics,
            extrinsics,
        ).astype(np.float32)

        # 1) Prediction path: save GUM depth into original camera folder.
        depth_gum_path = depth_gum_dir / rgb_path.name
        _save_pred_depth_png(
            pred_depth=pred_depth,
            output_path=depth_gum_path,
            near_clip=cfg.pred_near_clip,
            far_clip=cfg.pred_far_clip,
        )

        # 2) Compare path: load GT depth and compute visualization/metrics.
        if cfg.compare_with_gt_depth:
            depth_path = gt_depth_by_step[step]
            gt_depth = _load_gt_depth_for_compare(
                depth_path=depth_path,
                depth_scale=cfg.depth_scale,
                target_shape=pred_depth.shape[:2],
            )
            panel, mae, rmse = _compare_prediction_with_gt(pred_depth=pred_depth, gt_depth=gt_depth)
            if mae is not None and rmse is not None:
                mae_values.append(mae)
                rmse_values.append(rmse)

            label = f"step={step} | left:gt depth | center:gum pred | right:abs error"
            panel_img = Image.fromarray(panel)
            draw = ImageDraw.Draw(panel_img)
            draw.text((20, 20), label, fill=(255, 255, 255))
            panel_img.save(output_dir / f"compare_{step:06d}.png")

    mean_mae = float(np.mean(mae_values)) if mae_values else float("nan")
    mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
    summary = (
        f"frames_predicted: {len(rgb_frames)}\n"
        f"frames_compared: {len(mae_values)}\n"
        f"mean_mae: {mean_mae:.6f}\n"
        f"mean_rmse: {mean_rmse:.6f}\n"
        f"depth_gum_dir: {depth_gum_dir}\n"
        f"output_dir: {output_dir}\n"
    )
    (output_dir / "metrics.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
