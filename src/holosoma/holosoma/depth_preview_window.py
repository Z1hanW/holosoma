from __future__ import annotations

import argparse
import os
import signal
import time
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

import cv2
import numpy as np


def _parse_shape(shape: str, dim: int) -> tuple[int, int]:
    text = str(shape or "").strip().lower().replace(",", "x")
    if "x" in text:
        h_text, w_text = text.split("x", 1)
        height = int(h_text)
        width = int(w_text)
        if height > 0 and width > 0 and height * width == dim:
            return height, width
    if dim == 5046:
        return 58, 87
    if dim == 6360:
        return 60, 106
    root = int(round(float(dim) ** 0.5))
    if root > 0 and root * root == dim:
        return root, root
    return 1, dim


def _open_shm(name: str, dim: int) -> tuple[shared_memory.SharedMemory, np.ndarray] | None:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return None
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    expected_bytes = dim * np.dtype(np.float32).itemsize
    if len(shm.buf) < expected_bytes:
        shm.close()
        return None
    array = np.ndarray((dim,), dtype=np.float32, buffer=shm.buf)
    return shm, array


def _depth_to_bgr(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    frame = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(frame)
    if not np.any(finite):
        norm = np.zeros_like(frame, dtype=np.uint8)
    else:
        finite_values = frame[finite]
        lo = near
        hi = far
        if hi <= lo:
            lo = float(np.nanmin(finite_values))
            hi = float(np.nanmax(finite_values))
        denom = max(hi - lo, 1.0e-6)
        norm_f = np.clip((frame - lo) / denom, 0.0, 1.0)
        norm = (norm_f * 255.0).astype(np.uint8)
        norm[~finite] = 0
    return cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)


def _to_metric_depth(frame: np.ndarray, *, input_mode: str, near: float, far: float) -> np.ndarray:
    mode = str(input_mode or "normalized").strip().lower()
    if mode == "metric":
        return np.asarray(frame, dtype=np.float32)
    if mode == "auto":
        finite = frame[np.isfinite(frame)]
        if finite.size > 0 and float(finite.min()) >= -0.55 and float(finite.max()) <= 0.55 and far > near:
            mode = "normalized"
        else:
            return np.asarray(frame, dtype=np.float32)
    if mode != "normalized":
        raise ValueError(f"Unsupported depth input mode: {input_mode}")
    metric = (np.asarray(frame, dtype=np.float32) + 0.5) * (far - near) + near
    return np.clip(metric, near, far)


def _text_panel(lines: list[str], *, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    y = 22
    for index, line in enumerate(lines):
        color = (240, 240, 240) if index == 0 else (205, 205, 205)
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 18
        if y > height - 10:
            break
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview split MuJoCo depth shared memory in a small window.")
    parser.add_argument("--shm-name", default=os.environ.get("PERCEPTION_OBS_SHM_NAME", "depth_img_shm"))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("DEPTH_PREVIEW_DIM", "5046")))
    parser.add_argument("--shape", default=os.environ.get("DEPTH_PREVIEW_SHAPE", "58x87"))
    parser.add_argument("--scale", type=int, default=int(os.environ.get("DEPTH_PREVIEW_SCALE", "4")))
    parser.add_argument("--fps", type=float, default=float(os.environ.get("DEPTH_PREVIEW_FPS", "20")))
    parser.add_argument("--near", type=float, default=float(os.environ.get("PERCEPTION_CAMERA_NEAR", "0.3") or "0.3"))
    parser.add_argument("--far", type=float, default=float(os.environ.get("PERCEPTION_CAMERA_FAR", "3.0") or "3.0"))
    parser.add_argument(
        "--input-mode",
        choices=("normalized", "metric", "auto"),
        default=os.environ.get("DEPTH_PREVIEW_INPUT_MODE", "normalized"),
        help="Shared-memory depth format. Policy perception_obs is ONNX-normalized by default.",
    )
    parser.add_argument("--source", default=os.environ.get("PERCEPTION_CAMERA_SOURCE", "<default>"))
    parser.add_argument("--preset", default=os.environ.get("PERCEPTION_PRESET", "camera_depth_d435i"))
    parser.add_argument("--object-geometry-mode", default=os.environ.get("PERCEPTION_OBJECT_GEOMETRY_MODE", "<default>"))
    parser.add_argument("--raw-size", default=f"{os.environ.get('PERCEPTION_CAMERA_WIDTH', '<default>')}x{os.environ.get('PERCEPTION_CAMERA_HEIGHT', '<default>')}")
    parser.add_argument("--crop", default="")
    args = parser.parse_args()

    running = True

    def _stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    height, width = _parse_shape(args.shape, args.dim)
    shm: shared_memory.SharedMemory | None = None
    array: np.ndarray | None = None
    window_name = "MuJoCo depth"
    delay_ms = max(1, int(round(1000.0 / max(float(args.fps), 1.0))))
    last_log = 0.0

    try:
        while running:
            if shm is None or array is None:
                opened = _open_shm(args.shm_name, args.dim)
                if opened is None:
                    now = time.monotonic()
                    if now - last_log > 2.0:
                        print(f"[INFO] waiting for depth shm '{args.shm_name}' ({args.dim} float32 values)", flush=True)
                        last_log = now
                    time.sleep(0.1)
                    continue
                shm, array = opened
                print(
                    f"[INFO] depth preview attached shm={args.shm_name} shape={height}x{width} "
                    f"input_mode={args.input_mode} near={args.near:g} far={args.far:g}",
                    flush=True,
                )

            flat = array.copy()
            depth = flat.reshape(height, width)
            metric_depth = _to_metric_depth(
                depth,
                input_mode=str(args.input_mode),
                near=float(args.near),
                far=float(args.far),
            )
            bgr = _depth_to_bgr(metric_depth, near=float(args.near), far=float(args.far))
            if int(args.scale) > 1:
                bgr = cv2.resize(bgr, (width * int(args.scale), height * int(args.scale)), interpolation=cv2.INTER_NEAREST)

            finite = flat[np.isfinite(flat)]
            metric_finite = metric_depth[np.isfinite(metric_depth)]
            min_text = "nan" if finite.size == 0 else f"{float(finite.min()):.3f}"
            max_text = "nan" if finite.size == 0 else f"{float(finite.max()):.3f}"
            metric_min_text = "nan" if metric_finite.size == 0 else f"{float(metric_finite.min()):.3f}m"
            metric_max_text = "nan" if metric_finite.size == 0 else f"{float(metric_finite.max()):.3f}m"
            panel_lines = [
                "Depth preview",
                f"shm: {args.shm_name}",
                f"shape: {height}x{width} dim={args.dim}",
                f"tensor: {min_text}..{max_text}",
                f"metric: {metric_min_text}..{metric_max_text}",
                f"mode: {args.input_mode}",
                f"source: {args.source}",
                f"preset: {args.preset}",
                f"geometry: {args.object_geometry_mode}",
                f"raw: {args.raw_size}",
                f"crop: {args.crop or '<default>'}",
                f"near/far: {args.near:g}/{args.far:g}",
                "Esc/q: close",
            ]
            panel = _text_panel(panel_lines, width=260, height=bgr.shape[0])
            canvas = np.concatenate([bgr, panel], axis=1)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        if shm is not None:
            shm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
