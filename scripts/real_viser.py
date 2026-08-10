#!/usr/bin/env python3
"""Live Viser dashboard for real G1 policy and fixed-pose experiments."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
import webbrowser
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1" / "g1_29dof.urdf"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-path",
        type=Path,
        required=True,
        help="Atomic policy status JSON written by run_policy.",
    )
    parser.add_argument("--depth-shm-name", default="depth_img_shm")
    parser.add_argument("--depth-height", type=int, default=58)
    parser.add_argument("--depth-width", type=int, default=87)
    parser.add_argument("--depth-near", type=float, default=0.3)
    parser.add_argument("--depth-far", type=float, default=3.0)
    parser.add_argument("--horizontal-fov-deg", type=float, default=89.5)
    parser.add_argument("--vertical-fov-deg", type=float, default=58.6)
    parser.add_argument("--depth-profile", default="D435i")
    parser.add_argument("--depth-source-height", type=int)
    parser.add_argument("--depth-source-width", type=int)
    parser.add_argument("--depth-crop-y-start", type=int, default=0)
    parser.add_argument("--depth-crop-y-end", type=int, default=0)
    parser.add_argument("--depth-crop-x-start", type=int, default=0)
    parser.add_argument("--depth-crop-x-end", type=int, default=0)
    parser.add_argument(
        "--sim-gt-depth-shm-name",
        default="",
        help="Optional raw metric MuJoCo GT buffer to show beside real depth.",
    )
    parser.add_argument("--sim-gt-depth-height", type=int, default=60)
    parser.add_argument("--sim-gt-depth-width", type=int, default=106)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--root-height", type=float, default=0.78)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--no-depth", action="store_true", help="Hide the policy-depth panel and point cloud.")
    return parser.parse_args(argv)


def read_status(path: Path) -> dict[str, Any]:
    """Read one atomic status snapshot, returning an empty snapshot while it is unavailable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def normalized_depth_to_meters(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Undo the image server's [-0.5, 0.5] policy-depth normalization."""
    depth = np.asarray(depth, dtype=np.float32)
    return (np.clip(depth, -0.5, 0.5) + 0.5) * (far - near) + near


def meters_to_normalized_depth(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert metric depth to the image server's [-0.5, 0.5] policy range."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    clipped = np.clip(depth_m, near, far)
    return (clipped - near) / (far - near) - 0.5


def prepare_sim_gt_policy_depth(
    raw_depth_m: np.ndarray,
    *,
    near: float,
    far: float,
    output_height: int,
    output_width: int,
    crop_y_start: int = 0,
    crop_y_end: int = 0,
    crop_x_start: int = 0,
    crop_x_end: int = 0,
) -> np.ndarray:
    """Apply the 0mcqao8k crop, cubic resize, clip, and normalization to sim GT."""
    raw_depth_m = np.asarray(raw_depth_m, dtype=np.float32)
    if raw_depth_m.ndim != 2:
        raise ValueError(f"Expected a 2-D sim GT image, got shape {raw_depth_m.shape}")
    height, width = raw_depth_m.shape
    crop_y_stop = height if crop_y_end == 0 else crop_y_end
    crop_x_stop = width if crop_x_end == 0 else crop_x_end
    cropped = raw_depth_m[crop_y_start:crop_y_stop, crop_x_start:crop_x_stop]
    if cropped.size == 0:
        raise ValueError("Sim GT crop must leave a non-empty image")
    resized = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    return meters_to_normalized_depth(resized, near, far)


def depth_colors(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    """Map metric depth to a compact blue-to-red RGB visualization."""
    t = np.clip((np.asarray(depth_m, dtype=np.float32) - near) / max(far - near, 1.0e-6), 0.0, 1.0)
    # A small continuous approximation of a turbo-like map. Close points are
    # warm; far/no-return pixels are dark blue.
    inverse = 1.0 - t
    red = np.clip(1.5 - np.abs(4.0 * inverse - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * inverse - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * inverse - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return np.rint(rgb * 255.0).astype(np.uint8)


def robot_part_depth_comparison(
    real_depth: np.ndarray,
    sim_robot_depth: np.ndarray,
    *,
    near: float,
    far: float,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """Compare normalized depths only where sim GT predicts visible robot geometry."""
    real_depth = np.asarray(real_depth, dtype=np.float32)
    sim_robot_depth = np.asarray(sim_robot_depth, dtype=np.float32)
    if real_depth.shape != sim_robot_depth.shape:
        raise ValueError(f"Real/sim depth shapes differ: {real_depth.shape} != {sim_robot_depth.shape}")

    robot_mask = np.isfinite(sim_robot_depth) & (sim_robot_depth >= -0.5) & (sim_robot_depth < 0.499)
    real_valid = np.isfinite(real_depth) & (real_depth >= -0.5) & (real_depth < 0.499)
    comparable = robot_mask & real_valid
    missing = robot_mask & ~real_valid
    comparison = np.zeros((*real_depth.shape, 3), dtype=np.uint8)

    real_m = normalized_depth_to_meters(real_depth, near, far)
    sim_m = normalized_depth_to_meters(sim_robot_depth, near, far)
    error = np.abs(real_m - sim_m)
    good = comparable & (error <= 0.05)
    medium = comparable & (error > 0.05) & (error <= 0.15)
    bad = comparable & (error > 0.15)
    comparison[good] = (0, 220, 70)
    comparison[medium] = (255, 200, 0)
    comparison[bad] = (255, 45, 30)
    comparison[missing] = (220, 0, 255)

    robot_pixels = int(np.count_nonzero(robot_mask))
    compared_pixels = int(np.count_nonzero(comparable))
    compared_error = error[comparable]
    stats: dict[str, float | int | None] = {
        "robot_pixels": robot_pixels,
        "compared_pixels": compared_pixels,
        "coverage_percent": 100.0 * compared_pixels / robot_pixels if robot_pixels else 0.0,
        "mae_m": float(np.mean(compared_error)) if compared_pixels else None,
    }
    return comparison, stats


def depth_point_cloud(
    depth_normalized: np.ndarray,
    *,
    near: float,
    far: float,
    horizontal_fov_deg: float,
    vertical_fov_deg: float | None = None,
    source_height: int | None = None,
    source_width: int | None = None,
    crop_y_start: int = 0,
    crop_y_end: int = 0,
    crop_x_start: int = 0,
    crop_x_end: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an x-forward, y-left, z-up point cloud in camera coordinates."""
    depth_normalized = np.asarray(depth_normalized, dtype=np.float32)
    if depth_normalized.ndim != 2:
        raise ValueError(f"Expected a 2-D depth image, got shape {depth_normalized.shape}")

    height, width = depth_normalized.shape
    depth_m = normalized_depth_to_meters(depth_normalized, near, far)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    if source_height is None or source_width is None:
        focal = (0.5 * width) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
        x_ray = (u - (width - 1) * 0.5) / focal
        y_ray = (v - (height - 1) * 0.5) / focal
    else:
        vertical_fov_deg = horizontal_fov_deg if vertical_fov_deg is None else vertical_fov_deg
        crop_x_stop = source_width if crop_x_end == 0 else crop_x_end
        crop_y_stop = source_height if crop_y_end == 0 else crop_y_end
        if crop_x_stop < 0:
            crop_x_stop += source_width
        if crop_y_stop < 0:
            crop_y_stop += source_height
        cropped_width = crop_x_stop - crop_x_start
        cropped_height = crop_y_stop - crop_y_start
        if cropped_width <= 0 or cropped_height <= 0:
            raise ValueError("Depth crop must leave a positive image size")

        # Map output pixel centers back through crop+resize to the rendered
        # camera image, then use the original 0mcqao8k render intrinsics.
        source_u = (u + 0.5) * (cropped_width / width) - 0.5 + crop_x_start
        source_v = (v + 0.5) * (cropped_height / height) - 0.5 + crop_y_start
        focal_x = (0.5 * source_width) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
        focal_y = (0.5 * source_height) / math.tan(math.radians(vertical_fov_deg) * 0.5)
        x_ray = (source_u - (source_width - 1) * 0.5) / focal_x
        y_ray = (source_v - (source_height - 1) * 0.5) / focal_y

    x_optical = x_ray * depth_m
    y_optical = y_ray * depth_m

    valid = np.isfinite(depth_normalized)
    valid &= depth_normalized < 0.499
    valid &= depth_normalized >= -0.5
    points = np.stack((depth_m, -x_optical, -y_optical), axis=-1)[valid]
    colors = depth_colors(depth_m, near, far)[valid]
    return points.astype(np.float32, copy=False), colors


def joint_values_in_viser_order(
    values: Sequence[float],
    source_names: Sequence[str],
    viser_names: Sequence[str],
) -> np.ndarray:
    """Reorder inference joint vectors to ViserUrdf's actuated-joint order."""
    by_name = {str(name): float(value) for name, value in zip(source_names, values)}
    return np.asarray([by_name.get(str(name), 0.0) for name in viser_names], dtype=np.float32)


def normalized_wxyz(value: Sequence[float]) -> tuple[float, float, float, float]:
    quat = np.asarray(value, dtype=np.float32).reshape(-1)
    if quat.size != 4 or not bool(np.isfinite(quat).all()):
        return (1.0, 0.0, 0.0, 0.0)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-6:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(item) for item in quat / norm)


class DepthSharedMemoryReader:
    """Best-effort reader that never owns or unlinks the image-server buffer."""

    def __init__(self, name: str, height: int, width: int, source_hint: str = "start real_depth.sh") -> None:
        self.name = name
        self.shape = (1, 1, height, width)
        self.source_hint = source_hint
        self._shm: shared_memory.SharedMemory | None = None
        self._array: np.ndarray | None = None
        self.last_error = "waiting for depth shared memory"

    def read(self) -> np.ndarray | None:
        if self._array is None and not self._attach():
            return None
        assert self._array is not None
        return np.array(self._array[0, 0], dtype=np.float32, copy=True)

    def _attach(self) -> bool:
        try:
            shm = shared_memory.SharedMemory(name=self.name)
        except FileNotFoundError:
            self.last_error = f"waiting for /dev/shm/{self.name} ({self.source_hint})"
            return False

        # Python 3.10 has no SharedMemory(track=False). Unregister this
        # non-owning attachment immediately so every failure/exit path leaves
        # ownership of the buffer with the image server.
        resource_tracker.unregister(shm._name, "shared_memory")

        required_size = int(np.prod(self.shape)) * np.dtype(np.float32).itemsize
        if shm.size < required_size:
            self.last_error = f"depth buffer has {shm.size} bytes; expected at least {required_size}"
            shm.close()
            return False

        self._shm = shm
        self._array = np.ndarray(self.shape, dtype=np.float32, buffer=shm.buf)
        self.last_error = ""
        return True

    def close(self) -> None:
        self._array = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None


def _open_browser_later(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        webbrowser.open(url, new=2)

    threading.Thread(target=_open, daemon=True).start()


def _import_viser():
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        print(
            "[real_viser] Viser dependencies are missing. Activate hsinference and run "
            "`python3 -m pip install 'viser[urdf]>=1.0,<2.0'`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return viser, ViserUrdf


def run(args: argparse.Namespace) -> None:
    viser, ViserUrdf = _import_viser()
    urdf_path = args.urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"G1 URDF not found: {urdf_path}")

    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    try:
        server.gui.configure_theme(dark_mode=True, control_layout="collapsible", control_width="medium")
    except (AttributeError, TypeError):
        pass

    actual_root = server.scene.add_frame("/robot_actual", show_axes=False)
    target_root = server.scene.add_frame("/robot_target", show_axes=False)
    server.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))

    actual_urdf = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/robot_actual")
    target_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robot_target",
        mesh_color_override=(1.0, 0.5, 0.05, 0.32),
    )
    viser_joint_names = tuple(actual_urdf.get_actuated_joint_names())

    with server.gui.add_folder("Live experiment", order=10.0):
        status_md = server.gui.add_markdown("Waiting for policy telemetry...")
        show_actual = server.gui.add_checkbox("Measured robot", initial_value=True)
        show_target = server.gui.add_checkbox("Target overlay", initial_value=True)
        apply_imu = server.gui.add_checkbox("Apply IMU orientation", initial_value=True)

    depth_root = None
    depth_cloud = None
    depth_image = None
    depth_md = None
    depth_reader = None
    sim_gt_root = None
    sim_gt_cloud = None
    sim_gt_image = None
    sim_gt_md = None
    sim_gt_reader = None
    comparison_image = None
    comparison_md = None
    if not args.no_depth:
        depth_root = server.scene.add_frame(
            "/depth_camera_view",
            position=(0.0, -1.35, 0.75),
            show_axes=True,
        )
        depth_cloud = server.scene.add_point_cloud(
            "/depth_camera_view/policy_depth",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            point_size=0.012,
            point_shape="circle",
        )
        with server.gui.add_folder("Policy depth", order=20.0):
            show_depth = server.gui.add_checkbox("Show depth", initial_value=True)
            depth_image = server.gui.add_image(
                np.zeros((args.depth_height, args.depth_width, 3), dtype=np.uint8),
                label=f"{args.depth_profile}: exact normalized policy input",
            )
            depth_md = server.gui.add_markdown("Waiting for depth shared memory...")

        @show_depth.on_update
        def _(_event) -> None:
            visible = bool(show_depth.value)
            assert depth_root is not None and depth_image is not None
            depth_root.visible = visible
            depth_image.visible = visible

        depth_reader = DepthSharedMemoryReader(args.depth_shm_name, args.depth_height, args.depth_width)

    if not args.no_depth and args.sim_gt_depth_shm_name:
        sim_gt_root = server.scene.add_frame(
            "/sim_gt_depth_camera_view",
            position=(0.0, 1.35, 0.75),
            show_axes=True,
        )
        sim_gt_cloud = server.scene.add_point_cloud(
            "/sim_gt_depth_camera_view/ground_truth_depth",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            point_size=0.012,
            point_shape="circle",
        )
        with server.gui.add_folder("MuJoCo sim GT", order=30.0):
            show_sim_gt = server.gui.add_checkbox("Show sim GT", initial_value=True)
            sim_gt_image = server.gui.add_image(
                np.zeros((args.depth_height, args.depth_width, 3), dtype=np.uint8),
                label="Robot parts only: MuJoCo metric GT → 0mcqao8k preprocessing",
            )
            sim_gt_md = server.gui.add_markdown("Waiting for MuJoCo GT shared memory...")
            comparison_image = server.gui.add_image(
                np.zeros((args.depth_height, args.depth_width, 3), dtype=np.uint8),
                label="Real vs sim robot parts: green ≤5cm · yellow ≤15cm · red >15cm · magenta missing",
            )
            comparison_md = server.gui.add_markdown("Waiting for both real and robot-only sim depth...")

        @show_sim_gt.on_update
        def _(_event) -> None:
            visible = bool(show_sim_gt.value)
            assert sim_gt_root is not None and sim_gt_image is not None
            sim_gt_root.visible = visible
            sim_gt_image.visible = visible

        sim_gt_reader = DepthSharedMemoryReader(
            args.sim_gt_depth_shm_name,
            args.sim_gt_depth_height,
            args.sim_gt_depth_width,
            source_hint="start sim_gt_depth.sh",
        )

    @show_actual.on_update
    def _(_event) -> None:
        actual_root.visible = bool(show_actual.value)

    @show_target.on_update
    def _(_event) -> None:
        target_root.visible = bool(show_target.value)

    stop = threading.Event()

    def _stop(_signum=None, _frame=None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}"
    print(f"[real_viser] live viewer: {url}", flush=True)
    if args.open_browser:
        _open_browser_later(url)

    period = 1.0 / max(float(args.rate_hz), 1.0)
    last_status_text_update = 0.0
    try:
        while not stop.is_set():
            started = time.monotonic()
            status = read_status(args.state_path)
            source_names = status.get("dof_names", ())
            q_actual = status.get("q_actual", ())
            q_target = status.get("q_target", ())
            if (
                isinstance(source_names, list)
                and isinstance(q_actual, list)
                and isinstance(q_target, list)
                and len(source_names) == len(q_actual) == len(q_target)
            ):
                actual_cfg = joint_values_in_viser_order(q_actual, source_names, viser_joint_names)
                target_cfg = joint_values_in_viser_order(q_target, source_names, viser_joint_names)
                actual_urdf.update_cfg(actual_cfg)
                target_urdf.update_cfg(target_cfg)

                base_position = np.asarray(status.get("base_position", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(-1)
                root_position = np.array((0.0, 0.0, args.root_height), dtype=np.float32)
                if base_position.size == 3 and bool(np.isfinite(base_position).all()):
                    root_position[:2] = base_position[:2]
                    if abs(float(base_position[2])) > 0.1:
                        root_position[2] = base_position[2]
                root_wxyz = normalized_wxyz(status.get("base_wxyz", (1.0, 0.0, 0.0, 0.0)))
                if not bool(apply_imu.value):
                    root_wxyz = (1.0, 0.0, 0.0, 0.0)
                actual_root.position = tuple(float(item) for item in root_position)
                actual_root.wxyz = root_wxyz
                target_root.position = actual_root.position
                target_root.wxyz = root_wxyz

            depth = depth_reader.read() if depth_reader is not None else None
            if depth is not None and depth_image is not None and depth_cloud is not None:
                depth_m = normalized_depth_to_meters(depth, args.depth_near, args.depth_far)
                rgb = depth_colors(depth_m, args.depth_near, args.depth_far)
                depth_image.image = rgb
                points, colors = depth_point_cloud(
                    depth,
                    near=args.depth_near,
                    far=args.depth_far,
                    horizontal_fov_deg=args.horizontal_fov_deg,
                    vertical_fov_deg=args.vertical_fov_deg,
                    source_height=args.depth_source_height,
                    source_width=args.depth_source_width,
                    crop_y_start=args.depth_crop_y_start,
                    crop_y_end=args.depth_crop_y_end,
                    crop_x_start=args.depth_crop_x_start,
                    crop_x_end=args.depth_crop_x_end,
                )
                depth_cloud.points = points
                depth_cloud.colors = colors

            sim_gt_raw = sim_gt_reader.read() if sim_gt_reader is not None else None
            sim_gt_depth = None
            comparison_stats = None
            if sim_gt_raw is not None and sim_gt_image is not None and sim_gt_cloud is not None:
                sim_gt_depth = prepare_sim_gt_policy_depth(
                    sim_gt_raw,
                    near=args.depth_near,
                    far=args.depth_far,
                    output_height=args.depth_height,
                    output_width=args.depth_width,
                    crop_y_start=args.depth_crop_y_start,
                    crop_y_end=args.depth_crop_y_end,
                    crop_x_start=args.depth_crop_x_start,
                    crop_x_end=args.depth_crop_x_end,
                )
                sim_gt_m = normalized_depth_to_meters(sim_gt_depth, args.depth_near, args.depth_far)
                sim_gt_image.image = depth_colors(sim_gt_m, args.depth_near, args.depth_far)
                points, colors = depth_point_cloud(
                    sim_gt_depth,
                    near=args.depth_near,
                    far=args.depth_far,
                    horizontal_fov_deg=args.horizontal_fov_deg,
                    vertical_fov_deg=args.vertical_fov_deg,
                    source_height=args.sim_gt_depth_height,
                    source_width=args.sim_gt_depth_width,
                    crop_y_start=args.depth_crop_y_start,
                    crop_y_end=args.depth_crop_y_end,
                    crop_x_start=args.depth_crop_x_start,
                    crop_x_end=args.depth_crop_x_end,
                )
                sim_gt_cloud.points = points
                sim_gt_cloud.colors = colors
                if depth is not None and comparison_image is not None:
                    comparison_rgb, comparison_stats = robot_part_depth_comparison(
                        depth,
                        sim_gt_depth,
                        near=args.depth_near,
                        far=args.depth_far,
                    )
                    comparison_image.image = comparison_rgb

            now = time.time()
            if now - last_status_text_update >= 0.2:
                last_status_text_update = now
                timestamp = float(status.get("timestamp", 0.0) or 0.0)
                age = now - timestamp if timestamp else math.inf
                command = status.get("command", ())
                command_text = "waiting"
                if isinstance(command, list) and len(command) >= 3:
                    command_x, command_y, command_yaw = (float(value) for value in command[:3])
                    command_text = f"x={command_x:+.3f}, y={command_y:+.3f}, yaw={command_yaw:+.3f}"
                error_text = "n/a"
                if (
                    isinstance(q_actual, list)
                    and isinstance(q_target, list)
                    and len(q_actual) == len(q_target)
                    and q_actual
                ):
                    error = np.asarray(q_target, dtype=np.float32) - np.asarray(q_actual, dtype=np.float32)
                    error_absmax = float(np.max(np.abs(error)))
                    error_rms = float(np.sqrt(np.mean(error**2)))
                    error_text = f"max={error_absmax:.3f} rad, rms={error_rms:.3f} rad"
                state_label = "LIVE" if age < 0.5 else ("STALE" if math.isfinite(age) else "WAITING")
                mode_label = "STIFF HOLD" if bool(status.get("stiff_hold_active", False)) else "POLICY"
                status_md.content = (
                    f"**{state_label} · {mode_label}** · telemetry age `{age:.2f}s`  \n"
                    f"policy active `{bool(status.get('use_policy_action', False))}` · "
                    f"motion frame `{int(status.get('motion_timestep', 0) or 0)}`  \n"
                    f"command `{command_text}`  \n"
                    f"joint error `{error_text}`"
                )
                if depth_reader is not None and depth_md is not None:
                    if depth is None:
                        depth_md.content = depth_reader.last_error
                    else:
                        finite = np.isfinite(depth)
                        valid = finite & (depth < 0.499)
                        valid_percent = 100.0 * float(np.count_nonzero(valid)) / float(depth.size)
                        depth_md.content = (
                            f"profile `{args.depth_profile}` · buffer `{args.depth_shm_name}` · "
                            f"policy input `{args.depth_width}x{args.depth_height}` · "
                            f"valid `{valid_percent:.1f}%`"
                        )
                if sim_gt_reader is not None and sim_gt_md is not None:
                    if sim_gt_depth is None:
                        sim_gt_md.content = sim_gt_reader.last_error
                    else:
                        valid = np.isfinite(sim_gt_depth) & (sim_gt_depth < 0.499)
                        valid_percent = 100.0 * float(np.count_nonzero(valid)) / float(sim_gt_depth.size)
                        sim_gt_md.content = (
                            "geometry `G1 visual meshes only` · "
                            f"raw metric buffer `{args.sim_gt_depth_shm_name}` · "
                            f"render `{args.sim_gt_depth_width}x{args.sim_gt_depth_height}` · "
                            f"policy input `{args.depth_width}x{args.depth_height}` · "
                            f"valid `{valid_percent:.1f}%`"
                        )
                if comparison_md is not None:
                    if comparison_stats is None:
                        comparison_md.content = "Waiting for both real and robot-only sim depth..."
                    else:
                        mae = comparison_stats["mae_m"]
                        mae_text = "n/a" if mae is None else f"{float(mae) * 100.0:.1f} cm"
                        comparison_md.content = (
                            f"robot pixels `{comparison_stats['robot_pixels']}` · "
                            f"real coverage `{float(comparison_stats['coverage_percent']):.1f}%` · "
                            f"robot-only MAE `{mae_text}`"
                        )

            stop.wait(max(0.0, period - (time.monotonic() - started)))
    finally:
        if depth_reader is not None:
            depth_reader.close()
        if sim_gt_reader is not None:
            sim_gt_reader.close()
        try:
            server.stop()
        except AttributeError:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
