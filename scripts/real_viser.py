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


def depth_point_cloud(
    depth_normalized: np.ndarray,
    *,
    near: float,
    far: float,
    horizontal_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an x-forward, y-left, z-up point cloud in camera coordinates."""
    depth_normalized = np.asarray(depth_normalized, dtype=np.float32)
    if depth_normalized.ndim != 2:
        raise ValueError(f"Expected a 2-D depth image, got shape {depth_normalized.shape}")

    height, width = depth_normalized.shape
    depth_m = normalized_depth_to_meters(depth_normalized, near, far)
    focal = (0.5 * width) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x_optical = (u - (width - 1) * 0.5) * depth_m / focal
    y_optical = (v - (height - 1) * 0.5) * depth_m / focal

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

    def __init__(self, name: str, height: int, width: int) -> None:
        self.name = name
        self.shape = (1, 1, height, width)
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
            self.last_error = f"waiting for /dev/shm/{self.name} (start real_depth.sh)"
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
                label="Exact normalized policy input",
            )
            depth_md = server.gui.add_markdown("Waiting for depth shared memory...")

        @show_depth.on_update
        def _(_event) -> None:
            visible = bool(show_depth.value)
            assert depth_root is not None and depth_image is not None
            depth_root.visible = visible
            depth_image.visible = visible

        depth_reader = DepthSharedMemoryReader(args.depth_shm_name, args.depth_height, args.depth_width)

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
                )
                depth_cloud.points = points
                depth_cloud.colors = colors

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
                            f"buffer `{args.depth_shm_name}` · `{args.depth_width}x{args.depth_height}` · "
                            f"valid `{valid_percent:.1f}%`"
                        )

            stop.wait(max(0.0, period - (time.monotonic() - started)))
    finally:
        if depth_reader is not None:
            depth_reader.close()
        try:
            server.stop()
        except AttributeError:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
