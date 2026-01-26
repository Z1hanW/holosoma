from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402


@dataclass(frozen=True)
class JointViewerConfig:
    motion_dir: str
    port: int = 0
    fps: int | None = None
    autoplay: bool = True
    loop: bool = True
    preload: bool = True
    add_grid: bool = True
    grid_size: float = 10.0
    point_size: float = 0.02
    point_shape: str = "circle"
    start_clip: str | None = None


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _list_motion_files(motion_path: Path) -> tuple[list[str], dict[str, Path]]:
    if motion_path.is_file():
        return [motion_path.stem], {motion_path.stem: motion_path}
    motion_paths = sorted(list(motion_path.glob("*.npz")) + list(motion_path.glob("*.NPZ")))
    if not motion_paths:
        raise FileNotFoundError(f"No motion files found in: {motion_path}")
    motion_map = {path.stem: path for path in motion_paths}
    return sorted(motion_map.keys()), motion_map


def _load_joint_positions(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=True) as data:
        if "global_joint_positions" not in data:
            raise KeyError(f"Missing 'global_joint_positions' in {path}")
        joints = np.asarray(data["global_joint_positions"], dtype=np.float32)
        if joints.ndim == 3 and joints.shape[-1] == 3:
            pass
        elif joints.ndim == 3 and joints.shape[1] == 3:
            joints = np.transpose(joints, (0, 2, 1))
        elif joints.ndim == 2 and joints.shape[-1] % 3 == 0:
            joint_count = joints.shape[-1] // 3
            joints = joints.reshape(joints.shape[0], joint_count, 3)
        else:
            raise ValueError(f"Unsupported joints shape {joints.shape} in {path}")

        fps_val = data.get("mocap_framerate", data.get("fps", 30))
        fps = int(np.array(fps_val).reshape(-1)[0]) if fps_val is not None else 30
        return joints, fps


def _make_joint_colors(joint_count: int) -> np.ndarray:
    if joint_count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    colors = np.zeros((joint_count, 3), dtype=np.uint8)
    for idx in range(joint_count):
        hue = idx / max(1, joint_count - 1)
        colors[idx] = np.array(
            [int(255 * (1 - hue)), int(128 + 127 * hue), int(255 * hue)], dtype=np.uint8
        )
    return colors


def run_viewer(cfg: JointViewerConfig) -> None:
    motion_root = _resolve_data_path(cfg.motion_dir)
    if not motion_root.exists():
        raise FileNotFoundError(f"Motion dir not found: {motion_root}")

    clip_names, motion_map = _list_motion_files(motion_root)
    if cfg.start_clip and cfg.start_clip not in clip_names:
        raise ValueError(f"start_clip '{cfg.start_clip}' not found in motions.")

    port = resolve_viser_port(cfg.port)
    server = viser.ViserServer(port=port)

    if cfg.add_grid:
        server.scene.add_grid(
            "/grid",
            width=cfg.grid_size,
            height=cfg.grid_size,
            position=(0.0, 0.0, 0.0),
        )

    motion_cache: dict[str, dict[str, object]] = {}

    def _ensure_motion_loaded(name: str) -> dict[str, object]:
        if name in motion_cache:
            return motion_cache[name]
        joints, fps = _load_joint_positions(motion_map[name])
        if joints.shape[0] == 0:
            raise ValueError(f"Motion {name} has zero frames.")
        motion_cache[name] = {
            "joints": joints,
            "fps": int(fps),
            "n_frames": int(joints.shape[0]),
        }
        return motion_cache[name]

    if cfg.preload:
        for name in clip_names:
            _ensure_motion_loaded(name)

    joint_state: dict[str, object] = {}

    def _set_motion(name: str) -> None:
        state = _ensure_motion_loaded(name)
        joint_state.update({"name": name, **state})

    active_clip = cfg.start_clip or clip_names[0]
    _set_motion(active_clip)

    joint_colors = _make_joint_colors(int(joint_state["joints"].shape[1]))
    joint_handle = server.scene.add_point_cloud(
        "/joints",
        points=joint_state["joints"][0],
        colors=joint_colors,
        point_size=float(cfg.point_size),
        point_shape=str(cfg.point_shape),
    )

    with server.gui.add_folder("Motion"):
        clip_dropdown = server.gui.add_dropdown("Clip", options=tuple(clip_names), initial_value=active_clip)
        clip_info = server.gui.add_markdown("")

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, int(joint_state["n_frames"]) - 1),
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_initial = cfg.fps if cfg.fps is not None else int(joint_state["fps"])
        fps_in = server.gui.add_number("FPS", initial_value=int(fps_initial), min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=cfg.loop)

    def _update_clip_info() -> None:
        clip_info.content = (
            f"Clip: `{joint_state['name']}` | frames: {joint_state['n_frames']} | fps: {joint_state['fps']}"
        )

    _update_clip_info()

    @clip_dropdown.on_update
    def _(_evt) -> None:
        name = str(clip_dropdown.value)
        _set_motion(name)
        _update_clip_info()
        frame_slider.max = max(0, int(joint_state["n_frames"]) - 1)
        frame_slider.value = 0
        if cfg.fps is None:
            fps_in.value = int(joint_state["fps"])
        _apply_frame(0)

    playing = {"flag": bool(cfg.autoplay)}
    updating_slider = {"flag": False}

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_slider["flag"]:
            return
        _apply_frame(int(frame_slider.value))

    def _apply_frame(frame_idx: int) -> None:
        joints = joint_state["joints"][frame_idx]
        joint_handle.points = joints

    def _player_loop() -> None:
        while True:
            if playing["flag"]:
                fps_val = int(fps_in.value)
                if fps_val <= 0:
                    fps_val = 1
                next_frame = int(frame_slider.value) + 1
                last_frame = int(joint_state["n_frames"]) - 1
                if next_frame > last_frame:
                    if loop_cb.value:
                        next_frame = 0
                    else:
                        next_frame = last_frame
                        playing["flag"] = False
                updating_slider["flag"] = True
                frame_slider.value = next_frame
                updating_slider["flag"] = False
                _apply_frame(next_frame)
            time.sleep(1.0 / max(1.0, float(fps_in.value)))

    _apply_frame(0)
    threading.Thread(target=_player_loop, daemon=True).start()
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")

    while True:
        time.sleep(1.0)


def main() -> None:
    cfg = tyro.cli(JointViewerConfig, config=TYRO_CONIFG)
    run_viewer(cfg)


if __name__ == "__main__":
    main()
