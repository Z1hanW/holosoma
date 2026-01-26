from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.utils.viser_utils import ensure_viser_on_path  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_values import robot as robot_values  # noqa: E402
from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES, MotionLoader  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402
from holosoma.utils.viser_utils import resolve_viser_port  # noqa: E402


@dataclass(frozen=True)
class MotionGeometryViewerConfig:
    motion_dir: str
    geometry_dir: str
    robot: str = "g1_29dof"
    port: int = 0
    fps: int | None = None
    autoplay: bool = True
    loop: bool = True
    preload: bool = True
    show_meshes: bool = True
    show_geometry: bool = True
    add_grid: bool = True
    grid_size: float = 10.0
    start_clip: str | None = None


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _resolve_robot_config(name: str) -> RobotConfig:
    defaults = robot_values.DEFAULTS
    if name not in defaults:
        raise ValueError(f"Unknown robot '{name}'. Available: {sorted(defaults.keys())}")
    return defaults[name]


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> Path:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    return _resolve_data_path(os.path.join(str(asset_root), robot_config.asset.urdf_file))


def _list_pairs(motion_dir: Path, geometry_dir: Path) -> tuple[list[str], dict[str, Path], dict[str, Path]]:
    motion_paths = sorted(list(motion_dir.glob("*.npz")) + list(motion_dir.glob("*.NPZ")))
    geom_paths = sorted(list(geometry_dir.glob("*.obj")) + list(geometry_dir.glob("*.OBJ")))
    motion_map = {path.stem: path for path in motion_paths}
    geom_map = {path.stem: path for path in geom_paths}
    shared = sorted(set(motion_map) & set(geom_map))
    if not shared:
        raise FileNotFoundError(
            f"No matching motion/geometry pairs found. motions={len(motion_paths)}, geometry={len(geom_paths)}"
        )
    missing_geom = sorted(set(motion_map) - set(geom_map))
    missing_motion = sorted(set(geom_map) - set(motion_map))
    if missing_geom:
        logger.warning("No geometry for motions: {}", missing_geom[:10])
    if missing_motion:
        logger.warning("No motion for geometry: {}", missing_motion[:10])
    return shared, motion_map, geom_map


def _try_load_qpos_npz(path: Path) -> tuple[np.ndarray, int] | None:
    if not path.exists() or path.suffix.lower() != ".npz":
        return None
    with np.load(path, allow_pickle=True) as data:
        if "qpos" not in data:
            return None
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps_val = data.get("fps", 30)
        fps = int(np.array(fps_val).reshape(-1)[0]) if fps_val is not None else 30
        return qpos, fps


def _load_motion_qpos(
    motion_path: Path,
    robot_config: RobotConfig,
    viser_joint_names: list[str],
) -> tuple[np.ndarray, int]:
    qpos_payload = _try_load_qpos_npz(motion_path)
    if qpos_payload is not None:
        return qpos_payload

    robot_body_names = [FAKE_BODY_NAME_ALIASES.get(name, name) for name in robot_config.body_names]
    motion = MotionLoader(
        str(motion_path),
        robot_body_names,
        robot_config.dof_names,
        device="cpu",
    )

    name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    joint_order = [name_to_robot_idx[name] for name in viser_joint_names]

    joint_pos_robot = motion.joint_pos.cpu().numpy()
    joint_pos = joint_pos_robot[:, joint_order]

    root_pos = motion.body_pos_w[:, 0].cpu().numpy()
    root_quat_xyzw = motion.body_quat_w[:, 0].cpu().numpy()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    qpos = np.concatenate([root_pos, root_quat_wxyz, joint_pos], axis=1).astype(np.float32, copy=False)
    fps = int(motion.fps) if hasattr(motion, "fps") else 30
    return qpos, fps


def _load_obj_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded geometry is not a trimesh: {type(mesh)}")
    return mesh


def run_viewer(cfg: MotionGeometryViewerConfig) -> None:
    motion_dir = _resolve_data_path(cfg.motion_dir)
    geometry_dir = _resolve_data_path(cfg.geometry_dir)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion dir not found: {motion_dir}")
    if not geometry_dir.is_dir():
        raise FileNotFoundError(f"Geometry dir not found: {geometry_dir}")

    pair_names, motion_map, geom_map = _list_pairs(motion_dir, geometry_dir)

    if cfg.start_clip and cfg.start_clip not in pair_names:
        raise ValueError(f"start_clip '{cfg.start_clip}' not found in pairs.")

    robot_config = _resolve_robot_config(cfg.robot)
    urdf_path = _resolve_robot_urdf_path(robot_config)

    port = resolve_viser_port(cfg.port)
    server = viser.ViserServer(port=port)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    vr = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/robot")
    vr.show_visual = cfg.show_meshes

    viser_joint_names = list(vr.get_actuated_joint_names())
    name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    joint_order = [name_to_robot_idx[name] for name in viser_joint_names]

    if cfg.add_grid:
        server.scene.add_grid(
            "/grid",
            width=cfg.grid_size,
            height=cfg.grid_size,
            position=(0.0, 0.0, 0.0),
        )

    motion_cache: dict[str, dict[str, object]] = {}
    geometry_cache: dict[str, trimesh.Trimesh] = {}

    def _ensure_motion_loaded(name: str) -> dict[str, object]:
        if name in motion_cache:
            return motion_cache[name]
        qpos, fps = _load_motion_qpos(motion_map[name], robot_config, viser_joint_names)
        if qpos.shape[0] == 0:
            raise ValueError(f"Motion {name} has zero frames.")
        motion_cache[name] = {
            "qpos": qpos,
            "fps": int(fps),
            "n_frames": int(qpos.shape[0]),
        }
        return motion_cache[name]

    def _ensure_geometry_loaded(name: str) -> trimesh.Trimesh:
        if name in geometry_cache:
            return geometry_cache[name]
        mesh = _load_obj_mesh(geom_map[name])
        geometry_cache[name] = mesh
        return mesh

    if cfg.preload:
        for name in pair_names:
            _ensure_motion_loaded(name)
            _ensure_geometry_loaded(name)

    geometry_state: dict[str, viser.GlbHandle | None] = {"handle": None}
    motion_state: dict[str, object] = {}

    def _set_geometry(name: str) -> None:
        handle = geometry_state["handle"]
        if handle is not None:
            handle.remove()
            geometry_state["handle"] = None
        mesh = _ensure_geometry_loaded(name)
        geometry_state["handle"] = server.scene.add_mesh_trimesh("/geometry", mesh)

    def _set_motion(name: str) -> None:
        state = _ensure_motion_loaded(name)
        motion_state.update({"name": name, **state})

    active_clip = cfg.start_clip or pair_names[0]
    _set_motion(active_clip)
    _set_geometry(active_clip)

    with server.gui.add_folder("Motion"):
        clip_dropdown = server.gui.add_dropdown("Clip", options=tuple(pair_names), initial_value=active_clip)
        clip_info = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show robot meshes", initial_value=cfg.show_meshes)
        show_geom_cb = server.gui.add_checkbox("Show geometry", initial_value=cfg.show_geometry)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, int(motion_state["n_frames"]) - 1),
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_initial = cfg.fps if cfg.fps is not None else int(motion_state["fps"])
        fps_in = server.gui.add_number("FPS", initial_value=int(fps_initial), min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=cfg.loop)

    def _update_clip_info() -> None:
        clip_info.content = (
            f"Clip: `{motion_state['name']}` | frames: {motion_state['n_frames']} | fps: {motion_state['fps']}"
        )

    _update_clip_info()

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)

    @show_geom_cb.on_update
    def _(_evt) -> None:
        handle = geometry_state["handle"]
        if handle is not None:
            handle.visible = bool(show_geom_cb.value)

    @clip_dropdown.on_update
    def _(_evt) -> None:
        name = str(clip_dropdown.value)
        _set_motion(name)
        _set_geometry(name)
        _update_clip_info()
        frame_slider.max = max(0, int(motion_state["n_frames"]) - 1)
        frame_slider.value = 0
        if cfg.fps is None:
            fps_in.value = int(motion_state["fps"])
        _apply_frame(0)
        handle = geometry_state["handle"]
        if handle is not None:
            handle.visible = bool(show_geom_cb.value)

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

    joint_count = len(viser_joint_names)

    def _apply_frame(frame_idx: int) -> None:
        qpos = motion_state["qpos"]
        qpos_arr = qpos[frame_idx]
        if qpos_arr.shape[0] < 7 + joint_count:
            raise ValueError(f"qpos frame is too small: {qpos_arr.shape[0]} < {7 + joint_count}")
        root_pos = qpos_arr[:3]
        root_quat = qpos_arr[3:7]
        joints = qpos_arr[7 : 7 + joint_count]
        robot_root.position = root_pos
        robot_root.wxyz = root_quat
        vr.update_cfg(joints)

    def _player_loop() -> None:
        next_tick = time.time()
        frame_idx = 0
        while True:
            if playing["flag"]:
                fps_val = int(fps_in.value)
                if fps_val <= 0:
                    fps_val = 1
                now = time.time()
                if now >= next_tick:
                    next_tick = now + 1.0 / float(fps_val)
                    frame_idx = int(frame_slider.value) + 1
                    last_frame = int(motion_state["n_frames"]) - 1
                    if frame_idx > last_frame:
                        if loop_cb.value:
                            frame_idx = 0
                        else:
                            frame_idx = last_frame
                            playing["flag"] = False
                    updating_slider["flag"] = True
                    frame_slider.value = frame_idx
                    updating_slider["flag"] = False
                    _apply_frame(frame_idx)
            time.sleep(0.001)

    _apply_frame(0)
    threading.Thread(target=_player_loop, daemon=True).start()
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")

    while True:
        time.sleep(1.0)


def main() -> None:
    cfg = tyro.cli(MotionGeometryViewerConfig, config=TYRO_CONIFG)
    run_viewer(cfg)


if __name__ == "__main__":
    main()
