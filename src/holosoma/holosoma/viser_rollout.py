from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import trimesh

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[3]
VISER_SRC = REPO_ROOT / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.insert(0, str(VISER_SRC))

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


@dataclass(frozen=True)
class RolloutViewerConfig:
    rollout_dir: str
    rollout_file: str | None = None
    terrain_obj_path: str | None = None
    recenter: bool = True


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> str:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _decode_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _list_rollout_files(rollout_dir: Path) -> list[Path]:
    files = sorted(rollout_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No rollout files found in: {rollout_dir}")
    return files


def _load_rollout(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        if "qpos" not in data:
            raise ValueError(f"Missing 'qpos' in rollout file: {path}")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps_val = data.get("fps", 30)
        fps = float(np.array(fps_val).reshape(-1)[0]) if fps_val is not None else 30.0
        clip_name = _decode_str(data.get("clip_name"))
        env_origin = data.get("env_origin")
        if env_origin is not None:
            env_origin = np.asarray(env_origin, dtype=np.float32).reshape(-1)
        terrain_obj_path = _decode_str(data.get("terrain_obj_path"))
        terrain_rows = data.get("terrain_num_rows")
        terrain_cols = data.get("terrain_num_cols")
        terrain_rows_val = int(np.array(terrain_rows).reshape(-1)[0]) if terrain_rows is not None else None
        terrain_cols_val = int(np.array(terrain_cols).reshape(-1)[0]) if terrain_cols is not None else None
    return {
        "qpos": qpos,
        "fps": fps,
        "clip_name": clip_name,
        "env_origin": env_origin,
        "terrain_obj_path": terrain_obj_path,
        "terrain_num_rows": terrain_rows_val,
        "terrain_num_cols": terrain_cols_val,
    }


def _load_terrain_mesh(
    obj_path: str | None,
    *,
    clip_name: str | None,
    num_rows: int | None,
    num_cols: int | None,
) -> trimesh.Trimesh | None:
    if not obj_path:
        return None

    terrain_path = Path(_resolve_data_path(obj_path))

    def _load_mesh(path: Path) -> trimesh.Trimesh:
        base_mesh = trimesh.load(str(path), process=False)
        if isinstance(base_mesh, trimesh.Scene):
            base_mesh = base_mesh.dump(concatenate=True)
        if not isinstance(base_mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded terrain is not a trimesh: {type(base_mesh)}")
        return base_mesh

    if terrain_path.is_dir():
        obj_paths = sorted(list(terrain_path.glob("*.obj")) + list(terrain_path.glob("*.OBJ")))
        if not obj_paths:
            return None
        chosen = None
        if clip_name:
            for path in obj_paths:
                if path.stem == clip_name:
                    chosen = path
                    break
        if chosen is None:
            chosen = obj_paths[0]
        return _load_mesh(chosen)

    if not terrain_path.exists():
        return None

    base = _load_mesh(terrain_path)
    rows = int(num_rows or 1)
    cols = int(num_cols or 1)
    if rows * cols <= 1:
        return base

    gap = 1e-4
    stride = (base.bounds[1] - base.bounds[0]) + gap
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tile = base.copy()
            tile.apply_translation([c * stride[0], r * stride[1], 0.0])
            tiles.append(tile)
    return trimesh.util.concatenate(tiles)


def replay_rollout(cfg: ExperimentConfig, rollout_cfg: RolloutViewerConfig) -> None:
    rollout_dir = Path(rollout_cfg.rollout_dir).expanduser()
    files = _list_rollout_files(rollout_dir)
    file_map = {path.stem: path for path in files}

    if rollout_cfg.rollout_file:
        initial_path = Path(rollout_cfg.rollout_file).expanduser()
        if initial_path.is_dir():
            raise ValueError("rollout_file must be a file, not a directory.")
        if not initial_path.exists():
            raise FileNotFoundError(f"rollout_file not found: {initial_path}")
        initial_key = initial_path.stem
        file_map[initial_key] = initial_path
    else:
        initial_key = files[0].stem

    server = viser.ViserServer(port=int(os.environ.get("HOLOSOMA_VISER_PORT", "6060")))
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    robot_urdf_path = _resolve_robot_urdf_path(cfg.robot)
    vr = ViserUrdf(server, urdf_or_path=Path(robot_urdf_path), root_node_name="/robot")

    vo = None
    if cfg.robot.object.object_urdf_path:
        object_urdf_path = _resolve_data_path(cfg.robot.object.object_urdf_path)
        vo = ViserUrdf(server, urdf_or_path=Path(object_urdf_path), root_node_name="/object")

    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))
    ground_mesh = trimesh.creation.box(extents=(8.0, 8.0, 0.01))
    ground_mesh.apply_translation([0.0, 0.0, -0.005])
    ground_handle = server.scene.add_mesh_trimesh("/ground", ground_mesh)

    state: dict[str, object] = {"offset": np.zeros(3, dtype=np.float32)}
    terrain_handle: viser.GlbHandle | None = None

    with server.gui.add_folder("Rollout"):
        dropdown = server.gui.add_dropdown("File", options=tuple(sorted(file_map.keys())), initial_value=initial_key)
        reload_btn = server.gui.add_button("Reload")
        file_info = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=True)
        show_terrain_cb = server.gui.add_checkbox("Show terrain", initial_value=True)
        recenter_cb = server.gui.add_checkbox("Recenter", initial_value=rollout_cfg.recenter)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=30, min=1, max=240, step=1)
        interp_mult_in = server.gui.add_number("Visual FPS multiplier", initial_value=2, min=1, max=8, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=False)

    def _update_file_info() -> None:
        clip = state.get("clip_name")
        fps_val = int(state.get("fps", 0))
        n_frames = int(state.get("n_frames", 0))
        clip_str = clip if clip else "n/a"
        file_info.content = f"Clip: `{clip_str}` | frames: {n_frames} | fps: {fps_val}"

    def _set_terrain(clip_name: str | None, terrain_path: str | None, rows: int | None, cols: int | None) -> None:
        nonlocal terrain_handle
        if terrain_handle is not None:
            terrain_handle.remove()
            terrain_handle = None
        mesh = _load_terrain_mesh(terrain_path, clip_name=clip_name, num_rows=rows, num_cols=cols)
        if mesh is not None:
            terrain_handle = server.scene.add_mesh_trimesh("/terrain", mesh)
            terrain_handle.visible = bool(show_terrain_cb.value)
            ground_handle.visible = False
        else:
            ground_handle.visible = bool(show_terrain_cb.value)

    def _set_offset(qpos: np.ndarray, env_origin: np.ndarray | None) -> None:
        if not recenter_cb.value:
            state["offset"] = np.zeros(3, dtype=np.float32)
            return
        if env_origin is not None and env_origin.shape[0] >= 3:
            state["offset"] = env_origin[:3].astype(np.float32, copy=False)
            return
        state["offset"] = qpos[0, 0:3].astype(np.float32, copy=False)

    def _load_file(key: str) -> None:
        path = file_map[key]
        payload = _load_rollout(path)
        qpos = np.asarray(payload["qpos"], dtype=np.float32)
        fps_val = int(payload["fps"]) if payload["fps"] else 30
        clip_name = payload.get("clip_name")
        env_origin = payload.get("env_origin")

        terrain_path = rollout_cfg.terrain_obj_path or payload.get("terrain_obj_path")
        terrain_rows = payload.get("terrain_num_rows")
        terrain_cols = payload.get("terrain_num_cols")

        state.update(
            {
                "qpos": qpos,
                "fps": fps_val,
                "clip_name": clip_name,
                "env_origin": env_origin,
                "n_frames": int(qpos.shape[0]),
                "has_object": bool(vo is not None and qpos.shape[1] >= (7 + len(cfg.robot.dof_names) + 7)),
            }
        )
        _set_offset(qpos, env_origin if isinstance(env_origin, np.ndarray) else None)
        _set_terrain(clip_name if isinstance(clip_name, str) else None, terrain_path, terrain_rows, terrain_cols)

        frame_slider.max = max(0, int(qpos.shape[0] - 1))
        frame_slider.value = 0
        fps_in.value = fps_val
        _update_file_info()
        _apply_frame_from_float(0.0)

    def _apply_frame(frame: np.ndarray) -> None:
        offset = state.get("offset", np.zeros(3, dtype=np.float32))
        root_pos = frame[0:3] - offset
        root_quat_wxyz = frame[3:7]
        joints = frame[7 : 7 + len(cfg.robot.dof_names)]

        robot_root.position = root_pos
        robot_root.wxyz = root_quat_wxyz
        vr.update_cfg(joints.astype(np.float32, copy=False))

        if vo is None:
            return
        if state.get("has_object"):
            vo.show_visual = True
            object_root.position = frame[-7:-4] - offset
            object_root.wxyz = frame[-4:]
        else:
            vo.show_visual = False

    def _interp_qpos(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
        out = q0.copy()
        out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]
        out[3:7] = _slerp(q0[3:7], q1[3:7], u)
        dof = len(cfg.robot.dof_names)
        out[7 : 7 + dof] = (1.0 - u) * q0[7 : 7 + dof] + u * q1[7 : 7 + dof]
        if state.get("has_object"):
            out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]
            out[-4:] = _slerp(q0[-4:], q1[-4:], u)
        return out

    def _apply_frame_from_float(f_val: float) -> None:
        qpos = np.asarray(state.get("qpos"))
        n_frames = int(state.get("n_frames", 0))
        if qpos is None or n_frames == 0:
            return
        i0 = int(np.clip(np.floor(f_val), 0, n_frames - 1))
        i1 = min(i0 + 1, n_frames - 1)
        u = float(f_val - i0)
        if i0 == i1 or u <= 1.0e-6:
            _apply_frame(qpos[i0])
        else:
            _apply_frame(_interp_qpos(qpos[i0], qpos[i1], u))

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    @show_terrain_cb.on_update
    def _(_evt) -> None:
        if terrain_handle is not None:
            terrain_handle.visible = bool(show_terrain_cb.value)
        else:
            ground_handle.visible = bool(show_terrain_cb.value)

    @recenter_cb.on_update
    def _(_evt) -> None:
        qpos = np.asarray(state.get("qpos"))
        if qpos is None:
            return
        env_origin = state.get("env_origin")
        _set_offset(qpos, env_origin if isinstance(env_origin, np.ndarray) else None)
        _apply_frame_from_float(float(frame_slider.value))

    @dropdown.on_update
    def _(_evt) -> None:
        _load_file(str(dropdown.value))

    @reload_btn.on_click
    def _(_evt) -> None:
        _load_file(str(dropdown.value))

    playing = {"flag": False}
    frame_f = {"value": 0.0}
    updating_programmatically = {"flag": False}

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]

    @frame_slider.on_update
    def _(_evt) -> None:
        if not updating_programmatically["flag"]:
            playing["flag"] = False
            frame_f["value"] = float(frame_slider.value)
            _apply_frame_from_float(frame_f["value"])

    def _player_loop() -> None:
        next_tick = time.perf_counter()
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_mult_in.value))
                dt = 1.0 / (fps_val * mult)
                if now >= next_tick:
                    next_tick = now + dt
                    frame_f["value"] += 1.0 / float(mult)
                    last_frame = int(state.get("n_frames", 1)) - 1
                    if frame_f["value"] >= last_frame:
                        if loop_cb.value:
                            frame_f["value"] = 0.0
                        else:
                            frame_f["value"] = float(last_frame)
                            playing["flag"] = False
                    updating_programmatically["flag"] = True
                    frame_slider.value = int(frame_f["value"])
                    updating_programmatically["flag"] = False
                    _apply_frame_from_float(frame_f["value"])
            time.sleep(0.001)

    _load_file(initial_key)
    _apply_frame_from_float(0.0)
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")

    while True:
        time.sleep(1.0)


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = q0.astype(np.float64, copy=False)
    q1 = q1.astype(np.float64, copy=False)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + u * (q1 - q0)
        return out / max(np.linalg.norm(out), 1e-8)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / sin_theta


def main() -> None:
    exp_cfg, remaining = tyro.cli(
        AnnotatedExperimentConfig,
        config=TYRO_CONIFG,
        return_unknown_args=True,
        add_help=False,
    )
    rollout_cfg = tyro.cli(
        RolloutViewerConfig,
        args=remaining,
        add_help=False,
    )
    replay_rollout(exp_cfg, rollout_cfg)


if __name__ == "__main__":
    main()
