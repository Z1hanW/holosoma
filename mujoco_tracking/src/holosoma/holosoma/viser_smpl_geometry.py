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

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402


@dataclass(frozen=True)
class SmplGeometryViewerConfig:
    motion_dir: str = "/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene/vmm_25/gv/hmr"
    geometry_dir: str | None = "/home/ubuntu/FAR/holosoma/crisp/vmm_data/geo/obj/vmm_25/scene_mesh_sqs.obj"
    smpl_model_path: str | None = "/home/ubuntu/FAR/CRISP-Real2Sim/prep/data/smplx/models/smplx/SMPLX_NEUTRAL.pkl"
    smpl_model_type: str = "smplx"
    gender: str = "neutral"
    device: str = "auto"
    port: int = 0
    fps: int | None = None
    autoplay: bool = True
    loop: bool = True
    preload: bool = True
    show_mesh: bool = True
    show_joints: bool = False
    show_geometry: bool = True
    add_grid: bool = True
    grid_size: float = 10.0
    start_clip: str | None = None


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _decode_scalar(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _extract_key(data: dict[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _list_motion_files(motion_path: Path) -> tuple[list[str], dict[str, Path]]:
    if motion_path.is_file():
        return [motion_path.stem], {motion_path.stem: motion_path}
    motion_paths = sorted(list(motion_path.glob("*.npz")) + list(motion_path.glob("*.NPZ")))
    if not motion_paths:
        raise FileNotFoundError(f"No motion files found in: {motion_path}")
    motion_map = {path.stem: path for path in motion_paths}
    return sorted(motion_map.keys()), motion_map


def _resolve_geometry_inputs(
    geometry_dir: str | None,
) -> tuple[dict[str, Path] | None, Path | None]:
    if geometry_dir is None:
        return None, None
    geom_path = _resolve_data_path(geometry_dir)
    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry path not found: {geom_path}")

    if geom_path.is_file():
        return None, geom_path

    obj_files = sorted(list(geom_path.glob("*.obj")) + list(geom_path.glob("*.OBJ")))
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in geometry dir: {geom_path}")
    if len(obj_files) == 1:
        return None, obj_files[0]
    return {path.stem: path for path in obj_files}, None


def _load_obj_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded geometry is not a trimesh: {type(mesh)}")
    return mesh


def _normalize_pose_inputs(
    data: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, bool]:
    pose2rot = True

    poses = _extract_key(data, ("poses", "pose"))
    if poses is not None:
        arr = np.asarray(poses)
        if arr.ndim == 2:
            if arr.shape[1] < 6 or arr.shape[1] % 3 != 0:
                raise ValueError(f"Unsupported poses shape: {arr.shape}")
            global_orient = arr[:, :3]
            body_pose = arr[:, 3:]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            if arr.shape[1] < 2:
                raise ValueError(f"Unsupported poses shape: {arr.shape}")
            global_orient = arr[:, 0]
            body_pose = arr[:, 1:].reshape(arr.shape[0], -1)
        elif arr.ndim == 4 and arr.shape[2:] == (3, 3):
            if arr.shape[1] < 2:
                raise ValueError(f"Unsupported poses shape: {arr.shape}")
            global_orient = arr[:, 0]
            body_pose = arr[:, 1:]
            pose2rot = False
        else:
            raise ValueError(f"Unsupported poses shape: {arr.shape}")
        return global_orient, body_pose, pose2rot

    global_orient = _extract_key(data, ("global_orient", "root_orient"))
    body_pose = _extract_key(data, ("body_pose", "pose_body"))
    if global_orient is None or body_pose is None:
        raise ValueError("Missing SMPL pose data (poses or global_orient/body_pose).")

    global_orient = np.asarray(global_orient)
    body_pose = np.asarray(body_pose)

    if global_orient.ndim == 3 and global_orient.shape[1:] == (1, 3):
        global_orient = global_orient[:, 0]
    if global_orient.ndim == 3 and global_orient.shape[1:] == (3, 3):
        pose2rot = False

    if body_pose.ndim == 3 and body_pose.shape[-1] == 3:
        body_pose = body_pose.reshape(body_pose.shape[0], -1)
    elif body_pose.ndim == 4 and body_pose.shape[-2:] == (3, 3):
        pose2rot = False

    return global_orient, body_pose, pose2rot


def _resolve_model_path(cfg: SmplGeometryViewerConfig) -> Path:
    if cfg.smpl_model_path:
        path = _resolve_data_path(cfg.smpl_model_path)
    else:
        env_path = os.environ.get("SMPL_MODEL_PATH") or os.environ.get("SMPLX_MODEL_PATH")
        path = Path(env_path) if env_path else Path()
    if not path.exists():
        raise FileNotFoundError(
            "SMPL model path not found. Set --smpl-model-path or SMPL_MODEL_PATH/SMPLX_MODEL_PATH."
        )
    if path.is_file():
        parent = path.parent
        if parent.name.lower() in ("smpl", "smplx", "smplh", "smplxa", "smpla", "smil"):
            path = parent.parent
        else:
            path = parent
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"SMPL model directory not found: {path}")
    return path


def _load_motion_data(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        payload: dict[str, object] = {key: data[key] for key in data.files}
    return payload


def _build_smpl_sequence(
    cfg: SmplGeometryViewerConfig,
    motion_data: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, str]:
    try:
        import smplx  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError(f"smplx is required for SMPL visualization: {exc}") from exc

    global_orient, body_pose, pose2rot = _normalize_pose_inputs(motion_data)
    num_frames = int(global_orient.shape[0])

    trans = _extract_key(motion_data, ("trans", "transl", "root_trans", "root_transl"))
    if trans is None:
        trans_arr = np.zeros((num_frames, 3), dtype=np.float32)
    else:
        trans_arr = np.asarray(trans, dtype=np.float32)
        if trans_arr.ndim == 3 and trans_arr.shape[1] == 1 and trans_arr.shape[2] == 3:
            trans_arr = trans_arr[:, 0]

    betas = _extract_key(motion_data, ("betas", "beta"))
    if betas is None:
        betas_arr = np.zeros((1, 10), dtype=np.float32)
    else:
        betas_arr = np.asarray(betas, dtype=np.float32)
        if betas_arr.ndim == 1:
            betas_arr = betas_arr[None, :]

    if betas_arr.shape[0] == 1 and num_frames > 1:
        betas_arr = np.repeat(betas_arr, num_frames, axis=0)
    elif betas_arr.shape[0] != num_frames:
        raise ValueError(f"Betas batch mismatch: {betas_arr.shape[0]} != {num_frames}")

    gender_val = _decode_scalar(_extract_key(motion_data, ("gender",))) or cfg.gender

    model_path = _resolve_model_path(cfg)
    device = _resolve_device(cfg.device)
    smpl_model = smplx.create(
        model_path=str(model_path),
        model_type=cfg.smpl_model_type,
        gender=gender_val,
        num_betas=int(betas_arr.shape[1]),
        batch_size=num_frames,
    ).to(device)

    with torch.no_grad():
        global_orient_t = torch.from_numpy(global_orient).to(device=device, dtype=torch.float32)
        body_pose_t = torch.from_numpy(body_pose).to(device=device, dtype=torch.float32)
        betas_t = torch.from_numpy(betas_arr).to(device=device, dtype=torch.float32)
        trans_t = torch.from_numpy(trans_arr).to(device=device, dtype=torch.float32)

        smpl_out = smpl_model(
            body_pose=body_pose_t,
            betas=betas_t,
            global_orient=global_orient_t,
            transl=trans_t,
            pose2rot=pose2rot,
        )

    vertices = smpl_out.vertices.detach().cpu().numpy().astype(np.float32)
    joints = None
    if cfg.show_joints:
        joints = smpl_out.joints.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(smpl_model.faces, dtype=np.uint32)

    fps_val = cfg.fps
    if fps_val is None:
        fps_val = _extract_key(motion_data, ("mocap_framerate", "fps", "frame_rate"))
        if fps_val is None:
            fps_val = 30
    fps_val = int(np.array(fps_val).reshape(-1)[0])

    return vertices, faces, joints, int(fps_val), gender_val


def run_viewer(cfg: SmplGeometryViewerConfig) -> None:
    motion_root = _resolve_data_path(cfg.motion_dir)
    if not motion_root.exists():
        raise FileNotFoundError(f"Motion dir not found: {motion_root}")

    clip_names, motion_map = _list_motion_files(motion_root)
    if cfg.start_clip and cfg.start_clip not in clip_names:
        raise ValueError(f"start_clip '{cfg.start_clip}' not found in motions.")

    geom_map, default_geom = _resolve_geometry_inputs(cfg.geometry_dir)

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
    geometry_cache: dict[str, trimesh.Trimesh] = {}

    def _ensure_motion_loaded(name: str) -> dict[str, object]:
        if name in motion_cache:
            return motion_cache[name]
        motion_data = _load_motion_data(motion_map[name])
        verts, faces, joints, fps_val, gender_val = _build_smpl_sequence(cfg, motion_data)
        if verts.shape[0] == 0:
            raise ValueError(f"Motion {name} has zero frames.")
        motion_cache[name] = {
            "verts": verts,
            "faces": faces,
            "joints": joints,
            "fps": fps_val,
            "n_frames": int(verts.shape[0]),
            "gender": gender_val,
        }
        return motion_cache[name]

    def _ensure_geometry_loaded(name: str) -> trimesh.Trimesh | None:
        if geom_map is None and default_geom is None:
            return None
        if geom_map is None and default_geom is not None:
            key = default_geom.as_posix()
        else:
            geom_path = geom_map.get(name) if geom_map else None
            if geom_path is None:
                return None
            key = geom_path.as_posix()
        if key in geometry_cache:
            return geometry_cache[key]
        geom_path = Path(key)
        mesh = _load_obj_mesh(geom_path)
        geometry_cache[key] = mesh
        return mesh

    if cfg.preload:
        for name in clip_names:
            _ensure_motion_loaded(name)
            _ensure_geometry_loaded(name)

    motion_state: dict[str, object] = {}
    mesh_state: dict[str, object | None] = {"handle": None}
    joint_state: dict[str, object | None] = {"handle": None}
    geometry_state: dict[str, object | None] = {"handle": None}

    def _set_geometry(name: str) -> None:
        handle = geometry_state["handle"]
        if handle is not None:
            handle.remove()
            geometry_state["handle"] = None
        mesh = _ensure_geometry_loaded(name)
        if mesh is None:
            return
        geometry_state["handle"] = server.scene.add_mesh_trimesh("/geometry", mesh)

    def _set_motion(name: str) -> None:
        state = _ensure_motion_loaded(name)
        motion_state.update({"name": name, **state})
        verts = state["verts"]
        faces = state["faces"]
        handle = mesh_state["handle"]
        if handle is None:
            mesh_state["handle"] = server.scene.add_mesh_simple(
                "/smpl",
                vertices=verts[0],
                faces=faces,
                color=(255, 215, 0),
                wireframe=False,
                flat_shading=False,
                visible=cfg.show_mesh,
            )
        else:
            handle.vertices = verts[0]
            handle.faces = faces

        if cfg.show_joints and state.get("joints") is not None:
            joints = state["joints"][0]
            joint_handle = joint_state["handle"]
            if joint_handle is None:
                joint_state["handle"] = server.scene.add_point_cloud(
                    "/smpl_joints",
                    points=joints,
                    colors=np.array([[128, 0, 128]] * joints.shape[0]),
                    point_size=0.02,
                    point_shape="circle",
                )
            else:
                joint_handle.points = joints

    active_clip = cfg.start_clip or clip_names[0]
    _set_motion(active_clip)
    _set_geometry(active_clip)

    with server.gui.add_folder("Motion"):
        clip_dropdown = server.gui.add_dropdown("Clip", options=tuple(clip_names), initial_value=active_clip)
        clip_info = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_mesh_cb = server.gui.add_checkbox("Show SMPL mesh", initial_value=cfg.show_mesh)
        show_geom_cb = server.gui.add_checkbox("Show geometry", initial_value=cfg.show_geometry)
        show_joints_cb = None
        if cfg.show_joints:
            show_joints_cb = server.gui.add_checkbox("Show SMPL joints", initial_value=True)

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

    @show_mesh_cb.on_update
    def _(_evt) -> None:
        handle = mesh_state["handle"]
        if handle is not None:
            handle.visible = bool(show_mesh_cb.value)

    @show_geom_cb.on_update
    def _(_evt) -> None:
        handle = geometry_state["handle"]
        if handle is not None:
            handle.visible = bool(show_geom_cb.value)

    if show_joints_cb is not None:

        @show_joints_cb.on_update
        def _(_evt) -> None:
            handle = joint_state["handle"]
            if handle is not None:
                handle.visible = bool(show_joints_cb.value)

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

    def _apply_frame(frame_idx: int) -> None:
        verts = motion_state["verts"][frame_idx]
        handle = mesh_state["handle"]
        with server.atomic():
            if handle is not None:
                handle.vertices = verts
                handle.visible = bool(show_mesh_cb.value)
            if cfg.show_joints and motion_state.get("joints") is not None:
                joints = motion_state["joints"][frame_idx]
                joint_handle = joint_state["handle"]
                if joint_handle is not None:
                    joint_handle.points = joints.astype(np.float32, copy=False)
                    if show_joints_cb is not None:
                        joint_handle.visible = bool(show_joints_cb.value)

    def _player_loop() -> None:
        while True:
            if playing["flag"]:
                fps_val = int(fps_in.value)
                if fps_val <= 0:
                    fps_val = 1
                next_frame = int(frame_slider.value) + 1
                last_frame = int(motion_state["n_frames"]) - 1
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
    cfg = tyro.cli(SmplGeometryViewerConfig, config=TYRO_CONIFG)
    run_viewer(cfg)


if __name__ == "__main__":
    main()
