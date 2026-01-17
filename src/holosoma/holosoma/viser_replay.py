from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro
import trimesh

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[3]
VISER_SRC = REPO_ROOT / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.insert(0, str(VISER_SRC))

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.command import MotionConfig  # noqa: E402
from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_types.terrain import MeshType  # noqa: E402
from holosoma.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402
from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES, MotionLoader  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.rotations import get_euler_xyz, quat_from_euler_xyz  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> str:
    asset_root = robot_config.asset.asset_root
    asset_root = _resolve_data_path(asset_root)
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _get_motion_config(cfg: ExperimentConfig) -> MotionConfig:
    if cfg.command is None:
        raise ValueError("Experiment config has no command manager; motion replay requires motion_command.")
    setup_terms = cfg.command.setup_terms
    if "motion_command" not in setup_terms:
        raise ValueError("Command manager has no motion_command; use a WBT experiment config.")
    motion_params = setup_terms["motion_command"].params
    if "motion_config" not in motion_params:
        raise ValueError("motion_command is missing motion_config.")
    motion_cfg = motion_params["motion_config"]
    if isinstance(motion_cfg, MotionConfig):
        return motion_cfg
    return MotionConfig(**motion_cfg)


def _decode_h5_strings(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for item in values:
        if isinstance(item, (bytes, np.bytes_)):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _list_motion_clips(motion_cfg: MotionConfig) -> list[str]:
    motion_path = Path(_resolve_data_path(motion_cfg.motion_file))
    if motion_path.is_dir():
        files = sorted(list(motion_path.glob("*.npz")) + list(motion_path.glob("*.NPZ")))
        if not files:
            raise FileNotFoundError(f"No motion clips found in directory: {motion_path}")
        return [path.stem for path in files]

    if motion_path.suffix.lower() in (".h5", ".hdf5"):
        try:
            import h5py  # type: ignore[import-not-found]
        except Exception:
            return [motion_path.stem]
        with h5py.File(motion_path, "r") as h5f:
            clips = h5f.get("clips")
            if clips is None or "clip_ids" not in clips:
                return [motion_path.stem]
            clip_ids = _decode_h5_strings(np.asarray(clips["clip_ids"]))
            if not clip_ids:
                return [motion_path.stem]
            return clip_ids

    if motion_path.exists():
        return [motion_path.stem]

    raise FileNotFoundError(f"Motion file not found: {motion_path}")


def _select_initial_clip(motion_cfg: MotionConfig, clip_names: list[str]) -> str:
    if motion_cfg.motion_clip_name:
        if motion_cfg.motion_clip_name not in clip_names:
            raise ValueError(
                f"motion_clip_name '{motion_cfg.motion_clip_name}' not found in motion source."
            )
        return motion_cfg.motion_clip_name
    if motion_cfg.motion_clip_id is not None:
        clip_idx = int(motion_cfg.motion_clip_id)
        if clip_idx < 0 or clip_idx >= len(clip_names):
            raise IndexError(f"motion_clip_id {clip_idx} out of range for motion source.")
        return clip_names[clip_idx]
    return clip_names[0]


def _build_default_joint_pos(robot_config: RobotConfig) -> np.ndarray:
    defaults = robot_config.init_state.default_joint_angles
    joint_pos = np.zeros(len(robot_config.dof_names), dtype=np.float32)
    for idx, name in enumerate(robot_config.dof_names):
        joint_pos[idx] = float(defaults.get(name, 0.0))
    return joint_pos


def _build_default_root_pose(
    robot_config: RobotConfig,
    motion_root_pos: np.ndarray,
    motion_root_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    init_root_quat = torch.tensor(robot_config.init_state.rot, dtype=torch.float32).unsqueeze(0)
    init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

    motion_root_quat = torch.tensor(motion_root_quat_xyzw, dtype=torch.float32).unsqueeze(0)
    _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

    default_root_pos = np.array(
        [motion_root_pos[0], motion_root_pos[1], robot_config.init_state.pos[2]],
        dtype=np.float32,
    )
    default_root_quat_xyzw = quat_from_euler_xyz(
        init_roll.squeeze(0),
        init_pitch.squeeze(0),
        motion_yaw.squeeze(0),
    ).cpu().numpy()
    default_root_quat_wxyz = default_root_quat_xyzw[[3, 0, 1, 2]]
    return default_root_pos, default_root_quat_wxyz


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


def _split_qpos(
    qpos: np.ndarray,
    robot_dof: int,
    has_object: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    root_pos = qpos[0:3]
    root_quat = qpos[3:7]
    joints = qpos[7 : 7 + robot_dof]
    if has_object:
        obj_pos = qpos[-7:-4]
        obj_quat = qpos[-4:]
    else:
        obj_pos = None
        obj_quat = None
    return root_pos, root_quat, joints, obj_pos, obj_quat


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


def _build_transition_segment(
    start_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    robot_dof: int,
    has_object: bool,
    num_steps: int,
    prepend: bool,
) -> np.ndarray:
    if num_steps <= 1:
        return np.zeros((0, start_qpos.shape[0]), dtype=np.float32)

    alphas = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
    if prepend:
        alphas = alphas[:-1]
    else:
        alphas = alphas[1:]
    if alphas.size == 0:
        return np.zeros((0, start_qpos.shape[0]), dtype=np.float32)

    s_root_pos, s_root_quat, s_joints, s_obj_pos, s_obj_quat = _split_qpos(start_qpos, robot_dof, has_object)
    t_root_pos, t_root_quat, t_joints, t_obj_pos, t_obj_quat = _split_qpos(target_qpos, robot_dof, has_object)

    out = np.zeros((alphas.shape[0], start_qpos.shape[0]), dtype=np.float32)
    for i, alpha in enumerate(alphas):
        root_pos = (1.0 - alpha) * s_root_pos + alpha * t_root_pos
        root_quat = _slerp(s_root_quat, t_root_quat, float(alpha))
        joints = (1.0 - alpha) * s_joints + alpha * t_joints

        frame = np.concatenate([root_pos, root_quat, joints], axis=0)
        if has_object and s_obj_pos is not None and t_obj_pos is not None and s_obj_quat is not None and t_obj_quat is not None:
            obj_pos = (1.0 - alpha) * s_obj_pos + alpha * t_obj_pos
            obj_quat = _slerp(s_obj_quat, t_obj_quat, float(alpha))
            frame = np.concatenate([frame, obj_pos, obj_quat], axis=0)

        out[i] = frame.astype(np.float32, copy=False)
    return out


def _maybe_add_default_pose_transitions(
    qpos: np.ndarray,
    *,
    motion_cfg: MotionConfig,
    robot_config: RobotConfig,
    robot_dof: int,
    fps: float,
    default_joint_pos: np.ndarray,
) -> np.ndarray:
    if qpos.shape[0] == 0:
        return qpos

    has_object = qpos.shape[1] >= (7 + robot_dof + 7)
    if default_joint_pos.shape[0] != robot_dof:
        raise ValueError("default_joint_pos does not match robot_dof.")
    motion_root_pos_start = qpos[0, 0:3]
    motion_root_quat_start_wxyz = qpos[0, 3:7]
    motion_root_quat_start_xyzw = motion_root_quat_start_wxyz[[1, 2, 3, 0]]

    motion_root_pos_end = qpos[-1, 0:3]
    motion_root_quat_end_wxyz = qpos[-1, 3:7]
    motion_root_quat_end_xyzw = motion_root_quat_end_wxyz[[1, 2, 3, 0]]

    if motion_cfg.enable_default_pose_prepend and motion_cfg.default_pose_prepend_duration_s > 0.0:
        num_steps = int(round(motion_cfg.default_pose_prepend_duration_s * fps))
        default_root_pos, default_root_quat = _build_default_root_pose(
            robot_config,
            motion_root_pos_start,
            motion_root_quat_start_xyzw,
        )
        start = np.concatenate([default_root_pos, default_root_quat, default_joint_pos], axis=0)
        if has_object:
            start = np.concatenate([start, qpos[0, -7:]], axis=0)
        segment = _build_transition_segment(
            start,
            qpos[0],
            robot_dof=robot_dof,
            has_object=has_object,
            num_steps=num_steps,
            prepend=True,
        )
        if segment.shape[0] > 0:
            qpos = np.concatenate([segment, qpos], axis=0)

    if motion_cfg.enable_default_pose_append and motion_cfg.default_pose_append_duration_s > 0.0:
        num_steps = int(round(motion_cfg.default_pose_append_duration_s * fps))
        default_root_pos, default_root_quat = _build_default_root_pose(
            robot_config,
            motion_root_pos_end,
            motion_root_quat_end_xyzw,
        )
        target = np.concatenate([default_root_pos, default_root_quat, default_joint_pos], axis=0)
        if has_object:
            target = np.concatenate([target, qpos[-1, -7:]], axis=0)
        segment = _build_transition_segment(
            qpos[-1],
            target,
            robot_dof=robot_dof,
            has_object=has_object,
            num_steps=num_steps,
            prepend=False,
        )
        if segment.shape[0] > 0:
            qpos = np.concatenate([qpos, segment], axis=0)

    return qpos


def _load_motion_qpos(
    motion_cfg: MotionConfig,
    robot_config: RobotConfig,
    viser_joint_names: list[str],
    motion_clip_name: str | None = None,
    motion_clip_id: int | None = None,
) -> tuple[np.ndarray, int]:
    motion_path = Path(_resolve_data_path(motion_cfg.motion_file))
    if motion_clip_name and motion_path.is_dir():
        candidate = motion_path / f"{motion_clip_name}.npz"
        qpos_payload = _try_load_qpos_npz(candidate)
        if qpos_payload is not None:
            return qpos_payload
    if motion_path.is_file():
        qpos_payload = _try_load_qpos_npz(motion_path)
        if qpos_payload is not None:
            return qpos_payload

    robot_body_names = [FAKE_BODY_NAME_ALIASES.get(name, name) for name in robot_config.body_names]
    clip_name = motion_clip_name if motion_clip_name is not None else motion_cfg.motion_clip_name
    clip_id = motion_clip_id if motion_clip_id is not None else motion_cfg.motion_clip_id
    motion = MotionLoader(
        motion_cfg.motion_file,
        robot_body_names,
        robot_config.dof_names,
        device="cpu",
        motion_clip_id=clip_id,
        motion_clip_name=clip_name,
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

    parts = [root_pos, root_quat_wxyz, joint_pos]
    if motion.has_object:
        object_pos = motion.object_pos_w.cpu().numpy()
        object_quat_xyzw = motion.object_quat_w.cpu().numpy()
        object_quat_wxyz = object_quat_xyzw[:, [3, 0, 1, 2]]
        parts.extend([object_pos, object_quat_wxyz])

    qpos = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    fps = int(motion.fps) if hasattr(motion, "fps") else 30
    return qpos, fps


def _load_terrain_mesh(cfg: ExperimentConfig, clip_name: str | None = None) -> trimesh.Trimesh | None:
    terrain_cfg = cfg.terrain.terrain_term
    mesh_type = terrain_cfg.mesh_type
    mesh_type_value = mesh_type.value if isinstance(mesh_type, MeshType) else str(mesh_type)
    if mesh_type_value != "load_obj":
        return None
    if not terrain_cfg.obj_file_path:
        return None

    terrain_path = Path(_resolve_data_path(terrain_cfg.obj_file_path))

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

    num_rows = max(1, int(terrain_cfg.num_rows * terrain_cfg.scale_factor))
    num_cols = max(1, int(terrain_cfg.num_cols * terrain_cfg.scale_factor))
    if num_rows * num_cols <= 1:
        return base

    gap = 1e-4
    stride = (base.bounds[1] - base.bounds[0]) + gap
    tiles = []
    for r in range(num_rows):
        for c in range(num_cols):
            tile = base.copy()
            tile.apply_translation([c * stride[0], r * stride[1], 0.0])
            tiles.append(tile)
    return trimesh.util.concatenate(tiles)


def replay(cfg: ExperimentConfig) -> None:
    motion_cfg = _get_motion_config(cfg)
    robot_urdf_path = _resolve_robot_urdf_path(cfg.robot)
    object_urdf_path = None
    if cfg.robot.object.object_urdf_path:
        object_urdf_path = _resolve_data_path(cfg.robot.object.object_urdf_path)

    server = viser.ViserServer(port=6060)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    vr = ViserUrdf(server, urdf_or_path=Path(robot_urdf_path), root_node_name="/robot")

    vo = None
    if object_urdf_path:
        vo = ViserUrdf(server, urdf_or_path=Path(object_urdf_path), root_node_name="/object")

    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))

    motion_choices = _list_motion_clips(motion_cfg)
    active_clip = _select_initial_clip(motion_cfg, motion_choices)

    viser_joint_names = list(vr.get_actuated_joint_names())
    default_joint_robot = _build_default_joint_pos(cfg.robot)
    name_to_robot_idx = {name: idx for idx, name in enumerate(cfg.robot.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    default_joint_viser = np.array(
        [default_joint_robot[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32
    )

    terrain_state: dict[str, viser.GlbHandle | None] = {"handle": None}
    motion_state: dict[str, object] = {}

    def _set_terrain_for_clip(clip_name: str) -> None:
        handle = terrain_state["handle"]
        if handle is not None:
            handle.remove()
            terrain_state["handle"] = None
        terrain_mesh = _load_terrain_mesh(cfg, clip_name=clip_name)
        if terrain_mesh is None:
            return
        terrain_state["handle"] = server.scene.add_mesh_trimesh("/terrain", terrain_mesh)

    def _load_clip(clip_name: str) -> None:
        qpos, fps = _load_motion_qpos(
            motion_cfg,
            cfg.robot,
            viser_joint_names,
            motion_clip_name=clip_name,
        )
        qpos = _maybe_add_default_pose_transitions(
            qpos,
            motion_cfg=motion_cfg,
            robot_config=cfg.robot,
            robot_dof=len(viser_joint_names),
            fps=float(fps),
            default_joint_pos=default_joint_viser,
        )
        if qpos.shape[0] == 0:
            raise ValueError("Loaded motion has zero frames.")
        motion_state.update(
            {
                "clip": clip_name,
                "qpos": qpos,
                "fps": int(fps),
                "n_frames": int(qpos.shape[0]),
                "has_object": bool(vo is not None and qpos.shape[1] >= (7 + len(viser_joint_names) + 7)),
            }
        )

    _load_clip(active_clip)
    _set_terrain_for_clip(active_clip)

    with server.gui.add_folder("Motion"):
        clip_dropdown = server.gui.add_dropdown("Clip", options=tuple(motion_choices), initial_value=active_clip)
        reload_btn = server.gui.add_button("Reload clip")
        motion_source = server.gui.add_markdown(f"Source: `{motion_cfg.motion_file}`")
        clip_info = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=True)
        show_terrain_cb = server.gui.add_checkbox(
            "Show terrain", initial_value=terrain_state["handle"] is not None
        )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, int(motion_state["n_frames"]) - 1),
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=int(motion_state["fps"]), min=1, max=240, step=1)
        interp_mult_in = server.gui.add_number("Visual FPS multiplier", initial_value=2, min=1, max=8, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=False)

    def _update_clip_info() -> None:
        clip_name = str(motion_state["clip"])
        n_frames = int(motion_state["n_frames"])
        fps_val = int(motion_state["fps"])
        clip_info.content = f"Clip: `{clip_name}` | frames: {n_frames} | fps: {fps_val}"

    _update_clip_info()

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    @show_terrain_cb.on_update
    def _(_evt) -> None:
        handle = terrain_state["handle"]
        if handle is not None:
            handle.visible = bool(show_terrain_cb.value)

    def _apply_frame(frame: np.ndarray) -> None:
        root_pos = frame[0:3]
        root_quat_wxyz = frame[3:7]
        joints = frame[7 : 7 + len(viser_joint_names)]
        robot_root.position = root_pos
        robot_root.wxyz = root_quat_wxyz
        vr.update_cfg(joints.astype(np.float32, copy=False))

        if vo is None:
            return
        if motion_state["has_object"]:
            vo.show_visual = True
            object_root.position = frame[-7:-4]
            object_root.wxyz = frame[-4:]
        else:
            vo.show_visual = False

    def _interp_qpos(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
        out = q0.copy()
        out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]
        out[3:7] = _slerp(q0[3:7], q1[3:7], u)
        out[7 : 7 + len(viser_joint_names)] = (
            (1.0 - u) * q0[7 : 7 + len(viser_joint_names)]
            + u * q1[7 : 7 + len(viser_joint_names)]
        )
        if motion_state["has_object"]:
            out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]
            out[-4:] = _slerp(q0[-4:], q1[-4:], u)
        return out

    def _apply_frame_from_float(f_val: float) -> None:
        qpos = np.asarray(motion_state["qpos"])
        n_frames = int(motion_state["n_frames"])
        if n_frames <= 0:
            return
        i0 = int(np.clip(np.floor(f_val), 0, n_frames - 1))
        i1 = min(i0 + 1, n_frames - 1)
        u = float(f_val - i0)
        if i0 == i1 or u <= 1.0e-6:
            _apply_frame(qpos[i0])
        else:
            _apply_frame(_interp_qpos(qpos[i0], qpos[i1], u))

    playing = {"flag": False}
    frame_f = {"value": float(frame_slider.value)}
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

    def _set_clip(clip_name: str) -> None:
        playing["flag"] = False
        _load_clip(clip_name)
        _set_terrain_for_clip(clip_name)
        frame_f["value"] = 0.0
        updating_programmatically["flag"] = True
        frame_slider.max = max(0, int(motion_state["n_frames"]) - 1)
        frame_slider.value = 0
        updating_programmatically["flag"] = False
        fps_in.value = int(motion_state["fps"])
        _update_clip_info()
        _apply_frame_from_float(frame_f["value"])
        if terrain_state["handle"] is not None:
            terrain_state["handle"].visible = bool(show_terrain_cb.value)

    @clip_dropdown.on_update
    def _(_evt) -> None:
        _set_clip(str(clip_dropdown.value))

    @reload_btn.on_click
    def _(_evt) -> None:
        _set_clip(str(clip_dropdown.value))

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
                    last_frame = int(motion_state["n_frames"]) - 1
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

    _apply_frame_from_float(frame_f["value"])
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")

    while True:
        time.sleep(1.0)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
