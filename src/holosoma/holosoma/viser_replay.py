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
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


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


def _split_qpos(qpos: np.ndarray, robot_dof: int, has_object: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
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
) -> tuple[np.ndarray, int]:
    robot_body_names = [FAKE_BODY_NAME_ALIASES.get(name, name) for name in robot_config.body_names]
    motion = MotionLoader(
        motion_cfg.motion_file,
        robot_body_names,
        robot_config.dof_names,
        device="cpu",
        motion_clip_id=motion_cfg.motion_clip_id,
        motion_clip_name=motion_cfg.motion_clip_name,
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


def _load_terrain_mesh(cfg: ExperimentConfig) -> trimesh.Trimesh | None:
    terrain_cfg = cfg.terrain.terrain_term
    mesh_type = terrain_cfg.mesh_type
    mesh_type_value = mesh_type.value if isinstance(mesh_type, MeshType) else str(mesh_type)
    if mesh_type_value != "load_obj":
        return None
    if not terrain_cfg.obj_file_path:
        return None

    terrain_path = _resolve_data_path(terrain_cfg.obj_file_path)
    base = trimesh.load(terrain_path, process=False)
    if isinstance(base, trimesh.Scene):
        base = base.dump(concatenate=True)
    if not isinstance(base, trimesh.Trimesh):
        raise ValueError(f"Loaded terrain is not a trimesh: {type(base)}")

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

    terrain_mesh = _load_terrain_mesh(cfg)
    terrain_handle = None
    if terrain_mesh is not None:
        terrain_handle = server.scene.add_mesh_trimesh("/terrain", terrain_mesh)

    viser_joint_names = list(vr.get_actuated_joint_names())
    default_joint_robot = _build_default_joint_pos(cfg.robot)
    name_to_robot_idx = {name: idx for idx, name in enumerate(cfg.robot.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    default_joint_viser = np.array(
        [default_joint_robot[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32
    )
    qpos, fps = _load_motion_qpos(motion_cfg, cfg.robot, viser_joint_names)
    qpos = _maybe_add_default_pose_transitions(
        qpos,
        motion_cfg=motion_cfg,
        robot_config=cfg.robot,
        robot_dof=len(viser_joint_names),
        fps=float(fps),
        default_joint_pos=default_joint_viser,
    )

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=True)
        show_terrain_cb = server.gui.add_checkbox("Show terrain", initial_value=terrain_handle is not None)

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    @show_terrain_cb.on_update
    def _(_evt) -> None:
        if terrain_handle is not None:
            terrain_handle.visible = bool(show_terrain_cb.value)

    create_motion_control_sliders(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=len(viser_joint_names),
        viser_object=vo,
        object_base_frame=object_root if vo is not None else None,
        contains_object_in_qpos=bool(vo is not None and qpos.shape[1] >= (7 + len(viser_joint_names) + 7)),
        initial_fps=fps,
        initial_interp_mult=2,
        loop=False,
    )

    n_frames = qpos.shape[0]
    print(
        f"[viser_replay] Loaded {n_frames} frames | fps={fps} | object={'yes' if vo is not None else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")

    while True:
        time.sleep(1.0)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
