#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


SMPLX_22_EDGES = np.asarray(
    [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 12),
        (12, 15),
        (12, 13),
        (12, 14),
        (13, 16),
        (14, 17),
        (16, 18),
        (17, 19),
        (18, 20),
        (19, 21),
    ],
    dtype=np.int64,
)


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    human_joints = data["human_joints"] if "human_joints" in data else None
    return qpos, fps, human_joints


def _mesh_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10**9, path.name)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh from {path}, got {type(mesh)!r}")
    return mesh


def _add_terrain_mesh_pieces(server: viser.ViserServer, mesh_dir: str, scale: float, visible: bool):
    root = Path(mesh_dir)
    mesh_paths = sorted(root.glob("box*.obj"), key=_mesh_sort_key)
    if not mesh_paths:
        mesh_paths = sorted(root.glob("part_*.obj"), key=_mesh_sort_key)
    if not mesh_paths:
        raise FileNotFoundError(f"No box*.obj or part_*.obj terrain pieces found in {root}")

    handles = []
    for index, mesh_path in enumerate(mesh_paths, start=1):
        mesh = _load_mesh(mesh_path)
        handle = server.scene.add_mesh_simple(
            f"/terrain/piece_{index:03d}",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=(92, 138, 162),
            opacity=0.58,
            side="double",
            scale=float(scale),
            visible=visible,
        )
        handles.append(handle)
    return handles


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    fps: int | None = None,
    human_joints: np.ndarray | None = None,
):
    """
    qpos layout (MuJoCo order):
      [0:3]   robot base position (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end-4] (optional) object position (xyz)
      [end-4:end]   (optional) object quat (wxyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer(port=config.port)

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    vo = None
    terrain_handles = []
    if config.terrain_mesh_dir:
        terrain_handles = _add_terrain_mesh_pieces(
            server,
            mesh_dir=config.terrain_mesh_dir,
            scale=config.terrain_scale,
            visible=config.show_meshes,
        )
    elif config.object_urdf:
        object_urdf_y = yourdfpy.URDF.load(config.object_urdf, load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    # A tiny grid
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    # Use fps from config if not provided, otherwise use the one from npz file
    actual_fps = fps if fps is not None else config.fps

    # Set initial mesh visibility
    vr.show_visual = config.show_meshes
    if vo is not None:
        vo.show_visual = config.show_meshes
    for handle in terrain_handles:
        handle.visible = config.show_meshes

    human_points = None
    human_lines = None
    if human_joints is not None:
        human_joints = np.asarray(human_joints, dtype=np.float32)
        if human_joints.ndim != 3 or human_joints.shape[-1] != 3:
            raise ValueError(f"human_joints must have shape [T, J, 3], got {human_joints.shape}")
        human0 = human_joints[0]
        colors = np.tile(np.array([[255, 190, 60]], dtype=np.uint8), (human0.shape[0], 1))
        human_points = server.scene.add_point_cloud(
            "/human/joints",
            points=human0,
            colors=colors,
            point_size=0.035,
            visible=config.show_human,
        )
        edges = SMPLX_22_EDGES[SMPLX_22_EDGES.max(axis=1) < human0.shape[0]]
        human_lines = server.scene.add_line_segments(
            "/human/skeleton",
            points=human0[edges],
            colors=np.array([255, 190, 60], dtype=np.uint8),
            line_width=3.0,
            visible=config.show_human,
        )

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_robot_cb = server.gui.add_checkbox("Show G1", initial_value=config.show_meshes)
        show_object_cb = server.gui.add_checkbox("Show terrain", initial_value=config.show_meshes)
        show_human_cb = (
            server.gui.add_checkbox("Show HMR", initial_value=config.show_human) if human_joints is not None else None
        )

    @show_robot_cb.on_update
    def _(_):
        vr.show_visual = bool(show_robot_cb.value)

    @show_object_cb.on_update
    def _(_):
        if vo is not None:
            vo.show_visual = bool(show_object_cb.value)
        for handle in terrain_handles:
            handle.visible = bool(show_object_cb.value)

    if show_human_cb is not None:

        @show_human_cb.on_update
        def _(_):
            visible = bool(show_human_cb.value)
            if human_points is not None:
                human_points.visible = visible
            if human_lines is not None:
                human_lines.visible = visible

    def _update_human(_q: np.ndarray, frame_idx: int) -> None:
        if human_joints is None or human_points is None or human_lines is None:
            return
        idx = int(np.clip(frame_idx, 0, human_joints.shape[0] - 1))
        joints = human_joints[idx]
        human_points.points = joints
        edges = SMPLX_22_EDGES[SMPLX_22_EDGES.max(axis=1) < joints.shape[0]]
        human_lines.points = joints[edges]

    # ---------- Use reusable motion control sliders from viser_utils ----------
    create_motion_control_sliders(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        viser_object=vo if config.assume_object_in_qpos else None,
        object_base_frame=object_root if config.assume_object_in_qpos else None,
        contains_object_in_qpos=config.assume_object_in_qpos,
        initial_fps=actual_fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
        on_update=_update_human if human_joints is not None else None,
    )
    n_frames = int(qpos.shape[0])
    object_status = "no"
    if config.terrain_mesh_dir:
        object_status = f"static_direct_pieces:{len(terrain_handles)}"
    elif config.object_urdf:
        object_status = "dynamic" if config.assume_object_in_qpos else "static"
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={object_status} | "
        f"human={'yes' if human_joints is not None else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
    qpos, fps, human_joints = load_npz(cfg.qpos_npz)
    make_player(
        config=cfg,
        qpos=qpos,
        fps=fps,
        human_joints=human_joints,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
