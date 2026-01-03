#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    if "qpos" in data:
        qpos = data["qpos"]
    elif "joint_pos" in data:
        qpos = data["joint_pos"]
    else:
        raise KeyError(f"{npz_path} missing 'qpos' (or fallback 'joint_pos') for visualization")
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def _write_obj_urdf(obj_path: str) -> str:
    obj_abs = str(Path(obj_path).expanduser().resolve())
    urdf_text = (
        "<robot name=\"object\">\n"
        "  <link name=\"object\">\n"
        "    <visual>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        "      <geometry>\n"
        f"        <mesh filename=\"{obj_abs}\"/>\n"
        "      </geometry>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="viser_obj_"))
    urdf_path = tmp_dir / "object.urdf"
    urdf_path.write_text(urdf_text)
    return str(urdf_path)


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    fps: int | None = None,
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
    server = viser.ViserServer(port=7232)

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    vo = None
    object_urdf_path = config.object_urdf
    if object_urdf_path is None and config.object_obj:
        object_urdf_path = _write_obj_urdf(config.object_obj)
        print(f"[viser_player] Using object OBJ: {config.object_obj}")

    if object_urdf_path:
        object_urdf_y = yourdfpy.URDF.load(object_urdf_path, load_meshes=True, build_scene_graph=True)
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

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=config.show_meshes)

    @show_meshes_cb.on_update
    def _(_):
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

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
    )
    n_frames = int(qpos.shape[0])
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if object_urdf_path else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
    qpos, fps = load_npz(cfg.qpos_npz)
    make_player(
        config=cfg,
        qpos=qpos,
        fps=fps,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
