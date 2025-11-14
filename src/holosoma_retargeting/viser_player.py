#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    # expected: qpos [T, ?], and optional fps
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data else 30
    return qpos, fps


def make_player(
    robot_urdf: str,
    qpos: np.ndarray,
    object_urdf: str | None,
    fps: int = 30,
    assume_object_in_qpos: bool = True,
    loop: bool = False,
):
    """
    qpos layout (per your retargeter):
      [0:4]   robot base quat (wxyz)
      [4:7]   robot base position (xyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end] (optional) object quat (wxyz) + object pos (xyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer()

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_urdf_y = yourdfpy.URDF.load(robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    vo = None
    if object_urdf:
        object_urdf_y = yourdfpy.URDF.load(object_urdf, load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name="/object")

    # A tiny grid
    server.scene.add_grid("/grid", width=8, height=8, position=(0.0, 0.0, 0.0))

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    # ---------- GUI ----------
    n_frames = int(qpos.shape[0])
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=fps, min=1, max=240, step=1)
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=True)
    with server.gui.add_folder("Smoothing"):
        interp_mult_in = server.gui.add_number("Visual FPS multiplier", initial_value=2, min=1, max=8, step=1)

    vr.show_visual = True
    if vo is not None:
        vo.show_visual = True

    @show_meshes_cb.on_update
    def _(_):
        vr.show_visual = bool(show_meshes_cb.value)
        if vo is not None:
            vo.show_visual = bool(show_meshes_cb.value)

    # ---------- helpers (quat continuity + slerp + interpolation) ----------
    def _quat_normalize(q):
        q = np.asarray(q, float)
        n = float(np.linalg.norm(q))
        return q if n == 0 else q / n

    def _quat_continuous(prev_q, curr_q):
        q = _quat_normalize(curr_q)
        if prev_q is None:
            return q
        return -q if float(np.dot(prev_q, q)) < 0.0 else q

    def _slerp(q0, q1, u):
        q0 = _quat_normalize(q0)
        q1 = _quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + u * (q1 - q0)
            return _quat_normalize(q)
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / s

    def _interp_frame(qpos_arr, i0, i1, u):
        """Linear joints/positions, slerp base quat."""
        q0 = qpos_arr[i0]
        q1 = qpos_arr[i1]
        out = q0.copy()
        # base
        out[:4] = _slerp(q0[:4], q1[:4], u)
        out[4:7] = (1 - u) * q0[4:7] + u * q1[4:7]
        # remaining (joints + maybe object pose)
        if q0.shape[0] == 7 + robot_dof + 7:  # object pose is included (quat + pos)
            out[7:-7] = (1 - u) * q0[7:-7] + u * q1[7:-7]
            out[-7:-3] = _slerp(q0[-7:-3], q1[-7:-3], u)
            out[-3:] = (1 - u) * q0[-3:] + u * q1[-3:]
        else:
            out[7:] = (1 - u) * q0[7:] + u * q1[7:]

        return out

    # ---------- state ----------
    playing = {"flag": False}
    tick = {"next": time.perf_counter()}
    prev = {"robot_q": None, "obj_q": None}

    # ---------- controls ----------
    @play_btn.on_click
    def _(_evt):
        playing["flag"] = not playing["flag"]
        tick["next"] = time.perf_counter()
        # reset prev so continuity restarts from current frame
        prev["robot_q"] = None
        prev["obj_q"] = None
        # keep fractional index aligned with slider
        nonlocal_f["f"] = float(frame_slider.value)

    @fps_in.on_update
    def _(_evt):
        tick["next"] = time.perf_counter()

    @frame_slider.on_update
    def _(_evt):
        tick["next"] = time.perf_counter()
        # immediate draw from discrete frame
        apply_frame(int(frame_slider.value))
        # reset continuity starting from this frame
        prev["robot_q"] = None
        prev["obj_q"] = None
        nonlocal_f["f"] = float(frame_slider.value)

    # ---------- draw functions ----------
    def apply_frame_from_q(q):
        # joints
        joints = q[7 : 7 + robot_dof]
        if joints.shape[0] != robot_dof:
            joints = (
                joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
            )
        vr.update_cfg(joints)

        # robot base (continuous quat)
        robot_quat = _quat_continuous(prev["robot_q"], q[:4])
        prev["robot_q"] = robot_quat
        robot_root.wxyz = robot_quat
        robot_root.position = q[4:7]

        # object (optional) with continuity
        if assume_object_in_qpos and vo is not None:
            if q.shape[0] >= 7 + robot_dof + 7:
                obj_quat = _quat_continuous(prev["obj_q"], q[-7:-3])
                prev["obj_q"] = obj_quat
                obj_pos = q[-3:]
            else:
                obj_quat = np.array([1.0, 0.0, 0.0, 0.0])
                obj_pos = np.zeros(3)
            object_root.wxyz = obj_quat
            object_root.position = obj_pos

    def apply_frame(i: int):
        if n_frames == 0:
            return
        i = int(np.clip(i, 0, n_frames - 1))
        apply_frame_from_q(qpos[i])

    # ---------- player loop with interpolation ----------
    nonlocal_f = {"f": float(frame_slider.value)}

    def _player_loop():
        if n_frames <= 1:
            print("[viser] only one (or zero) frame; nothing to play.")
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_mult_in.value))
                dt = 1.0 / (fps_val * mult)
                if now >= tick["next"]:
                    # advance fractional frame
                    if loop:
                        f = (nonlocal_f["f"] + 1.0 / mult) % max(1, n_frames)
                    else:
                        f = min(nonlocal_f["f"] + 1.0 / mult, float(n_frames - 1))
                    nonlocal_f["f"] = f
                    k0 = int(np.floor(f))

                    if loop:
                        k1 = (k0 + 1) % max(1, n_frames)
                    else:
                        k1 = min(k0 + 1, n_frames - 1)

                    u = float(f - k0)
                    q_interp = _interp_frame(qpos, k0, k1, u)
                    apply_frame_from_q(q_interp)
                    # keep slider roughly in sync when mult==1
                    if mult == 1:
                        frame_slider.value = k0
                    tick["next"] = now + dt
                else:
                    time.sleep(min(0.002, max(0.0, tick["next"] - now)))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player_loop, daemon=True).start()

    # draw first frame
    apply_frame(0)
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if (object_urdf and assume_object_in_qpos) else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


@dataclass
class Args:
    """Play retargeted qpos with viser."""
    
    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Path to robot URDF"""
    
    qpos_npz: str = "rt_results/OMOMO_new/box_parallel/sub8_largebox_051_original.npz"
    """Path to .npz with qpos"""
    
    object_urdf: str | None = None
    """Path to object URDF (optional)"""


def main(cfg: Args) -> None:
    qpos, fps = load_npz(cfg.qpos_npz)
    make_player(
        robot_urdf=cfg.robot_urdf,
        qpos=qpos,
        object_urdf=cfg.object_urdf,
        fps=fps,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    main(cfg)
