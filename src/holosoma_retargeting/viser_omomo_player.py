#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tyro
import trimesh
import viser  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.src.utils import load_intermimic_data  # noqa: E402


@dataclass(frozen=True)
class OmomoVisConfig:
    data_dir: Path = Path("demo_data/OMOMO_new")
    glob: str = "*.pt"
    recursive: bool = False
    fps: int = 30
    joint_radius: float = 0.02
    show_object_frame: bool = True
    loop: bool = True


def _collect_pt_paths(cfg: OmomoVisConfig) -> Tuple[list[str], Dict[str, Path]]:
    root = cfg.data_dir
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob(cfg.glob) if cfg.recursive else root.glob(cfg.glob))

    paths = [p for p in paths if p.is_file() and p.suffix == ".pt"]
    if not paths:
        raise FileNotFoundError(f"No .pt files found at: {root}")

    labels: list[str] = []
    label_to_path: Dict[str, Path] = {}
    for p in paths:
        label = p.relative_to(root).as_posix() if root in p.parents else p.name
        if label in label_to_path:
            label = p.name
        labels.append(label)
        label_to_path[label] = p
    return labels, label_to_path


def _make_joint_handle(server: viser.ViserServer, points: np.ndarray, radius: float):
    sphere = trimesh.primitives.Sphere(radius=float(radius))
    vertices = sphere.vertices.astype(np.float32)
    faces = sphere.faces.astype(np.int32)
    color = (30, 200, 255)
    return server.scene.add_batched_meshes_simple(
        "/human/joints",
        vertices=vertices,
        faces=faces,
        batched_positions=points,
        batched_wxyzs=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (points.shape[0], 1)),
        batched_colors=color,
        opacity=1.0,
    )


def main(cfg: OmomoVisConfig) -> None:
    labels, label_to_path = _collect_pt_paths(cfg)

    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _load_label(label: str) -> Tuple[np.ndarray, np.ndarray]:
        if label in cache:
            return cache[label]
        human_joints, object_poses = load_intermimic_data(str(label_to_path[label]))
        human_joints = np.asarray(human_joints, dtype=float)
        object_poses = np.asarray(object_poses, dtype=float)
        cache[label] = (human_joints, object_poses)
        return cache[label]

    first_label = labels[0]
    human_joints, object_poses = _load_label(first_label)

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=8, height=8, position=(0.0, 0.0, 0.0))

    obj_frame = server.scene.add_frame("/object", show_axes=cfg.show_object_frame)

    state = {
        "human": human_joints,
        "object": object_poses,
        "n_frames": int(human_joints.shape[0]),
        "frame": 0,
        "fps": int(cfg.fps),
    }

    joint_handle = _make_joint_handle(server, human_joints[0], cfg.joint_radius)

    updating_programmatically = {"flag": False}
    playing = {"flag": False}
    tick = {"next": time.perf_counter()}

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, state["n_frames"] - 1),
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=state["fps"], min=1, max=240, step=1)

    with server.gui.add_folder("Sequence"):
        seq_dropdown = server.gui.add_dropdown("Motion", options=labels, initial_value=first_label)

    def _apply_frame(i: int) -> None:
        nonlocal joint_handle
        i = int(np.clip(i, 0, state["n_frames"] - 1))
        state["frame"] = i
        points = state["human"][i]
        try:
            joint_handle.remove()
        except Exception:
            pass
        new_handle = _make_joint_handle(server, points, cfg.joint_radius)
        joint_handle = new_handle

        if cfg.show_object_frame and state["object"].size > 0:
            obj = state["object"][i]
            obj_frame.position = obj[4:7]
            obj_frame.wxyz = obj[0:4]

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_programmatically["flag"]:
            return
        playing["flag"] = False
        _apply_frame(int(frame_slider.value))

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        tick["next"] = time.perf_counter()

    @fps_in.on_update
    def _(_evt) -> None:
        try:
            state["fps"] = int(fps_in.value)
        except Exception:
            state["fps"] = int(cfg.fps)

    @seq_dropdown.on_update
    def _(_evt) -> None:
        label = seq_dropdown.value
        human_joints_new, object_poses_new = _load_label(label)
        state["human"] = human_joints_new
        state["object"] = object_poses_new
        state["n_frames"] = int(human_joints_new.shape[0])
        state["frame"] = 0
        updating_programmatically["flag"] = True
        frame_slider.max = max(0, state["n_frames"] - 1)
        frame_slider.value = 0
        updating_programmatically["flag"] = False
        _apply_frame(0)

    def _player_loop() -> None:
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                if now >= tick["next"]:
                    i = state["frame"] + 1
                    if i >= state["n_frames"]:
                        if cfg.loop:
                            i = 0
                        else:
                            playing["flag"] = False
                            i = state["n_frames"] - 1
                    updating_programmatically["flag"] = True
                    frame_slider.value = i
                    updating_programmatically["flag"] = False
                    _apply_frame(i)
                    tick["next"] = now + 1.0 / max(1, state["fps"])
            time.sleep(0.005)

    threading.Thread(target=_player_loop, daemon=True).start()

    print(f"[viser_omomo] Loaded {state['n_frames']} frames from {first_label}")
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(OmomoVisConfig)
    main(cfg)
