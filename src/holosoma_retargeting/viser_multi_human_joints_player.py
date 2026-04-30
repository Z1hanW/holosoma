#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser  # type: ignore[import-not-found]


RETARGETING_ROOT = Path(__file__).resolve().parent


SMPLH_JOINTS = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]


SKELETON_EDGES_BY_NAME = [
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Toe"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Toe"),
    ("Pelvis", "Torso"),
    ("Torso", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "L_Thorax"),
    ("L_Thorax", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Chest", "R_Thorax"),
    ("R_Thorax", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
    ("L_Wrist", "L_Index1"),
    ("L_Index1", "L_Index2"),
    ("L_Index2", "L_Index3"),
    ("L_Wrist", "L_Middle1"),
    ("L_Middle1", "L_Middle2"),
    ("L_Middle2", "L_Middle3"),
    ("L_Wrist", "L_Ring1"),
    ("L_Ring1", "L_Ring2"),
    ("L_Ring2", "L_Ring3"),
    ("L_Wrist", "L_Pinky1"),
    ("L_Pinky1", "L_Pinky2"),
    ("L_Pinky2", "L_Pinky3"),
    ("L_Wrist", "L_Thumb1"),
    ("L_Thumb1", "L_Thumb2"),
    ("L_Thumb2", "L_Thumb3"),
    ("R_Wrist", "R_Index1"),
    ("R_Index1", "R_Index2"),
    ("R_Index2", "R_Index3"),
    ("R_Wrist", "R_Middle1"),
    ("R_Middle1", "R_Middle2"),
    ("R_Middle2", "R_Middle3"),
    ("R_Wrist", "R_Ring1"),
    ("R_Ring1", "R_Ring2"),
    ("R_Ring2", "R_Ring3"),
    ("R_Wrist", "R_Pinky1"),
    ("R_Pinky1", "R_Pinky2"),
    ("R_Pinky2", "R_Pinky3"),
    ("R_Wrist", "R_Thumb1"),
    ("R_Thumb1", "R_Thumb2"),
    ("R_Thumb2", "R_Thumb3"),
]

SKELETON_EDGES = np.asarray(
    [(SMPLH_JOINTS.index(a), SMPLH_JOINTS.index(b)) for a, b in SKELETON_EDGES_BY_NAME],
    dtype=np.int64,
)


@dataclass(frozen=True)
class MultiHumanJointsConfig:
    motion_root: str = (
        "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/"
        "demo_results_parallel/g1/object_interaction/omomo-ca-process"
    )
    port: int = 1087
    limit: int = 0
    autoplay: bool = True
    loop: bool = True
    align_xy_to_first_pelvis: bool = True
    spacing: float = 2.35
    columns: int = 0
    fps: int = 30
    point_size: float = 0.035
    line_width: float = 2.0
    show_points: bool = True
    show_lines: bool = True
    show_objects: bool = True
    show_labels: bool = True
    show_grid: bool = True
    grid_size: float = 18.0


@dataclass
class HumanClip:
    stem: str
    path: Path
    joints: np.ndarray
    object_pos: np.ndarray | None
    object_quat: np.ndarray | None
    fps: int
    offset: np.ndarray
    points_handle: viser.PointCloudHandle
    lines_handle: viser.LineSegmentsHandle
    object_handle: viser.MeshHandle | viser.GlbHandle | None

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[object] = []
    for part in parts:
        if part:
            key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_") or "clip"


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object mesh is not a trimesh: {type(mesh)} from {path}")
    return mesh


def _resolve_object_mesh(data: np.lib.npyio.NpzFile, object_name: str) -> Path:
    if "object_mesh_path" in data:
        raw = str(np.asarray(data["object_mesh_path"]).item()).strip()
        if raw:
            path = Path(raw).expanduser()
            candidates = [path] if path.is_absolute() else [RETARGETING_ROOT / path, Path.cwd() / path]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()
    fallback = RETARGETING_ROOT / "models" / object_name / f"{object_name}.obj"
    if not fallback.is_file():
        raise FileNotFoundError(f"Object mesh not found for {object_name}: {fallback}")
    return fallback.resolve()


def _load_clip_data(
    path: Path,
    *,
    align_xy: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Path | None, tuple[float, float, float], str, int]:
    with np.load(path, allow_pickle=True) as data:
        if "human_joints" not in data:
            raise KeyError(f"Missing human_joints in {path}")
        joints = np.asarray(data["human_joints"], dtype=np.float32)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30
        object_pos = None
        object_quat = None
        object_mesh = None
        object_scale = (1.0, 1.0, 1.0)
        object_name = ""
        if "object_name" in data:
            object_name = str(np.asarray(data["object_name"]).item())
        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float32)
            if qpos.ndim == 2 and qpos.shape[1] >= 14:
                object_pos = qpos[:, -7:-4].copy()
                object_quat = qpos[:, -4:].copy()
        if object_name:
            object_mesh = _resolve_object_mesh(data, object_name)
        if "object_mesh_scale" in data:
            scale_arr = np.asarray(data["object_mesh_scale"], dtype=np.float32).reshape(-1)
            if scale_arr.size == 1:
                scale_arr = np.repeat(scale_arr, 3)
            if scale_arr.size == 3:
                object_scale = tuple(float(v) for v in scale_arr.tolist())
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Unsupported human_joints shape in {path}: {joints.shape}")
    if joints.shape[1] < len(SMPLH_JOINTS):
        raise ValueError(f"Expected at least {len(SMPLH_JOINTS)} joints in {path}, got {joints.shape[1]}")
    joints = joints[:, : len(SMPLH_JOINTS), :].copy()
    if align_xy:
        anchor = joints[0, 0].copy()
        anchor[2] = 0.0
        joints -= anchor.reshape(1, 1, 3)
        if object_pos is not None:
            object_pos -= anchor.reshape(1, 3)
    return joints, object_pos, object_quat, object_mesh, object_scale, object_name, max(1, fps)


def _clip_color(idx: int, total: int) -> np.ndarray:
    phase = idx / max(1, total)
    channels = np.asarray(
        [
            0.55 + 0.45 * math.sin(2.0 * math.pi * phase + 0.0),
            0.55 + 0.45 * math.sin(2.0 * math.pi * phase + 2.1),
            0.55 + 0.45 * math.sin(2.0 * math.pi * phase + 4.2),
        ],
        dtype=np.float32,
    )
    return np.clip(channels * 255.0, 50.0, 255.0).astype(np.uint8)


def _line_segments(joints: np.ndarray) -> np.ndarray:
    return joints[SKELETON_EDGES]


def main(cfg: MultiHumanJointsConfig) -> None:
    motion_root = Path(cfg.motion_root).expanduser().resolve()
    if not motion_root.is_dir():
        raise FileNotFoundError(f"Motion root not found: {motion_root}")

    paths = sorted(motion_root.glob("*.npz"), key=_natural_sort_key)
    if cfg.limit > 0:
        paths = paths[: cfg.limit]
    if not paths:
        raise FileNotFoundError(f"No .npz files found under {motion_root}")

    columns = int(cfg.columns) if cfg.columns > 0 else int(math.ceil(math.sqrt(len(paths))))
    rows = int(math.ceil(len(paths) / columns))

    server = viser.ViserServer(port=int(cfg.port))
    if cfg.show_grid:
        server.scene.add_grid(
            "/grid",
            width=max(float(cfg.grid_size), float(columns) * float(cfg.spacing)),
            height=max(float(cfg.grid_size), float(rows) * float(cfg.spacing)),
            position=(0.0, 0.0, 0.0),
        )

    clips: list[HumanClip] = []
    mesh_cache: dict[Path, trimesh.Trimesh] = {}
    for idx, path in enumerate(paths):
        joints, object_pos, object_quat, object_mesh_path, object_scale, object_name, fps = _load_clip_data(
            path,
            align_xy=bool(cfg.align_xy_to_first_pelvis),
        )
        row = idx // columns
        col = idx % columns
        offset = np.asarray(
            [
                (col - (columns - 1) * 0.5) * float(cfg.spacing),
                -float(row) * float(cfg.spacing),
                0.0,
            ],
            dtype=np.float32,
        )
        color = _clip_color(idx, len(paths))
        point_colors = np.tile(color.reshape(1, 3), (len(SMPLH_JOINTS), 1))
        line_colors = tuple(int(v) for v in color.tolist())
        frame0 = joints[0] + offset.reshape(1, 3)
        node = _safe_name(path.stem)
        points_handle = server.scene.add_point_cloud(
            f"/humans/{node}/points",
            points=frame0,
            colors=point_colors,
            point_size=float(cfg.point_size),
            point_shape="circle",
            visible=bool(cfg.show_points),
        )
        lines_handle = server.scene.add_line_segments(
            f"/humans/{node}/skeleton",
            points=_line_segments(frame0),
            colors=line_colors,
            line_width=float(cfg.line_width),
            visible=bool(cfg.show_lines),
        )
        object_handle: viser.MeshHandle | viser.GlbHandle | None = None
        if object_pos is not None and object_quat is not None and object_mesh_path is not None:
            if object_mesh_path not in mesh_cache:
                mesh_cache[object_mesh_path] = _load_mesh(object_mesh_path)
            object_handle = server.scene.add_mesh_trimesh(
                f"/humans/{node}/object",
                mesh=mesh_cache[object_mesh_path],
                scale=object_scale,
                position=object_pos[0] + offset,
                wxyz=object_quat[0],
                visible=bool(cfg.show_objects),
            )
        if cfg.show_labels:
            label_pos = offset + np.asarray([0.0, -0.72, 1.55], dtype=np.float32)
            server.scene.add_label(
                f"/humans/{node}/label",
                text=f"{path.stem} ({object_name or 'object'})",
                position=label_pos,
                font_size_mode="scene",
                font_scene_height=0.075,
                anchor="center-center",
            )
        clips.append(
            HumanClip(
                stem=path.stem,
                path=path,
                joints=joints,
                object_pos=object_pos,
                object_quat=object_quat,
                fps=fps,
                offset=offset,
                points_handle=points_handle,
                lines_handle=lines_handle,
                object_handle=object_handle,
            )
        )

    base_fps = int(cfg.fps) if cfg.fps > 0 else max(clip.fps for clip in clips)
    max_duration_s = max((max(clip.n_frames - 1, 0) / float(clip.fps)) for clip in clips)
    max_global_frame = max(0, int(math.ceil(max_duration_s * float(base_fps))))

    with server.gui.add_folder("Scene"):
        server.gui.add_markdown(
            "\n".join(
                [
                    f"Loaded human sequences: `{len(clips)}`",
                    f"Layout: `{rows} x {columns}`",
                    f"Motion root: `{motion_root}`",
                    f"Alignment: `{'first pelvis XY' if cfg.align_xy_to_first_pelvis else 'raw world'}`",
                ]
            )
        )

    with server.gui.add_folder("Display"):
        show_points_cb = server.gui.add_checkbox("Show points", initial_value=bool(cfg.show_points))
        show_lines_cb = server.gui.add_checkbox("Show skeleton lines", initial_value=bool(cfg.show_lines))
        show_objects_cb = server.gui.add_checkbox("Show objects", initial_value=bool(cfg.show_objects))

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Global frame",
            min=0,
            max=max_global_frame,
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=base_fps, min=1, max=120, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=bool(cfg.loop))
        clock_md = server.gui.add_markdown("t = 0.00s")

    playing = {"flag": bool(cfg.autoplay)}
    updating_slider = {"flag": False}

    @show_points_cb.on_update
    def _(_evt) -> None:
        for clip in clips:
            clip.points_handle.visible = bool(show_points_cb.value)

    @show_lines_cb.on_update
    def _(_evt) -> None:
        for clip in clips:
            clip.lines_handle.visible = bool(show_lines_cb.value)

    @show_objects_cb.on_update
    def _(_evt) -> None:
        for clip in clips:
            if clip.object_handle is not None:
                clip.object_handle.visible = bool(show_objects_cb.value)

    def _apply_global_frame(global_frame: int) -> None:
        t_s = float(global_frame) / float(max(1, int(fps_in.value)))
        with server.atomic():
            for clip in clips:
                local = int(round(t_s * float(clip.fps)))
                if loop_cb.value and clip.n_frames > 0:
                    local = local % clip.n_frames
                else:
                    local = min(max(local, 0), clip.n_frames - 1)
                joints = clip.joints[local] + clip.offset.reshape(1, 3)
                clip.points_handle.points = joints.astype(np.float32, copy=False)
                clip.lines_handle.points = _line_segments(joints).astype(np.float32, copy=False)
                if clip.object_handle is not None and clip.object_pos is not None and clip.object_quat is not None:
                    clip.object_handle.position = clip.object_pos[local] + clip.offset
                    clip.object_handle.wxyz = clip.object_quat[local]
        clock_md.content = f"t = {t_s:.2f}s"

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_slider["flag"]:
            return
        playing["flag"] = False
        _apply_global_frame(int(frame_slider.value))

    def _player_loop() -> None:
        while True:
            if not playing["flag"]:
                time.sleep(0.02)
                continue
            fps_val = max(1, int(fps_in.value))
            next_frame = int(frame_slider.value) + 1
            if next_frame > max_global_frame:
                if loop_cb.value:
                    next_frame = 0
                else:
                    next_frame = max_global_frame
                    playing["flag"] = False
            updating_slider["flag"] = True
            frame_slider.value = next_frame
            updating_slider["flag"] = False
            _apply_global_frame(next_frame)
            time.sleep(1.0 / float(fps_val))

    _apply_global_frame(0)
    threading.Thread(target=_player_loop, daemon=True).start()
    print(f"[viser_multi_human] Loaded {len(clips)} human sequences from {motion_root}")
    print(f"[viser_multi_human] Open http://localhost:{cfg.port}")
    print("[viser_multi_human] Close the process with Ctrl+C to exit.")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(MultiHumanJointsConfig))
