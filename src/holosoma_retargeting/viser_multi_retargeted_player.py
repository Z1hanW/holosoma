#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MultiRetargetedConfig:
    motion_root: str = str(REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared")
    robot_urdf: str = str(REPO_ROOT / "src/holosoma_retargeting/models/g1/g1_29dof.urdf")
    default_object_urdf: str = ""
    port: int = 18086
    limit: int = 0
    clip_names_csv: str = ""
    autoplay: bool = False
    loop: bool = True
    show_robot_meshes: bool = True
    show_object_meshes: bool = True
    show_grid: bool = True
    grid_width: float = 8.0
    grid_height: float = 8.0
    align_anchor: str = "robot"
    align_xy_only: bool = True
    playback_fps: int = 0
    visual_fps_multiplier: int = 2


@dataclass
class MotionClip:
    stem: str
    path: Path
    qpos: np.ndarray
    fps: int
    object_urdf: Path | None
    robot_frame: viser.FrameHandle
    robot_viser: ViserUrdf
    object_frame: viser.FrameHandle | None
    object_viser: ViserUrdf | None
    prev_robot_quat: np.ndarray | None = None
    prev_object_quat: np.ndarray | None = None
    last_local_frame: float | None = None

    @property
    def n_frames(self) -> int:
        return int(self.qpos.shape[0])


def _scalar_string(value: object) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    flat = arr.reshape(-1)
    if flat.size == 1:
        return str(flat[0].item())
    return str(value)


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def _safe_node_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")
    return cleaned or "clip"


def _parse_clip_names(csv_value: str) -> list[str]:
    return [part.strip() for part in csv_value.split(",") if part.strip()]


def _load_clip_object_map(motion_root: Path) -> dict[str, dict[str, object]]:
    map_path = motion_root / "_clip_object_urdf_map.json"
    if not map_path.is_file():
        return {}
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    clips = payload.get("clips", payload) if isinstance(payload, dict) else {}
    if not isinstance(clips, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for key, value in clips.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def _resolve_object_urdf(
    data: np.lib.npyio.NpzFile,
    clip_path: Path,
    clip_object_map: dict[str, dict[str, object]],
    default_object_urdf: Path | None,
) -> Path | None:
    candidate_raw = ""
    if "object_urdf_path" in data:
        candidate_raw = _scalar_string(data["object_urdf_path"]).strip()
    elif clip_path.stem in clip_object_map:
        mapped = clip_object_map[clip_path.stem].get("object_urdf_path")
        if isinstance(mapped, str):
            candidate_raw = mapped.strip()

    if candidate_raw:
        candidate = Path(candidate_raw).expanduser()
        if not candidate.is_absolute():
            candidate = (clip_path.parent / candidate).resolve()
        if candidate.is_file():
            return candidate

    if "object_name" in data:
        object_name = _scalar_string(data["object_name"]).strip()
        if object_name:
            for base_dir in (
                REPO_ROOT / "src/holosoma_retargeting/models",
                REPO_ROOT / "src/holosoma_retargeting/models/behave_objects",
            ):
                candidate = base_dir / object_name / f"{object_name}.urdf"
                if candidate.is_file():
                    return candidate.resolve()

    return default_object_urdf if default_object_urdf and default_object_urdf.is_file() else None


def _load_joint_names(data: np.lib.npyio.NpzFile) -> list[str] | None:
    if "joint_names" not in data:
        return None
    return [str(item) for item in np.asarray(data["joint_names"]).reshape(-1)]


def _order_joint_block(
    joint_block: np.ndarray,
    clip_joint_names: list[str] | None,
    viser_joint_names: list[str],
    clip_path: Path,
) -> np.ndarray:
    joint_block = np.asarray(joint_block, dtype=np.float32)
    if clip_joint_names is None:
        if joint_block.shape[1] < len(viser_joint_names):
            raise ValueError(
                f"{clip_path} joint block too small: {joint_block.shape[1]} < expected {len(viser_joint_names)}"
            )
        return joint_block[:, : len(viser_joint_names)]

    name_to_idx = {name: idx for idx, name in enumerate(clip_joint_names)}
    missing = [name for name in viser_joint_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"{clip_path} missing joints for viser robot: {missing}")
    ordered = joint_block[:, [name_to_idx[name] for name in viser_joint_names]]
    return np.asarray(ordered, dtype=np.float32)


def _build_qpos(
    data: np.lib.npyio.NpzFile,
    clip_path: Path,
    viser_joint_names: list[str],
) -> np.ndarray:
    if "qpos" not in data:
        raise ValueError(
            f"{clip_path} does not contain qpos. "
            "Viewer fallback to reconstruct qpos from raw motion is disabled."
        )

    clip_joint_names = _load_joint_names(data)
    raw_qpos = np.asarray(data["qpos"], dtype=np.float32)
    if raw_qpos.ndim != 2 or raw_qpos.shape[1] < 7:
        raise ValueError(f"Invalid qpos array in {clip_path}: shape={raw_qpos.shape}")
    raw_joint_count = len(clip_joint_names) if clip_joint_names is not None else min(
        len(viser_joint_names), max(0, raw_qpos.shape[1] - 7)
    )
    if raw_qpos.shape[1] < 7 + raw_joint_count:
        raise ValueError(f"qpos joint slice is invalid in {clip_path}: shape={raw_qpos.shape}")
    ordered_joints = _order_joint_block(
        raw_qpos[:, 7 : 7 + raw_joint_count],
        clip_joint_names,
        viser_joint_names,
        clip_path,
    )
    tail = raw_qpos[:, 7 + raw_joint_count :]
    return np.concatenate((raw_qpos[:, :7], ordered_joints, tail), axis=1, dtype=np.float32)


def _align_qpos(qpos: np.ndarray, *, anchor: str, xy_only: bool, has_object: bool) -> np.ndarray:
    anchor_lc = anchor.strip().lower()
    if anchor_lc == "none":
        return qpos

    out = np.array(qpos, dtype=np.float32, copy=True)
    if anchor_lc == "object" and has_object:
        offset = np.array(out[0, -7:-4], dtype=np.float32, copy=True)
    else:
        offset = np.array(out[0, :3], dtype=np.float32, copy=True)
    if xy_only:
        offset[2] = 0.0

    out[:, :3] -= offset
    if has_object:
        out[:, -7:-4] -= offset
    return out


def _resolve_motion_paths(cfg: MultiRetargetedConfig, motion_root: Path) -> list[Path]:
    if not motion_root.is_dir():
        raise FileNotFoundError(f"Motion root not found: {motion_root}")

    clip_paths = sorted(
        (path.resolve() for path in motion_root.glob("*.npz") if path.is_file()),
        key=_natural_sort_key,
    )
    if not clip_paths:
        raise ValueError(f"No .npz files found under {motion_root}")

    selected_names = _parse_clip_names(cfg.clip_names_csv)
    if selected_names:
        by_stem = {path.stem: path for path in clip_paths}
        missing = [name for name in selected_names if name not in by_stem]
        if missing:
            raise ValueError(f"Unknown clip names in clip_names_csv: {missing}")
        clip_paths = [by_stem[name] for name in selected_names]

    if cfg.limit > 0:
        clip_paths = clip_paths[: cfg.limit]

    return clip_paths


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    return q if norm == 0.0 else q / norm


def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
    curr = _quat_normalize(curr_q)
    if prev_q is None:
        return curr
    return -curr if float(np.dot(prev_q, curr)) < 0.0 else curr


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0 + u * (q1 - q0))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    denom = math.sin(theta)
    return (math.sin((1.0 - u) * theta) * q0 + math.sin(u * theta) * q1) / denom


def _interp_clip_frame(qpos: np.ndarray, local_frame: float, joint_count: int) -> np.ndarray:
    if qpos.shape[0] == 1:
        return np.array(qpos[0], dtype=np.float32, copy=True)

    i0 = int(math.floor(local_frame))
    i1 = min(i0 + 1, qpos.shape[0] - 1)
    u = float(local_frame - i0)
    q0 = qpos[i0]
    q1 = qpos[i1]
    out = np.array(q0, dtype=np.float32, copy=True)

    out[:3] = (1.0 - u) * q0[:3] + u * q1[:3]
    out[3:7] = _slerp(q0[3:7], q1[3:7], u)
    out[7 : 7 + joint_count] = (1.0 - u) * q0[7 : 7 + joint_count] + u * q1[7 : 7 + joint_count]
    if qpos.shape[1] >= 7 + joint_count + 7:
        out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]
        out[-4:] = _slerp(q0[-4:], q1[-4:], u)
    return out


def _reset_clip_continuity(clips: list[MotionClip]) -> None:
    for clip in clips:
        clip.prev_robot_quat = None
        clip.prev_object_quat = None
        clip.last_local_frame = None


def _apply_clip_pose(
    clip: MotionClip,
    q: np.ndarray,
    *,
    joint_count: int,
) -> None:
    clip.robot_viser.update_cfg(q[7 : 7 + joint_count])
    clip.robot_frame.position = q[:3]
    clip.prev_robot_quat = _quat_continuous(clip.prev_robot_quat, q[3:7])
    clip.robot_frame.wxyz = clip.prev_robot_quat

    if clip.object_frame is not None and clip.object_viser is not None and q.shape[0] >= 7 + joint_count + 7:
        clip.object_frame.position = q[-7:-4]
        clip.prev_object_quat = _quat_continuous(clip.prev_object_quat, q[-4:])
        clip.object_frame.wxyz = clip.prev_object_quat


def main(cfg: MultiRetargetedConfig) -> None:
    motion_root = Path(cfg.motion_root).expanduser().resolve()
    robot_urdf = Path(cfg.robot_urdf).expanduser().resolve()
    default_object_urdf = (
        Path(cfg.default_object_urdf).expanduser().resolve() if cfg.default_object_urdf.strip() else None
    )

    if not robot_urdf.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {robot_urdf}")

    clip_paths = _resolve_motion_paths(cfg, motion_root)
    clip_object_map = _load_clip_object_map(motion_root)

    robot_urdf_y = yourdfpy.URDF.load(str(robot_urdf), load_meshes=True, build_scene_graph=True)
    viser_joint_names = [joint.name for joint in robot_urdf_y.robot.joints if joint.type != "fixed"]
    joint_count = len(viser_joint_names)
    if joint_count == 0:
        raise ValueError(f"No actuated joints found in robot URDF: {robot_urdf}")

    server = viser.ViserServer(port=cfg.port)
    if cfg.show_grid:
        server.scene.add_grid("/grid", width=cfg.grid_width, height=cfg.grid_height, position=(0.0, 0.0, 0.0))

    object_urdf_cache: dict[Path, yourdfpy.URDF] = {}
    clips: list[MotionClip] = []
    loaded_lines: list[str] = []

    for idx, clip_path in enumerate(clip_paths):
        with np.load(clip_path, allow_pickle=True) as data:
            qpos = _build_qpos(data, clip_path, viser_joint_names)
            fps = int(np.asarray(data.get("fps", 30)).reshape(-1)[0])
            if fps <= 0:
                fps = 30
            object_urdf = _resolve_object_urdf(data, clip_path, clip_object_map, default_object_urdf)

        has_object = bool(object_urdf is not None and qpos.shape[1] >= (7 + joint_count + 7))
        qpos = _align_qpos(qpos, anchor=cfg.align_anchor, xy_only=cfg.align_xy_only, has_object=has_object)

        clip_name = f"{idx:02d}_{clip_path.stem}"
        root_name = _safe_node_name(clip_name)
        robot_root_path = f"/clips/{root_name}/robot"
        object_root_path = f"/clips/{root_name}/object"

        robot_frame = server.scene.add_frame(robot_root_path, show_axes=False)
        robot_viser = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name=robot_root_path)
        robot_viser.show_visual = cfg.show_robot_meshes

        object_frame: viser.FrameHandle | None = None
        object_viser: ViserUrdf | None = None
        if has_object and object_urdf is not None:
            object_frame = server.scene.add_frame(object_root_path, show_axes=False)
            if object_urdf not in object_urdf_cache:
                object_urdf_cache[object_urdf] = yourdfpy.URDF.load(
                    str(object_urdf),
                    load_meshes=True,
                    build_scene_graph=True,
                )
            object_viser = ViserUrdf(
                server,
                urdf_or_path=object_urdf_cache[object_urdf],
                root_node_name=object_root_path,
            )
            object_viser.show_visual = cfg.show_object_meshes

        clip = MotionClip(
            stem=clip_path.stem,
            path=clip_path,
            qpos=qpos,
            fps=fps,
            object_urdf=object_urdf,
            robot_frame=robot_frame,
            robot_viser=robot_viser,
            object_frame=object_frame,
            object_viser=object_viser,
        )
        clips.append(clip)
        loaded_lines.append(
            f"- {clip.stem}: frames={clip.n_frames}, fps={clip.fps}, object={'yes' if has_object else 'no'}"
        )

    if not clips:
        raise ValueError("No clips loaded.")

    base_playback_fps = int(cfg.playback_fps) if cfg.playback_fps > 0 else max(clip.fps for clip in clips)
    max_duration_s = max((max(clip.n_frames - 1, 0) / max(1, clip.fps)) for clip in clips)
    max_global_frame = max(0, int(math.ceil(max_duration_s * base_playback_fps)))

    with server.gui.add_folder("Scene"):
        summary_md = server.gui.add_markdown(
            "\n".join(
                [
                    f"Loaded clips: {len(clips)}",
                    f"Playback fps base: {base_playback_fps}",
                    f"Max duration: {max_duration_s:.2f}s",
                    f"Align anchor: {cfg.align_anchor}",
                ]
            )
        )
        loaded_md = server.gui.add_markdown("\n".join(loaded_lines))

    with server.gui.add_folder("Display"):
        show_robot_meshes_cb = server.gui.add_checkbox("Show robot meshes", initial_value=cfg.show_robot_meshes)
        show_object_meshes_cb = server.gui.add_checkbox("Show object meshes", initial_value=cfg.show_object_meshes)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Global frame",
            min=0,
            max=max_global_frame,
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=base_playback_fps, min=1, max=240, step=1)
        interp_mult_in = server.gui.add_number(
            "Visual FPS multiplier",
            initial_value=int(cfg.visual_fps_multiplier),
            min=1,
            max=8,
            step=1,
        )
        loop_cb = server.gui.add_checkbox("Loop", initial_value=cfg.loop)
        clock_md = server.gui.add_markdown("t = 0.00s")

    playing = {"flag": bool(cfg.autoplay)}
    cursor = {"frame": float(frame_slider.value)}
    tick = {"next": time.perf_counter()}
    updating_slider = {"flag": False}

    def _apply_global_frame(global_frame: float) -> None:
        time_s = float(global_frame) / float(base_playback_fps)
        for clip in clips:
            local_frame = time_s * float(clip.fps)
            if loop_cb.value and clip.n_frames > 0:
                local_frame = math.fmod(local_frame, float(clip.n_frames))
                if local_frame < 0.0:
                    local_frame += float(clip.n_frames)
            else:
                local_frame = min(local_frame, float(max(clip.n_frames - 1, 0)))

            if clip.last_local_frame is not None and local_frame < clip.last_local_frame:
                clip.prev_robot_quat = None
                clip.prev_object_quat = None
            clip.last_local_frame = local_frame

            q = _interp_clip_frame(clip.qpos, local_frame, joint_count)
            _apply_clip_pose(clip, q, joint_count=joint_count)

        clock_md.content = f"t = {time_s:.2f}s"

    @show_robot_meshes_cb.on_update
    def _(_evt) -> None:
        value = bool(show_robot_meshes_cb.value)
        for clip in clips:
            clip.robot_viser.show_visual = value

    @show_object_meshes_cb.on_update
    def _(_evt) -> None:
        value = bool(show_object_meshes_cb.value)
        for clip in clips:
            if clip.object_viser is not None:
                clip.object_viser.show_visual = value

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        tick["next"] = time.perf_counter()
        _reset_clip_continuity(clips)
        cursor["frame"] = float(frame_slider.value)

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_slider["flag"]:
            return
        playing["flag"] = False
        tick["next"] = time.perf_counter()
        cursor["frame"] = float(frame_slider.value)
        _reset_clip_continuity(clips)
        _apply_global_frame(cursor["frame"])

    @fps_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @interp_mult_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    def _player_loop() -> None:
        while True:
            if not playing["flag"]:
                time.sleep(0.02)
                continue

            now = time.perf_counter()
            fps_val = max(1, int(fps_in.value))
            mult = max(1, int(interp_mult_in.value))
            dt = 1.0 / float(fps_val * mult)

            if now < tick["next"]:
                time.sleep(min(0.002, max(0.0, tick["next"] - now)))
                continue

            next_cursor = cursor["frame"] + (1.0 / float(mult))
            if loop_cb.value and max_global_frame > 0:
                if next_cursor > float(max_global_frame):
                    next_cursor = 0.0
                    _reset_clip_continuity(clips)
            else:
                if next_cursor >= float(max_global_frame):
                    next_cursor = float(max_global_frame)
                    playing["flag"] = False

            cursor["frame"] = next_cursor
            _apply_global_frame(next_cursor)

            updating_slider["flag"] = True
            frame_slider.value = int(math.floor(next_cursor))
            updating_slider["flag"] = False
            tick["next"] = now + dt

    threading.Thread(target=_player_loop, daemon=True).start()

    _reset_clip_continuity(clips)
    _apply_global_frame(0.0)

    print(f"[viser_multi_retargeted] Loaded {len(clips)} clips from {motion_root}")
    print(f"[viser_multi_retargeted] Robot URDF: {robot_urdf}")
    print(f"[viser_multi_retargeted] Open http://localhost:{cfg.port}")
    print(
        "[viser_multi_retargeted] "
        f"Origin alignment anchor={cfg.align_anchor}, xy_only={cfg.align_xy_only}, loop={cfg.loop}"
    )
    print("[viser_multi_retargeted] Close the process with Ctrl+C to exit.")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(MultiRetargetedConfig))
