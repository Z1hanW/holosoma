#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReviewConfig:
    motion_dirs_csv: str = ",".join(
        (
            str(REPO_ROOT / "src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry_aug_extra"),
            str(REPO_ROOT / "src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry_aug_extra"),
        )
    )
    robot_urdf: str = str(REPO_ROOT / "src/holosoma_retargeting/models/g1/g1_29dof.urdf")
    default_object_urdf: str = str(REPO_ROOT / "src/holosoma_retargeting/models/largebox/largebox.urdf")
    port: int = 18080
    autoplay: bool = False
    loop: bool = True
    show_meshes: bool = True
    show_object: bool = True
    start_clip: str = ""
    fps: int | None = None
    decision_log_path: str = str(REPO_ROOT / "logs/object_interaction_review_decisions.json")
    auto_advance_after_keep: bool = True


@dataclass(frozen=True)
class ClipEntry:
    label: str
    stem: str
    path: Path
    source_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_motion_dirs(csv_value: str) -> list[Path]:
    parts = [part.strip() for part in csv_value.split(",") if part.strip()]
    if not parts:
        raise ValueError("motion_dirs_csv is empty.")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for part in parts:
        path = Path(part).expanduser().resolve()
        if path in seen:
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Motion dir not found: {path}")
        resolved.append(path)
        seen.add(path)
    return resolved


def _build_entries(motion_dirs: list[Path]) -> list[ClipEntry]:
    paths: list[Path] = []
    for motion_dir in motion_dirs:
        for path in sorted(motion_dir.glob("*.npz")):
            if path.name.startswith("."):
                continue
            paths.append(path.resolve())

    if not paths:
        raise ValueError("No .npz files found in motion dirs.")

    stem_counts: dict[str, int] = {}
    for path in paths:
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1

    entries: list[ClipEntry] = []
    for path in sorted(paths):
        label = path.stem if stem_counts[path.stem] == 1 else f"{path.parent.name}/{path.stem}"
        entries.append(
            ClipEntry(
                label=label,
                stem=path.stem,
                path=path,
                source_dir=path.parent,
            )
        )
    return entries


def _scalar_string(value: object) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    flat = arr.reshape(-1)
    if flat.size == 1:
        return str(flat[0].item())
    return str(value)


def _load_npz_payload(path: Path, default_object_urdf: Path | None) -> tuple[np.ndarray, int, Path | None]:
    with np.load(path, allow_pickle=True) as data:
        if "qpos" not in data:
            raise ValueError(f"Missing qpos in {path}")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps_val = int(np.asarray(data.get("fps", 30)).reshape(-1)[0])

        object_urdf: Path | None = None
        if "object_urdf_path" in data:
            candidate = Path(_scalar_string(data["object_urdf_path"])).expanduser()
            if not candidate.is_absolute():
                candidate = (path.parent / candidate).resolve()
            if candidate.exists():
                object_urdf = candidate

        if object_urdf is None and "object_name" in data:
            object_name = _scalar_string(data["object_name"]).strip()
            if object_name:
                candidate = REPO_ROOT / "src/holosoma_retargeting/models" / object_name / f"{object_name}.urdf"
                if candidate.exists():
                    object_urdf = candidate.resolve()

        if object_urdf is None and default_object_urdf is not None:
            stem_lc = path.stem.lower()
            if "largebox" in stem_lc or "box" not in stem_lc:
                object_urdf = default_object_urdf

    return qpos, fps_val, object_urdf


def _load_decisions(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        clips = payload["clips"]
        return {str(k): dict(v) for k, v in clips.items() if isinstance(v, dict)}
    if isinstance(payload, dict):
        return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
    raise ValueError(f"Unexpected decision log payload: {path}")


def _write_decisions(path: Path, decisions: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": _utc_now(),
        "clips": decisions,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _trash_path_for(entry: ClipEntry) -> Path:
    trash_dir = entry.source_dir.parent / f"{entry.source_dir.name}_trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    candidate = trash_dir / entry.path.name
    if not candidate.exists():
        return candidate
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return trash_dir / f"{entry.path.stem}_{suffix}{entry.path.suffix}"


def run_review(cfg: ReviewConfig) -> None:
    motion_dirs = _resolve_motion_dirs(cfg.motion_dirs_csv)
    entries = _build_entries(motion_dirs)
    entries_by_label = {entry.label: entry for entry in entries}
    labels: list[str] = [entry.label for entry in entries]

    decision_log_path = Path(cfg.decision_log_path).expanduser().resolve()
    decisions = _load_decisions(decision_log_path)

    default_object_urdf = Path(cfg.default_object_urdf).expanduser().resolve() if cfg.default_object_urdf else None
    if default_object_urdf is not None and not default_object_urdf.exists():
        default_object_urdf = None

    robot_urdf = Path(cfg.robot_urdf).expanduser().resolve()
    if not robot_urdf.exists():
        raise FileNotFoundError(f"Robot URDF not found: {robot_urdf}")

    server = viser.ViserServer(port=cfg.port)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_state: dict[str, object | None] = {"frame": None, "key": None}

    robot_urdf_y = yourdfpy.URDF.load(str(robot_urdf), load_meshes=True, build_scene_graph=True)
    robot_viser = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")
    robot_viser.show_visual = cfg.show_meshes
    joint_count = len(robot_viser.get_actuated_joint_limits())

    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))

    motion_cache: dict[str, dict[str, object]] = {}
    object_cache: dict[str, tuple[viser.FrameHandle, ViserUrdf]] = {}
    motion_state: dict[str, object] = {}
    active_label = {"value": ""}
    playing = {"flag": bool(cfg.autoplay)}
    updating_slider = {"flag": False}

    with server.gui.add_folder("Motion"):
        clip_dropdown = server.gui.add_dropdown(
            "Clip",
            options=tuple(labels),
            initial_value=cfg.start_clip if cfg.start_clip in labels else labels[0],
        )
        clip_info = server.gui.add_markdown("")
        clip_path_md = server.gui.add_markdown("")
        clip_status_md = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show robot meshes", initial_value=cfg.show_meshes)
        show_object_cb = server.gui.add_checkbox("Show object meshes", initial_value=cfg.show_object)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_initial = int(cfg.fps) if cfg.fps is not None else 30
        fps_in = server.gui.add_number("FPS", initial_value=fps_initial, min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=cfg.loop)

    with server.gui.add_folder("Review"):
        stats_md = server.gui.add_markdown("")
        action_md = server.gui.add_markdown("No actions yet.")
        keep_btn = server.gui.add_button("Keep")
        delete_btn = server.gui.add_button("Delete To Trash")

    def _decision_for(entry: ClipEntry | None) -> dict[str, object] | None:
        if entry is None:
            return None
        return decisions.get(str(entry.path))

    def _update_stats() -> None:
        kept_active = 0
        undecided_active = 0
        for label in labels:
            entry = entries_by_label[label]
            action = str(decisions.get(str(entry.path), {}).get("action", "")).strip().lower()
            if action == "keep":
                kept_active += 1
            else:
                undecided_active += 1
        deleted_total = sum(1 for payload in decisions.values() if str(payload.get("action", "")).lower() == "delete")
        stats_md.content = (
            f"Active clips: {len(labels)} | undecided: {undecided_active} | kept: {kept_active} | deleted: {deleted_total}"
        )

    def _update_clip_info() -> None:
        label = active_label["value"]
        entry = entries_by_label.get(label)
        if entry is None:
            clip_info.content = "No clips available."
            clip_path_md.content = ""
            clip_status_md.content = ""
            return
        clip_info.content = (
            f"Clip: `{entry.stem}` | frames: {motion_state['n_frames']} | fps: {motion_state['fps']}"
        )
        clip_path_md.content = f"Path: `{entry.path}`"
        decision = _decision_for(entry)
        if decision is None:
            clip_status_md.content = "Status: undecided"
            return
        action = str(decision.get("action", "undecided"))
        when = str(decision.get("timestamp_utc", ""))
        clip_status_md.content = f"Status: {action} | updated: {when}"

    def _ensure_motion_loaded(label: str) -> dict[str, object]:
        if label in motion_cache:
            return motion_cache[label]
        entry = entries_by_label[label]
        qpos, fps_val, object_urdf = _load_npz_payload(entry.path, default_object_urdf)
        if qpos.shape[0] == 0:
            raise ValueError(f"Motion {entry.path} has zero frames.")
        motion_cache[label] = {
            "qpos": qpos,
            "fps": int(fps_val),
            "n_frames": int(qpos.shape[0]),
            "object_urdf": object_urdf,
        }
        return motion_cache[label]

    def _set_object(label: str) -> None:
        entry_state = _ensure_motion_loaded(label)
        object_urdf = entry_state.get("object_urdf")
        prev_key = object_state["key"]
        if isinstance(prev_key, str) and prev_key in object_cache:
            object_cache[prev_key][1].show_visual = False
        if not isinstance(object_urdf, Path):
            object_state["key"] = None
            object_state["frame"] = None
            return

        object_key = str(object_urdf)
        if object_key not in object_cache:
            object_root_path = f"/object/{len(object_cache)}"
            object_frame = server.scene.add_frame(object_root_path, show_axes=False)
            object_urdf_y = yourdfpy.URDF.load(str(object_urdf), load_meshes=True, build_scene_graph=True)
            object_viser = ViserUrdf(server, urdf_or_path=object_urdf_y, root_node_name=object_root_path)
            object_cache[object_key] = (object_frame, object_viser)
        object_frame, object_viser = object_cache[object_key]
        object_viser.show_visual = bool(show_object_cb.value)
        object_state["key"] = object_key
        object_state["frame"] = object_frame

    def _apply_frame(frame_idx: int) -> None:
        if "qpos" not in motion_state:
            return
        qpos = np.asarray(motion_state["qpos"])
        qpos_arr = qpos[int(frame_idx)]
        robot_root.position = qpos_arr[:3]
        robot_root.wxyz = qpos_arr[3:7]
        robot_viser.update_cfg(qpos_arr[7 : 7 + joint_count])
        object_frame = object_state["frame"]
        if object_frame is not None and qpos_arr.shape[0] >= 7 + joint_count + 7:
            object_frame.position = qpos_arr[-7:-4]
            object_frame.wxyz = qpos_arr[-4:]

    def _select_label(label: str) -> None:
        if label not in entries_by_label:
            return
        state = _ensure_motion_loaded(label)
        motion_state.clear()
        motion_state.update({"label": label, **state})
        active_label["value"] = label
        _set_object(label)
        frame_slider.max = max(0, int(motion_state["n_frames"]) - 1)
        updating_slider["flag"] = True
        frame_slider.value = 0
        updating_slider["flag"] = False
        if cfg.fps is None:
            fps_in.value = int(motion_state["fps"])
        _apply_frame(0)
        _update_clip_info()

    def _next_label_after(current_label: str) -> str | None:
        if not labels:
            return None
        if current_label not in labels:
            return labels[0]
        idx = labels.index(current_label)
        if idx + 1 < len(labels):
            return labels[idx + 1]
        if idx > 0:
            return labels[idx - 1]
        return labels[0]

    def _persist_decision(entry: ClipEntry, action: str, extra: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = {
            "action": action,
            "clip_label": entry.label,
            "clip_name": entry.stem,
            "path": str(entry.path),
            "source_dir": str(entry.source_dir),
            "timestamp_utc": _utc_now(),
        }
        if extra:
            payload.update(extra)
        decisions[str(entry.path)] = payload
        _write_decisions(decision_log_path, decisions)
        _update_stats()
        _update_clip_info()

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        robot_viser.show_visual = bool(show_meshes_cb.value)

    @show_object_cb.on_update
    def _(_evt) -> None:
        object_key = object_state["key"]
        if isinstance(object_key, str) and object_key in object_cache:
            object_cache[object_key][1].show_visual = bool(show_object_cb.value)

    @clip_dropdown.on_update
    def _(_evt) -> None:
        _select_label(str(clip_dropdown.value))

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_slider["flag"]:
            return
        _apply_frame(int(frame_slider.value))

    @keep_btn.on_click
    def _(_evt) -> None:
        entry = entries_by_label.get(active_label["value"])
        if entry is None:
            return
        _persist_decision(entry, "keep")
        action_md.content = f"Kept `{entry.stem}`."
        if cfg.auto_advance_after_keep:
            next_label = _next_label_after(entry.label)
            if next_label is not None and next_label != entry.label:
                clip_dropdown.value = next_label

    @delete_btn.on_click
    def _(_evt) -> None:
        entry = entries_by_label.get(active_label["value"])
        if entry is None:
            return
        destination = _trash_path_for(entry)
        shutil.move(str(entry.path), str(destination))
        _persist_decision(entry, "delete", {"trash_path": str(destination)})
        motion_cache.pop(entry.label, None)
        labels.remove(entry.label)
        del entries_by_label[entry.label]
        action_md.content = f"Moved `{entry.stem}` to `{destination}`."
        if not labels:
            playing["flag"] = False
            clip_dropdown.options = ("<empty>",)
            clip_dropdown.value = "<empty>"
            active_label["value"] = ""
            motion_state.clear()
            clip_info.content = "No clips available."
            clip_path_md.content = ""
            clip_status_md.content = ""
            _update_stats()
            return
        next_label = _next_label_after(entry.label)
        clip_dropdown.options = tuple(labels)
        clip_dropdown.value = next_label or labels[0]

    def _player_loop() -> None:
        next_tick = time.time()
        while True:
            if playing["flag"] and "n_frames" in motion_state:
                fps_val = max(1, int(fps_in.value))
                now = time.time()
                if now >= next_tick:
                    next_tick = now + 1.0 / float(fps_val)
                    frame_idx = int(frame_slider.value) + 1
                    last_frame = int(motion_state["n_frames"]) - 1
                    if frame_idx > last_frame:
                        if loop_cb.value:
                            frame_idx = 0
                        else:
                            frame_idx = last_frame
                            playing["flag"] = False
                    updating_slider["flag"] = True
                    frame_slider.value = frame_idx
                    updating_slider["flag"] = False
                    _apply_frame(frame_idx)
            time.sleep(0.001)

    threading.Thread(target=_player_loop, daemon=True).start()

    initial_label = cfg.start_clip if cfg.start_clip in labels else labels[0]
    for label in labels:
        entry = entries_by_label[label]
        action = str(decisions.get(str(entry.path), {}).get("action", "")).lower()
        if action != "keep":
            initial_label = label
            break
    _update_stats()
    _select_label(initial_label)

    print(f"[viser_review_bank] Reviewing {len(labels)} clips")
    print(f"[viser_review_bank] Decision log: {decision_log_path}")
    print(f"[viser_review_bank] Open http://localhost:{server.get_port()}")
    print("[viser_review_bank] Keep leaves the clip in place and records the decision.")
    print("[viser_review_bank] Delete moves the clip into a sibling *_trash directory.")

    while True:
        time.sleep(1.0)


def main() -> None:
    cfg = tyro.cli(ReviewConfig)
    run_review(cfg)


if __name__ == "__main__":
    main()
