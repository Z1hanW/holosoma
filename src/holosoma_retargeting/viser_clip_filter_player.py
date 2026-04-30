#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).resolve().parents[2]
RETARGETING_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ClipFilterConfig:
    source_dir: str = str(
        REPO_ROOT
        / "src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo-ca-process-no-boxlike"
    )
    output_dir: str = str(
        REPO_ROOT
        / "src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo-ca-process-no-boxlike-curated"
    )
    robot_urdf: str = str(REPO_ROOT / "src/holosoma_retargeting/models/g1/g1_29dof.urdf")
    port: int = 1090
    start_clip: str = ""
    autoplay: bool = True
    loop: bool = True
    copy_mode: str = "copy"
    show_robot_meshes: bool = True
    show_object_meshes: bool = True
    show_grid: bool = True
    grid_size: float = 6.0


def _safe_node_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_") or "clip"


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[object] = []
    for part in parts:
        if part:
            key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def _scalar_string(value: object) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    flat = arr.reshape(-1)
    if flat.size == 1:
        return str(flat[0].item())
    return str(value)


def _load_clip_object_map(source_dir: Path) -> dict[str, dict[str, Any]]:
    map_path = source_dir / "_viser_object_urdf_map.json"
    if not map_path.is_file():
        map_path = source_dir / "_clip_object_urdf_map.json"
    if not map_path.is_file():
        return {}
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    clips = payload.get("clips", payload) if isinstance(payload, dict) else {}
    if not isinstance(clips, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in clips.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def _resolve_urdf(raw: str, base_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path if path.is_file() else None


def _resolve_object_urdf(
    data: np.lib.npyio.NpzFile,
    clip_path: Path,
    object_map: dict[str, dict[str, Any]],
) -> Path | None:
    stem = clip_path.stem
    mapped = object_map.get(stem)
    if mapped is not None:
        urdf = _resolve_urdf(str(mapped.get("object_urdf_path", "")).strip(), clip_path.parent)
        if urdf is not None:
            return urdf
    if "object_urdf_path" in data:
        urdf = _resolve_urdf(_scalar_string(data["object_urdf_path"]).strip(), clip_path.parent)
        if urdf is not None:
            return urdf
    if "object_name" in data:
        object_name = _scalar_string(data["object_name"]).strip()
        candidate = RETARGETING_ROOT / "models" / object_name / f"{object_name}.urdf"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _list_clip_paths(source_dir: Path) -> list[Path]:
    paths = sorted((path for path in source_dir.glob("*.npz") if path.is_file()), key=_natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No .npz files found under {source_dir}")
    return paths


def _load_metrics(source_dir: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    path = source_dir / "_filter_selected_metrics.csv"
    if not path.is_file():
        return [], {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = {row.get("task", ""): row for row in reader if row.get("task")}
    return fieldnames, rows


def _load_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "_review_state.json"
    if not path.is_file():
        return {"kept": [], "rejected": [], "actions": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"kept": [], "rejected": [], "actions": []}
    if not isinstance(payload, dict):
        return {"kept": [], "rejected": [], "actions": []}
    payload.setdefault("kept", [])
    payload.setdefault("rejected", [])
    payload.setdefault("actions", [])
    return payload


def _task_from_stem(stem: str) -> str:
    return stem[: -len("_original")] if stem.endswith("_original") else stem


class ClipReviewSession:
    def __init__(
        self,
        *,
        source_dir: Path,
        output_dir: Path,
        clip_paths: list[Path],
        object_map: dict[str, dict[str, Any]],
        metrics_fieldnames: list[str],
        metrics_rows: dict[str, dict[str, str]],
        copy_mode: str,
    ) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.clip_paths = clip_paths
        self.object_map = object_map
        self.metrics_fieldnames = metrics_fieldnames
        self.metrics_rows = metrics_rows
        self.copy_mode = copy_mode.lower().strip()
        self.object_urdf_dir = self.output_dir / "_viser_object_urdfs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.object_urdf_dir.mkdir(parents=True, exist_ok=True)
        self.state = _load_state(self.output_dir)
        self._normalize_state()
        self._rewrite_outputs()

    @property
    def kept(self) -> set[str]:
        return set(str(item) for item in self.state.get("kept", []))

    @property
    def rejected(self) -> set[str]:
        return set(str(item) for item in self.state.get("rejected", []))

    def _normalize_state(self) -> None:
        valid = {path.stem for path in self.clip_paths}
        kept = [stem for stem in self.state.get("kept", []) if stem in valid]
        rejected = [stem for stem in self.state.get("rejected", []) if stem in valid and stem not in set(kept)]
        self.state["kept"] = sorted(set(kept), key=lambda stem: [path.stem for path in self.clip_paths].index(stem))
        self.state["rejected"] = sorted(
            set(rejected),
            key=lambda stem: [path.stem for path in self.clip_paths].index(stem),
        )

    def decision(self, stem: str) -> str:
        if stem in self.kept:
            return "KEEP"
        if stem in self.rejected:
            return "DELETE"
        return "UNREVIEWED"

    def keep(self, stem: str) -> None:
        src = self.source_dir / f"{stem}.npz"
        dst = self.output_dir / src.name
        if not src.is_file():
            raise FileNotFoundError(src)
        if self.copy_mode == "hardlink":
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        self._ensure_output_object_urdf(stem, src)
        kept = self.kept
        rejected = self.rejected
        kept.add(stem)
        rejected.discard(stem)
        self._set_sets(kept, rejected)
        self._record_action(stem, "keep")
        self._rewrite_outputs()

    def reject(self, stem: str) -> None:
        for path in (
            self.output_dir / f"{stem}.npz",
            self.object_urdf_dir / f"{stem}.urdf",
        ):
            if path.exists() or path.is_symlink():
                path.unlink()
        kept = self.kept
        rejected = self.rejected
        kept.discard(stem)
        rejected.add(stem)
        self._set_sets(kept, rejected)
        self._record_action(stem, "delete")
        self._rewrite_outputs()

    def clear(self, stem: str) -> None:
        kept = self.kept
        rejected = self.rejected
        kept.discard(stem)
        rejected.discard(stem)
        self._set_sets(kept, rejected)
        self._record_action(stem, "clear")
        self._rewrite_outputs()

    def _set_sets(self, kept: set[str], rejected: set[str]) -> None:
        order = [path.stem for path in self.clip_paths]
        self.state["kept"] = [stem for stem in order if stem in kept]
        self.state["rejected"] = [stem for stem in order if stem in rejected and stem not in kept]

    def _record_action(self, stem: str, action: str) -> None:
        actions = self.state.setdefault("actions", [])
        if isinstance(actions, list):
            actions.append({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "clip": stem, "action": action})

    def _save_state(self) -> None:
        (self.output_dir / "_review_state.json").write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _ensure_output_object_urdf(self, stem: str, src_npz: Path) -> dict[str, Any]:
        dst_urdf = self.object_urdf_dir / f"{stem}.urdf"
        mapped = self.object_map.get(stem)
        src_urdf = None
        if mapped is not None:
            src_urdf = _resolve_urdf(str(mapped.get("object_urdf_path", "")).strip(), self.source_dir)
        if src_urdf is not None:
            shutil.copy2(src_urdf, dst_urdf)
            out = dict(mapped or {})
            out["object_urdf_path"] = str(dst_urdf)
            return out

        with np.load(src_npz, allow_pickle=True) as data:
            object_name = _scalar_string(data["object_name"]) if "object_name" in data else "object"
            if "object_mesh_path" in data:
                mesh_raw = _scalar_string(data["object_mesh_path"]).strip()
                mesh_path = Path(mesh_raw)
                if not mesh_path.is_absolute():
                    mesh_path = (RETARGETING_ROOT / mesh_path).resolve()
            else:
                mesh_path = RETARGETING_ROOT / "models" / object_name / f"{object_name}.obj"
            scale = np.asarray(data["object_mesh_scale"], dtype=float).reshape(-1) if "object_mesh_scale" in data else np.ones(3)
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            if scale.size != 3:
                scale = np.ones(3)
        sx, sy, sz = [float(v) for v in scale.tolist()]
        dst_urdf.write_text(
            f"""<?xml version=\"1.0\" ?>
<robot name=\"{html.escape(object_name, quote=True)}\">
  <link name=\"{html.escape(object_name, quote=True)}_link\">
    <visual>
      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>
      <geometry><mesh filename=\"{html.escape(str(mesh_path), quote=True)}\" scale=\"{sx:.9g} {sy:.9g} {sz:.9g}\"/></geometry>
      <material name=\"mat\"><color rgba=\"0.7 0.8 0.9 0.85\"/></material>
    </visual>
    <collision>
      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>
      <geometry><mesh filename=\"{html.escape(str(mesh_path), quote=True)}\" scale=\"{sx:.9g} {sy:.9g} {sz:.9g}\"/></geometry>
    </collision>
  </link>
</robot>
""",
            encoding="utf-8",
        )
        return {
            "object_name": object_name,
            "object_urdf_path": str(dst_urdf),
            "object_mesh_path": str(mesh_path),
            "object_mesh_scale": [sx, sy, sz],
        }

    def _rewrite_outputs(self) -> None:
        self._save_state()
        kept_order = [path.stem for path in self.clip_paths if path.stem in self.kept]
        viewer_map: dict[str, dict[str, Any]] = {}
        metrics_rows: list[dict[str, str]] = []
        object_counts: dict[str, int] = {}
        for stem in kept_order:
            src_npz = self.source_dir / f"{stem}.npz"
            if not (self.output_dir / f"{stem}.npz").is_file():
                continue
            viewer_map[stem] = self._ensure_output_object_urdf(stem, src_npz)
            object_name = str(viewer_map[stem].get("object_name", ""))
            if object_name:
                object_counts[object_name] = object_counts.get(object_name, 0) + 1
            task = _task_from_stem(stem)
            if task in self.metrics_rows:
                row = dict(self.metrics_rows[task])
                row["raw_path"] = str(self.output_dir / f"{stem}.npz")
                metrics_rows.append(row)

        (self.output_dir / "_viser_object_urdf_map.json").write_text(
            json.dumps({"clips": viewer_map}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if self.metrics_fieldnames:
            fieldnames = list(self.metrics_fieldnames)
            if "raw_path" not in fieldnames:
                fieldnames.append("raw_path")
            with (self.output_dir / "_filter_selected_metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in metrics_rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})

        manifest = {
            "created_or_updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_dir": str(self.source_dir),
            "target_dir": str(self.output_dir),
            "total_source_clips": len(self.clip_paths),
            "kept_count": len(kept_order),
            "rejected_count": len(self.rejected),
            "unreviewed_count": len(self.clip_paths) - len(self.kept) - len(self.rejected),
            "copy_mode": self.copy_mode,
            "object_counts": dict(sorted(object_counts.items())),
            "notes": [
                "Generated by viser_clip_filter_player.py.",
                "Keep copies the current clip into this folder; Delete removes it from this folder.",
            ],
        }
        (self.output_dir / "_filter_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    return q if norm == 0.0 else q / norm


def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
    curr = _quat_normalize(curr_q)
    if prev_q is None:
        return curr
    return -curr if float(np.dot(prev_q, curr)) < 0.0 else curr


def main(cfg: ClipFilterConfig) -> None:
    source_dir = Path(cfg.source_dir).expanduser().resolve()
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    robot_urdf_path = Path(cfg.robot_urdf).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    if not robot_urdf_path.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {robot_urdf_path}")

    clip_paths = _list_clip_paths(source_dir)
    clip_names = [path.stem for path in clip_paths]
    clip_to_idx = {name: idx for idx, name in enumerate(clip_names)}
    start_idx = clip_to_idx.get(cfg.start_clip, 0) if cfg.start_clip else 0
    object_map = _load_clip_object_map(source_dir)
    metrics_fieldnames, metrics_rows = _load_metrics(source_dir)
    review = ClipReviewSession(
        source_dir=source_dir,
        output_dir=output_dir,
        clip_paths=clip_paths,
        object_map=object_map,
        metrics_fieldnames=metrics_fieldnames,
        metrics_rows=metrics_rows,
        copy_mode=cfg.copy_mode,
    )

    robot_urdf_y = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True, build_scene_graph=True)
    robot_joint_names = [joint.name for joint in robot_urdf_y.robot.joints if joint.type != "fixed"]
    joint_count = len(robot_joint_names)

    server = viser.ViserServer(port=int(cfg.port))
    if cfg.show_grid:
        server.scene.add_grid("/grid", width=float(cfg.grid_size), height=float(cfg.grid_size), position=(0.0, 0.0, 0.0))

    robot_frame = server.scene.add_frame("/robot", show_axes=False)
    robot_viser = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")
    robot_viser.show_visual = bool(cfg.show_robot_meshes)

    object_urdf_cache: dict[Path, yourdfpy.URDF] = {}
    object_viser_cache: dict[str, ViserUrdf] = {}
    object_frame_cache: dict[str, viser.FrameHandle] = {}

    current: dict[str, Any] = {
        "idx": start_idx,
        "qpos": None,
        "fps": 30,
        "object_stem": None,
        "prev_robot_q": None,
        "prev_object_q": None,
    }

    with server.gui.add_folder("Review"):
        clip_dropdown = server.gui.add_dropdown("Clip", options=tuple(clip_names), initial_value=clip_names[start_idx])
        clip_info = server.gui.add_markdown("")
        stats_info = server.gui.add_markdown("")
        keep_btn = server.gui.add_button("Keep + Next")
        delete_btn = server.gui.add_button("Delete + Next")
        clear_btn = server.gui.add_button("Clear Decision")
        prev_btn = server.gui.add_button("Prev")
        next_btn = server.gui.add_button("Next")
        next_unreviewed_btn = server.gui.add_button("Next Unreviewed")

    with server.gui.add_folder("Display"):
        show_robot_cb = server.gui.add_checkbox("Show robot meshes", initial_value=bool(cfg.show_robot_meshes))
        show_object_cb = server.gui.add_checkbox("Show object meshes", initial_value=bool(cfg.show_object_meshes))

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=30, min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=bool(cfg.loop))

    playing = {"flag": bool(cfg.autoplay)}
    updating_gui = {"flag": False}

    def _hide_current_object() -> None:
        stem = current.get("object_stem")
        if stem is not None and stem in object_viser_cache:
            object_viser_cache[stem].show_visual = False

    def _ensure_object_for_clip(stem: str, data: np.lib.npyio.NpzFile, clip_path: Path) -> None:
        object_urdf = _resolve_object_urdf(data, clip_path, object_map)
        current["object_stem"] = None
        if object_urdf is None or current["qpos"].shape[1] < 7 + joint_count + 7:
            return
        if stem not in object_viser_cache:
            frame_path = f"/objects/{_safe_node_name(stem)}"
            object_frame_cache[stem] = server.scene.add_frame(frame_path, show_axes=False)
            if object_urdf not in object_urdf_cache:
                object_urdf_cache[object_urdf] = yourdfpy.URDF.load(
                    str(object_urdf),
                    load_meshes=True,
                    build_scene_graph=True,
                )
            object_viser_cache[stem] = ViserUrdf(
                server,
                urdf_or_path=object_urdf_cache[object_urdf],
                root_node_name=frame_path,
            )
        for cached_stem, cached_viser in object_viser_cache.items():
            cached_viser.show_visual = bool(show_object_cb.value) if cached_stem == stem else False
        current["object_stem"] = stem

    def _update_info() -> None:
        stem = clip_names[int(current["idx"])]
        idx = int(current["idx"])
        decision = review.decision(stem)
        qpos = current["qpos"]
        n_frames = int(qpos.shape[0]) if qpos is not None else 0
        with np.load(clip_paths[idx], allow_pickle=True) as data:
            object_name = _scalar_string(data["object_name"]) if "object_name" in data else "object"
        clip_info.content = (
            f"Clip `{idx + 1}/{len(clip_names)}`: `{stem}`  \n"
            f"Object: `{object_name}`  \n"
            f"Frames: `{n_frames}` | Decision: `{decision}`"
        )
        stats_info.content = (
            f"Kept: `{len(review.kept)}` | Deleted: `{len(review.rejected)}` | "
            f"Unreviewed: `{len(clip_names) - len(review.kept) - len(review.rejected)}`  \n"
            f"Output: `{output_dir}`"
        )

    def _apply_frame(frame_idx: int) -> None:
        qpos = current["qpos"]
        if qpos is None:
            return
        frame_idx = int(np.clip(frame_idx, 0, qpos.shape[0] - 1))
        q = qpos[frame_idx]
        robot_viser.update_cfg(q[7 : 7 + joint_count])
        robot_frame.position = q[:3]
        robot_q = _quat_continuous(current.get("prev_robot_q"), q[3:7])
        current["prev_robot_q"] = robot_q
        robot_frame.wxyz = robot_q
        object_stem = current.get("object_stem")
        if object_stem is not None and object_stem in object_frame_cache and q.shape[0] >= 7 + joint_count + 7:
            frame = object_frame_cache[object_stem]
            frame.position = q[-7:-4]
            object_q = _quat_continuous(current.get("prev_object_q"), q[-4:])
            current["prev_object_q"] = object_q
            frame.wxyz = object_q

    def _set_clip(idx: int, *, update_dropdown: bool = True) -> None:
        idx = int(idx) % len(clip_names)
        _hide_current_object()
        current["idx"] = idx
        current["prev_robot_q"] = None
        current["prev_object_q"] = None
        clip_path = clip_paths[idx]
        with np.load(clip_path, allow_pickle=True) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float32)
            fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30
            current["qpos"] = qpos
            current["fps"] = max(1, fps)
            _ensure_object_for_clip(clip_path.stem, data, clip_path)
        updating_gui["flag"] = True
        if update_dropdown:
            clip_dropdown.value = clip_path.stem
        frame_slider.max = max(0, int(current["qpos"].shape[0]) - 1)
        frame_slider.value = 0
        fps_in.value = int(current["fps"])
        updating_gui["flag"] = False
        _apply_frame(0)
        _update_info()

    def _next_index(from_idx: int, *, unreviewed_only: bool = False) -> int:
        for step in range(1, len(clip_names) + 1):
            idx = (from_idx + step) % len(clip_names)
            if not unreviewed_only or review.decision(clip_names[idx]) == "UNREVIEWED":
                return idx
        return from_idx

    @show_robot_cb.on_update
    def _(_evt) -> None:
        robot_viser.show_visual = bool(show_robot_cb.value)

    @show_object_cb.on_update
    def _(_evt) -> None:
        stem = current.get("object_stem")
        for cached_stem, cached_viser in object_viser_cache.items():
            cached_viser.show_visual = bool(show_object_cb.value) if cached_stem == stem else False

    @clip_dropdown.on_update
    def _(_evt) -> None:
        if updating_gui["flag"]:
            return
        _set_clip(clip_to_idx[str(clip_dropdown.value)], update_dropdown=False)

    @keep_btn.on_click
    def _(_evt) -> None:
        stem = clip_names[int(current["idx"])]
        review.keep(stem)
        _set_clip(_next_index(int(current["idx"]), unreviewed_only=True))

    @delete_btn.on_click
    def _(_evt) -> None:
        stem = clip_names[int(current["idx"])]
        review.reject(stem)
        _set_clip(_next_index(int(current["idx"]), unreviewed_only=True))

    @clear_btn.on_click
    def _(_evt) -> None:
        stem = clip_names[int(current["idx"])]
        review.clear(stem)
        _update_info()

    @prev_btn.on_click
    def _(_evt) -> None:
        _set_clip(int(current["idx"]) - 1)

    @next_btn.on_click
    def _(_evt) -> None:
        _set_clip(int(current["idx"]) + 1)

    @next_unreviewed_btn.on_click
    def _(_evt) -> None:
        _set_clip(_next_index(int(current["idx"]), unreviewed_only=True))

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]
        current["prev_robot_q"] = None
        current["prev_object_q"] = None

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_gui["flag"]:
            return
        playing["flag"] = False
        current["prev_robot_q"] = None
        current["prev_object_q"] = None
        _apply_frame(int(frame_slider.value))

    def _player_loop() -> None:
        while True:
            if not playing["flag"] or current["qpos"] is None:
                time.sleep(0.02)
                continue
            fps = max(1, int(fps_in.value))
            next_frame = int(frame_slider.value) + 1
            last_frame = int(current["qpos"].shape[0]) - 1
            if next_frame > last_frame:
                if loop_cb.value:
                    next_frame = 0
                    current["prev_robot_q"] = None
                    current["prev_object_q"] = None
                else:
                    next_frame = last_frame
                    playing["flag"] = False
            updating_gui["flag"] = True
            frame_slider.value = next_frame
            updating_gui["flag"] = False
            _apply_frame(next_frame)
            time.sleep(1.0 / float(fps))

    _set_clip(start_idx)
    threading.Thread(target=_player_loop, daemon=True).start()
    print(f"[viser_clip_filter] Source: {source_dir}")
    print(f"[viser_clip_filter] Output: {output_dir}")
    print(f"[viser_clip_filter] Loaded {len(clip_names)} clips")
    print(f"[viser_clip_filter] Open http://localhost:{cfg.port}")
    print("[viser_clip_filter] Use Keep + Next or Delete + Next in the Review panel.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(ClipFilterConfig))
