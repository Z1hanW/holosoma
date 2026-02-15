from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
import tyro


def _prepare_depth_for_visualization(depth: np.ndarray, near_clip: float, far_clip: float) -> np.ndarray:
    """Clip depth then scale to uint8 grayscale for visualization."""
    depth = np.clip(depth, near_clip, far_clip)
    depth = (depth - near_clip) / (far_clip - near_clip)
    return (depth * 255.0).astype(np.uint8)


def _parse_saved_frame_step(frame_path: Path) -> int | None:
    """Extract integer step id from saved frame file name."""
    match = re.match(r"^.+_(\d+)_\d{8}_\d{6}_\d{6}\.png$", frame_path.name)
    if match is None:
        return None
    return int(match.group(1))


def _get_ordered_camera_names(camera_names: list[str]) -> list[str]:
    """Return deterministic camera ordering with front/back preference."""
    if not camera_names:
        raise ValueError("No camera directories found for this session")

    sorted_names = sorted(camera_names)
    front_names = [name for name in sorted_names if "front" in name.lower()]
    back_names = [name for name in sorted_names if "back" in name.lower()]
    remainder = [
        name for name in sorted_names if name not in set(front_names) and name not in set(back_names)
    ]
    return front_names + back_names + remainder


def _add_tile_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Overlay tile label text with a filled background strip."""
    labeled = frame.copy()
    frame_h, frame_w = labeled.shape[:2]
    bar_height = max(30, int(frame_h * 0.055))
    font_scale = max(0.6, frame_h / 1300.0)
    thickness = max(1, int(frame_h / 520))
    x_pad = max(10, int(frame_w * 0.01))

    (_text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    text_y = max(text_h + 2, min(bar_height - baseline - 2, (bar_height + text_h) // 2))

    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], bar_height), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (x_pad, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return labeled


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert grayscale/single-channel frame to BGR."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _create_missing_tile(
    tile_w: int,
    tile_h: int,
    camera_name: str,
    modality: str,
) -> np.ndarray:
    """Create a visible placeholder tile for a missing modality frame."""
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    cv2.putText(
        tile,
        "MISSING",
        (max(10, tile_w // 12), max(40, tile_h // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, tile_h / 1100.0),
        (0, 0, 255),
        max(1, tile_h // 500),
        cv2.LINE_AA,
    )
    return _add_tile_label(tile, f"{camera_name}_{modality}")


def write_combined_video_for_session(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    fps: int = 10,
    near_clip: float = 0.1,
    far_clip: float = 2.0,
) -> Path:
    """Build one combined video for a recording session.

    Layout per frame:
      - one row per camera
      - each row is [side_by_side_cameras, depth?, depth_gum?]
      - optional modalities are included only if found in at least one camera
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

    camera_names = [p.name for p in session_dir.iterdir() if p.is_dir()]
    ordered_camera_names = _get_ordered_camera_names(camera_names)

    camera_modalities: dict[tuple[str, str], dict[int, Path]] = {}
    optional_modalities = ("depth", "depth_gum")
    available_optional_modalities: set[str] = set()

    for camera_name in ordered_camera_names:
        for modality in ("rgb", *optional_modalities):
            modality_dir = session_dir / camera_name / modality
            if not modality_dir.exists():
                if modality == "rgb":
                    raise FileNotFoundError(f"Missing required modality directory: {modality_dir}")
                continue
            indexed_paths: dict[int, Path] = {}
            for frame_path in modality_dir.iterdir():
                if not frame_path.is_file() or frame_path.suffix.lower() != ".png":
                    continue
                step = _parse_saved_frame_step(frame_path)
                if step is not None:
                    indexed_paths[step] = frame_path
            if not indexed_paths:
                if modality == "rgb":
                    raise FileNotFoundError(f"No frames found in required modality: {modality_dir}")
                continue
            camera_modalities[(camera_name, modality)] = indexed_paths
            if modality in optional_modalities:
                available_optional_modalities.add(modality)

    ordered_modalities = ["rgb"] + [
        modality for modality in optional_modalities if modality in available_optional_modalities
    ]

    first_camera = ordered_camera_names[0]
    common_steps = set(camera_modalities[(first_camera, "rgb")].keys())
    for camera_name in ordered_camera_names:
        common_steps &= set(camera_modalities[(camera_name, "rgb")].keys())
    ordered_steps = sorted(common_steps)
    if not ordered_steps:
        raise RuntimeError(f"No aligned frame steps found in session {session_dir}")

    first_rgb_path = camera_modalities[(first_camera, "rgb")][ordered_steps[0]]
    first_rgb = cv2.imread(str(first_rgb_path), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError(f"Failed to read RGB frame: {first_rgb_path}")
    row_h, _ = first_rgb.shape[:2]

    modality_widths: dict[str, int] = {"rgb": max(1, first_rgb.shape[1])}
    for modality in ordered_modalities:
        if modality == "rgb":
            continue
        sample_width: int | None = None
        for camera_name in ordered_camera_names:
            modality_steps = camera_modalities.get((camera_name, modality))
            if not modality_steps:
                continue
            sample_path = next(iter(modality_steps.values()))
            sample_frame = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
            if sample_frame is None:
                continue
            if sample_frame.ndim == 3:
                sample_frame = sample_frame[..., 0]
            if sample_frame.shape[0] != row_h:
                raise RuntimeError(
                    f"Expected {modality} height {row_h}, got {sample_frame.shape[0]} at {sample_path}"
                )
            sample_width = int(sample_frame.shape[1])
            break
        modality_widths[modality] = sample_width if sample_width is not None else row_h

    if output_path is None:
        output_path = session_dir / "combined.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (sum(modality_widths[modality] for modality in ordered_modalities), row_h * len(ordered_camera_names)),
        isColor=True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    for step in ordered_steps:
        row_tiles = []
        for camera_name in ordered_camera_names:
            rgb_path = camera_modalities[(camera_name, "rgb")][step]

            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Failed to read RGB frame: {rgb_path}")
            if rgb.shape[0] != row_h:
                raise RuntimeError(f"Expected RGB height {row_h}, got {rgb.shape[0]} at {rgb_path}")
            if rgb.shape[1] != modality_widths["rgb"]:
                raise RuntimeError(
                    f"Expected RGB width {modality_widths['rgb']}, got {rgb.shape[1]} at {rgb_path}"
                )
            rgb = _add_tile_label(rgb, f"{camera_name}_side_by_side_cameras")

            modality_tiles = [rgb]
            for modality in ordered_modalities:
                if modality == "rgb":
                    continue
                modality_steps = camera_modalities.get((camera_name, modality))
                modality_path = None if modality_steps is None else modality_steps.get(step)
                if modality_path is None:
                    modality_tiles.append(
                        _create_missing_tile(
                            tile_w=modality_widths[modality],
                            tile_h=row_h,
                            camera_name=camera_name,
                            modality=modality,
                        )
                    )
                    continue

                modality_raw = cv2.imread(str(modality_path), cv2.IMREAD_UNCHANGED)
                if modality_raw is None:
                    modality_tiles.append(
                        _create_missing_tile(
                            tile_w=modality_widths[modality],
                            tile_h=row_h,
                            camera_name=camera_name,
                            modality=modality,
                        )
                    )
                    continue
                if modality_raw.ndim == 3:
                    modality_raw = modality_raw[..., 0]
                if modality_raw.shape[0] != row_h:
                    raise RuntimeError(
                        f"Expected {modality} height {row_h}, got {modality_raw.shape[0]} at {modality_path}"
                    )
                if modality_raw.shape[1] != modality_widths[modality]:
                    raise RuntimeError(
                        f"Expected {modality} width {modality_widths[modality]}, got {modality_raw.shape[1]} at {modality_path}"
                    )
                modality_vis = _ensure_bgr(modality_raw)
                modality_vis = _add_tile_label(modality_vis, f"{camera_name}_{modality}")
                modality_tiles.append(modality_vis)

            row_tiles.append(cv2.hconcat(modality_tiles))

        frame_grid = row_tiles[0] if len(row_tiles) == 1 else cv2.vconcat(row_tiles)
        writer.write(frame_grid)

    writer.release()
    return output_path


def write_combined_videos_for_sessions(
    image_root_dir: str | Path,
    output_dir: str | Path | None = None,
    session: str | None = None,
    fps: int = 10,
    near_clip: float = 0.1,
    far_clip: float = 2.0,
) -> list[Path]:
    """Create combined mp4(s) under image_root_dir.

    If session is provided, only matched session(s) are processed. The value can be:
      - session folder keyword under image_root_dir (fuzzy match, case-insensitive)
      - exact session folder name under image_root_dir
      - absolute or relative path to a session directory
    """
    image_root_dir = Path(image_root_dir)
    if not image_root_dir.exists():
        raise FileNotFoundError(f"Image root directory does not exist: {image_root_dir}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session_dirs = sorted([p for p in image_root_dir.iterdir() if p.is_dir()])
    else:
        session_path = Path(session)
        if session_path.is_dir():
            session_dirs = [session_path]
        else:
            # Prefer exact folder match under image_root_dir first.
            exact_match_dir = image_root_dir / session
            if exact_match_dir.exists() and exact_match_dir.is_dir():
                session_dirs = [exact_match_dir]
            else:
                # Fuzzy match by keyword in session folder name.
                session_keyword = session.lower()
                matched_session_dirs = sorted(
                    [
                        p
                        for p in image_root_dir.iterdir()
                        if p.is_dir() and session_keyword in p.name.lower()
                    ]
                )
                if not matched_session_dirs:
                    raise FileNotFoundError(
                        f"No session directories match keyword '{session}' under {image_root_dir}"
                    )
                session_dirs = matched_session_dirs

    created_outputs: list[Path] = []
    for session_dir in session_dirs:
        try:
            if output_dir is None:
                session_output = session_dir / "combined.mp4"
            else:
                session_output = output_dir / f"{session_dir.name}_combined.mp4"
            out_path = write_combined_video_for_session(
                session_dir=session_dir,
                output_path=session_output,
                fps=fps,
                near_clip=near_clip,
                far_clip=far_clip,
            )
            created_outputs.append(out_path)
            print(f"[Offline Video Writer] Combined video created: {out_path}")
        except Exception as exc:
            print(f"[Offline Video Writer] Skipped session {session_dir}: {exc}")

    return created_outputs


@dataclass(frozen=True)
class OfflineVideoWriterConfig:
    image_root_dir: str = "image_server_images"
    output_dir: str | None = None
    session: str | None = None
    fps: int = 10
    near_clip: float = 0.1
    far_clip: float = 2.0


if __name__ == "__main__":
    cfg = tyro.cli(OfflineVideoWriterConfig)
    outputs = write_combined_videos_for_sessions(
        image_root_dir=cfg.image_root_dir,
        output_dir=cfg.output_dir,
        session=cfg.session,
        fps=cfg.fps,
        near_clip=cfg.near_clip,
        far_clip=cfg.far_clip,
    )
    print(f"[Offline Video Writer] Created {len(outputs)} videos")
