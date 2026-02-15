from dataclasses import dataclass
from pathlib import Path

import cv2
import tyro


@dataclass(frozen=True)
class ClipMp4VideosConfig:
    input_file: str
    output_file: str
    start_sec: float
    end_sec: float
    fps: float


def clip_mp4_video(
    input_file: str | Path,
    output_file: str | Path,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> Path:
    """Clip an MP4 file between [start_sec, end_sec) and save to output_file."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if start_sec < 0:
        raise ValueError(f"start_sec must be >= 0, got {start_sec}")
    if end_sec <= start_sec:
        raise ValueError(
            f"end_sec must be greater than start_sec, got start_sec={start_sec}, end_sec={end_sec}"
        )

    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))

    if start_frame >= frame_count:
        cap.release()
        raise ValueError(
            f"start frame {start_frame} is beyond video length ({frame_count} frames)"
        )
    if end_frame <= start_frame:
        cap.release()
        raise ValueError(
            f"Computed end frame must be greater than start frame, got start={start_frame}, end={end_frame}"
        )
    end_frame = min(end_frame, frame_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=True,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_idx = start_frame
    written = 0
    while current_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        current_idx += 1
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        raise RuntimeError(
            f"No frames were written. Check start/end/fps: start_sec={start_sec}, end_sec={end_sec}, fps={fps}"
        )

    return output_path


if __name__ == "__main__":
    cfg = tyro.cli(ClipMp4VideosConfig)
    output = clip_mp4_video(
        input_file=cfg.input_file,
        output_file=cfg.output_file,
        start_sec=cfg.start_sec,
        end_sec=cfg.end_sec,
        fps=cfg.fps,
    )
    print(f"[Clip MP4] Wrote clip to: {output}")
