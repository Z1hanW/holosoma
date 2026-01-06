#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Config:
    """Pack converted_res .npz motions into a single HDF5 motion bank."""

    input_dir: str
    output_h5: str
    pattern: str = "*.npz"
    recursive: bool = False
    max_clips: int | None = None
    chunk_frames: int = 1024
    compression: str | None = "gzip"
    compression_level: int = 4


REQUIRED_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
    "fps",
)

OPTIONAL_OBJECT_KEYS = (
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
)


def _decode_strings(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for item in values:
        if isinstance(item, (bytes, np.bytes_)):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _list_npz_files(cfg: Config) -> list[Path]:
    base = Path(cfg.input_dir).expanduser().resolve()
    if cfg.recursive:
        files = sorted(base.rglob(cfg.pattern))
    else:
        files = sorted(base.glob(cfg.pattern))
    if cfg.max_clips is not None:
        files = files[: cfg.max_clips]
    return files


def _resize_and_append(ds, data: np.ndarray) -> int:
    start = ds.shape[0]
    end = start + data.shape[0]
    ds.resize((end,) + ds.shape[1:])
    ds[start:end] = data
    return start


def main(cfg: Config) -> None:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("h5py is required to write HDF5 motion banks.") from exc

    files = _list_npz_files(cfg)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {cfg.input_dir} with pattern {cfg.pattern}")

    output_path = Path(cfg.output_h5).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_ids: list[str] = []
    clip_offsets: list[int] = []
    clip_lengths: list[int] = []
    clip_fps: list[float] = []
    clip_sources: list[str] = []

    with h5py.File(output_path, "w") as h5f:
        meta = h5f.create_group("meta")
        data_group = h5f.create_group("data")

        has_object = None
        joint_names: list[str] = []
        body_names: list[str] = []
        fps_ref: float | None = None

        datasets = {}

        for file_path in files:
            with np.load(file_path, allow_pickle=True) as npz:
                missing = [key for key in REQUIRED_KEYS if key not in npz]
                if missing:
                    raise KeyError(f"Missing keys in {file_path}: {missing}")

                if has_object is None:
                    has_object = all(key in npz for key in OPTIONAL_OBJECT_KEYS)
                elif has_object != all(key in npz for key in OPTIONAL_OBJECT_KEYS):
                    raise ValueError("Object fields are inconsistent across clips.")

                if not joint_names:
                    joint_names = _decode_strings(np.asarray(npz["joint_names"]))
                if not body_names:
                    body_names = _decode_strings(np.asarray(npz["body_names"]))

                joint_names_clip = _decode_strings(np.asarray(npz["joint_names"]))
                body_names_clip = _decode_strings(np.asarray(npz["body_names"]))
                if joint_names_clip != joint_names:
                    raise ValueError(f"Joint names mismatch in {file_path}")
                if body_names_clip != body_names:
                    raise ValueError(f"Body names mismatch in {file_path}")

                fps_arr = np.array(npz["fps"]).reshape(-1)
                fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0
                if fps_ref is None:
                    fps_ref = fps
                elif abs(fps_ref - fps) > 1e-6:
                    raise ValueError(f"FPS mismatch in {file_path}: {fps} != {fps_ref}")

                if not datasets:
                    compression = cfg.compression
                    compression_opts = cfg.compression_level if compression else None
                    string_dtype = h5py.string_dtype(encoding="utf-8")

                    meta.create_dataset("joint_names", data=np.array(joint_names, dtype=object), dtype=string_dtype)
                    meta.create_dataset("body_names", data=np.array(body_names, dtype=object), dtype=string_dtype)
                    meta.create_dataset("fps", data=np.array([fps_ref], dtype=np.float32))
                    meta.attrs["schema_version"] = 1
                    meta.attrs["has_object"] = bool(has_object)

                    def _create_ds(key: str, arr: np.ndarray):
                        shape = (0,) + arr.shape[1:]
                        maxshape = (None,) + arr.shape[1:]
                        chunks = (min(cfg.chunk_frames, max(1, arr.shape[0])),) + arr.shape[1:]
                        datasets[key] = data_group.create_dataset(
                            key,
                            shape=shape,
                            maxshape=maxshape,
                            dtype=arr.dtype,
                            chunks=chunks,
                            compression=compression,
                            compression_opts=compression_opts,
                            shuffle=bool(compression),
                        )

                    for key in (
                        "joint_pos",
                        "joint_vel",
                        "body_pos_w",
                        "body_quat_w",
                        "body_lin_vel_w",
                        "body_ang_vel_w",
                    ):
                        _create_ds(key, np.asarray(npz[key]))

                    if has_object:
                        for key in OPTIONAL_OBJECT_KEYS:
                            _create_ds(key, np.asarray(npz[key]))

                clip_id = file_path.stem
                clip_ids.append(clip_id)
                clip_sources.append(str(file_path))

                joint_pos = np.asarray(npz["joint_pos"])
                clip_offsets.append(_resize_and_append(datasets["joint_pos"], joint_pos))
                clip_lengths.append(joint_pos.shape[0])
                clip_fps.append(fps)

                _resize_and_append(datasets["joint_vel"], np.asarray(npz["joint_vel"]))
                _resize_and_append(datasets["body_pos_w"], np.asarray(npz["body_pos_w"]))
                _resize_and_append(datasets["body_quat_w"], np.asarray(npz["body_quat_w"]))
                _resize_and_append(datasets["body_lin_vel_w"], np.asarray(npz["body_lin_vel_w"]))
                _resize_and_append(datasets["body_ang_vel_w"], np.asarray(npz["body_ang_vel_w"]))

                if has_object:
                    _resize_and_append(datasets["object_pos_w"], np.asarray(npz["object_pos_w"]))
                    _resize_and_append(datasets["object_quat_w"], np.asarray(npz["object_quat_w"]))
                    _resize_and_append(datasets["object_lin_vel_w"], np.asarray(npz["object_lin_vel_w"]))
                    _resize_and_append(datasets["object_ang_vel_w"], np.asarray(npz["object_ang_vel_w"]))

        clips = h5f.create_group("clips")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        clips.create_dataset("clip_ids", data=np.array(clip_ids, dtype=object), dtype=string_dtype)
        clips.create_dataset("offsets", data=np.asarray(clip_offsets, dtype=np.int64))
        clips.create_dataset("lengths", data=np.asarray(clip_lengths, dtype=np.int64))
        clips.create_dataset("clip_fps", data=np.asarray(clip_fps, dtype=np.float32))
        clips.create_dataset("source_paths", data=np.array(clip_sources, dtype=object), dtype=string_dtype)

    print(f"[convert_npz_to_h5] Wrote {len(clip_ids)} clips to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
