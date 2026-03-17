#!/usr/bin/env python3
"""Read object/scene metadata embedded in a converted motion clip."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _scalar_str(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _scalar_str(value.item())
        if value.size == 1:
            return _scalar_str(value.reshape(-1)[0])
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _normalize_scale(raw: Any) -> list[float] | None:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        value = float(arr[0])
        return [value, value, value]
    if arr.size >= 3:
        return [float(arr[0]), float(arr[1]), float(arr[2])]
    return None


def _resolve_metadata_path(raw_path: str, *, motion_file: Path) -> str:
    raw_path = str(raw_path).strip()
    if not raw_path:
        return ""

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    repo_root = _repo_root()
    candidates = [
        (motion_file.parent / candidate).resolve(),
        (repo_root / candidate).resolve(),
    ]
    if raw_path.startswith("holosoma/"):
        candidates.append((repo_root / "src" / candidate).resolve())

    for resolved in candidates:
        if resolved.exists():
            return str(resolved)
    return str(candidates[0])


def read_motion_clip_metadata(motion_file: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "has_object_pose": False,
        "object_name": "",
        "object_urdf_path": "",
        "scene_xml_file": "",
        "object_scale": None,
    }

    with np.load(motion_file, allow_pickle=True) as data:
        metadata["has_object_pose"] = "object_pos_w" in data and "object_quat_w" in data
        if "object_name" in data:
            metadata["object_name"] = _scalar_str(data["object_name"])
        if "object_urdf_path" in data:
            metadata["object_urdf_path"] = _resolve_metadata_path(_scalar_str(data["object_urdf_path"]), motion_file=motion_file)
        if "scene_xml_file" in data:
            metadata["scene_xml_file"] = _resolve_metadata_path(_scalar_str(data["scene_xml_file"]), motion_file=motion_file)

        for key in ("object_scale", "object_size", "object_mesh_scale"):
            if key not in data:
                continue
            object_scale = _normalize_scale(data[key])
            if object_scale is not None:
                metadata["object_scale"] = object_scale
                break

    return metadata


def _to_shell(payload: dict[str, Any]) -> str:
    mapping = {
        "SIM2SIM_CLIP_HAS_OBJECT": "1" if payload.get("has_object_pose") else "0",
        "SIM2SIM_CLIP_OBJECT_NAME": str(payload.get("object_name") or ""),
        "SIM2SIM_CLIP_OBJECT_URDF_PATH": str(payload.get("object_urdf_path") or ""),
        "SIM2SIM_CLIP_SCENE_XML_FILE": str(payload.get("scene_xml_file") or ""),
        "SIM2SIM_CLIP_OBJECT_SCALE": (
            ",".join(f"{float(v):g}" for v in payload["object_scale"])
            if isinstance(payload.get("object_scale"), list)
            else ""
        ),
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in mapping.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", required=True, help="Converted motion clip (.npz)")
    parser.add_argument(
        "--format",
        choices=("json", "shell"),
        default="json",
        help="Output format. 'shell' emits KEY='value' assignments suitable for eval.",
    )
    args = parser.parse_args()

    motion_file = Path(args.motion_file).expanduser().resolve()
    payload = read_motion_clip_metadata(motion_file)
    if args.format == "shell":
        print(_to_shell(payload))
        return
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
