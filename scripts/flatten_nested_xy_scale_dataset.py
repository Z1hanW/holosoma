#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten a nested XY-scale converted bank into a single training directory."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory with per-scale folders such as xy050/xy060/... containing *_mj_w_obj.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Flat output directory to populate with *_mj_w_obj__xyNNN.npz files.",
    )
    parser.add_argument(
        "--expected-tags",
        nargs="*",
        default=None,
        help="Optional explicit list of scale tags to require, e.g. xy050 xy060 ... xy150.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output-dir before writing the flattened bank.",
    )
    return parser.parse_args()


def scalar_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    if arr.shape == ():
        item = arr.item()
    else:
        item = arr.reshape(-1)[0]
        if hasattr(item, "item"):
            item = item.item()
    return str(item).strip()


def scale_from_tag(tag: str) -> np.ndarray:
    if len(tag) != 5 or not tag.startswith("xy") or not tag[2:].isdigit():
        raise ValueError(f"Invalid scale tag: {tag}")
    scale = int(tag[2:]) / 100.0
    return np.array([scale, scale, 1.0], dtype=np.float32)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.expected_tags:
        tags = list(args.expected_tags)
    else:
        tags = sorted(p.name for p in input_root.iterdir() if p.is_dir() and p.name.startswith("xy"))
    if not tags:
        raise FileNotFoundError(f"No scale-tag directories found under: {input_root}")

    clip_map: dict[str, dict[str, str]] = {}
    total_files = 0

    for tag in tags:
        scale_xyz = scale_from_tag(tag)
        tag_dir = input_root / tag
        if not tag_dir.is_dir():
            raise FileNotFoundError(f"Missing tag directory: {tag_dir}")

        src_files = sorted(tag_dir.glob("*_mj_w_obj.npz"))
        if not src_files:
            raise FileNotFoundError(f"No *_mj_w_obj.npz files found in: {tag_dir}")

        for src in src_files:
            with np.load(src, allow_pickle=True) as data:
                payload = {key: data[key] for key in data.files}

            object_name = scalar_str(payload.get("object_name", np.array("")))
            object_urdf_path = scalar_str(payload.get("object_urdf_path", np.array("")))
            if not object_name:
                raise ValueError(f"Missing object_name in {src}")
            if not object_urdf_path:
                raise ValueError(f"Missing object_urdf_path in {src}")

            urdf_path = Path(object_urdf_path)
            if not urdf_path.is_absolute():
                urdf_path = (tag_dir / urdf_path).resolve()
            if not urdf_path.exists():
                raise FileNotFoundError(f"Missing object URDF for {src}: {urdf_path}")

            out_stem = f"{src.stem}__{tag}"
            out_path = output_dir / f"{out_stem}.npz"

            out_payload = dict(payload)
            out_payload["object_name"] = np.array(object_name)
            out_payload["object_urdf_path"] = np.array(str(urdf_path))
            out_payload["object_scale"] = scale_xyz
            np.savez(out_path, **out_payload)

            clip_map[out_stem] = {
                "object_name": object_name,
                "object_urdf_path": str(urdf_path),
            }
            total_files += 1

    payload = {
        "clips": clip_map,
        "notes": "Flattened from real per-scale converted outputs",
        "source_converted_root": str(input_root),
        "tags": tags,
    }
    map_path = output_dir / "_clip_object_urdf_map.json"
    map_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_dir": str(output_dir),
                "tags": tags,
                "files": total_files,
                "map_file": str(map_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
