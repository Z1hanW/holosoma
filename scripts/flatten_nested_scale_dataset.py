#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten per-scale converted motion folders into a single scale-variant bank."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing per-scale tag subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Flat output directory to populate.",
    )
    parser.add_argument(
        "--expected-tags",
        nargs="*",
        default=None,
        help="Optional explicit list of scale tags to require.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output-dir before writing.",
    )
    parser.add_argument(
        "--name-style",
        type=str,
        default="scale_suffix",
        choices=("scale_suffix", "tag_suffix"),
        help="How to name flattened output files.",
    )
    parser.add_argument(
        "--tag-label",
        action="append",
        default=None,
        help="Optional tag-to-label mapping like s070=0p7.",
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


def parse_tag_label_args(raw_items: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw_items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --tag-label entry: {item!r}")
        tag, label = item.split("=", 1)
        tag = tag.strip()
        label = label.strip()
        if not tag or not label:
            raise ValueError(f"Invalid --tag-label entry: {item!r}")
        mapping[tag] = label
    return mapping


def make_flat_stem(src_stem: str, tag: str, *, name_style: str, tag_label_map: dict[str, str]) -> str:
    suffix = "_mj_w_obj"
    if name_style == "scale_suffix":
        scale_label = tag_label_map.get(tag, tag)
        if src_stem.endswith(suffix):
            return f"{src_stem[:-len(suffix)]}_scale_{scale_label}{suffix}"
        return f"{src_stem}_scale_{scale_label}"
    if src_stem.endswith(suffix):
        return f"{src_stem[:-len(suffix)]}__{tag}{suffix}"
    return f"{src_stem}__{tag}"


def load_existing_clip_map(output_dir: Path) -> dict[str, dict[str, str]]:
    map_path = output_dir / "_clip_object_urdf_map.json"
    if not map_path.is_file():
        return {}

    payload = json.loads(map_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        payload = payload["clips"]
    if not isinstance(payload, dict):
        return {}

    existing_file_stems = {path.stem for path in output_dir.glob("*_mj_w_obj.npz")}
    clip_map: dict[str, dict[str, str]] = {}
    for clip_name, entry in payload.items():
        if clip_name not in existing_file_stems:
            continue
        if isinstance(entry, dict):
            clip_map[clip_name] = {
                "object_name": str(entry.get("object_name", "")).strip(),
                "object_urdf_path": str(entry.get("object_urdf_path", "")).strip(),
            }
    return clip_map


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag_label_map = parse_tag_label_args(args.tag_label)

    if args.expected_tags:
        tags = list(args.expected_tags)
    else:
        tags = sorted(p.name for p in input_root.iterdir() if p.is_dir())
    if not tags:
        raise FileNotFoundError(f"No scale-tag directories found under: {input_root}")

    clip_map = load_existing_clip_map(output_dir)
    summary: dict[str, int] = {}
    total_files = 0

    for tag in tags:
        tag_dir = input_root / tag
        if not tag_dir.is_dir():
            raise FileNotFoundError(f"Missing tag directory: {tag_dir}")

        src_files = sorted(tag_dir.glob("*_mj_w_obj.npz"))
        if not src_files:
            raise FileNotFoundError(f"No *_mj_w_obj.npz files found in: {tag_dir}")

        summary[tag] = len(src_files)

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

            out_stem = make_flat_stem(src.stem, tag, name_style=args.name_style, tag_label_map=tag_label_map)
            out_path = output_dir / f"{out_stem}.npz"

            out_payload = dict(payload)
            out_payload["object_name"] = np.array(object_name)
            out_payload["object_urdf_path"] = np.array(str(urdf_path))
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
        "counts_by_tag": summary,
    }
    map_path = output_dir / "_clip_object_urdf_map.json"
    map_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_dir": str(output_dir),
                "tags": tags,
                "counts_by_tag": summary,
                "files": total_files,
                "map_file": str(map_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
