#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create XY-scale-augmented datasets from filtered converted motion directories."
    )
    parser.add_argument(
        "--omomo-input-dir",
        type=Path,
        default=Path("src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"),
    )
    parser.add_argument(
        "--behave-input-dir",
        type=Path,
        default=Path("src/holosoma_retargeting/converted_res/behave_sq_carry"),
    )
    parser.add_argument(
        "--omomo-output-dir",
        type=Path,
        default=Path("src/holosoma_retargeting/converted_res/object_interaction/omomo_carry_xy_0p5_1p5"),
    )
    parser.add_argument(
        "--behave-output-dir",
        type=Path,
        default=Path("src/holosoma_retargeting/converted_res/behave_sq_carry_xy_0p5_1p5"),
    )
    parser.add_argument(
        "--omomo-base-urdf",
        type=Path,
        default=Path("src/holosoma_retargeting/models/largebox/largebox.urdf"),
    )
    parser.add_argument("--omomo-object-name", type=str, default="largebox")
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--scale-step", type=float, default=0.1)
    return parser.parse_args()


def scale_values(scale_min: float, scale_max: float, scale_step: float) -> list[float]:
    vals: list[float] = []
    cur = scale_min
    while cur <= scale_max + 1e-9:
        vals.append(round(cur, 6))
        cur += scale_step
    return vals


def normalize_scale(raw: np.ndarray | list[float] | tuple[float, ...] | float) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size != 3:
        raise ValueError(f"Expected scalar or 3 elements, got shape={np.asarray(raw).shape}")
    return arr


def to_scalar_str(value: np.ndarray | str | object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0].item())


def parse_urdf_mesh_scale(urdf_path: Path) -> np.ndarray:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mesh = root.find(".//mesh")
    if mesh is None:
        return np.ones(3, dtype=np.float64)
    raw = mesh.get("scale", "").strip()
    if not raw:
        return np.ones(3, dtype=np.float64)
    parts = [float(x) for x in raw.split()]
    if len(parts) == 1:
        parts = [parts[0], parts[0], parts[0]]
    if len(parts) != 3:
        raise ValueError(f"Invalid mesh scale in {urdf_path}: '{raw}'")
    return np.array(parts, dtype=np.float64)


def resolve_mesh_filename(base_urdf: Path, mesh_filename: str) -> str:
    mesh_filename = mesh_filename.strip()
    if not mesh_filename:
        return mesh_filename

    if "://" in mesh_filename:
        return mesh_filename

    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_filename

    candidate = (base_urdf.parent / mesh_path)
    if candidate.exists():
        return str(candidate.absolute())

    return mesh_filename


def write_scaled_urdf(base_urdf: Path, out_urdf: Path, scale_xyz: np.ndarray) -> None:
    tree = ET.parse(base_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        raw = (mesh.get("scale") or "").strip()
        if raw:
            base = normalize_scale([float(x) for x in raw.split()])
        else:
            base = np.ones(3, dtype=np.float64)
        scaled = base * scale_xyz
        mesh.set("scale", f"{scaled[0]:.8g} {scaled[1]:.8g} {scaled[2]:.8g}")
        mesh_filename = resolve_mesh_filename(base_urdf, mesh.get("filename", ""))
        if mesh_filename:
            mesh.set("filename", mesh_filename)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_urdf)


def urdf_cache_key(base_urdf: Path, sx: float, sy: float, sz: float) -> str:
    return f"{base_urdf.resolve()}::{sx:.6f}::{sy:.6f}::{sz:.6f}"


def make_scale_tag(xy_scale: float) -> str:
    return f"xy{int(round(xy_scale * 100)):03d}"


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    scales_xy: list[float],
    *,
    default_object_name: str | None = None,
    default_object_urdf: Path | None = None,
) -> dict[str, object]:
    input_files = sorted(input_dir.glob("*_mj_w_obj.npz"))
    if not input_files:
        raise FileNotFoundError(f"No *_mj_w_obj.npz files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_out_root = output_dir / "_scaled_urdf"
    urdf_out_root.mkdir(parents=True, exist_ok=True)
    urdf_cache: dict[str, Path] = {}

    clip_map: dict[str, dict[str, str]] = {}
    produced = 0

    for src in input_files:
        with np.load(src, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}

        object_name = to_scalar_str(payload.get("object_name", np.array(default_object_name or ""))).strip()
        if not object_name:
            object_name = default_object_name or "object"

        raw_urdf = to_scalar_str(payload.get("object_urdf_path", np.array(""))).strip()
        if raw_urdf:
            base_urdf = Path(raw_urdf)
        elif default_object_urdf is not None:
            base_urdf = default_object_urdf
        else:
            raise ValueError(f"Missing object_urdf_path for {src}")
        if not base_urdf.exists():
            raise FileNotFoundError(f"Missing base URDF: {base_urdf} (from {src})")

        base_mesh_scale = (
            normalize_scale(payload["object_mesh_scale"]) if "object_mesh_scale" in payload else parse_urdf_mesh_scale(base_urdf)
        )

        stem = src.stem
        for s in scales_xy:
            scale_xyz = np.array([s, s, 1.0], dtype=np.float64)
            scale_tag = make_scale_tag(s)
            out_stem = f"{stem}__{scale_tag}"
            out_path = output_dir / f"{out_stem}.npz"

            key = urdf_cache_key(base_urdf, float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2]))
            if key not in urdf_cache:
                urdf_name = f"{base_urdf.stem}__{scale_tag}.urdf"
                out_urdf = urdf_out_root / object_name / urdf_name
                write_scaled_urdf(base_urdf, out_urdf, scale_xyz)
                urdf_cache[key] = out_urdf.absolute()
            scaled_urdf_path = urdf_cache[key]

            out_payload = dict(payload)
            new_mesh_scale = (base_mesh_scale * scale_xyz).astype(np.float32)
            out_payload["object_name"] = np.array(object_name)
            out_payload["object_urdf_path"] = np.array(str(scaled_urdf_path))
            out_payload["object_mesh_scale"] = new_mesh_scale
            out_payload["object_scale"] = scale_xyz.astype(np.float32)
            out_payload["object_size"] = scale_xyz.astype(np.float32)
            if "scene_xml_file" not in out_payload:
                out_payload["scene_xml_file"] = np.array("")

            np.savez(out_path, **out_payload)
            clip_map[out_stem] = {
                "object_name": object_name,
                "object_urdf_path": str(scaled_urdf_path),
            }
            produced += 1

    map_payload = {
        "clips": clip_map,
        "notes": "XY-scale augmentation generated from filtered converted dataset",
        "source_converted_dir": str(input_dir.resolve()),
        "scales_xy": scales_xy,
        "z_scale": 1.0,
    }
    map_path = output_dir / "_clip_object_urdf_map.json"
    map_path.write_text(json.dumps(map_payload, indent=2, ensure_ascii=False))

    return {
        "input_dir": str(input_dir.absolute()),
        "output_dir": str(output_dir.absolute()),
        "source_files": len(input_files),
        "produced_files": produced,
        "map_file": str(map_path.absolute()),
    }


def main() -> None:
    args = parse_args()
    scales_xy = scale_values(args.scale_min, args.scale_max, args.scale_step)
    if not scales_xy:
        raise ValueError("No scale values generated.")

    omomo_stats = augment_dataset(
        input_dir=args.omomo_input_dir,
        output_dir=args.omomo_output_dir,
        scales_xy=scales_xy,
        default_object_name=args.omomo_object_name,
        default_object_urdf=args.omomo_base_urdf,
    )
    behave_stats = augment_dataset(
        input_dir=args.behave_input_dir,
        output_dir=args.behave_output_dir,
        scales_xy=scales_xy,
    )

    print("[augment_xy_scale] OMOMO:", json.dumps(omomo_stats, indent=2))
    print("[augment_xy_scale] BEHAVE:", json.dumps(behave_stats, indent=2))


if __name__ == "__main__":
    main()
