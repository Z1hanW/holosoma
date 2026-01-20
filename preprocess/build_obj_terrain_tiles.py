#!/usr/bin/env python3
"""Prebuild a combined OBJ terrain and metadata for motion pairing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)  # type: ignore[assignment]
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a valid Trimesh: {type(mesh)}")
    return mesh


def _resolve_obj_paths(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_dir():
        matches = list(path.glob("*.obj")) + list(path.glob("*.OBJ"))
        return sorted(matches)
    return [path] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-dir", required=True, help="Directory containing per-tile OBJ meshes.")
    parser.add_argument("--out-obj", required=True, help="Output combined OBJ path.")
    parser.add_argument("--out-meta", required=True, help="Output metadata JSON path.")
    parser.add_argument("--num-rows", type=int, default=1, help="Number of rows to repeat the column set.")
    parser.add_argument("--gap", type=float, default=1e-4, help="Gap between tiles to avoid overlap.")
    args = parser.parse_args()

    obj_paths = _resolve_obj_paths(args.obj_dir)
    if not obj_paths:
        raise FileNotFoundError(f"No OBJ files found in: {args.obj_dir}")

    tile_names = [path.stem for path in obj_paths]
    if len(set(tile_names)) != len(tile_names):
        raise ValueError("OBJ stems must be unique for pairing.")

    meshes = []
    spans = []
    tile_max_z = []
    for path in obj_paths:
        mesh = _load_mesh(path)
        meshes.append(mesh)
        spans.append(mesh.bounds[1] - mesh.bounds[0])
        tile_max_z.append(float(mesh.vertices[:, 2].max() if mesh.vertices.size else 0.0))

    spans = np.vstack(spans)
    stride = spans.max(axis=0) + args.gap

    tiles = []
    tile_offsets = []
    for col, mesh in enumerate(meshes):
        col_offset = np.array([col * stride[0], 0.0, 0.0], dtype=np.float64)
        tile_offsets.append(col_offset)
        for row in range(max(1, args.num_rows)):
            offset = col_offset + np.array([0.0, row * stride[1], 0.0], dtype=np.float64)
            tile = mesh.copy()
            tile.apply_translation(offset)
            tiles.append(tile)

    combined = trimesh.util.concatenate(tiles)
    out_obj = Path(args.out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out_obj))

    meta = {
        "tile_names": tile_names,
        "tile_offsets": np.asarray(tile_offsets, dtype=np.float32).tolist(),
        "tile_stride": stride.astype(np.float32).tolist(),
        "tile_rows": int(max(1, args.num_rows)),
        "tile_cols": int(len(tile_names)),
        "tile_max_z": np.asarray(tile_max_z, dtype=np.float32).tolist(),
    }

    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"[INFO] Wrote combined mesh: {out_obj}")
    print(f"[INFO] Wrote metadata: {out_meta}")


if __name__ == "__main__":
    main()
