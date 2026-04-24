#!/usr/bin/env python3
"""Extract per-tile OBJ meshes from a fused terrain OBJ plus metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = mesh.dump(concatenate=False)
        if not meshes:
            raise ValueError(f"Loaded terrain scene has no geometries: {path}")
        mesh = max(meshes, key=lambda item: len(getattr(item, "faces", [])) or len(item.vertices))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded terrain is not a trimesh: {type(mesh)}")
    return mesh


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in metadata: {path}")
    return payload


def _canonicalize_offsets(tile_offsets: np.ndarray, tile_stride: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tile_offsets.ndim != 2 or tile_offsets.shape[1] < 2:
        raise ValueError("tile_offsets must be an Nx2/Nx3 array.")
    if tile_offsets.shape[1] == 2:
        tile_offsets = np.concatenate([tile_offsets, np.zeros((tile_offsets.shape[0], 1), dtype=np.float64)], axis=1)
    elif tile_offsets.shape[1] > 3:
        tile_offsets = tile_offsets[:, :3]

    if tile_stride.ndim != 1 or tile_stride.size < 2:
        raise ValueError("tile_stride must provide at least X/Y entries.")
    if tile_stride.size == 2:
        tile_stride = np.array([tile_stride[0], tile_stride[1], 0.0], dtype=np.float64)
    elif tile_stride.size > 3:
        tile_stride = tile_stride[:3]
    return tile_offsets, tile_stride


def _tile_offset_for_col(
    *,
    tile_offsets: np.ndarray,
    tile_rows: int,
    tile_cols: int,
    tile_stride: np.ndarray,
    row: int,
    col: int,
) -> np.ndarray:
    if tile_offsets.shape[0] == tile_cols:
        return np.asarray(tile_offsets[col], dtype=np.float64) + np.array([0.0, row * tile_stride[1], 0.0], dtype=np.float64)
    expected = tile_rows * tile_cols
    if tile_offsets.shape[0] != expected:
        raise ValueError(
            f"tile_offsets length mismatch: got {tile_offsets.shape[0]}, expected cols={tile_cols} or rows*cols={expected}."
        )
    return np.asarray(tile_offsets[row * tile_cols + col], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-obj", required=True, help="Combined OBJ path.")
    parser.add_argument("--in-meta", required=True, help="Metadata JSON path for the combined OBJ.")
    parser.add_argument("--out-dir", required=True, help="Output directory for per-tile OBJ files.")
    parser.add_argument("--row", type=int, default=0, help="Which row to extract when the fused OBJ repeats rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing OBJ files.")
    args = parser.parse_args()

    in_obj = Path(args.in_obj).expanduser().resolve()
    in_meta = Path(args.in_meta).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(in_meta)
    tile_names = [str(name) for name in metadata.get("tile_names", [])]
    if not tile_names:
        raise ValueError("Metadata must include non-empty tile_names.")

    tile_offsets = np.asarray(metadata.get("tile_offsets", []), dtype=np.float64)
    tile_stride = np.asarray(metadata.get("tile_stride", []), dtype=np.float64).reshape(-1)
    tile_rows = max(1, int(metadata.get("tile_rows", 1) or 1))
    tile_cols = max(1, int(metadata.get("tile_cols", len(tile_names)) or len(tile_names)))
    if len(tile_names) != tile_cols:
        raise ValueError(f"tile_names length {len(tile_names)} must match tile_cols {tile_cols}.")
    if not (0 <= int(args.row) < tile_rows):
        raise ValueError(f"--row must be in [0, {tile_rows - 1}], got {args.row}.")
    tile_offsets, tile_stride = _canonicalize_offsets(tile_offsets, tile_stride)

    print(f"[INFO] Loading fused OBJ: {in_obj}", flush=True)
    mesh = _load_mesh(in_obj)
    print(f"[INFO] Loaded fused OBJ with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.", flush=True)
    print("[INFO] Computing face centroids.", flush=True)
    face_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    if face_centroids.ndim != 2 or face_centroids.shape[1] < 2:
        raise ValueError("Failed to compute terrain face centroids.")
    print(f"[INFO] Extracting {len(tile_names)} tile OBJ(s) from row {int(args.row)}.", flush=True)

    pad = max(1.0e-4, float(max(abs(tile_stride[0]), abs(tile_stride[1]))) * 1.0e-4)
    written = 0
    for col, clip_name in enumerate(tile_names):
        tile_offset = _tile_offset_for_col(
            tile_offsets=tile_offsets,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            tile_stride=tile_stride,
            row=int(args.row),
            col=col,
        )
        x_min = float(tile_offset[0])
        y_min = float(tile_offset[1])
        x_max = x_min + float(tile_stride[0])
        y_max = y_min + float(tile_stride[1])
        mask = (
            (face_centroids[:, 0] >= (x_min - pad))
            & (face_centroids[:, 0] <= (x_max + pad))
            & (face_centroids[:, 1] >= (y_min - pad))
            & (face_centroids[:, 1] <= (y_max + pad))
        )
        face_indices = np.flatnonzero(mask)
        if face_indices.size == 0:
            raise ValueError(f"No terrain faces found for clip '{clip_name}' at row={args.row}, col={col}.")

        tile_mesh = mesh.submesh([face_indices], append=True, repair=False)
        if isinstance(tile_mesh, trimesh.Scene):
            meshes = tile_mesh.dump(concatenate=False)
            if not meshes:
                raise ValueError(f"Submesh extraction returned no geometries for clip '{clip_name}'.")
            tile_mesh = max(meshes, key=lambda item: len(getattr(item, "faces", [])) or len(item.vertices))
        if not isinstance(tile_mesh, trimesh.Trimesh) or len(tile_mesh.faces) == 0:
            raise ValueError(f"Extracted tile mesh is empty for clip '{clip_name}'.")

        out_path = out_dir / f"{clip_name}.obj"
        if out_path.exists() and not args.overwrite:
            continue
        tile_mesh = tile_mesh.copy()
        tile_mesh.apply_translation((-tile_offset).tolist())
        tile_mesh.export(str(out_path))
        written += 1
        print(f"[INFO] Wrote {out_path}")

    summary = {
        "source_obj": str(in_obj),
        "source_metadata": str(in_meta),
        "row": int(args.row),
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tile_count": len(tile_names),
        "tile_names": tile_names,
    }
    summary_path = out_dir / "tiles.summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[INFO] Wrote {summary_path}")
    print(f"[INFO] Extracted {written} tile OBJ(s) to {out_dir}")


if __name__ == "__main__":
    main()
