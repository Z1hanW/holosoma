#!/usr/bin/env python3
"""
Generate a thin ground box from an OBJ bounding box.

The top surface is at z = 0 and spans from (0, 0) to (x, y), where
x/y are the bbox extents of the input mesh. Thickness extends downward.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_vertices(obj_path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            vertices.append((x, y, z))
    if not vertices:
        raise ValueError(f"No vertices found in {obj_path}")
    return vertices


def compute_bbox(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float, float, float, float]:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def write_ground_box(
    output_path: Path,
    *,
    size_x: float,
    size_y: float,
    thickness: float,
    bbox: tuple[float, float, float, float, float, float],
    source_obj: Path,
) -> None:
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"Invalid extents from bbox: size_x={size_x}, size_y={size_y}")
    if thickness <= 0.0:
        raise ValueError(f"Thickness must be positive; got {thickness}")

    z_top = 0.0
    z_bottom = -thickness

    verts = [
        (0.0, 0.0, z_top),
        (size_x, 0.0, z_top),
        (size_x, size_y, z_top),
        (0.0, size_y, z_top),
        (0.0, 0.0, z_bottom),
        (size_x, 0.0, z_bottom),
        (size_x, size_y, z_bottom),
        (0.0, size_y, z_bottom),
    ]

    # Triangulated faces. Top face normal points +Z.
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 7, 6),
        (5, 8, 7),
        (2, 7, 3),
        (2, 6, 7),
        (1, 4, 8),
        (1, 8, 5),
        (3, 8, 4),
        (3, 7, 8),
        (1, 6, 2),
        (1, 5, 6),
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated ground box\n")
        handle.write(f"# source_obj={source_obj}\n")
        handle.write(
            "# bbox_min=({:.6f}, {:.6f}, {:.6f}) bbox_max=({:.6f}, {:.6f}, {:.6f})\n".format(
                bbox[0], bbox[2], bbox[4], bbox[1], bbox[3], bbox[5]
            )
        )
        handle.write(f"# size_x={size_x:.6f} size_y={size_y:.6f} thickness={thickness:.6f}\n")
        for vx, vy, vz in verts:
            handle.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for a, b, c in faces:
            handle.write(f"f {a} {b} {c}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a thin ground box at z=0 using the XY bbox extents of an OBJ."
        )
    )
    parser.add_argument("input_obj", type=Path, help="Path to input OBJ file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output OBJ path. Defaults to <input>_ground.obj",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.05,
        help="Thickness in meters (extends downward from z=0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_obj: Path = args.input_obj
    if not input_obj.exists():
        raise FileNotFoundError(f"Input OBJ not found: {input_obj}")

    vertices = load_vertices(input_obj)
    min_x, max_x, min_y, max_y, min_z, max_z = compute_bbox(vertices)
    # Use bbox span so negative minima expand the plane size (plane spans 0..(max-min)).
    size_x = max_x - min_x
    size_y = max_y - min_y

    output_path = args.output
    if output_path is None:
        output_path = input_obj.with_name(f"{input_obj.stem}_ground.obj")

    write_ground_box(
        output_path,
        size_x=size_x,
        size_y=size_y,
        thickness=args.thickness,
        bbox=(min_x, max_x, min_y, max_y, min_z, max_z),
        source_obj=input_obj,
    )

    print(f"Wrote ground box OBJ: {output_path}")
    print(f"BBox: x=[{min_x:.6f}, {max_x:.6f}] y=[{min_y:.6f}, {max_y:.6f}] z=[{min_z:.6f}, {max_z:.6f}]")
    print(f"Plane: (0,0) to ({size_x:.6f}, {size_y:.6f}) at z=0, thickness={args.thickness:.6f}")


if __name__ == "__main__":
    main()
