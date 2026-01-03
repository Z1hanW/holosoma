#!/usr/bin/env python3
"""
Generate a fixed-size floor plane centered at the origin.

Top surface is at z=0 and spans from (-extent, -extent) to (+extent, +extent).
Thickness extends downward.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def write_floor_box(
    output_path: Path,
    *,
    extent: float | None,
    thickness: float,
    min_x: float | None,
    max_x: float | None,
    min_y: float | None,
    max_y: float | None,
) -> None:
    if thickness <= 0.0:
        raise ValueError(f"Thickness must be positive; got {thickness}")

    if any(v is not None for v in (min_x, max_x, min_y, max_y)):
        if None in (min_x, max_x, min_y, max_y):
            raise ValueError("Must provide all of --min-x, --max-x, --min-y, --max-y.")
        assert min_x is not None
        assert max_x is not None
        assert min_y is not None
        assert max_y is not None
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("Max must be greater than min for both axes.")
    else:
        if extent is None:
            raise ValueError("Extent must be provided if explicit bounds are not set.")
        if extent <= 0.0:
            raise ValueError(f"Extent must be positive; got {extent}")
        min_x, max_x = -extent, extent
        min_y, max_y = -extent, extent
    z_top = 0.0
    z_bottom = -thickness

    verts = [
        (min_x, min_y, z_top),
        (max_x, min_y, z_top),
        (max_x, max_y, z_top),
        (min_x, max_y, z_top),
        (min_x, min_y, z_bottom),
        (max_x, min_y, z_bottom),
        (max_x, max_y, z_bottom),
        (min_x, max_y, z_bottom),
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
        handle.write("# Generated fixed floor box\n")
        handle.write(f"# extent={extent:.6f} thickness={thickness:.6f}\n")
        for vx, vy, vz in verts:
            handle.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for a, b, c in faces:
            handle.write(f"f {a} {b} {c}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-size floor box centered at the origin."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output OBJ path.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=5.0,
        help="Half-size in meters (plane spans -extent..+extent).",
    )
    parser.add_argument("--min-x", type=float, default=None, help="Minimum X bound.")
    parser.add_argument("--max-x", type=float, default=None, help="Maximum X bound.")
    parser.add_argument("--min-y", type=float, default=None, help="Minimum Y bound.")
    parser.add_argument("--max-y", type=float, default=None, help="Maximum Y bound.")
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.05,
        help="Thickness in meters (extends downward from z=0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_floor_box(
        args.output,
        extent=args.extent,
        thickness=args.thickness,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
    )
    print(f"Wrote fixed floor OBJ: {args.output}")
    if any(v is not None for v in (args.min_x, args.max_x, args.min_y, args.max_y)):
        print(
            "Plane: x in [{}, {}], y in [{}, {}] at z=0, thickness={}".format(
                args.min_x, args.max_x, args.min_y, args.max_y, args.thickness
            )
        )
    else:
        print(f"Plane: x/y in [-{args.extent}, {args.extent}] at z=0, thickness={args.thickness}")


if __name__ == "__main__":
    main()
