#!/usr/bin/env python3
"""Convert a gsplat PLY (3D Gaussians) into a heightfield OBJ for simulator loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PLY_TYPE_MAP: dict[str, np.dtype] = {
    "char": np.dtype("<i1"),
    "uchar": np.dtype("<u1"),
    "short": np.dtype("<i2"),
    "ushort": np.dtype("<u2"),
    "int": np.dtype("<i4"),
    "uint": np.dtype("<u4"),
    "float": np.dtype("<f4"),
    "double": np.dtype("<f8"),
}


def _read_ply_header(handle) -> tuple[str, int, list[tuple[str, str]], int]:
    fmt = ""
    vertex_count = -1
    vertex_props: list[tuple[str, str]] = []
    current_element = None
    header_bytes = 0

    while True:
        line = handle.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header.")
        header_bytes += len(line)
        text = line.decode("ascii", errors="ignore").strip()
        if text == "end_header":
            break
        if not text:
            continue
        parts = text.split()
        if parts[0] == "format":
            if len(parts) < 2:
                raise ValueError(f"Malformed PLY format line: {text}")
            fmt = parts[1]
        elif parts[0] == "element":
            if len(parts) < 3:
                raise ValueError(f"Malformed PLY element line: {text}")
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) < 3:
                raise ValueError(f"Malformed PLY property line: {text}")
            if parts[1] == "list":
                # Skip list properties for vertex data.
                continue
            prop_type = parts[1]
            prop_name = parts[2]
            vertex_props.append((prop_name, prop_type))

    if not fmt:
        raise ValueError("PLY header missing format line.")
    if vertex_count < 0:
        raise ValueError("PLY header missing vertex count.")
    if not vertex_props:
        raise ValueError("PLY header has no vertex properties.")

    return fmt, vertex_count, vertex_props, header_bytes


def _load_ply_vertices(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        fmt, vertex_count, vertex_props, header_bytes = _read_ply_header(handle)

        names = [name for name, _ in vertex_props]
        types = [ptype for _, ptype in vertex_props]

        if fmt == "ascii":
            data = np.zeros((vertex_count, len(names)), dtype=np.float32)
            for idx in range(vertex_count):
                line = handle.readline()
                if not line:
                    raise ValueError("Unexpected EOF while reading ASCII PLY vertices.")
                parts = line.strip().split()
                if len(parts) < len(names):
                    raise ValueError("PLY vertex line has fewer columns than header.")
                for col in range(len(names)):
                    data[idx, col] = float(parts[col])
            return {name: data[:, i] for i, name in enumerate(names)}

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt}")

        dtypes = []
        for name, ptype in zip(names, types):
            if ptype not in PLY_TYPE_MAP:
                raise ValueError(f"Unsupported PLY property type: {ptype}")
            dtypes.append((name, PLY_TYPE_MAP[ptype]))
        dtype = np.dtype(dtypes)

        handle.seek(header_bytes)
        raw = handle.read(vertex_count * dtype.itemsize)
        if len(raw) < vertex_count * dtype.itemsize:
            raise ValueError("PLY vertex buffer is shorter than expected.")
        verts = np.frombuffer(raw, dtype=dtype, count=vertex_count)

        return {name: verts[name].astype(np.float32, copy=False) for name in names}


def _shift_with_nan(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    y0 = max(0, -dy)
    y1 = arr.shape[0] - max(0, dy)
    x0 = max(0, -dx)
    x1 = arr.shape[1] - max(0, dx)
    out[y0 + dy : y1 + dy, x0 + dx : x1 + dx] = arr[y0:y1, x0:x1]
    return out


def _fill_missing_neighbors(height: np.ndarray, max_iters: int) -> np.ndarray:
    height = height.copy()
    height[~np.isfinite(height)] = np.nan
    for _ in range(max_iters):
        missing = ~np.isfinite(height)
        if not missing.any():
            break
        neighbors = [
            _shift_with_nan(height, -1, 0),
            _shift_with_nan(height, 1, 0),
            _shift_with_nan(height, 0, -1),
            _shift_with_nan(height, 0, 1),
            _shift_with_nan(height, -1, -1),
            _shift_with_nan(height, -1, 1),
            _shift_with_nan(height, 1, -1),
            _shift_with_nan(height, 1, 1),
        ]
        stack = np.stack(neighbors, axis=0)
        valid = np.isfinite(stack)
        counts = valid.sum(axis=0)
        if not counts.any():
            break
        stack = np.where(valid, stack, np.nan)
        avg = np.nanmean(stack, axis=0)
        fill_mask = missing & (counts > 0)
        height[fill_mask] = avg[fill_mask]
    return height


def _build_heightfield(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    grid_res: float,
    pad: float,
    fill_mode: str,
    neighbor_iters: int,
    align_ground: bool,
    shift_xy_to_zero: bool,
    max_vertices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_y, max_y = float(np.min(y)), float(np.max(y))
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    xs = np.arange(min_x, max_x + grid_res * 0.5, grid_res, dtype=np.float32)
    ys = np.arange(min_y, max_y + grid_res * 0.5, grid_res, dtype=np.float32)
    nx, ny = xs.shape[0], ys.shape[0]
    if nx * ny > max_vertices:
        raise ValueError(
            f"Grid is too large ({nx}x{ny}={nx * ny} vertices). "
            "Increase --grid-res or raise --max-vertices."
        )

    height = np.full((ny, nx), -np.inf, dtype=np.float32)
    ix = np.floor((x - min_x) / grid_res).astype(np.int64)
    iy = np.floor((y - min_y) / grid_res).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[valid]
    iy = iy[valid]
    z = z[valid]
    np.maximum.at(height, (iy, ix), z)

    if fill_mode == "min":
        fill_value = float(np.min(z)) if z.size > 0 else 0.0
        height[~np.isfinite(height)] = fill_value
    elif fill_mode == "zero":
        height[~np.isfinite(height)] = 0.0
    elif fill_mode == "neighbor":
        height = _fill_missing_neighbors(height, neighbor_iters)
        if not np.isfinite(height).all():
            fill_value = float(np.nanmin(height[np.isfinite(height)])) if np.isfinite(height).any() else 0.0
            height[~np.isfinite(height)] = fill_value
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    if align_ground:
        height -= float(np.min(height))

    if shift_xy_to_zero:
        xs = xs - float(xs.min())
        ys = ys - float(ys.min())

    return xs, ys, height


def _write_obj(path: Path, xs: np.ndarray, ys: np.ndarray, height: np.ndarray) -> None:
    nx = xs.shape[0]
    ny = ys.shape[0]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Heightfield mesh generated from gsplat PLY\n")
        handle.write(f"# grid: {nx} x {ny}\n")
        for iy in range(ny):
            for ix in range(nx):
                handle.write(f"v {xs[ix]:.6f} {ys[iy]:.6f} {height[iy, ix]:.6f}\n")

        for iy in range(ny - 1):
            for ix in range(nx - 1):
                v00 = iy * nx + ix + 1
                v10 = iy * nx + (ix + 1) + 1
                v01 = (iy + 1) * nx + ix + 1
                v11 = (iy + 1) * nx + (ix + 1) + 1
                handle.write(f"f {v00} {v10} {v11}\n")
                handle.write(f"f {v00} {v11} {v01}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a gsplat PLY (3D Gaussians) to a heightfield OBJ mesh that "
            "can be loaded by holosoma's terrain_load_obj."
        )
    )
    parser.add_argument("input_ply", type=Path, help="Path to gsplat PLY file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output OBJ path. Defaults to <input>_terrain.obj",
    )
    parser.add_argument("--grid-res", type=float, default=0.05, help="Grid resolution in meters.")
    parser.add_argument("--pad", type=float, default=0.0, help="Padding added to XY bounds (meters).")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor applied to XYZ coordinates.")
    parser.add_argument("--opacity-min", type=float, default=None, help="Minimum opacity to keep points.")
    parser.add_argument("--opacity-max", type=float, default=None, help="Maximum opacity to keep points.")
    parser.add_argument("--z-min", type=float, default=None, help="Minimum Z to keep points.")
    parser.add_argument("--z-max", type=float, default=None, help="Maximum Z to keep points.")
    parser.add_argument(
        "--fill-mode",
        choices=["min", "zero", "neighbor"],
        default="min",
        help="How to fill empty grid cells.",
    )
    parser.add_argument("--neighbor-iters", type=int, default=16, help="Iterations for neighbor fill mode.")
    parser.add_argument(
        "--align-ground",
        dest="align_ground",
        action="store_true",
        default=True,
        help="Shift heightfield so the minimum height is zero (default: enabled).",
    )
    parser.add_argument(
        "--no-align-ground",
        dest="align_ground",
        action="store_false",
        help="Disable ground alignment.",
    )
    parser.add_argument(
        "--shift-xy-to-zero",
        action="store_true",
        help="Shift XY so the minimum corner is at (0, 0).",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=5_000_000,
        help="Maximum number of vertices allowed in the output mesh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_ply: Path = args.input_ply
    if not input_ply.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_ply}")

    data = _load_ply_vertices(input_ply)
    for key in ("x", "y", "z"):
        if key not in data:
            raise ValueError(f"PLY is missing required '{key}' property.")

    x = data["x"] * args.scale
    y = data["y"] * args.scale
    z = data["z"] * args.scale

    mask = np.ones_like(z, dtype=bool)
    if args.opacity_min is not None or args.opacity_max is not None:
        if "opacity" not in data:
            raise ValueError("opacity filter requested but PLY has no opacity property.")
        opacity = data["opacity"]
        if args.opacity_min is not None:
            mask &= opacity >= args.opacity_min
        if args.opacity_max is not None:
            mask &= opacity <= args.opacity_max
    if args.z_min is not None:
        mask &= z >= args.z_min
    if args.z_max is not None:
        mask &= z <= args.z_max

    x = x[mask]
    y = y[mask]
    z = z[mask]
    if x.size == 0:
        raise ValueError("No points left after filtering.")

    xs, ys, height = _build_heightfield(
        x,
        y,
        z,
        grid_res=args.grid_res,
        pad=args.pad,
        fill_mode=args.fill_mode,
        neighbor_iters=args.neighbor_iters,
        align_ground=args.align_ground,
        shift_xy_to_zero=args.shift_xy_to_zero,
        max_vertices=args.max_vertices,
    )

    output_path = args.output
    if output_path is None:
        output_path = input_ply.with_name(f"{input_ply.stem}_terrain.obj")
    _write_obj(output_path, xs, ys, height)

    print(f"Wrote OBJ: {output_path}")
    print(f"Grid: {xs.shape[0]} x {ys.shape[0]} (res={args.grid_res})")


if __name__ == "__main__":
    main()
