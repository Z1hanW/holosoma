#!/usr/bin/env python3
"""Prepare a MuJoCo object URDF with convex collision meshes.

This helper is intentionally standalone so the tracking source tree can stay
unchanged. It prefers existing ``meshes_collision_convex/part_*.obj`` assets
from the CarryAny object directories. If they are missing, it can run the
Python ``coacd`` package to generate them.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _parse_xyz(value: str | None, default: str = "1 1 1") -> str:
    if value is None or not value.strip():
        return default
    return " ".join(value.split())


def _mesh_path(filename: str, urdf_dir: Path) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return urdf_dir / path


def _path_for_urdf(path: Path, output_dir: Path, absolute_paths: bool) -> str:
    path = path.resolve()
    if absolute_paths:
        return str(path)
    try:
        return str(path.relative_to(output_dir.resolve()))
    except ValueError:
        return str(path)


def _find_visual_mesh(link: ET.Element, urdf_dir: Path) -> tuple[ET.Element, Path, str]:
    visual = link.find("visual")
    if visual is None:
        raise ValueError("URDF link has no <visual>; pass a URDF with a visual mesh")
    mesh = visual.find("./geometry/mesh")
    if mesh is None or not mesh.get("filename"):
        raise ValueError("URDF visual has no mesh filename")
    return mesh, _mesh_path(mesh.get("filename", ""), urdf_dir), _parse_xyz(mesh.get("scale"))


def _existing_convex_parts(object_dir: Path) -> list[Path]:
    convex_dir = object_dir / "meshes_collision_convex"
    if not convex_dir.is_dir():
        return []
    return sorted(convex_dir.glob("part_*.obj"))


def _write_obj(path: Path, vertices, faces) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
        for face in faces:
            # OBJ is 1-indexed.
            idx = [int(i) + 1 for i in face]
            f.write("f " + " ".join(str(i) for i in idx) + "\n")


def _run_coacd(source_mesh: Path, output_dir: Path) -> list[Path]:
    try:
        import coacd  # type: ignore
        import trimesh  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No existing convex collision parts were found and the Python "
            "'coacd'/'trimesh' packages are not installed in this environment."
        ) from exc

    mesh = trimesh.load_mesh(source_mesh, process=False)
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise RuntimeError(f"Could not load a triangle mesh from {source_mesh}")

    coacd.set_log_level("error")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh)
    if not parts:
        raise RuntimeError(f"COACD returned no convex parts for {source_mesh}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (vertices, faces) in enumerate(parts):
        path = output_dir / f"part_{i:03d}.obj"
        _write_obj(path, vertices, faces)
        paths.append(path)
    return paths


def _replace_collisions(
    link: ET.Element,
    convex_parts: list[Path],
    scale: str,
    output_dir: Path,
    absolute_paths: bool,
) -> None:
    for collision in list(link.findall("collision")):
        link.remove(collision)

    for i, part in enumerate(convex_parts):
        collision = ET.SubElement(link, "collision", {"name": f"coacd_part_{i:03d}"})
        ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {
                "filename": _path_for_urdf(part, output_dir, absolute_paths),
                "scale": scale,
            },
        )


def prepare_urdf(
    urdf: Path,
    output: Path,
    force_coacd: bool,
    absolute_paths: bool,
) -> tuple[Path, int, str]:
    urdf = urdf.resolve()
    output = output.resolve()
    tree = ET.parse(urdf)
    root = tree.getroot()
    link = root.find("link")
    if link is None:
        raise ValueError("URDF has no <link>")

    visual_mesh_elem, source_mesh, visual_scale = _find_visual_mesh(link, urdf.parent)
    convex_parts = [] if force_coacd else _existing_convex_parts(urdf.parent)
    source = "existing"
    if not convex_parts:
        convex_parts = _run_coacd(source_mesh, urdf.parent / "meshes_collision_convex")
        source = "coacd"

    if absolute_paths:
        visual_mesh_elem.set("filename", str(source_mesh.resolve()))

    _replace_collisions(link, convex_parts, visual_scale, output.parent, absolute_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output, len(convex_parts), source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path, help="Input object URDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output URDF path. Defaults to '<input>_mujoco_convex.urdf'.",
    )
    parser.add_argument(
        "--force-coacd",
        action="store_true",
        help="Run COACD even if meshes_collision_convex/part_*.obj already exists.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute mesh paths, useful when output URDF is outside the object directory.",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.urdf.with_name(f"{args.urdf.stem}_mujoco_convex.urdf")

    try:
        out, count, source = prepare_urdf(args.urdf, output, args.force_coacd, args.absolute_paths)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {out}")
    print(f"collision_parts={count} source={source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
