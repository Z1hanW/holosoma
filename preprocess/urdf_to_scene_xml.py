#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

try:
    import yourdfpy  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover - import error path
    raise SystemExit(f"yourdfpy is required: {exc}") from exc


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return cleaned.strip("_") or "obj"


def _parse_package_map(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --package '{item}', expected pkg=path")
        pkg, path = item.split("=", 1)
        out[pkg] = Path(path).expanduser().resolve()
    return out


def _resolve_mesh_path(mesh_path: str, urdf_dir: Path, package_map: dict[str, Path]) -> Path:
    if mesh_path.startswith("package://"):
        rel = mesh_path[len("package://") :]
        pkg, _, sub = rel.partition("/")
        if pkg not in package_map:
            raise ValueError(f"Missing --package mapping for '{pkg}' (from {mesh_path})")
        return (package_map[pkg] / sub).resolve()
    p = Path(mesh_path)
    if p.is_absolute():
        return p
    return (urdf_dir / p).resolve()


def _iter_meshes(
    urdf: yourdfpy.URDF,
    source: str,
) -> Iterable[tuple[str, tuple[float, float, float]]]:
    meshes: list[tuple[str, tuple[float, float, float]]] = []

    def collect_from(geoms):
        for geom in geoms:
            if geom is None or not hasattr(geom, "geometry"):
                continue
            g = geom.geometry
            if not hasattr(g, "filename"):
                continue
            filename = g.filename
            scale = g.scale if getattr(g, "scale", None) is not None else (1.0, 1.0, 1.0)
            if isinstance(scale, (list, tuple)) and len(scale) == 3:
                scale_t = (float(scale[0]), float(scale[1]), float(scale[2]))
            else:
                scale_t = (1.0, 1.0, 1.0)
            meshes.append((filename, scale_t))

    for link in urdf.links:
        visuals = getattr(link, "visuals", []) or []
        collisions = getattr(link, "collisions", []) or []
        if source == "visual":
            collect_from(visuals)
        elif source == "collision":
            collect_from(collisions)
        elif source == "both":
            collect_from(visuals)
            collect_from(collisions)
        elif source == "auto":
            if collisions:
                collect_from(collisions)
            else:
                collect_from(visuals)
        else:
            raise ValueError(f"Unknown source '{source}'")

    # De-duplicate while preserving order
    seen = set()
    for filename, scale in meshes:
        key = (filename, scale)
        if key in seen:
            continue
        seen.add(key)
        yield filename, scale


def _find_or_create(parent: ET.Element, tag: str) -> ET.Element:
    elem = parent.find(tag)
    if elem is None:
        elem = ET.SubElement(parent, tag)
    return elem


def _unique_name(existing: set[str], base: str) -> str:
    if base not in existing:
        existing.add(base)
        return base
    idx = 1
    while f"{base}_{idx}" in existing:
        idx += 1
    name = f"{base}_{idx}"
    existing.add(name)
    return name


def _build_scene_xml(
    base_root: ET.Element,
    mesh_path: Path,
    mesh_scale: tuple[float, float, float],
    output_path: Path,
    *,
    add_freejoint: bool,
    object_mass: float,
    object_diaginertia: tuple[float, float, float],
    use_absolute_paths: bool,
) -> None:
    root = copy.deepcopy(base_root)
    asset = _find_or_create(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF is missing <worldbody>")

    existing_mesh_names = {m.attrib.get("name", "") for m in asset.findall("mesh")}
    existing_geom_names = {
        g.attrib.get("name", "") for g in worldbody.findall(".//geom") if g.attrib.get("name")
    }

    obj_stem = _sanitize_name(mesh_path.stem)
    mesh_name = _unique_name(existing_mesh_names, f"{obj_stem}_mesh")
    geom_name = _unique_name(existing_geom_names, obj_stem)
    body_name = _unique_name(existing_geom_names, f"{obj_stem}_link")

    mesh_file = mesh_path.resolve()
    if not use_absolute_paths:
        mesh_file = Path("./") / mesh_file.name

    mesh_elem = ET.SubElement(asset, "mesh")
    mesh_elem.set("name", mesh_name)
    mesh_elem.set("file", str(mesh_file))
    mesh_elem.set("scale", f"{mesh_scale[0]} {mesh_scale[1]} {mesh_scale[2]}")

    body = ET.SubElement(worldbody, "body")
    body.set("name", body_name)
    if add_freejoint:
        ET.SubElement(body, "freejoint")

    inertial = ET.SubElement(body, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", f"{object_mass}")
    inertial.set("diaginertia", f"{object_diaginertia[0]} {object_diaginertia[1]} {object_diaginertia[2]}")

    geom = ET.SubElement(body, "geom")
    geom.set("name", geom_name)
    geom.set("type", "mesh")
    geom.set("mesh", mesh_name)
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    geom.set("pos", "0 0 0")
    geom.set("quat", "1 0 0 0")
    geom.set("rgba", "0.7 0.8 0.9 0.7")
    geom.set("friction", "0.9 0.5 0.5")
    geom.set("solref", "0.02 1")
    geom.set("solimp", "0.9 0.95 0.001")

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(output_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MJCF scene XMLs from URDF meshes.")
    parser.add_argument("--urdf", required=True, help="Input URDF path containing mesh references.")
    parser.add_argument("--robot-xml", required=True, help="Base robot MJCF (e.g., g1_29dof.xml).")
    parser.add_argument("--output-dir", required=True, help="Directory to write output MJCFs.")
    parser.add_argument(
        "--mesh-source",
        choices=["auto", "visual", "collision", "both"],
        default="auto",
        help="Which URDF meshes to convert (default: auto prefers collision).",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="URDF package mapping: pkg=/abs/path (repeatable).",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Do not add freejoint to object body (static object).",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Use relative mesh paths (files must be colocated with XML).",
    )
    parser.add_argument("--object-mass", type=float, default=0.1)
    parser.add_argument("--object-diaginertia", type=float, nargs=3, default=(0.002, 0.002, 0.002))

    args = parser.parse_args()

    urdf_path = Path(args.urdf).expanduser().resolve()
    robot_xml = Path(args.robot_xml).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    package_map = _parse_package_map(args.package)

    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_scene_graph=False)
    urdf_dir = urdf_path.parent

    base_tree = ET.parse(robot_xml)
    base_root = base_tree.getroot()

    base_stem = robot_xml.stem
    wrote = 0
    for mesh_file, scale in _iter_meshes(urdf, args.mesh_source):
        mesh_path = _resolve_mesh_path(mesh_file, urdf_dir, package_map)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        obj_stem = _sanitize_name(mesh_path.stem)
        out_name = f"{base_stem}_w_{obj_stem}.xml"
        out_path = output_dir / out_name

        _build_scene_xml(
            base_root,
            mesh_path,
            scale,
            out_path,
            add_freejoint=not args.static,
            object_mass=float(args.object_mass),
            object_diaginertia=tuple(float(x) for x in args.object_diaginertia),
            use_absolute_paths=not args.relative,
        )
        wrote += 1

    print(f"Wrote {wrote} scene XML file(s) to {output_dir}")


if __name__ == "__main__":
    main()
