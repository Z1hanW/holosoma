#!/usr/bin/env python3
"""Export representative GLBs overlaying legacy and canonical object axes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


DEFAULT_SOURCE_BANK = Path(
    "/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/by-source/"
    "307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef"
)
DEFAULT_CANONICAL_BANK = Path(
    "/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_canonical_object_frame_v1/"
    "by-source/6ea4e78886a463b0dd59c1d69e6b15d13fb90fed88fbd397323a9eb441233616"
)
DEFAULT_OUTPUT = Path("outputs/canonical_object_frame_axis_comparison_glbs_20260812_v2")

SELECTION = (
    ("01", "box_10", "aligned_box_control"),
    ("02", "noscale__any_bin_39", "tilted_bin"),
    ("03", "scale__any_bin_35", "axisymmetric_bin"),
    ("04", "scale__any_barrel_6", "strongly_tilted_barrel"),
    ("05", "scale__any_barrel_68", "ambiguous_barrel"),
    ("06", "scaledown__any_ball_26", "tilted_near_spherical_object"),
    ("07", "unscale__any_ball_24", "near_spherical_object"),
)

OLD_COLORS = {
    "X": (255, 150, 145, 185),
    "Y": (145, 235, 155, 185),
    "Z": (145, 175, 255, 185),
}
NEW_COLORS = {
    "X": (235, 35, 30, 255),
    "Y": (25, 185, 55, 255),
    "Z": (25, 75, 240, 255),
}
AXES = {
    "X": np.asarray([1.0, 0.0, 0.0]),
    "Y": np.asarray([0.0, 1.0, 0.0]),
    "Z": np.asarray([0.0, 0.0, 1.0]),
}


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type {type(loaded).__name__}: {path}")
    return loaded


def _resolve(raw: str, base: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _urdf_mesh_and_scale(urdf_path: Path) -> tuple[Path, float]:
    root = ET.parse(urdf_path).getroot()
    mesh_node = root.find(".//visual/geometry/mesh")
    if mesh_node is None:
        raise ValueError(f"No visual mesh in {urdf_path}")
    scale = np.asarray([float(value) for value in mesh_node.get("scale", "1 1 1").split()])
    if scale.shape != (3,) or not np.allclose(scale, scale[0], atol=1.0e-10):
        raise ValueError(f"Expected uniform scale in {urdf_path}, got {scale}")
    mesh_path = _resolve(str(mesh_node.get("filename", "")), urdf_path.parent)
    return mesh_path, float(scale[0])


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    z_axis = np.asarray([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if math.isclose(dot, 1.0, abs_tol=1.0e-10):
        return np.eye(4)
    if math.isclose(dot, -1.0, abs_tol=1.0e-10):
        return trimesh.transformations.rotation_matrix(math.pi, [1.0, 0.0, 0.0])
    axis = np.cross(z_axis, direction)
    axis /= np.linalg.norm(axis)
    return trimesh.transformations.rotation_matrix(math.acos(dot), axis)


def _paint(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _arrow(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    length: float,
    radius: float,
    color: tuple[int, int, int, int],
    old_style: bool,
) -> trimesh.Trimesh:
    shaft_fraction = 0.82 if old_style else 0.74
    shaft_length = length * shaft_fraction
    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_length, sections=20)
    shaft.apply_translation([0.0, 0.0, shaft_length * 0.5])
    head = trimesh.creation.cone(
        radius=radius * (1.75 if old_style else 2.35),
        height=length - shaft_length,
        sections=20,
    )
    head.apply_translation([0.0, 0.0, shaft_length + (length - shaft_length) * 0.5])
    result = trimesh.util.concatenate((shaft, head))
    result.apply_transform(_rotation_from_z(direction))
    result.apply_translation(origin)
    return _paint(result, color)


def _sphere(origin: np.ndarray, radius: float, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(origin)
    return _paint(sphere, color)


def _connector(start: np.ndarray, end: np.ndarray, radius: float) -> trimesh.Trimesh | None:
    delta = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(delta))
    if length < 1.0e-7:
        return None
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    cylinder.apply_translation([0.0, 0.0, length * 0.5])
    cylinder.apply_transform(_rotation_from_z(delta))
    cylinder.apply_translation(start)
    return _paint(cylinder, (255, 205, 35, 230))


def _build_scene(
    mesh: trimesh.Trimesh,
    *,
    old_origin: np.ndarray,
    new_origin: np.ndarray,
    old_from_new: np.ndarray,
    clip_id: str,
) -> trimesh.Scene:
    extent = float(np.max(mesh.extents))
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError(f"Invalid mesh extent for {clip_id}: {mesh.extents}")
    mesh = mesh.copy()
    _paint(mesh, (178, 187, 202, 175))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name=f"OBJECT__{clip_id}", node_name=f"OBJECT__{clip_id}")

    old_length = extent * 0.62
    new_length = extent * 0.90
    for axis_index, (axis_name, axis) in enumerate(AXES.items()):
        old_arrow = _arrow(
            old_origin,
            axis,
            length=old_length,
            radius=extent * 0.008,
            color=OLD_COLORS[axis_name],
            old_style=True,
        )
        new_arrow = _arrow(
            new_origin,
            old_from_new[:, axis_index],
            length=new_length,
            radius=extent * 0.014,
            color=NEW_COLORS[axis_name],
            old_style=False,
        )
        scene.add_geometry(
            old_arrow,
            geom_name=f"OLD_{axis_name}__PALE_SHORT",
            node_name=f"OLD_{axis_name}__PALE_SHORT",
        )
        scene.add_geometry(
            new_arrow,
            geom_name=f"NEW_{axis_name}__SATURATED_LONG",
            node_name=f"NEW_{axis_name}__SATURATED_LONG",
        )

    scene.add_geometry(
        _sphere(old_origin, extent * 0.025, (245, 245, 245, 255)),
        geom_name="OLD_ORIGIN__WHITE",
        node_name="OLD_ORIGIN__WHITE",
    )
    scene.add_geometry(
        _sphere(new_origin, extent * 0.032, (255, 200, 20, 255)),
        geom_name="NEW_COM_ORIGIN__GOLD",
        node_name="NEW_COM_ORIGIN__GOLD",
    )
    connector = _connector(old_origin, new_origin, extent * 0.0045)
    if connector is not None:
        scene.add_geometry(
            connector,
            geom_name="OLD_ORIGIN_TO_NEW_COM__GOLD",
            node_name="OLD_ORIGIN_TO_NEW_COM__GOLD",
        )
    return scene


def export(source_bank: Path, canonical_bank: Path, output_dir: Path) -> Path:
    source_bank = source_bank.expanduser().resolve()
    canonical_bank = canonical_bank.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_map = json.loads((source_bank / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))["clips"]
    canonical_map = json.loads(
        (canonical_bank / "_clip_object_urdf_map.json").read_text(encoding="utf-8")
    )["clips"]
    canonical_manifest = json.loads(
        (canonical_bank / "_canonical_object_frame_manifest.json").read_text(encoding="utf-8")
    )
    reports = {report["clip_id"]: report for report in canonical_manifest["clip_reports"]}
    outputs: list[dict[str, object]] = []
    for sequence, clip_id, description in SELECTION:
        entry = source_map[clip_id]
        source_urdf = _resolve(str(entry["object_urdf_path"]), source_bank)
        source_mesh, scale = _urdf_mesh_and_scale(source_urdf)
        mesh = _load_mesh(source_mesh)
        mesh.apply_scale(scale)
        report = reports[clip_id]
        old_from_new = np.asarray(report["rotation_old_from_canonical"], dtype=np.float64)
        new_origin = np.asarray(report["origin_old_scaled_m"], dtype=np.float64)
        old_origin = np.zeros(3, dtype=np.float64)
        scene = _build_scene(
            mesh,
            old_origin=old_origin,
            new_origin=new_origin,
            old_from_new=old_from_new,
            clip_id=clip_id,
        )
        output_name = f"{sequence}_{clip_id.replace('__', '_')}__{description}__old_vs_canonical_axes.glb"
        output_path = output_dir / output_name
        scene.export(output_path)
        outputs.append(
            {
                "sequence": int(sequence),
                "clip_id": clip_id,
                "description": description,
                "file": output_name,
                "size_bytes": output_path.stat().st_size,
                "source_mesh": str(source_mesh),
                "uniform_scale": scale,
                "old_origin_m": old_origin.tolist(),
                "new_com_origin_in_old_frame_m": new_origin.tolist(),
                "rotation_old_from_canonical": old_from_new.tolist(),
                "canonical_symmetry": canonical_map[clip_id]["canonical_symmetry"],
                "legend": {
                    "old_axes": "pale, short, thin RGB arrows",
                    "canonical_axes": "saturated, long, thick RGB arrows",
                    "old_origin": "white sphere",
                    "canonical_com_origin": "gold sphere",
                    "origin_offset": "gold connector",
                    "axis_colors": "X=red, Y=green, Z=blue",
                },
            }
        )
        print(output_path)

    manifest = {
        "schema_version": 1,
        "semantics": "legacy_and_canonical_object_frames_overlaid_on_legacy_mesh_coordinates",
        "source_bank": str(source_bank),
        "canonical_bank": str(canonical_bank),
        "count": len(outputs),
        "legend": outputs[0]["legend"],
        "files": outputs,
    }
    manifest_path = output_dir / "axis_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_SOURCE_BANK)
    parser.add_argument("--canonical-bank", type=Path, default=DEFAULT_CANONICAL_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export(args.source_bank, args.canonical_bank, args.output_dir)
    print(manifest)


if __name__ == "__main__":
    main()
