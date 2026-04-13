#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJ = REPO_ROOT / "src" / "holosoma_retargeting" / "models" / "largebox" / "largebox.obj"
DEFAULT_OUT_DIR = Path("/home/ubuntu/FAR")


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        dumped = loaded.dump(concatenate=True)
        if not isinstance(dumped, trimesh.Trimesh):
            raise TypeError(f"Unsupported scene contents in {path}")
        loaded = dumped
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(loaded).__name__}")
    return loaded


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if math.isclose(dot, 1.0, abs_tol=1.0e-8):
        return np.eye(4, dtype=np.float64)
    if math.isclose(dot, -1.0, abs_tol=1.0e-8):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
        return transform
    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(dot)
    return trimesh.transformations.rotation_matrix(angle, axis)


def _make_arrow(direction: np.ndarray, length: float, radius: float, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    shaft_length = length * 0.76
    head_length = length - shaft_length
    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_length, sections=24)
    shaft.apply_translation([0.0, 0.0, shaft_length * 0.5])
    head = trimesh.creation.cone(radius=radius * 2.2, height=head_length, sections=24)
    head.apply_translation([0.0, 0.0, shaft_length + head_length * 0.5])
    arrow = trimesh.util.concatenate([shaft, head])
    arrow.apply_transform(_rotation_from_z(direction))
    vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(arrow.vertices), 1))
    arrow.visual.vertex_colors = vertex_colors
    return arrow


def _build_scene(mesh: trimesh.Trimesh) -> tuple[trimesh.Scene, trimesh.Trimesh]:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    axis_length = float(np.max(extents) * 0.8)
    axis_radius = float(np.max(extents) * 0.015)

    mesh_colored = mesh.copy()
    base_color = np.array([180, 190, 205, 255], dtype=np.uint8)
    mesh_colored.visual.vertex_colors = np.tile(base_color, (len(mesh_colored.vertices), 1))

    x_arrow = _make_arrow(np.array([1.0, 0.0, 0.0]), axis_length, axis_radius, (230, 70, 60, 255))
    y_arrow = _make_arrow(np.array([0.0, 1.0, 0.0]), axis_length, axis_radius, (60, 180, 75, 255))
    z_arrow = _make_arrow(np.array([0.0, 0.0, 1.0]), axis_length, axis_radius, (70, 115, 230, 255))

    scene = trimesh.Scene()
    scene.add_geometry(mesh_colored, geom_name="largebox_mesh")
    scene.add_geometry(x_arrow, geom_name="axis_x")
    scene.add_geometry(y_arrow, geom_name="axis_y")
    scene.add_geometry(z_arrow, geom_name="axis_z")

    combined = trimesh.util.concatenate([mesh_colored, x_arrow, y_arrow, z_arrow])
    return scene, combined


def _render_preview(mesh: trimesh.Trimesh, out_path: Path) -> None:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    fig = plt.figure(figsize=(8, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    light = LightSource(azdeg=25, altdeg=35)
    face_normals = mesh.face_normals
    shade = light.shade_normals(face_normals)
    base_rgb = np.array([0.68, 0.72, 0.80], dtype=np.float64)
    face_colors = np.clip(base_rgb[None, :] * (0.55 + 0.45 * shade[:, None]), 0.0, 1.0)
    tris = verts[faces]
    poly = Poly3DCollection(tris, facecolors=face_colors, edgecolors=(0.35, 0.35, 0.38, 0.25), linewidths=0.15)
    ax.add_collection3d(poly)

    axis_len = float(np.max(mesh.extents) * 0.95)
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="#d94841", linewidth=2.2, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="#2b8a3e", linewidth=2.2, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="#2f59d9", linewidth=2.2, arrow_length_ratio=0.12)
    ax.text(axis_len * 1.05, 0, 0, "+X", color="#d94841", fontsize=12, weight="bold")
    ax.text(0, axis_len * 1.05, 0, "+Y", color="#2b8a3e", fontsize=12, weight="bold")
    ax.text(0, 0, axis_len * 1.05, "+Z", color="#2f59d9", fontsize=12, weight="bold")

    bounds = mesh.bounds
    corner_min = bounds[0] - 0.22 * np.max(mesh.extents)
    corner_max = bounds[1] + 0.22 * np.max(mesh.extents)
    ax.set_xlim(corner_min[0], corner_max[0])
    ax.set_ylim(corner_min[1], corner_max[1])
    ax.set_zlim(corner_min[2], corner_max[2])
    ax.set_box_aspect(corner_max - corner_min)
    ax.view_init(elev=22, azim=40)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("largebox.obj with object-frame axes")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    obj_path = DEFAULT_OBJ
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = _load_mesh(obj_path)
    scene, combined = _build_scene(mesh)

    glb_path = out_dir / "largebox_with_axes.glb"
    ply_path = out_dir / "largebox_with_axes.ply"
    png_path = out_dir / "largebox_axes_preview.png"

    scene.export(glb_path)
    combined.export(ply_path)
    _render_preview(mesh, png_path)

    extents = np.asarray(mesh.extents, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    print(f"mesh: {obj_path}")
    print(f"bounds_min: {bounds[0].tolist()}")
    print(f"bounds_max: {bounds[1].tolist()}")
    print(f"extents_xyz: {extents.tolist()}")
    print(f"origin_to_centroid: {mesh.centroid.tolist()}")
    print(f"wrote: {glb_path}")
    print(f"wrote: {ply_path}")
    print(f"wrote: {png_path}")


if __name__ == "__main__":
    main()
