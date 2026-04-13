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
from scipy.spatial.transform import Rotation as R

from holosoma.utils.object_geometry import load_urdf_box_primitive_metadata
from holosoma.utils.object_pose_correction import get_omomo_largebox_primitive_fit_local_correction_wxyz_np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "data_demo" / "objects" / "objects_largebox.urdf"
DEFAULT_OBJ = REPO_ROOT / "data_demo" / "objects" / "largebox.obj"
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


def _rotate_mesh_to_primitive_aligned_frame(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    correction_wxyz = get_omomo_largebox_primitive_fit_local_correction_wxyz_np().astype(np.float64)
    rot = R.from_quat([correction_wxyz[1], correction_wxyz[2], correction_wxyz[3], correction_wxyz[0]]).inv()
    rotated = mesh.copy()
    rotated.vertices = rot.apply(np.asarray(rotated.vertices, dtype=np.float64))
    return rotated


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
    arrow.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(arrow.vertices), 1))
    return arrow


def _box_corners(extents: np.ndarray, center: np.ndarray) -> np.ndarray:
    half = 0.5 * extents
    corners = np.array(
        [
            [sx, sy, sz]
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ],
        dtype=np.float64,
    )
    return corners + center.reshape(1, 3)


def _make_box_wireframe(extents: np.ndarray, center: np.ndarray, radius: float) -> trimesh.Trimesh:
    corners = _box_corners(extents, center)
    edges = (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    )
    segments: list[trimesh.Trimesh] = []
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for i0, i1 in edges:
        p0 = corners[i0]
        p1 = corners[i1]
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
        direction = delta / max(length, 1.0e-8)
        transform = _rotation_from_z(direction)
        transform[:3, 3] = 0.5 * (p0 + p1)
        cyl.apply_transform(transform)
        segments.append(cyl)
    wire = trimesh.util.concatenate(segments)
    wire.visual.vertex_colors = np.tile(np.array([220, 70, 60, 255], dtype=np.uint8), (len(wire.vertices), 1))
    return wire


def _build_scene(mesh: trimesh.Trimesh, extents: np.ndarray, center: np.ndarray) -> tuple[trimesh.Scene, trimesh.Trimesh]:
    scene = trimesh.Scene()

    mesh_colored = mesh.copy()
    mesh_colored.visual.vertex_colors = np.tile(np.array([180, 190, 205, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    scene.add_geometry(mesh_colored, geom_name="largebox_mesh")

    axis_length = float(max(np.max(mesh.extents), np.max(extents)) * 0.9)
    axis_radius = float(max(np.max(mesh.extents), np.max(extents)) * 0.015)
    axis_x = _make_arrow(np.array([1.0, 0.0, 0.0]), axis_length, axis_radius, (230, 70, 60, 255))
    axis_y = _make_arrow(np.array([0.0, 1.0, 0.0]), axis_length, axis_radius, (60, 180, 75, 255))
    axis_z = _make_arrow(np.array([0.0, 0.0, 1.0]), axis_length, axis_radius, (70, 115, 230, 255))
    scene.add_geometry(axis_x, geom_name="axis_x")
    scene.add_geometry(axis_y, geom_name="axis_y")
    scene.add_geometry(axis_z, geom_name="axis_z")

    box_solid = trimesh.creation.box(extents=extents)
    box_solid.apply_translation(center)
    box_solid.visual.vertex_colors = np.tile(np.array([220, 70, 60, 60], dtype=np.uint8), (len(box_solid.vertices), 1))
    box_wire = _make_box_wireframe(extents, center, radius=axis_radius * 0.45)
    scene.add_geometry(box_solid, geom_name="iou_box_solid")
    scene.add_geometry(box_wire, geom_name="iou_box_wire")

    combined = trimesh.util.concatenate(
        [
            mesh_colored,
            axis_x,
            axis_y,
            axis_z,
            box_solid,
            box_wire,
        ]
    )
    return scene, combined


def _render_preview(mesh: trimesh.Trimesh, extents: np.ndarray, center: np.ndarray, out_path: Path) -> None:
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
    poly = Poly3DCollection(tris, facecolors=face_colors, edgecolors=(0.35, 0.35, 0.38, 0.20), linewidths=0.12)
    ax.add_collection3d(poly)

    box_mesh = trimesh.creation.box(extents=extents)
    box_mesh.apply_translation(center)
    box_tris = np.asarray(box_mesh.vertices)[np.asarray(box_mesh.faces)]
    box_poly = Poly3DCollection(
        box_tris,
        facecolors=(0.86, 0.27, 0.24, 0.10),
        edgecolors=(0.86, 0.27, 0.24, 0.90),
        linewidths=1.1,
    )
    ax.add_collection3d(box_poly)

    axis_len = float(max(np.max(mesh.extents), np.max(extents)) * 0.95)
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="#d94841", linewidth=2.2, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="#2b8a3e", linewidth=2.2, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="#2f59d9", linewidth=2.2, arrow_length_ratio=0.12)
    ax.text(axis_len * 1.05, 0, 0, "+X", color="#d94841", fontsize=12, weight="bold")
    ax.text(0, axis_len * 1.05, 0, "+Y", color="#2b8a3e", fontsize=12, weight="bold")
    ax.text(0, 0, axis_len * 1.05, "+Z", color="#2f59d9", fontsize=12, weight="bold")

    mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    box_bounds = np.asarray(box_mesh.bounds, dtype=np.float64)
    bounds_min = np.minimum(mesh_bounds[0], box_bounds[0])
    bounds_max = np.maximum(mesh_bounds[1], box_bounds[1])
    pad = 0.22 * max(np.max(mesh.extents), np.max(extents))
    bounds_min = bounds_min - pad
    bounds_max = bounds_max + pad
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])
    ax.set_box_aspect(bounds_max - bounds_min)
    ax.view_init(elev=22, azim=40)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("largebox.obj primitive-aligned mesh with yaw-aware IoU-fit box")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = _rotate_mesh_to_primitive_aligned_frame(_load_mesh(DEFAULT_OBJ))
    metadata = load_urdf_box_primitive_metadata(DEFAULT_URDF)
    if metadata is None:
        raise RuntimeError(f"Failed to resolve primitive metadata from {DEFAULT_URDF}")

    extents = np.asarray(metadata.extents, dtype=np.float64)
    center = np.asarray(metadata.center_offset, dtype=np.float64)
    scene, combined = _build_scene(mesh, extents, center)

    glb_path = out_dir / "largebox_with_axes_and_iou_box.glb"
    ply_path = out_dir / "largebox_with_axes_and_iou_box.ply"
    png_path = out_dir / "largebox_iou_box_overlay_preview.png"

    scene.export(glb_path)
    combined.export(ply_path)
    _render_preview(mesh, extents, center, png_path)

    print(f"mesh: {DEFAULT_OBJ}")
    print(f"primitive_extents_xyz: {extents.tolist()}")
    print(f"primitive_center_offset_xyz: {center.tolist()}")
    print(f"wrote: {glb_path}")
    print(f"wrote: {ply_path}")
    print(f"wrote: {png_path}")


if __name__ == "__main__":
    main()
