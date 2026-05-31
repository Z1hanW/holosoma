#!/usr/bin/env python3
"""Export a canonical G1 collision mesh highlighting allowed contact bodies.

The default profile matches train_as_general.sh's current scene/floor/global
contact penalty:

  - red: bodies penalized by lower_body_undesired_contacts
  - green: collision bodies not penalized by that scene/floor/global term

Foot/ankle-roll object contact is a separate ObjectUndesiredContacts penalty in
the reward config; it is listed in the manifest but not colored red here because
this mesh is specifically for scene/floor/global contact.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"
DEFAULT_OUTPUT = Path("/home/ubuntu/FAR/g1_train_as_general_scene_floor_allowed_contacts_canonical.glb")

SCENE_FLOOR_PENALIZED_BODIES = {
    "torso_link",
    "head_link",  # fixed-collapsed into torso_link during training
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
}

OBJECT_ONLY_PENALIZED_BODIES = {
    "left_foot_contact_point",
    "right_foot_contact_point",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
}

GT_WOBJ_ALLOWED_BODIES = {
    "left_foot_contact_point",
    "right_foot_contact_point",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
}

GREEN_ALLOWED = np.array([35, 220, 95, 255], dtype=np.uint8)
RED_PENALIZED = np.array([225, 65, 65, 235], dtype=np.uint8)
CYAN_CURRENT_ONLY = np.array([35, 185, 235, 255], dtype=np.uint8)
YELLOW_GT_ONLY = np.array([245, 190, 45, 255], dtype=np.uint8)


def _resolve_profile(profile: str) -> dict:
    if profile == "train_as_general":
        return {
            "profile": "train_as_general_scene_floor_global_contact",
            "default_output": DEFAULT_OUTPUT,
            "is_penalized": lambda link_name: link_name in SCENE_FLOOR_PENALIZED_BODIES,
            "scene_floor_penalized_bodies": sorted(SCENE_FLOOR_PENALIZED_BODIES),
            "object_only_penalized_bodies_not_encoded_as_red_here": sorted(OBJECT_ONLY_PENALIZED_BODIES),
            "color_legend": {
                "green": "allowed for scene/floor/global contact under current train_as_general lower_body_undesired_contacts",
                "red": "penalized by lower_body_undesired_contacts; head_link is red because it is fixed-collapsed into torso_link",
            },
        }
    if profile == "gt_wobj":
        return {
            "profile": "gt_wobj_undesired_contacts",
            "default_output": Path("/home/ubuntu/FAR/g1_gt_wobj_allowed_contacts_canonical.glb"),
            "is_penalized": lambda link_name: link_name not in GT_WOBJ_ALLOWED_BODIES,
            "gt_wobj_allowed_bodies": sorted(GT_WOBJ_ALLOWED_BODIES),
            "color_legend": {
                "green": "excluded by GT w-obj undesired_contacts regex, so global contact is allowed",
                "red": "selected by GT w-obj undesired_contacts regex, so contact with object/floor/scene is penalized",
            },
        }
    if profile == "compare_gt_train":
        return {
            "profile": "compare_gt_wobj_vs_train_as_general_scene_floor_global_contact",
            "default_output": Path("/home/ubuntu/FAR/g1_gt_vs_train_as_general_allowed_contacts_canonical.glb"),
            "compare": True,
            "color_legend": {
                "green": "allowed by both GT w-obj and current train_as_general scene/floor/global rule",
                "red": "penalized by both GT w-obj and current train_as_general scene/floor/global rule",
                "cyan": "allowed only by current train_as_general; GT w-obj would penalize it",
                "yellow": "allowed only by GT w-obj; current train_as_general would penalize it",
            },
            "gt_wobj_allowed_bodies": sorted(GT_WOBJ_ALLOWED_BODIES),
            "train_as_general_scene_floor_penalized_bodies": sorted(SCENE_FLOOR_PENALIZED_BODIES),
            "object_only_penalized_bodies_not_encoded_as_current_scene_floor_red": sorted(OBJECT_ONLY_PENALIZED_BODIES),
        }
    raise ValueError(f"unsupported profile: {profile}")


def _comparison_class(link_name: str) -> tuple[str, np.ndarray]:
    gt_allowed = link_name in GT_WOBJ_ALLOWED_BODIES
    train_allowed = link_name not in SCENE_FLOOR_PENALIZED_BODIES
    if gt_allowed and train_allowed:
        return "both_allowed", GREEN_ALLOWED
    if (not gt_allowed) and (not train_allowed):
        return "both_penalized", RED_PENALIZED
    if train_allowed and not gt_allowed:
        return "train_only_allowed", CYAN_CURRENT_ONLY
    return "gt_only_allowed", YELLOW_GT_ONLY


def _parse_vec(raw: str | None, length: int, default: float = 0.0) -> np.ndarray:
    if not raw:
        return np.full(length, default, dtype=np.float64)
    values = [float(value) for value in raw.split()]
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {raw!r}")
    return np.asarray(values, dtype=np.float64)


def _origin_matrix(node: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if node is None:
        return matrix
    xyz = _parse_vec(node.get("xyz"), 3, 0.0)
    rpy = _parse_vec(node.get("rpy"), 3, 0.0)
    matrix = trimesh.transformations.euler_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz")
    matrix[:3, 3] = xyz
    return matrix


def _as_mesh(loaded: object, path: Path) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            raise ValueError(f"no mesh geometry in {path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise TypeError(f"unsupported mesh type for {path}: {type(loaded).__name__}")


def _load_geometry(geometry: ET.Element, urdf_dir: Path) -> trimesh.Trimesh:
    mesh_node = geometry.find("mesh")
    if mesh_node is not None:
        filename = str(mesh_node.get("filename", "")).strip()
        if not filename:
            raise ValueError("mesh geometry is missing filename")
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf_dir / mesh_path).resolve()
        mesh = _as_mesh(trimesh.load(mesh_path, process=False), mesh_path)
        mesh = mesh.copy()
        scale_raw = mesh_node.get("scale")
        if scale_raw:
            scale = _parse_vec(scale_raw, 3, 1.0)
            scale_matrix = np.eye(4, dtype=np.float64)
            scale_matrix[0, 0] = scale[0]
            scale_matrix[1, 1] = scale[1]
            scale_matrix[2, 2] = scale[2]
            mesh.apply_transform(scale_matrix)
        return mesh

    box_node = geometry.find("box")
    if box_node is not None:
        return trimesh.creation.box(extents=_parse_vec(box_node.get("size"), 3, 0.0))

    sphere_node = geometry.find("sphere")
    if sphere_node is not None:
        radius = float(sphere_node.get("radius", "0"))
        return trimesh.creation.icosphere(subdivisions=3, radius=radius)

    cylinder_node = geometry.find("cylinder")
    if cylinder_node is not None:
        radius = float(cylinder_node.get("radius", "0"))
        height = float(cylinder_node.get("length", "0"))
        return trimesh.creation.cylinder(radius=radius, height=height, sections=32)

    raise ValueError(f"unsupported collision geometry: {ET.tostring(geometry, encoding='unicode')}")


def _compute_link_transforms(root: ET.Element) -> dict[str, np.ndarray]:
    link_names = [link.get("name") for link in root.findall("link") if link.get("name")]
    children: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    child_links: set[str] = set()

    for joint in root.findall("joint"):
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if parent_node is None or child_node is None:
            continue
        parent = parent_node.get("link")
        child = child_node.get("link")
        if not parent or not child:
            continue
        children[parent].append((child, _origin_matrix(joint.find("origin"))))
        child_links.add(child)

    roots = [name for name in link_names if name not in child_links]
    if not roots:
        raise ValueError("could not find root link")

    transforms: dict[str, np.ndarray] = {}
    queue: deque[tuple[str, np.ndarray]] = deque((name, np.eye(4, dtype=np.float64)) for name in roots)
    while queue:
        link_name, transform = queue.popleft()
        transforms[link_name] = transform
        for child, joint_transform in children.get(link_name, ()):
            queue.append((child, transform @ joint_transform))
    return transforms


def _iter_collision_meshes(urdf_path: Path) -> tuple[list[dict], set[str]]:
    root = ET.parse(urdf_path).getroot()
    link_transforms = _compute_link_transforms(root)
    urdf_dir = urdf_path.parent
    pieces: list[dict] = []
    collision_links: set[str] = set()

    for link in root.findall("link"):
        link_name = str(link.get("name", "")).strip()
        if not link_name:
            continue
        link_transform = link_transforms.get(link_name)
        if link_transform is None:
            continue
        for collision_index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            mesh = _load_geometry(geometry, urdf_dir)
            mesh.apply_transform(_origin_matrix(collision.find("origin")))
            mesh.apply_transform(link_transform)
            collision_links.add(link_name)
            pieces.append(
                {
                    "link_name": link_name,
                    "collision_index": collision_index,
                    "mesh": mesh,
                }
            )
    return pieces, collision_links


def _color_mesh(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    return mesh


def _export_preview(meshes: list[tuple[trimesh.Trimesh, np.ndarray]], png_path: Path, max_faces: int) -> None:
    rng = np.random.default_rng(7)
    tris_parts = []
    color_parts = []
    remaining = max_faces

    for mesh, color in meshes:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if faces.size == 0 or vertices.size == 0 or remaining <= 0:
            continue
        face_count = faces.shape[0]
        take = min(face_count, max(1, int(max_faces * face_count / max(1, sum(len(m.faces) for m, _ in meshes)))))
        take = min(take, remaining)
        if take < face_count:
            indices = rng.choice(face_count, size=take, replace=False)
            faces = faces[indices]
        tris_parts.append(vertices[faces])
        color_parts.append(np.tile(color.astype(np.float64) / 255.0, (faces.shape[0], 1)))
        remaining -= faces.shape[0]

    if not tris_parts:
        return

    triangles = np.concatenate(tris_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    points = triangles.reshape(-1, 3)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    radius = float(np.max(bounds_max - bounds_min) * 0.58)

    fig = plt.figure(figsize=(8, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(triangles, facecolors=colors, linewidths=0.0, edgecolors="none")
    ax.add_collection3d(collection)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius * 0.75, center[2] + radius * 0.95)
    ax.set_box_aspect((1, 1, 1.7))
    ax.view_init(elev=14, azim=-63)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(png_path, transparent=False, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--profile",
        choices=["train_as_general", "gt_wobj", "compare_gt_train"],
        default="train_as_general",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-preview-faces", type=int, default=70000)
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)
    urdf_path = args.urdf.expanduser().resolve()
    output_path = (args.output or profile["default_output"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pieces, collision_links = _iter_collision_meshes(urdf_path)
    scene = trimesh.Scene()
    preview_meshes: list[tuple[trimesh.Trimesh, np.ndarray]] = []
    allowed_links: set[str] = set()
    penalized_links: set[str] = set()
    comparison_links: dict[str, set[str]] = {
        "both_allowed": set(),
        "both_penalized": set(),
        "train_only_allowed": set(),
        "gt_only_allowed": set(),
    }

    for piece in pieces:
        link_name = str(piece["link_name"])
        if profile.get("compare"):
            class_name, color = _comparison_class(link_name)
            comparison_links[class_name].add(link_name)
            is_penalized = class_name in {"both_penalized", "gt_only_allowed"}
        else:
            is_penalized = bool(profile["is_penalized"](link_name))
            color = RED_PENALIZED if is_penalized else GREEN_ALLOWED
            if is_penalized:
                penalized_links.add(link_name)
            else:
                allowed_links.add(link_name)
        mesh = _color_mesh(piece["mesh"], color)
        geom_status = class_name if profile.get("compare") else ("penalized" if is_penalized else "allowed")
        geom_name = f"{link_name}_collision_{piece['collision_index']:02d}_{geom_status}"
        scene.add_geometry(mesh, geom_name=geom_name)
        preview_meshes.append((mesh, color))

    scene.export(output_path)

    manifest = {
        "profile": profile["profile"],
        "urdf": str(urdf_path),
        "glb": str(output_path),
        "color_legend": profile["color_legend"],
        "allowed_collision_bodies": sorted(allowed_links),
        "penalized_collision_bodies": sorted(penalized_links),
        "collision_body_count": len(collision_links),
        "collision_geometry_count": len(pieces),
    }
    if profile.get("compare"):
        manifest["comparison_collision_bodies"] = {
            key: sorted(value)
            for key, value in comparison_links.items()
        }
        manifest["allowed_collision_bodies"] = sorted(
            comparison_links["both_allowed"] | comparison_links["train_only_allowed"]
        )
        manifest["penalized_collision_bodies"] = sorted(
            comparison_links["both_penalized"] | comparison_links["gt_only_allowed"]
        )
    for key in (
        "scene_floor_penalized_bodies",
        "object_only_penalized_bodies_not_encoded_as_red_here",
        "gt_wobj_allowed_bodies",
        "train_as_general_scene_floor_penalized_bodies",
        "object_only_penalized_bodies_not_encoded_as_current_scene_floor_red",
    ):
        if key in profile:
            manifest[key] = profile[key]

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    png_path = output_path.with_suffix(".png")
    if not args.no_preview:
        _export_preview(preview_meshes, png_path, max(1, int(args.max_preview_faces)))
        manifest["png"] = str(png_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
