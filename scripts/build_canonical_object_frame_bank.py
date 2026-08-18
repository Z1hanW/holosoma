#!/usr/bin/env python3
"""Build a separate motion bank with physically canonical object frames.

The source bank is treated as immutable.  For each unique object mesh this
builder moves the link origin to the physical COM and constructs a deterministic
geometry-aligned frame:

* +Z is the principal-inertia direction closest to reference world-up;
* +X is the longer principal direction in the plane orthogonal to +Z;
* +Y completes a right-handed frame;
* near-degenerate eigenspaces use projected legacy +X/world-up only as a
  deterministic tie-break, and the ambiguity is recorded in the manifest.

Meshes, URDF inertials, motion poses/velocities, object sizes, rollout-reference
object states, and object-frame contact targets are transformed together.  The
world-space geometry and dynamics therefore remain unchanged while object-frame
coordinates acquire consistent semantics across assets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np


CONTRACT = "canonical_object_frame_com_geometry_axes_v1"
DEGENERACY_REL_TOL = 0.02
CONTACT_POINT_SUFFIX = "_contact_points.npy"
REMOVED_STALE_ROLLOUT_FIELDS = ("actor_obs_raw", "actor_obs_norm")


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_vec(raw: str | None, *, default: tuple[float, float, float]) -> np.ndarray:
    if not raw:
        return np.asarray(default, dtype=np.float64)
    values = [float(value) for value in raw.split()]
    if len(values) != 3:
        raise ValueError(f"Expected three values, got {raw!r}")
    return np.asarray(values, dtype=np.float64)


def _format_vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _resolve_relative(raw: str, base: Path) -> Path:
    value = str(raw).strip()
    if value.startswith("file://"):
        value = value[7:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_zero_origin(node: ET.Element | None, *, label: str, path: Path) -> None:
    if node is None:
        return
    xyz = _parse_vec(node.get("xyz"), default=(0.0, 0.0, 0.0))
    rpy = _parse_vec(node.get("rpy"), default=(0.0, 0.0, 0.0))
    if not np.allclose(xyz, 0.0, atol=1.0e-10) or not np.allclose(rpy, 0.0, atol=1.0e-10):
        raise ValueError(f"{label} origin is not zero in {path}: xyz={xyz}, rpy={rpy}")


def _quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Quaternion must end in 4, got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("Quaternion contains a zero norm")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q *= -1.0
    return q


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    result = np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _rotate_wxyz(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", _quat_to_matrix_wxyz(quaternion), vectors)


@dataclass(frozen=True)
class UrdfInfo:
    clip_id: str
    category: str
    path: Path
    output_name: str
    mesh_path: Path
    scale: float
    mass: float
    com_old_scaled: np.ndarray
    inertia_old: np.ndarray
    up_old_samples: np.ndarray


@dataclass
class CanonicalMesh:
    source_mesh: Path
    clip_ids: list[str]
    categories: list[str]
    origin_old_unscaled: np.ndarray
    rotation_old_from_canonical: np.ndarray
    rotation_quat_wxyz: np.ndarray
    inertia_eigenvalues_per_mass_unscaled: np.ndarray
    symmetry: str
    z_up_alignment_deg: float
    xy_moment_gap_relative: float
    source_sha256: str = ""
    output_sha256: str = ""
    output_name: str = ""
    canonical_bounds_min: np.ndarray | None = None
    canonical_bounds_max: np.ndarray | None = None
    vertex_count: int = 0
    sample_vertices_old: np.ndarray | None = None


def _read_urdf_info(bank: Path, clip_id: str, entry: dict[str, Any]) -> UrdfInfo:
    raw_urdf = str(entry.get("object_urdf_path", "")).strip()
    if not raw_urdf:
        raise ValueError(f"Map entry {clip_id!r} omits object_urdf_path")
    urdf_path = _resolve_relative(raw_urdf, bank)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Missing URDF for {clip_id}: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(f"Expected one object link in {urdf_path}, found {len(links)}")
    link = links[0]
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"Missing inertial in {urdf_path}")
    _require_zero_origin(
        ET.Element("origin", {"rpy": inertial.find("origin").get("rpy", "0 0 0")})
        if inertial.find("origin") is not None
        else None,
        label="inertial rotation",
        path=urdf_path,
    )
    origin = inertial.find("origin")
    com = _parse_vec(origin.get("xyz") if origin is not None else None, default=(0.0, 0.0, 0.0))
    mass_node = inertial.find("mass")
    inertia_node = inertial.find("inertia")
    if mass_node is None or inertia_node is None:
        raise ValueError(f"Incomplete inertial in {urdf_path}")
    mass = float(mass_node.get("value", "nan"))
    ixx = float(inertia_node.get("ixx", "nan"))
    iyy = float(inertia_node.get("iyy", "nan"))
    izz = float(inertia_node.get("izz", "nan"))
    ixy = float(inertia_node.get("ixy", "0"))
    ixz = float(inertia_node.get("ixz", "0"))
    iyz = float(inertia_node.get("iyz", "0"))
    inertia = np.asarray(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64
    )
    if not math.isfinite(mass) or mass <= 0.0 or not np.all(np.linalg.eigvalsh(inertia) > 0.0):
        raise ValueError(f"Invalid mass/inertia in {urdf_path}")

    mesh_nodes = link.findall("./visual/geometry/mesh") + link.findall("./collision/geometry/mesh")
    if not mesh_nodes:
        raise ValueError(f"No mesh geometry in {urdf_path}")
    mesh_paths: set[Path] = set()
    scales: list[np.ndarray] = []
    for parent_tag in ("visual", "collision"):
        for parent in link.findall(parent_tag):
            _require_zero_origin(parent.find("origin"), label=parent_tag, path=urdf_path)
            mesh_node = parent.find("./geometry/mesh")
            if mesh_node is None:
                raise ValueError(f"Non-mesh {parent_tag} geometry in {urdf_path}")
            mesh_paths.add(_resolve_relative(str(mesh_node.get("filename", "")), urdf_path.parent))
            scales.append(_parse_vec(mesh_node.get("scale"), default=(1.0, 1.0, 1.0)))
    if len(mesh_paths) != 1:
        raise ValueError(f"Visual/collision meshes differ in {urdf_path}: {mesh_paths}")
    if any(not np.allclose(scale, scales[0], atol=1.0e-10) for scale in scales[1:]):
        raise ValueError(f"Visual/collision scales differ in {urdf_path}: {scales}")
    scale_vec = scales[0]
    if not np.allclose(scale_vec, scale_vec[0], atol=1.0e-10) or scale_vec[0] <= 0.0:
        raise ValueError(f"Canonical builder requires positive uniform scale in {urdf_path}: {scale_vec}")
    mesh_path = next(iter(mesh_paths))
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    motion_path = bank / f"{clip_id}.npz"
    with np.load(motion_path, allow_pickle=False) as motion:
        quaternion = np.asarray(motion["object_quat_w"][:10], dtype=np.float64)
    matrices = _quat_to_matrix_wxyz(quaternion)
    # R_WO.T @ world_up equals the third row of R_WO.
    up_old_samples = matrices[:, 2, :]
    category = str(entry.get("mesh_physics_category", entry.get("object_category", "other")))
    return UrdfInfo(
        clip_id=clip_id,
        category=category,
        path=urdf_path,
        output_name=urdf_path.name,
        mesh_path=mesh_path,
        scale=float(scale_vec[0]),
        mass=mass,
        com_old_scaled=com,
        inertia_old=inertia,
        up_old_samples=up_old_samples,
    )


def _normalize(vector: np.ndarray, *, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-10:
        raise ValueError(f"Cannot normalize near-zero {label}: {vector}")
    return np.asarray(vector, dtype=np.float64) / norm


def _canonical_axes(inertia: np.ndarray, up_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    inertia = 0.5 * (np.asarray(inertia, dtype=np.float64) + np.asarray(inertia, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    scale = max(float(eigenvalues[-1]), 1.0e-12)
    if not np.all(eigenvalues > 0.0):
        raise ValueError(f"Inertia is not positive definite: {eigenvalues}")

    up = np.asarray(up_samples, dtype=np.float64).sum(axis=0)
    if np.linalg.norm(up) < 0.25:
        up = np.asarray(up_samples[0], dtype=np.float64)
    up = _normalize(up, label="reference up")
    z_index = int(np.argmax(np.abs(eigenvectors.T @ up)))
    z_value = float(eigenvalues[z_index])
    degenerate_z = [
        index
        for index, value in enumerate(eigenvalues)
        if abs(float(value) - z_value) / scale <= DEGENERACY_REL_TOL
    ]
    if len(degenerate_z) > 1:
        subspace = eigenvectors[:, degenerate_z]
        projected_up = subspace @ (subspace.T @ up)
        z_axis = _normalize(projected_up, label="up projection in degenerate eigenspace")
    else:
        z_axis = eigenvectors[:, z_index]
    if float(np.dot(z_axis, up)) < 0.0:
        z_axis = -z_axis

    legacy_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    projected_x = legacy_x - z_axis * float(np.dot(legacy_x, z_axis))
    if np.linalg.norm(projected_x) < 1.0e-6:
        legacy_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        projected_x = legacy_x - z_axis * float(np.dot(legacy_x, z_axis))
    plane_x = _normalize(projected_x, label="legacy x projection")
    plane_y = _normalize(np.cross(z_axis, plane_x), label="canonical plane y")
    plane_basis = np.column_stack((plane_x, plane_y))
    plane_inertia = plane_basis.T @ inertia @ plane_basis
    plane_values, plane_vectors = np.linalg.eigh(plane_inertia)
    xy_gap = abs(float(plane_values[1] - plane_values[0])) / scale
    if xy_gap <= DEGENERACY_REL_TOL:
        x_axis = plane_x
    else:
        # Lower moment about an axis usually corresponds to the longer geometric
        # direction in the perpendicular canonical plane.
        x_axis = _normalize(plane_basis @ plane_vectors[:, 0], label="canonical x")
        if float(np.dot(x_axis, plane_x)) < 0.0:
            x_axis = -x_axis
    y_axis = _normalize(np.cross(z_axis, x_axis), label="canonical y")
    x_axis = _normalize(np.cross(y_axis, z_axis), label="canonical x reorthogonalized")
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9) or np.linalg.det(rotation) < 0.999999:
        raise ValueError(f"Invalid canonical rotation:\n{rotation}")

    gaps = np.diff(eigenvalues) / scale
    if bool(np.all(gaps <= DEGENERACY_REL_TOL)):
        symmetry = "near_spherical_axes_ambiguous"
    elif bool(np.any(gaps <= DEGENERACY_REL_TOL)):
        symmetry = "near_axisymmetric_one_axis_ambiguous"
    else:
        symmetry = "asymmetric_principal_axes"
    alignment = math.degrees(math.acos(float(np.clip(np.dot(z_axis, up), -1.0, 1.0))))
    return rotation, eigenvalues, symmetry, alignment, xy_gap


def _build_canonical_mesh_contracts(infos: dict[str, UrdfInfo]) -> dict[Path, CanonicalMesh]:
    grouped: dict[Path, list[UrdfInfo]] = {}
    for info in infos.values():
        grouped.setdefault(info.mesh_path, []).append(info)
    contracts: dict[Path, CanonicalMesh] = {}
    for mesh_path, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        normalized_coms = [info.com_old_scaled / info.scale for info in group]
        normalized_inertias = [info.inertia_old / (info.mass * info.scale * info.scale) for info in group]
        reference_com = normalized_coms[0]
        reference_inertia = normalized_inertias[0]
        for info, com, inertia in zip(group[1:], normalized_coms[1:], normalized_inertias[1:]):
            if not np.allclose(com, reference_com, rtol=1.0e-5, atol=2.0e-6):
                raise ValueError(f"Shared mesh has inconsistent normalized COM: {mesh_path}, {info.clip_id}")
            if not np.allclose(inertia, reference_inertia, rtol=2.0e-4, atol=2.0e-7):
                raise ValueError(f"Shared mesh has inconsistent normalized inertia: {mesh_path}, {info.clip_id}")
        up_samples = np.concatenate([info.up_old_samples for info in group], axis=0)
        rotation, eigenvalues, symmetry, alignment, xy_gap = _canonical_axes(reference_inertia, up_samples)
        contracts[mesh_path] = CanonicalMesh(
            source_mesh=mesh_path,
            clip_ids=sorted(info.clip_id for info in group),
            categories=sorted({info.category for info in group}),
            origin_old_unscaled=reference_com,
            rotation_old_from_canonical=rotation,
            rotation_quat_wxyz=_matrix_to_quat_wxyz(rotation),
            inertia_eigenvalues_per_mass_unscaled=eigenvalues,
            symmetry=symmetry,
            z_up_alignment_deg=alignment,
            xy_moment_gap_relative=xy_gap,
        )
    return contracts


def _transform_obj(source: Path, temporary_output: Path, contract: CanonicalMesh) -> None:
    rotation_t = contract.rotation_old_from_canonical.T
    origin = contract.origin_old_unscaled
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    samples: list[np.ndarray] = []
    vertex_count = 0
    with source.open("r", encoding="utf-8", errors="strict") as reader, temporary_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line in reader:
            if line.startswith("v "):
                values = line.rstrip("\r\n").split()
                if len(values) < 4:
                    raise ValueError(f"Malformed OBJ vertex in {source}: {line!r}")
                vertex_old = np.asarray([float(values[1]), float(values[2]), float(values[3])], dtype=np.float64)
                vertex_new = rotation_t @ (vertex_old - origin)
                bounds_min = np.minimum(bounds_min, vertex_new)
                bounds_max = np.maximum(bounds_max, vertex_new)
                if len(samples) < 16:
                    samples.append(vertex_old)
                vertex_count += 1
                extras = "" if len(values) == 4 else " " + " ".join(values[4:])
                writer.write(f"v {_format_vec(vertex_new)}{extras}\n")
            elif line.startswith("vn "):
                values = line.rstrip("\r\n").split()
                if len(values) != 4:
                    raise ValueError(f"Malformed OBJ normal in {source}: {line!r}")
                normal_old = np.asarray([float(values[1]), float(values[2]), float(values[3])], dtype=np.float64)
                normal_new = rotation_t @ normal_old
                writer.write(f"vn {_format_vec(normal_new)}\n")
            else:
                writer.write(line if line.endswith(("\n", "\r")) else line + "\n")
    if vertex_count == 0:
        raise ValueError(f"OBJ contains no vertices: {source}")
    contract.vertex_count = vertex_count
    contract.canonical_bounds_min = bounds_min
    contract.canonical_bounds_max = bounds_max
    contract.sample_vertices_old = np.stack(samples, axis=0)
    contract.source_sha256 = _sha256_file(source)
    contract.output_sha256 = _sha256_file(temporary_output)
    contract.output_name = f"{contract.output_sha256}.obj"


def _rewrite_urdf(info: UrdfInfo, contract: CanonicalMesh, output: Path) -> None:
    tree = ET.parse(info.path)
    root = tree.getroot()
    link = root.find("link")
    assert link is not None
    inertial = link.find("inertial")
    assert inertial is not None
    origin = inertial.find("origin")
    if origin is None:
        origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", "0 0 0")
    inertia_node = inertial.find("inertia")
    assert inertia_node is not None
    inertia_new = contract.rotation_old_from_canonical.T @ info.inertia_old @ contract.rotation_old_from_canonical
    for key, value in {
        "ixx": inertia_new[0, 0],
        "ixy": inertia_new[0, 1],
        "ixz": inertia_new[0, 2],
        "iyy": inertia_new[1, 1],
        "iyz": inertia_new[1, 2],
        "izz": inertia_new[2, 2],
    }.items():
        inertia_node.set(key, f"{float(value):.17g}")
    for mesh_node in link.findall("./visual/geometry/mesh") + link.findall("./collision/geometry/mesh"):
        mesh_node.set("filename", f"../_mesh_assets/{contract.output_name}")
        mesh_node.set("scale", _format_vec((info.scale, info.scale, info.scale)))
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _transform_pose_velocity(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    *,
    offset_old: np.ndarray,
    rotation_quat_wxyz: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(position)
    quaternion_wxyz = np.asarray(quaternion_wxyz)
    linear_velocity = np.asarray(linear_velocity)
    angular_velocity = np.asarray(angular_velocity)
    mask = np.ones(position.shape[0], dtype=np.bool_) if valid_mask is None else np.asarray(valid_mask, dtype=np.bool_)
    out_position = position.copy()
    out_quaternion = quaternion_wxyz.copy()
    out_linear = linear_velocity.copy()
    out_angular = angular_velocity.copy()
    rotated_offset = _rotate_wxyz(quaternion_wxyz[mask], np.broadcast_to(offset_old, (int(mask.sum()), 3)))
    out_position[mask] = (position[mask].astype(np.float64) + rotated_offset).astype(position.dtype)
    out_quaternion[mask] = _quat_mul_wxyz(quaternion_wxyz[mask], rotation_quat_wxyz).astype(
        quaternion_wxyz.dtype
    )
    out_linear[mask] = (
        linear_velocity[mask].astype(np.float64)
        + np.cross(angular_velocity[mask].astype(np.float64), rotated_offset)
    ).astype(linear_velocity.dtype)
    return out_position, out_quaternion, out_linear, out_angular


def _canonical_size(info: UrdfInfo, contract: CanonicalMesh) -> np.ndarray:
    assert contract.canonical_bounds_min is not None and contract.canonical_bounds_max is not None
    return ((contract.canonical_bounds_max - contract.canonical_bounds_min) * info.scale).astype(np.float32)


def _transform_motion_npz(
    source: Path,
    output: Path,
    *,
    info: UrdfInfo,
    contract: CanonicalMesh,
    relative_urdf: str,
) -> dict[str, float]:
    with np.load(source, allow_pickle=False) as loaded:
        payload = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {"object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "object_size"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{source} omits object fields: {missing}")
    unknown_object_fields = sorted(
        key
        for key in payload
        if key.startswith("object_")
        and key
        not in {
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
            "object_ang_vel_w",
            "object_size",
            "object_name",
            "object_urdf_path",
        }
    )
    if unknown_object_fields:
        raise ValueError(f"Unhandled object-frame fields in {source}: {unknown_object_fields}")
    offset = info.com_old_scaled
    old_pos = payload["object_pos_w"]
    old_quat = payload["object_quat_w"]
    old_lin = payload["object_lin_vel_w"]
    old_ang = payload["object_ang_vel_w"]
    new_pos, new_quat, new_lin, new_ang = _transform_pose_velocity(
        old_pos,
        old_quat,
        old_lin,
        old_ang,
        offset_old=offset,
        rotation_quat_wxyz=contract.rotation_quat_wxyz,
    )
    payload["object_pos_w"] = new_pos
    payload["object_quat_w"] = new_quat
    payload["object_lin_vel_w"] = new_lin
    payload["object_ang_vel_w"] = new_ang
    payload["object_size"] = _canonical_size(info, contract)
    payload["object_urdf_path"] = np.asarray(relative_urdf)
    np.savez_compressed(output, **payload)

    assert contract.sample_vertices_old is not None
    samples_old = contract.sample_vertices_old
    samples_new = (contract.rotation_old_from_canonical.T @ (samples_old - contract.origin_old_unscaled).T).T
    sample_count = samples_old.shape[0]
    old_rot = _quat_to_matrix_wxyz(old_quat)
    new_rot = _quat_to_matrix_wxyz(new_quat)
    old_world = old_pos[:, None, :] + np.einsum(
        "tij,sj->tsi", old_rot, samples_old * info.scale
    )
    new_world = new_pos[:, None, :] + np.einsum(
        "tij,sj->tsi", new_rot, samples_new * info.scale
    )
    geometry_error = float(np.max(np.abs(old_world - new_world)))
    if geometry_error > 3.0e-6:
        raise ValueError(f"World geometry changed for {info.clip_id}: {geometry_error}")
    old_surface_velocity = old_lin[:, None, :] + np.cross(
        old_ang[:, None, :], old_world - old_pos[:, None, :]
    )
    new_surface_velocity = new_lin[:, None, :] + np.cross(
        new_ang[:, None, :], new_world - new_pos[:, None, :]
    )
    velocity_error = float(np.max(np.abs(old_surface_velocity - new_surface_velocity)))
    if velocity_error > 5.0e-5:
        raise ValueError(f"World surface velocity changed for {info.clip_id}: {velocity_error}")
    inertia_new = contract.rotation_old_from_canonical.T @ info.inertia_old @ contract.rotation_old_from_canonical
    frame_indices = np.unique(np.linspace(0, old_pos.shape[0] - 1, min(8, old_pos.shape[0]), dtype=np.int64))
    old_world_inertia = np.einsum(
        "tij,jk,tlk->til", old_rot[frame_indices], info.inertia_old, old_rot[frame_indices]
    )
    new_world_inertia = np.einsum(
        "tij,jk,tlk->til", new_rot[frame_indices], inertia_new, new_rot[frame_indices]
    )
    inertia_error = float(np.max(np.abs(old_world_inertia - new_world_inertia)))
    if inertia_error > 2.0e-7:
        raise ValueError(f"World inertia changed for {info.clip_id}: {inertia_error}")
    return {
        "world_geometry_max_abs_error_m": geometry_error,
        "world_surface_velocity_max_abs_error_mps": velocity_error,
        "world_inertia_max_abs_error_kg_m2": inertia_error,
        "verification_vertex_samples": int(sample_count),
    }


def _transform_xyzw_rollout_object_fields(
    payload: dict[str, np.ndarray],
    prefix: str,
    *,
    info: UrdfInfo,
    contract: CanonicalMesh,
    valid_mask: np.ndarray,
) -> None:
    names = (
        f"{prefix}object_pos_local",
        f"{prefix}object_quat_w",
        f"{prefix}object_lin_vel_w",
        f"{prefix}object_ang_vel_w",
    )
    if not all(name in payload for name in names):
        if any(name in payload for name in names):
            raise ValueError(f"Partial rollout object fields for prefix {prefix!r}: {names}")
        return
    quat_xyzw = payload[names[1]]
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    pos, quat, lin, ang = _transform_pose_velocity(
        payload[names[0]],
        quat_wxyz,
        payload[names[2]],
        payload[names[3]],
        offset_old=info.com_old_scaled,
        rotation_quat_wxyz=contract.rotation_quat_wxyz,
        valid_mask=valid_mask,
    )
    payload[names[0]] = pos
    payload[names[1]] = quat[..., [1, 2, 3, 0]].astype(quat_xyzw.dtype)
    payload[names[2]] = lin
    payload[names[3]] = ang


def _transform_teacher_reference(
    source: Path, output: Path, *, info: UrdfInfo, contract: CanonicalMesh
) -> None:
    with np.load(source, allow_pickle=False) as loaded:
        payload = {key: np.asarray(loaded[key]) for key in loaded.files}
    valid_mask = np.asarray(payload["valid_steps"], dtype=np.bool_)
    _transform_xyzw_rollout_object_fields(
        payload, "", info=info, contract=contract, valid_mask=valid_mask
    )
    _transform_xyzw_rollout_object_fields(
        payload, "target_", info=info, contract=contract, valid_mask=valid_mask
    )
    for stale_field in REMOVED_STALE_ROLLOUT_FIELDS:
        payload.pop(stale_field, None)
    np.savez_compressed(output, **payload)


def _transform_contact_points(source: Path, output: Path, *, info: UrdfInfo, contract: CanonicalMesh) -> None:
    points = np.load(source, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Contact points must be Nx3: {source} has {points.shape}")
    transformed = (
        contract.rotation_old_from_canonical.T
        @ (points.astype(np.float64) - info.com_old_scaled).T
    ).T.astype(points.dtype)
    np.save(output, transformed, allow_pickle=False)


def _copy_contact_export(
    source_root: Path,
    output_root: Path,
    *,
    infos: dict[str, UrdfInfo],
    contracts: dict[Path, CanonicalMesh],
    new_map: dict[str, dict[str, Any]],
) -> dict[str, int]:
    source_clips = source_root / "clips"
    if not source_clips.is_dir():
        raise FileNotFoundError(f"Contact export omits clips/: {source_root}")
    output_clips = output_root / "clips"
    output_clips.mkdir(parents=True)
    seen: set[str] = set()
    transformed_point_files = 0
    removed_rollout_arrays = 0
    for source_dir in sorted(path for path in source_clips.iterdir() if path.is_dir()):
        rollout_path = source_dir / "teacher_rollout_reference.npz"
        if not rollout_path.is_file():
            raise FileNotFoundError(rollout_path)
        with np.load(rollout_path, allow_pickle=False) as rollout:
            clip_id = str(np.asarray(rollout["clip_id"]).item())
            removed_rollout_arrays += sum(field in rollout.files for field in REMOVED_STALE_ROLLOUT_FIELDS)
        if clip_id not in infos or clip_id in seen:
            raise ValueError(f"Unexpected or duplicate contact clip {clip_id!r} at {source_dir}")
        seen.add(clip_id)
        info = infos[clip_id]
        contract = contracts[info.mesh_path]
        output_dir = output_clips / source_dir.name
        output_dir.mkdir()
        for source_file in sorted(path for path in source_dir.iterdir() if path.is_file()):
            output_file = output_dir / source_file.name
            if source_file.name == "teacher_rollout_reference.npz":
                _transform_teacher_reference(source_file, output_file, info=info, contract=contract)
            elif source_file.name.endswith(CONTACT_POINT_SUFFIX):
                _transform_contact_points(source_file, output_file, info=info, contract=contract)
                transformed_point_files += 1
            elif source_file.name == "metadata.json":
                metadata = json.loads(source_file.read_text(encoding="utf-8"))
                entry = new_map[clip_id]
                metadata["object_urdf_path"] = f"../../../{entry['object_urdf_path']}"
                metadata["object_mesh_path"] = f"../../../{entry['object_mesh_path']}"
                metadata["primitive_extents_xyz"] = entry["object_size"]
                metadata["teacher_rollout_motion_bank_path"] = f"../../../{clip_id}.npz"
                metadata["canonical_object_frame_contract"] = CONTRACT
                metadata["canonical_object_frame_source_clip"] = clip_id
                _write_json(output_file, metadata)
            else:
                shutil.copy2(source_file, output_file)
    if seen != set(infos):
        missing = sorted(set(infos).difference(seen))
        raise ValueError(f"Contact export missing clips: {missing}")
    return {
        "clip_count": len(seen),
        "transformed_contact_point_files": transformed_point_files,
        "removed_stale_actor_observation_arrays": removed_rollout_arrays,
    }


def _tree_metadata_snapshot(root: Path) -> str:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        info = path.lstat()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "kind": "symlink" if path.is_symlink() else "dir" if path.is_dir() else "file",
                "mode": stat.S_IMODE(info.st_mode),
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "link": os.readlink(path) if path.is_symlink() else None,
            }
        )
    return _sha256_bytes(_strict_json_bytes(records))


def _payload_records(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((path for path in root.rglob("*") if path.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        if path.is_symlink():
            raise ValueError(f"Output contains a symlink: {path}")
        records.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    return records


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise ValueError(f"Refusing to publish symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(source_bank: Path, output_root: Path, expected_clips: int) -> Path:
    source_bank = source_bank.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_bank.is_dir():
        raise FileNotFoundError(source_bank)
    source_snapshot_before = _tree_metadata_snapshot(source_bank)
    map_path = source_bank / "_clip_object_urdf_map.json"
    source_map_payload = json.loads(map_path.read_text(encoding="utf-8"))
    source_map = source_map_payload.get("clips")
    if not isinstance(source_map, dict):
        raise ValueError(f"Invalid clip map: {map_path}")
    motion_ids = {path.stem for path in source_bank.glob("*.npz")}
    if len(motion_ids) != expected_clips or set(source_map) != motion_ids:
        raise ValueError(
            f"Expected exact {expected_clips}-clip bank; motions={len(motion_ids)} map={len(source_map)}"
        )

    infos = {
        clip_id: _read_urdf_info(source_bank, clip_id, dict(source_map[clip_id]))
        for clip_id in sorted(motion_ids)
    }
    contracts = _build_canonical_mesh_contracts(infos)

    by_source = output_root / "by-source"
    by_source.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{CONTRACT}.staging-", dir=by_source))
    print(f"[INFO] staging={staging}", file=sys.stderr, flush=True)
    mesh_output = staging / "_mesh_assets"
    urdf_output = staging / "_single_slot_urdfs"
    mesh_output.mkdir()
    urdf_output.mkdir()

    for index, contract in enumerate(contracts.values(), start=1):
        temporary_mesh = mesh_output / f".{index:04d}.obj.tmp"
        _transform_obj(contract.source_mesh, temporary_mesh, contract)
        final_mesh = mesh_output / contract.output_name
        if final_mesh.exists():
            if _sha256_file(final_mesh) != contract.output_sha256:
                raise ValueError(f"Canonical mesh hash collision: {final_mesh}")
            temporary_mesh.unlink()
        else:
            temporary_mesh.rename(final_mesh)
        print(
            f"[INFO] mesh {index}/{len(contracts)} vertices={contract.vertex_count} "
            f"symmetry={contract.symmetry} source={contract.source_mesh.name}",
            file=sys.stderr,
            flush=True,
        )

    new_map: dict[str, dict[str, Any]] = {}
    clip_reports: list[dict[str, Any]] = []
    output_npz_hashes: dict[str, str] = {}
    for index, clip_id in enumerate(sorted(motion_ids), start=1):
        info = infos[clip_id]
        contract = contracts[info.mesh_path]
        relative_urdf = f"_single_slot_urdfs/{info.output_name}"
        relative_mesh = f"_mesh_assets/{contract.output_name}"
        _rewrite_urdf(info, contract, urdf_output / info.output_name)
        size = _canonical_size(info, contract)
        entry = dict(source_map[clip_id])
        entry.update(
            {
                "object_urdf_path": relative_urdf,
                "object_mesh_path": relative_mesh,
                "object_size": [float(value) for value in size],
                "canonical_object_frame_contract": CONTRACT,
                "canonical_origin": "physical_center_of_mass",
                "canonical_axis_semantics": {"x": "longer_horizontal_principal_axis", "y": "right_handed", "z": "principal_axis_closest_to_reference_world_up"},
                "canonical_symmetry": contract.symmetry,
            }
        )
        new_map[clip_id] = entry
        source_motion = source_bank / f"{clip_id}.npz"
        output_motion = staging / source_motion.name
        verification = _transform_motion_npz(
            source_motion,
            output_motion,
            info=info,
            contract=contract,
            relative_urdf=relative_urdf,
        )
        output_hash = _sha256_file(output_motion)
        output_npz_hashes[clip_id] = output_hash
        clip_reports.append(
            {
                "clip_id": clip_id,
                "category": info.category,
                "source_npz_sha256": _sha256_file(source_motion),
                "canonical_npz_sha256": output_hash,
                "source_urdf": info.path.relative_to(source_bank).as_posix(),
                "canonical_urdf": relative_urdf,
                "source_mesh_sha256": contract.source_sha256,
                "canonical_mesh_sha256": contract.output_sha256,
                "uniform_mesh_scale": info.scale,
                "origin_old_scaled_m": [float(value) for value in info.com_old_scaled],
                "rotation_old_from_canonical": contract.rotation_old_from_canonical.tolist(),
                "canonical_size_m": [float(value) for value in size],
                **verification,
            }
        )
        if index % 10 == 0 or index == len(motion_ids):
            print(f"[INFO] motion {index}/{len(motion_ids)}", file=sys.stderr, flush=True)

    new_map_payload = {
        key: value for key, value in source_map_payload.items() if key != "clips"
    }
    new_map_payload["clips"] = new_map
    new_map_payload["canonical_object_frame_contract"] = {
        "name": CONTRACT,
        "source_bank": str(source_bank),
        "source_map_sha256": _sha256_file(map_path),
        "origin": "physical_center_of_mass",
        "axes": {"x": "longer_horizontal_principal_axis", "y": "z_cross_x", "z": "principal_axis_closest_to_reference_world_up"},
        "degeneracy_relative_tolerance": DEGENERACY_REL_TOL,
    }
    _write_json(staging / "_clip_object_urdf_map.json", new_map_payload)

    contact_roots = [
        path
        for path in source_bank.iterdir()
        if path.is_dir() and path.name.startswith("contact_export")
    ]
    if len(contact_roots) != 1:
        raise ValueError(f"Expected exactly one contact export directory, found {contact_roots}")
    contact_report = _copy_contact_export(
        contact_roots[0],
        staging / contact_roots[0].name,
        infos=infos,
        contracts=contracts,
        new_map=new_map,
    )

    source_command_manifest_path = source_bank / "manifest.json"
    source_command_manifest = json.loads(source_command_manifest_path.read_text(encoding="utf-8"))
    command_manifest = dict(source_command_manifest)
    command_manifest["canonical_object_frame_contract"] = CONTRACT
    command_manifest["canonical_source_bank"] = str(source_bank)
    command_manifest["canonical_source_bank_manifest_sha256"] = _sha256_file(source_command_manifest_path)
    command_manifest["invariants"] = dict(command_manifest.get("invariants", {}))
    command_manifest["invariants"].pop("source_arrays_exactly_preserved", None)
    command_manifest["invariants"].update(
        {
            "non_object_frame_arrays_exactly_preserved": True,
            "world_geometry_preserved": True,
            "world_dynamics_preserved": True,
            "object_frame_fields_transformed_together": True,
        }
    )
    for clip in command_manifest.get("clips", []):
        clip_id = str(clip["clip_id"])
        clip["canonical_source_npz_sha256"] = clip.get("derived_npz_sha256")
        clip["derived_npz_sha256"] = output_npz_hashes[clip_id]

    marker = {
        "contract": CONTRACT,
        "source_bank": str(source_bank),
        "clip_count": len(motion_ids),
    }
    _write_json(staging / ".generated_by_canonical_object_frame_builder", marker)
    preliminary_records = _payload_records(
        staging,
        exclude={"manifest.json", "_canonical_object_frame_manifest.json"},
    )
    command_manifest["generated_records"] = preliminary_records
    command_manifest["derived_payload_digest"] = _sha256_bytes(_strict_json_bytes(preliminary_records))
    _write_json(staging / "manifest.json", command_manifest)

    payload_records = _payload_records(staging, exclude={"_canonical_object_frame_manifest.json"})
    payload_digest = _sha256_bytes(_strict_json_bytes(payload_records))
    max_errors = {
        key: max(float(report[key]) for report in clip_reports)
        for key in (
            "world_geometry_max_abs_error_m",
            "world_surface_velocity_max_abs_error_mps",
            "world_inertia_max_abs_error_kg_m2",
        )
    }
    mesh_reports = [
        {
            "source_mesh": str(contract.source_mesh),
            "source_sha256": contract.source_sha256,
            "canonical_mesh": f"_mesh_assets/{contract.output_name}",
            "canonical_sha256": contract.output_sha256,
            "clip_ids": contract.clip_ids,
            "categories": contract.categories,
            "vertex_count": contract.vertex_count,
            "origin_old_unscaled_m": [float(value) for value in contract.origin_old_unscaled],
            "rotation_old_from_canonical": contract.rotation_old_from_canonical.tolist(),
            "inertia_eigenvalues_per_mass_unscaled": [float(value) for value in contract.inertia_eigenvalues_per_mass_unscaled],
            "canonical_bounds_min_unscaled_m": [float(value) for value in contract.canonical_bounds_min],
            "canonical_bounds_max_unscaled_m": [float(value) for value in contract.canonical_bounds_max],
            "symmetry": contract.symmetry,
            "z_up_alignment_deg": contract.z_up_alignment_deg,
            "xy_moment_gap_relative": contract.xy_moment_gap_relative,
        }
        for contract in contracts.values()
    ]
    source_snapshot_after = _tree_metadata_snapshot(source_bank)
    if source_snapshot_after != source_snapshot_before:
        raise RuntimeError(
            "Source bank metadata changed during canonicalization; refusing publication: "
            f"before={source_snapshot_before} after={source_snapshot_after}"
        )
    manifest = {
        "schema_version": 1,
        "contract": CONTRACT,
        "source_bank": str(source_bank),
        "source_bank_directory_digest": source_bank.name,
        "source_bank_manifest_sha256": _sha256_file(source_command_manifest_path),
        "source_clip_map_sha256": _sha256_file(map_path),
        "source_tree_metadata_sha256_before": source_snapshot_before,
        "source_tree_metadata_sha256_after": source_snapshot_after,
        "source_bank_unchanged": True,
        "clip_count": len(motion_ids),
        "mesh_count": len(contracts),
        "category_counts": source_command_manifest.get("category_counts", {}),
        "coordinate_contract": {
            "origin": "physical_center_of_mass",
            "x_axis": "longer principal direction in canonical horizontal plane",
            "y_axis": "right_handed z cross x",
            "z_axis": "principal-inertia direction closest to mean reference world-up",
            "axis_sign_tie_break": "positive z toward reference world-up; positive x toward projected legacy +X",
            "near_degenerate_tie_break": "projected reference world-up and legacy +X",
            "degeneracy_relative_tolerance": DEGENERACY_REL_TOL,
            "quaternion_main_motion": "WXYZ",
            "quaternion_teacher_rollout_reference": "XYZW",
        },
        "transformed_fields": {
            "motion_npz": ["object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "object_size", "object_urdf_path"],
            "teacher_rollout_reference": ["object_*", "target_object_*"],
            "contact_sidecars": ["*_contact_points.npy"],
            "physics": ["mesh vertices/normals", "URDF inertial origin/tensor"],
        },
        "removed_stale_fields": {
            "teacher_rollout_reference": list(REMOVED_STALE_ROLLOUT_FIELDS),
            "reason": "opaque teacher observations encode the legacy object frame and cannot be relabeled as canonical",
        },
        "contact_report": contact_report,
        "verification": {
            "world_geometry_and_dynamics_checked_for_every_frame": True,
            "max_errors": max_errors,
            "source_bank_metadata_unchanged": True,
            "output_contains_symlinks": False,
        },
        "mesh_reports": mesh_reports,
        "clip_reports": clip_reports,
        "payload_record_count": len(payload_records),
        "payload_digest": payload_digest,
        "payload_records": payload_records,
    }
    _write_json(staging / "_canonical_object_frame_manifest.json", manifest)
    final_path = by_source / payload_digest
    if final_path.exists():
        raise FileExistsError(
            f"Canonical payload already exists at {final_path}; staging retained at {staging}"
        )
    staging.rename(final_path)
    _make_read_only(final_path)
    print(f"[INFO] published={final_path}", file=sys.stderr, flush=True)
    return final_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-clips", type=int, default=109)
    args = parser.parse_args()
    try:
        result = build(args.source_bank, args.output_root, args.expected_clips)
    except Exception as exc:
        print(f"[ERROR] canonical object-frame bank build failed: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
