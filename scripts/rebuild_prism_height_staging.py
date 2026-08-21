#!/usr/bin/env python3
"""Extract frame-0 mesh heights and rebuild a PRISM retarget staging tree."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


HEIGHT_SOURCE = "authoritative_grounded_incam_world_smpl_mesh_frame0_z_extent"
HEIGHT_FORMULA = "vertices_world[0,:,2].max() - vertices_world[0,:,2].min()"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def scalar_string(value: object) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def frame0_z_extent(vertices: np.ndarray) -> tuple[float, float, float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[0] == 0 or vertices.shape[2] != 3:
        raise ValueError(f"Expected vertices (T,V,3), got {vertices.shape}")
    z = vertices[0, :, 2]
    if z.size == 0 or not np.isfinite(z).all():
        raise ValueError("Frame-0 mesh Z coordinates must be finite and non-empty")
    minimum = float(z.min())
    maximum = float(z.max())
    height = maximum - minimum
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError(f"Invalid frame-0 mesh height: {height}")
    return minimum, maximum, height


def extract_manifest(mesh_root: Path, output: Path, expected_count: int | None) -> dict:
    mesh_paths = sorted(
        mesh_root.glob("gpu_*/prism_cf_*/hmr/solver_human_mesh_grounded.npz")
    )
    if not mesh_paths:
        mesh_paths = sorted(
            mesh_root.rglob("solver_human_mesh_grounded.npz")
        )
    rows = []
    seen = set()
    for mesh_path in mesh_paths:
        sequence = mesh_path.parents[1].name
        if sequence in seen:
            raise ValueError(f"Duplicate authoritative mesh for {sequence}")
        seen.add(sequence)
        with np.load(mesh_path, allow_pickle=True) as mesh:
            vertices = np.asarray(mesh["vertices_world"], dtype=np.float64)
            source_hmr = (
                scalar_string(mesh["source_hmr_results"])
                if "source_hmr_results" in mesh.files
                else ""
            )
        minimum, maximum, height = frame0_z_extent(vertices)
        rows.append(
            {
                "sequence": sequence,
                "human_height_m": height,
                "frame0_min_z_m": minimum,
                "frame0_max_z_m": maximum,
                "frame_count": int(vertices.shape[0]),
                "vertex_count": int(vertices.shape[1]),
                "source_mesh": str(mesh_path.resolve()),
                "source_hmr_results": source_hmr,
            }
        )
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} meshes, found {len(rows)}")
    rows.sort(key=lambda row: row["sequence"])
    heights = np.asarray([row["human_height_m"] for row in rows], dtype=np.float64)
    payload = {
        "created_at": utc_now(),
        "formula": HEIGHT_FORMULA,
        "coordinate_frame": (
            "support-plane-normalized Z-up grounded authoritative incam_world "
            "SMPL mesh"
        ),
        "source_root": str(mesh_root.resolve()),
        "sequence_count": len(rows),
        "summary": {
            "height_min_m": float(heights.min()),
            "height_median_m": float(np.median(heights)),
            "height_max_m": float(heights.max()),
        },
        "rows": rows,
    }
    atomic_json(output, payload)
    return payload


def pose_rotations(poses: np.ndarray, quaternion_order: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    quaternion = poses[:, :4]
    order = quaternion_order.strip().lower()
    if order == "wxyz":
        quaternion = np.concatenate(
            [quaternion[:, 1:4], quaternion[:, 0:1]], axis=1
        )
    elif order != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {quaternion_order}")
    return Rotation.from_quat(quaternion).as_matrix()


def full_mesh_bottom_z(
    vertices: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    bottoms = np.empty(len(rotations), dtype=np.float64)
    for start in range(0, len(rotations), 32):
        stop = min(start + 32, len(rotations))
        rotated_z = vertices @ rotations[start:stop, 2, :].T
        bottoms[start:stop] = (
            rotated_z.min(axis=0) + translations[start:stop, 2]
        )
    return bottoms


def reground_object_track(
    poses: np.ndarray,
    quaternion_order: str,
    local_vertices: np.ndarray,
    human_scale: float,
    *,
    ground_z: float,
    clearance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Raise Z only until source and retarget-space object meshes clear ground."""
    source = np.asarray(poses)
    vertices = np.asarray(local_vertices, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 7 or len(source) == 0:
        raise ValueError(f"Expected poses (T,7), got {source.shape}")
    if not np.isfinite(human_scale) or human_scale <= 0.0:
        raise ValueError(f"Invalid human scale: {human_scale}")

    rotations = pose_rotations(source, quaternion_order)
    source_translations = np.asarray(source[:, 4:7], dtype=np.float64)
    rotated_min_z = full_mesh_bottom_z(
        vertices, rotations, np.zeros_like(source_translations)
    )
    target_bottom_z = float(ground_z) + float(clearance_m)
    desired_frame0_center_z = target_bottom_z - rotated_min_z[0]
    source_required_z = target_bottom_z - rotated_min_z
    retarget_required_z = (
        desired_frame0_center_z
        + (
            target_bottom_z
            - rotated_min_z
            - desired_frame0_center_z
        )
        / float(human_scale)
    )
    requested_z = np.maximum.reduce(
        [source_translations[:, 2], source_required_z, retarget_required_z]
    )
    requested_z[0] = max(requested_z[0], desired_frame0_center_z)

    output = source.copy()
    output[:, 6] = requested_z.astype(source.dtype)
    for _ in range(16):
        stored_rotations = pose_rotations(output, quaternion_order)
        stored_translations = np.asarray(output[:, 4:7], dtype=np.float64)
        source_bottoms = full_mesh_bottom_z(
            vertices, stored_rotations, stored_translations
        )
        frame0_center_z = target_bottom_z - float(
            full_mesh_bottom_z(
                vertices,
                stored_rotations[:1],
                np.zeros((1, 3), dtype=np.float64),
            )[0]
        )
        staged_z = (
            stored_translations[:, 2]
            + frame0_center_z
            - stored_translations[0, 2]
        )
        retarget_z = (
            frame0_center_z
            + (staged_z - frame0_center_z) * float(human_scale)
        )
        retarget_translations = stored_translations.copy()
        retarget_translations[:, 2] = retarget_z
        retarget_bottoms = full_mesh_bottom_z(
            vertices, stored_rotations, retarget_translations
        )
        source_deficit = np.maximum(float(ground_z) - source_bottoms, 0.0)
        retarget_deficit = np.maximum(
            float(ground_z) - retarget_bottoms, 0.0
        )
        increment = np.maximum(
            source_deficit, retarget_deficit / float(human_scale)
        )
        violating = increment > 0.0
        if not np.any(violating):
            break
        old_z = output[violating, 6].copy()
        output[violating, 6] = (
            np.asarray(old_z, dtype=np.float64) + increment[violating]
        ).astype(source.dtype)
        unchanged = output[violating, 6] <= old_z
        if np.any(unchanged):
            replacement = output[violating, 6].copy()
            replacement[unchanged] = np.nextafter(
                replacement[unchanged],
                np.asarray(np.inf, dtype=source.dtype),
            )
            output[violating, 6] = replacement
    else:
        raise RuntimeError("Object grounding did not converge")

    rotations = pose_rotations(output, quaternion_order)
    translations = np.asarray(output[:, 4:7], dtype=np.float64)
    source_bottoms = full_mesh_bottom_z(vertices, rotations, translations)
    frame0_center_z = target_bottom_z - float(
        full_mesh_bottom_z(
            vertices,
            rotations[:1],
            np.zeros((1, 3), dtype=np.float64),
        )[0]
    )
    staged_z = translations[:, 2] + frame0_center_z - translations[0, 2]
    retarget_z = (
        frame0_center_z
        + (staged_z - frame0_center_z) * float(human_scale)
    )
    retarget_translations = translations.copy()
    retarget_translations[:, 2] = retarget_z
    retarget_bottoms = full_mesh_bottom_z(
        vertices, rotations, retarget_translations
    )
    if source_bottoms.min() < float(ground_z) - 1e-9:
        raise RuntimeError(f"Source object crosses ground: {source_bottoms.min()}")
    if retarget_bottoms.min() < float(ground_z) - 1e-9:
        raise RuntimeError(
            f"Retarget object crosses ground: {retarget_bottoms.min()}"
        )
    if not np.array_equal(output[:, :6], source[:, :6]):
        raise RuntimeError("Grounding changed quaternion or translation XY")
    lift = (
        np.asarray(output[:, 6], dtype=np.float64)
        - source_translations[:, 2]
    )
    return output, {
        "vertical_lift_m": lift,
        "source_bottom_z_after_m": source_bottoms,
        "predicted_retarget_bottom_z_after_m": retarget_bottoms,
        "modified_frame_count": int(np.count_nonzero(lift > 0.0)),
        "max_vertical_lift_m": float(lift.max(initial=0.0)),
    }


def update_world_contact_aliases(
    payload: dict[str, np.ndarray],
    poses: np.ndarray,
    quaternion_order: str,
) -> list[str]:
    rotations = pose_rotations(poses, quaternion_order)
    translations = np.asarray(poses[:, 4:7], dtype=np.float64)
    updated = []
    for prefix in ("object_contact_points", "palm_contact_points"):
        local_key = f"{prefix}_local"
        world_key = f"{prefix}_world"
        if local_key not in payload or world_key not in payload:
            continue
        local = np.asarray(payload[local_key], dtype=np.float64)
        original = np.asarray(payload[world_key])
        if local.shape[:2] != original.shape[:2] or len(local) != len(poses):
            raise ValueError(
                f"Incompatible {local_key}/{world_key}: "
                f"{local.shape}/{original.shape}"
            )
        payload[world_key] = (
            np.einsum("tij,tpj->tpi", rotations, local)
            + translations[:, None, :]
        ).astype(original.dtype)
        updated.append(world_key)
    return updated


def remap_string_array(
    array: np.ndarray, source_root: Path, output_root: Path
) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind not in {"U", "S", "O"}:
        return value.copy()
    source = str(source_root)
    destination = str(output_root)

    def remap(item: object) -> object:
        if isinstance(item, bytes):
            decoded = item.decode("utf-8")
            return decoded.replace(source, destination).encode("utf-8")
        if isinstance(item, str):
            return item.replace(source, destination)
        return item

    if value.shape == ():
        return np.asarray(remap(value.item()))
    flat = [remap(item) for item in value.reshape(-1)]
    return np.asarray(flat, dtype=value.dtype).reshape(value.shape)


def load_mesh_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [
            item
            for item in mesh.geometry.values()
            if isinstance(item, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry in {path}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(mesh)}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid mesh vertices from {path}: {vertices.shape}")
    return vertices


def clone_staging(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(output_root)
    try:
        shutil.copytree(source_root, output_root, copy_function=os.link)
    except OSError:
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.copytree(source_root, output_root)


def rebuild_staging(
    source_root: Path,
    height_manifest: Path,
    output_root: Path,
    *,
    robot_height_m: float,
    expected_count: int | None,
    reground_object: bool,
) -> dict:
    manifest = json.loads(height_manifest.read_text(encoding="utf-8"))
    if manifest.get("formula") != HEIGHT_FORMULA:
        raise ValueError(f"Unexpected height formula: {manifest.get('formula')}")
    heights = {
        row["sequence"]: row
        for row in manifest["rows"]
    }
    inputs = sorted(source_root.glob("*/input_for_retarget.npz"))
    if expected_count is not None and len(inputs) != expected_count:
        raise ValueError(f"Expected {expected_count} inputs, found {len(inputs)}")
    input_sequences = {path.parent.name for path in inputs}
    missing = sorted(input_sequences.difference(heights))
    if missing:
        raise ValueError(f"Missing mesh heights for: {', '.join(missing)}")

    clone_staging(source_root, output_root)
    created_at = utc_now()
    reports = []
    for source_input in inputs:
        sequence = source_input.parent.name
        output_input = output_root / sequence / source_input.name
        with np.load(output_input, allow_pickle=True) as data:
            payload = {
                key: remap_string_array(
                    np.asarray(data[key]), source_root, output_root
                )
                for key in data.files
            }
        old_height = float(np.asarray(payload["human_height_m"]).reshape(-1)[0])
        row = heights[sequence]
        height = float(row["human_height_m"])
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError(f"{sequence}: invalid height {height}")
        if abs(float(row["frame0_min_z_m"])) > 1e-6:
            raise ValueError(
                f"{sequence}: authoritative frame-0 mesh is not grounded: "
                f"{row['frame0_min_z_m']}"
            )
        mesh_ground = np.asarray(payload["human_mesh_min_z_after_m"])
        if mesh_ground.size == 0 or abs(float(mesh_ground.reshape(-1)[0])) > 1e-6:
            raise ValueError(f"{sequence}: staged frame-0 human mesh is not grounded")

        human_scale = float(robot_height_m) / height
        object_report = {
            "modified_frame_count": 0,
            "max_vertical_lift_m": 0.0,
        }
        updated_world_keys: list[str] = []
        if reground_object:
            poses = np.asarray(payload["object_poses"])
            quaternion_order = scalar_string(
                payload.get("object_pose_quat_order", np.asarray("xyzw"))
            )
            mesh_path = Path(scalar_string(payload["mesh_file"])).resolve()
            vertices = load_mesh_vertices(mesh_path)
            ground_z = float(
                np.asarray(
                    payload.get(
                        "allframe_object_grounded_ground_z_m",
                        np.asarray(0.0),
                    )
                ).reshape(-1)[0]
            )
            clearance = float(
                np.asarray(
                    payload.get(
                        "allframe_object_grounded_clearance_m",
                        np.asarray(1e-5),
                    )
                ).reshape(-1)[0]
            )
            new_poses, object_report = reground_object_track(
                poses,
                quaternion_order,
                vertices,
                human_scale,
                ground_z=ground_z,
                clearance_m=clearance,
            )
            payload["object_poses"] = new_poses
            if "object_pose" in payload:
                payload["object_pose"] = new_poses.copy()
            updated_world_keys = update_world_contact_aliases(
                payload, new_poses, quaternion_order
            )
            previous_lift = np.asarray(
                payload.get(
                    "allframe_object_grounded_vertical_lift_m",
                    np.zeros(len(new_poses)),
                ),
                dtype=np.float64,
            )
            incremental_lift = np.asarray(
                object_report["vertical_lift_m"], dtype=np.float64
            )
            payload["allframe_object_grounded_vertical_lift_m"] = (
                previous_lift + incremental_lift
            ).astype(np.float32)
            payload["allframe_object_grounded_source_bottom_z_m"] = np.asarray(
                object_report["source_bottom_z_after_m"], dtype=np.float32
            )
            payload[
                "allframe_object_grounded_predicted_retarget_bottom_z_m"
            ] = np.asarray(
                object_report["predicted_retarget_bottom_z_after_m"],
                dtype=np.float32,
            )
            payload["human_height_update_object_z_increment_m"] = (
                incremental_lift.astype(np.float32)
            )

        payload["human_height_previous_m"] = np.asarray(
            old_height, dtype=np.float32
        )
        payload["human_height_m"] = np.asarray(height, dtype=np.float32)
        payload["human_height_source"] = np.asarray(HEIGHT_SOURCE)
        payload["human_height_formula"] = np.asarray(HEIGHT_FORMULA)
        payload["human_height_reference_frame"] = np.asarray(
            "support_plane_normalized_zup_world"
        )
        payload["human_height_reference_frame_index"] = np.asarray(
            0, dtype=np.int32
        )
        payload["human_height_reference_mesh"] = np.asarray(
            str(row["source_mesh"])
        )
        payload["human_height_update_applied"] = np.asarray(1, dtype=np.int32)
        payload["human_height_updated_at"] = np.asarray(created_at)
        payload["allframe_object_grounded_human_scale"] = np.asarray(
            human_scale, dtype=np.float32
        )
        payload["human_height_update_world_point_keys"] = np.asarray(
            updated_world_keys
        )
        atomic_npz(output_input, payload)
        reports.append(
            {
                "sequence": sequence,
                "input": str(output_input),
                "old_height_m": old_height,
                "new_height_m": height,
                "height_delta_m": height - old_height,
                "old_human_scale": float(robot_height_m) / old_height,
                "new_human_scale": human_scale,
                "frame0_min_z_m": float(row["frame0_min_z_m"]),
                "frame0_max_z_m": float(row["frame0_max_z_m"]),
                "object_modified_frame_count": int(
                    object_report["modified_frame_count"]
                ),
                "object_max_vertical_lift_m": float(
                    object_report["max_vertical_lift_m"]
                ),
                "updated_world_point_keys": updated_world_keys,
            }
        )

    reports.sort(key=lambda row: row["sequence"])
    new_heights = np.asarray(
        [row["new_height_m"] for row in reports], dtype=np.float64
    )
    deltas = np.asarray(
        [row["height_delta_m"] for row in reports], dtype=np.float64
    )
    scales = np.asarray(
        [row["new_human_scale"] for row in reports], dtype=np.float64
    )
    report = {
        "status": "pass",
        "created_at": created_at,
        "source_staging_root": str(source_root),
        "output_staging_root": str(output_root),
        "height_manifest": str(height_manifest),
        "height_source": HEIGHT_SOURCE,
        "height_formula": HEIGHT_FORMULA,
        "robot_height_m": robot_height_m,
        "sequence_count": len(reports),
        "object_reground_applied": reground_object,
        "summary": {
            "height_min_m": float(new_heights.min()),
            "height_median_m": float(np.median(new_heights)),
            "height_max_m": float(new_heights.max()),
            "height_delta_min_m": float(deltas.min()),
            "height_delta_median_m": float(np.median(deltas)),
            "height_delta_max_m": float(deltas.max()),
            "human_scale_min": float(scales.min()),
            "human_scale_median": float(np.median(scales)),
            "human_scale_max": float(scales.max()),
            "object_modified_sequence_count": int(
                np.count_nonzero(
                    [row["object_modified_frame_count"] for row in reports]
                )
            ),
            "object_max_vertical_lift_m": float(
                max(
                    row["object_max_vertical_lift_m"]
                    for row in reports
                )
            ),
        },
        "rows": reports,
    }
    atomic_json(output_root / "height_update_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--mesh-root", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--expected-count", type=int)

    rebuild = subparsers.add_parser("rebuild")
    rebuild.add_argument("--source-staging-root", type=Path, required=True)
    rebuild.add_argument("--height-manifest", type=Path, required=True)
    rebuild.add_argument("--output-staging-root", type=Path, required=True)
    rebuild.add_argument("--robot-height-m", type=float, default=1.32)
    rebuild.add_argument("--expected-count", type=int)
    rebuild.add_argument("--skip-object-reground", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "extract":
        payload = extract_manifest(
            args.mesh_root.expanduser().resolve(),
            args.output.expanduser().resolve(),
            args.expected_count,
        )
    else:
        payload = rebuild_staging(
            args.source_staging_root.expanduser().resolve(),
            args.height_manifest.expanduser().resolve(),
            args.output_staging_root.expanduser().resolve(),
            robot_height_m=float(args.robot_height_m),
            expected_count=args.expected_count,
            reground_object=not args.skip_object_reground,
        )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
