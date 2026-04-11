#!/usr/bin/env python3
"""Align G1 motions to paired terrain by lifting the whole trajectory in +Z.

For each motion clip:
- Use first-frame body points from ``body_pos_w[0]``.
- Cast vertical rays downward onto paired terrain ``<clip>.obj``.
- Compute minimum global lift ``dz`` so all valid sampled points are not below
  the terrain surface at their XY.
- Apply ``dz`` to the entire clip (joint/qpos/body/object world positions).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import mujoco  # type: ignore[import-not-found]


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type at {path}: {type(mesh)}")
    if mesh.vertices.shape[0] == 0:
        raise ValueError(f"Empty mesh vertices: {path}")
    return mesh


def _compute_required_lift(
    *,
    mesh: trimesh.Trimesh,
    body_points_first_frame: np.ndarray,
    clearance: float,
    top_margin: float,
) -> tuple[float, int]:
    points = np.asarray(body_points_first_frame, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"body points must be (N,3), got {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    if not np.any(finite_mask):
        return 0.0, 0
    points = points[finite_mask]

    top_z = float(np.max(mesh.vertices[:, 2])) + float(top_margin)
    origins = np.column_stack([points[:, 0], points[:, 1], np.full((points.shape[0],), top_z, dtype=np.float64)])
    directions = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float64), (points.shape[0], 1))

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, ray_index, _tri_index = intersector.intersects_location(origins, directions, multiple_hits=True)

    if locations.shape[0] == 0:
        return 0.0, 0

    hit_surface_z = np.full((points.shape[0],), -np.inf, dtype=np.float64)
    np.maximum.at(hit_surface_z, ray_index, locations[:, 2])
    valid_hit = np.isfinite(hit_surface_z)
    if not np.any(valid_hit):
        return 0.0, 0

    penetration = hit_surface_z[valid_hit] + float(clearance) - points[valid_hit, 2]
    dz = max(0.0, float(np.max(penetration)))
    return dz, int(np.count_nonzero(valid_hit))


def _apply_global_z_shift(payload: dict[str, np.ndarray], dz: float) -> None:
    if dz <= 0.0:
        return

    if "qpos" in payload:
        qpos = np.asarray(payload["qpos"])
        if qpos.ndim == 2 and qpos.shape[1] >= 3:
            qpos = qpos.copy()
            qpos[:, 2] += dz
            payload["qpos"] = qpos

    if "joint_pos" in payload:
        joint_pos = np.asarray(payload["joint_pos"])
        if joint_pos.ndim == 2 and joint_pos.shape[1] >= 3:
            joint_pos = joint_pos.copy()
            joint_pos[:, 2] += dz
            payload["joint_pos"] = joint_pos

    if "body_pos_w" in payload:
        body_pos_w = np.asarray(payload["body_pos_w"])
        if body_pos_w.ndim == 3 and body_pos_w.shape[2] >= 3:
            body_pos_w = body_pos_w.copy()
            body_pos_w[:, :, 2] += dz
            payload["body_pos_w"] = body_pos_w

    if "object_pos_w" in payload:
        object_pos_w = np.asarray(payload["object_pos_w"])
        if object_pos_w.ndim == 2 and object_pos_w.shape[1] >= 3:
            object_pos_w = object_pos_w.copy()
            object_pos_w[:, 2] += dz
            payload["object_pos_w"] = object_pos_w


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, **payload)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _collect_motion_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _first_frame_body_points(
    *,
    payload: dict[str, np.ndarray],
    mj_model: mujoco.MjModel | None,
    mj_data: mujoco.MjData | None,
) -> np.ndarray | None:
    if "body_pos_w" in payload:
        body_pos_w = np.asarray(payload["body_pos_w"], dtype=np.float64)
        if body_pos_w.ndim == 3 and body_pos_w.shape[0] > 0 and body_pos_w.shape[2] == 3:
            body_points = body_pos_w[0]
            if "body_names" in payload:
                body_names = [
                    name.decode("utf-8") if isinstance(name, bytes) else str(name)
                    for name in np.asarray(payload["body_names"]).reshape(-1)
                ]
                if len(body_names) == body_points.shape[0]:
                    keep = np.array([name.lower() != "world" for name in body_names], dtype=bool)
                    if np.any(keep):
                        body_points = body_points[keep]
            elif body_points.shape[0] > 1:
                # Most generated body_pos_w arrays put world body at index 0.
                body_points = body_points[1:]
            return body_points
        return None

    if "qpos" not in payload or mj_model is None or mj_data is None:
        return None

    qpos = np.asarray(payload["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] < mj_model.nq:
        return None

    mj_data.qpos[:] = qpos[0, : mj_model.nq]
    mujoco.mj_forward(mj_model, mj_data)
    body_points = np.asarray(mj_data.xpos, dtype=np.float64).copy()
    if body_points.shape[0] > 1:
        return body_points[1:]
    return body_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", required=True, help="Directory with motion npz files.")
    parser.add_argument("--geometry-dir", required=True, help="Directory with terrain obj files.")
    parser.add_argument("--out-dir", default="", help="Output motion directory. Empty means in-place.")
    parser.add_argument("--clearance", type=float, default=0.0, help="Extra clearance added above terrain.")
    parser.add_argument("--top-margin", type=float, default=5.0, help="Ray origin z margin above mesh max-z.")
    parser.add_argument(
        "--robot-xml",
        default="",
        help="Optional MuJoCo robot xml used when motion lacks body_pos_w and only has qpos.",
    )
    parser.add_argument(
        "--min-shift-eps",
        type=float,
        default=1e-6,
        help="Ignore tiny shifts below this threshold.",
    )
    parser.add_argument("--write", action="store_true", help="Actually write shifted files.")
    parser.add_argument("--report-json", default="", help="Optional JSON report output path.")
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir).resolve()
    geometry_dir = Path(args.geometry_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else motion_dir

    if not motion_dir.is_dir():
        raise FileNotFoundError(f"motion dir not found: {motion_dir}")
    if not geometry_dir.is_dir():
        raise FileNotFoundError(f"geometry dir not found: {geometry_dir}")

    motion_paths = sorted(motion_dir.glob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"no motion clips under: {motion_dir}")

    mj_model: mujoco.MjModel | None = None
    mj_data: mujoco.MjData | None = None
    if args.robot_xml:
        robot_xml = Path(args.robot_xml).resolve()
        if not robot_xml.is_file():
            raise FileNotFoundError(f"robot xml not found: {robot_xml}")
        mj_model = mujoco.MjModel.from_xml_path(str(robot_xml))
        mj_data = mujoco.MjData(mj_model)

    summary: list[dict[str, Any]] = []
    adjusted = 0
    processed = 0
    missing_geom = 0
    missing_body_or_qpos = 0
    max_shift = 0.0

    for motion_path in motion_paths:
        clip = motion_path.stem
        geom_path = geometry_dir / f"{clip}.obj"
        if not geom_path.is_file():
            missing_geom += 1
            summary.append(
                {"clip": clip, "status": "skip_missing_geometry", "motion_path": str(motion_path), "dz": 0.0}
            )
            continue

        payload = _collect_motion_payload(motion_path)
        body_points0 = _first_frame_body_points(payload=payload, mj_model=mj_model, mj_data=mj_data)
        if body_points0 is None:
            missing_body_or_qpos += 1
            summary.append(
                {
                    "clip": clip,
                    "status": "skip_missing_body_pos_w_and_qpos_fallback",
                    "motion_path": str(motion_path),
                    "dz": 0.0,
                }
            )
            continue

        mesh = _load_mesh(geom_path)
        dz, num_hits = _compute_required_lift(
            mesh=mesh,
            body_points_first_frame=body_points0,
            clearance=float(args.clearance),
            top_margin=float(args.top_margin),
        )
        if dz < float(args.min_shift_eps):
            dz = 0.0

        out_path = out_dir / motion_path.name
        status = "unchanged"
        if dz > 0.0:
            adjusted += 1
            max_shift = max(max_shift, dz)
            status = "adjusted"
            _apply_global_z_shift(payload, dz)

        if args.write:
            _write_npz_atomic(out_path, payload)

        processed += 1
        summary.append(
            {
                "clip": clip,
                "status": status,
                "dz": dz,
                "ray_hits": num_hits,
                "motion_path": str(motion_path),
                "geometry_path": str(geom_path),
                "output_path": str(out_path),
            }
        )

    result = {
        "motion_dir": str(motion_dir),
        "geometry_dir": str(geometry_dir),
        "out_dir": str(out_dir),
        "write": bool(args.write),
        "clips_total": len(motion_paths),
        "clips_processed": processed,
        "clips_adjusted": adjusted,
        "clips_missing_geometry": missing_geom,
        "clips_missing_body_pos_w_and_qpos_fallback": missing_body_or_qpos,
        "max_shift": float(max_shift),
        "clearance": float(args.clearance),
        "min_shift_eps": float(args.min_shift_eps),
        "items": summary,
    }

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "[INFO] total={clips_total} processed={clips_processed} adjusted={clips_adjusted} "
        "missing_geom={clips_missing_geometry} missing_body={clips_missing_body_pos_w_and_qpos_fallback} "
        "max_shift={max_shift:.6f} write={write}".format(**result)
    )

    top = sorted((item for item in summary if item.get("dz", 0.0) > 0.0), key=lambda x: x["dz"], reverse=True)[:10]
    if top:
        print("[INFO] top_shifts:")
        for item in top:
            print(f"  {item['clip']}: dz={item['dz']:.6f}")


if __name__ == "__main__":
    main()
