from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


TYPE_COLORS = {
    mujoco.mjtGeom.mjGEOM_MESH: np.array([70, 150, 255, 255], dtype=np.uint8),
    mujoco.mjtGeom.mjGEOM_SPHERE: np.array([255, 180, 70, 255], dtype=np.uint8),
    mujoco.mjtGeom.mjGEOM_CAPSULE: np.array([110, 220, 130, 255], dtype=np.uint8),
    mujoco.mjtGeom.mjGEOM_CYLINDER: np.array([255, 110, 110, 255], dtype=np.uint8),
    mujoco.mjtGeom.mjGEOM_BOX: np.array([190, 120, 255, 255], dtype=np.uint8),
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: np.array([120, 220, 220, 255], dtype=np.uint8),
    mujoco.mjtGeom.mjGEOM_PLANE: np.array([150, 150, 150, 255], dtype=np.uint8),
}


def _mesh_full_local_vertices(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    v0 = int(model.mesh_vertadr[mesh_id])
    nv = int(model.mesh_vertnum[mesh_id])
    return model.mesh_vert[v0 : v0 + nv].astype(np.float64, copy=True)


def _mesh_convex_hull_local_vf(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None, None

    vertices_full = _mesh_full_local_vertices(model, mesh_id)
    graph_adr = int(model.mesh_graphadr[mesh_id])

    if graph_adr != -1:
        graph = np.asarray(model.mesh_graph, dtype=np.int32)
        idx = graph_adr
        numvert = int(graph[idx])
        idx += 1
        numface = int(graph[idx])
        idx += 1
        idx += numvert
        idx += numvert
        idx += numvert + 3 * numface
        face_global = graph[idx : idx + 3 * numface].copy().reshape(numface, 3)
        used = np.unique(face_global.reshape(-1))
        vertices_hull = vertices_full[used]
        faces_hull = np.searchsorted(used, face_global).astype(np.int32)
        return vertices_hull, faces_hull

    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return None, None

    if vertices_full.shape[0] < 4:
        return None, None

    hull = ConvexHull(vertices_full)
    face_global = hull.simplices.astype(np.int32)
    used = np.unique(face_global.reshape(-1))
    vertices_hull = vertices_full[used]
    faces_hull = np.searchsorted(used, face_global).astype(np.int32)
    return vertices_hull, faces_hull


def _part_color(index: int, total: int) -> np.ndarray:
    if total <= 0:
        total = 1
    hue = (index / total + 0.61803398875) % 1.0
    sat = 0.65
    val = 0.95
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return np.array([int(255 * c) for c in rgb] + [255], dtype=np.uint8)


def _paint_mesh(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
    mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))
    mesh.visual.material = PBRMaterial(
        name=f"rgba_{int(color[0])}_{int(color[1])}_{int(color[2])}_{int(color[3])}",
        baseColorFactor=color.tolist(),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return mesh


def _build_local_mesh(model: mujoco.MjModel, geom_id: int, *, color: np.ndarray) -> trimesh.Trimesh | None:
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        vertices, faces = _mesh_convex_hull_local_vf(model, geom_id)
        if vertices is None or faces is None:
            return None
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(height=2.0 * float(size[1]), radius=float(size[0]))
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
    elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        base = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh = trimesh.Trimesh(vertices=base.vertices * size, faces=base.faces, process=False)
    elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        sx = float(size[0]) if size[0] > 0 else 5.0
        sy = float(size[1]) if size[1] > 0 else 5.0
        thickness = max(1e-3, 0.002 * max(sx, sy))
        mesh = trimesh.creation.box(extents=(2.0 * sx, 2.0 * sy, thickness))
    else:
        return None

    return _paint_mesh(mesh, color)


def _active_collision_geom_ids(model: mujoco.MjModel, *, include_ground: bool) -> list[int]:
    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not include_ground and "ground" in name.lower():
            continue
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        geom_ids.append(geom_id)
    return geom_ids


def _apply_pose(mesh: trimesh.Trimesh, data: mujoco.MjData, geom_id: int) -> trimesh.Trimesh:
    world_mesh = mesh.copy()
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
    transform[:3, 3] = data.geom_xpos[geom_id]
    world_mesh.apply_transform(transform)
    return world_mesh


def _geom_color(
    model: mujoco.MjModel,
    geom_id: int,
    *,
    color_mode: str,
    geom_ids: list[int],
) -> np.ndarray:
    if color_mode == "type":
        gtype = int(model.geom_type[geom_id])
        return TYPE_COLORS.get(gtype, np.array([220, 220, 220, 255], dtype=np.uint8))

    color_index = geom_ids.index(geom_id)
    return _part_color(color_index, len(geom_ids))


def _load_qpos(model: mujoco.MjModel, pose_file: Path | None, frame: int) -> np.ndarray:
    qpos = np.array(model.qpos0, dtype=float, copy=True)
    if pose_file is None:
        return qpos

    data = np.load(str(pose_file))
    if "qpos" not in data:
        raise KeyError(f"{pose_file} does not contain 'qpos'")
    qpos_seq = np.asarray(data["qpos"], dtype=float)
    if qpos_seq.ndim == 1:
        qpos = qpos_seq
    else:
        qpos = qpos_seq[frame]
    if qpos.shape[0] != model.nq:
        raise ValueError(f"Pose nq mismatch: got {qpos.shape[0]}, expected {model.nq}")
    return qpos


def export_collision_mesh(
    xml_path: Path,
    output_path: Path,
    *,
    pose_file: Path | None = None,
    frame: int = 0,
    include_ground: bool = False,
    color_mode: str = "part",
) -> tuple[int, dict[str, int]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = _load_qpos(model, pose_file, frame)
    mujoco.mj_forward(model, data)

    meshes: list[trimesh.Trimesh] = []
    mesh_names: list[str] = []
    counts: dict[str, int] = {}
    geom_ids = _active_collision_geom_ids(model, include_ground=include_ground)
    for geom_id in geom_ids:
        color = _geom_color(model, geom_id, color_mode=color_mode, geom_ids=geom_ids)
        local_mesh = _build_local_mesh(model, geom_id, color=color)
        if local_mesh is None:
            continue
        world_mesh = _apply_pose(local_mesh, data, geom_id)
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        geom_type = mujoco.mjtGeom(model.geom_type[geom_id]).name.replace("mjGEOM_", "").lower()
        counts[geom_type] = counts.get(geom_type, 0) + 1
        meshes.append(world_mesh)
        mesh_names.append(geom_name)

    if not meshes:
        raise RuntimeError(f"No active collision geoms found in {xml_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".glb", ".gltf"}:
        scene = trimesh.Scene()
        for mesh_name, mesh in zip(mesh_names, meshes):
            scene.add_geometry(mesh, node_name=mesh_name, geom_name=mesh_name)
        scene.export(str(output_path))
    else:
        merged = trimesh.util.concatenate(meshes)
        merged.export(str(output_path))
    return len(meshes), counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export active MuJoCo collision geoms as a single mesh.")
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=Path("src/holosoma_retargeting/models/g1/g1_29dof.xml"),
        help="MuJoCo XML to read.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("src/holosoma_retargeting/models/g1/g1_29dof_collision_primitives.ply"),
        help="Output mesh path. PLY/OBJ/GLB are all supported by trimesh.",
    )
    parser.add_argument(
        "--pose-file",
        type=Path,
        default=None,
        help="Optional .npz containing qpos. If omitted, exports the XML default pose.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index used when --pose-file stores a qpos sequence.")
    parser.add_argument("--include-ground", action="store_true", help="Include active ground geom in the export.")
    parser.add_argument(
        "--color-mode",
        choices=("part", "type"),
        default="part",
        help="Use a unique color per collision part/geom, or color only by primitive type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, by_type = export_collision_mesh(
        xml_path=args.xml_path,
        output_path=args.output_path,
        pose_file=args.pose_file,
        frame=args.frame,
        include_ground=args.include_ground,
        color_mode=args.color_mode,
    )
    print(f"Exported {count} active collision geoms from {args.xml_path} to {args.output_path}")
    print(f"By type: {by_type}")


if __name__ == "__main__":
    main()
