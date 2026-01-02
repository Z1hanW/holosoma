import os
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

from .viser_visualizer import ViserHelper


def _parse_floats(text: Optional[str]) -> List[float]:
    return [float(x) for x in text.split()] if text else []


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def _quat_rotate_vec_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return (R @ v.reshape(3, 1)).reshape(3)


def _make_segment_mesh(trimesh_module, fromto, size, local_pos, local_quat, shape: str):
    if len(fromto) == 6 and len(size) >= 1:
        p1 = np.array(fromto[:3], dtype=np.float32)
        p2 = np.array(fromto[3:], dtype=np.float32)
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return None, local_pos, local_quat

        radius = float(size[0])
        if shape == "capsule":
            mesh = trimesh_module.creation.capsule(radius=radius, height=max(length, 1e-4))
        else:
            mesh = trimesh_module.creation.cylinder(radius=radius, height=max(length, 1e-4), sections=32)

        direction = vec / length
        axis = np.cross([0.0, 0.0, 1.0], direction)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-8:
            if np.dot([0.0, 0.0, 1.0], direction) > 0:
                local_quat = np.array([0, 0, 0, 1], dtype=np.float32)
            else:
                local_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            axis = axis / norm_axis
            angle = float(np.arccos(np.clip(np.dot([0.0, 0.0, 1.0], direction), -1.0, 1.0)))
            s = np.sin(angle / 2.0)
            local_quat = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)], dtype=np.float32)
        local_pos = (p1 + p2) * 0.5
        return mesh, local_pos, local_quat

    if len(size) >= 2:
        radius, half = float(size[0]), float(size[1])
        length = 2.0 * half
        if shape == "capsule":
            mesh = trimesh_module.creation.capsule(radius=radius, height=max(length, 1e-4))
        else:
            mesh = trimesh_module.creation.cylinder(radius=radius, height=max(length, 1e-4), sections=32)
        return mesh, local_pos, local_quat

    return None, local_pos, local_quat


class RobotMjcfViser:
    """Visualize a MuJoCo MJCF humanoid by parsing simple geoms."""

    def __init__(self, viser: ViserHelper, mjcf_path: str, body_names: Optional[List[str]]):
        self._viser = viser
        self._mjcf_path = mjcf_path
        self._body_names = body_names
        self._geom_specs: Dict[int, List[Tuple[str, np.ndarray, np.ndarray]]] = {}

        self._load_geoms()

    def _load_geoms(self):
        if not os.path.exists(self._mjcf_path):
            print(f"[RobotMjcfViser] MJCF not found: {self._mjcf_path}")
            return
        try:
            import trimesh
        except Exception as e:
            print(f"[RobotMjcfViser] trimesh not available: {e}")
            return

        tree = ET.parse(self._mjcf_path)
        root = tree.getroot()

        bodies: List[ET.Element] = []

        def visit(node: ET.Element):
            if node.tag == "body":
                bodies.append(node)
            for ch in list(node):
                visit(ch)

        world = root.find("worldbody")
        visit(world if world is not None else root)

        name_to_idx = {n: i for i, n in enumerate(self._body_names)} if self._body_names else None

        for enum_idx, body in enumerate(bodies):
            body_name = body.attrib.get("name", f"body_{enum_idx}")
            if name_to_idx is not None:
                if body_name not in name_to_idx:
                    continue
                body_idx = name_to_idx[body_name]
            else:
                body_idx = enum_idx

            for geom in body.findall("geom"):
                gtype = geom.attrib.get("type", "capsule")
                size = _parse_floats(geom.attrib.get("size"))
                lpos = (
                    np.array(_parse_floats(geom.attrib.get("pos")), dtype=np.float32)
                    if geom.attrib.get("pos")
                    else np.zeros(3, dtype=np.float32)
                )
                if geom.attrib.get("quat"):
                    q_wxyz = np.array(_parse_floats(geom.attrib.get("quat")), dtype=np.float32)
                    if q_wxyz.shape[0] == 4:
                        lquat = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
                    else:
                        lquat = np.array([0, 0, 0, 1], dtype=np.float32)
                else:
                    lquat = np.array([0, 0, 0, 1], dtype=np.float32)
                fromto = _parse_floats(geom.attrib.get("fromto")) if geom.attrib.get("fromto") else []

                mesh = None
                if gtype == "box" and len(size) == 3:
                    extents = 2 * np.array(size, dtype=np.float32)
                    mesh = trimesh.creation.box(extents=extents)
                elif gtype == "sphere" and len(size) >= 1:
                    mesh = trimesh.creation.icosphere(subdivisions=1, radius=size[0])
                elif gtype in ("capsule", "cylinder"):
                    mesh, lpos, lquat = _make_segment_mesh(trimesh, fromto, size, lpos, lquat, gtype)
                else:
                    continue

                if mesh is None:
                    continue

                name = f"/robot/{body_idx}/{len(self._geom_specs.get(body_idx, []))}"
                self._viser.add_mesh_simple(name, mesh.vertices, mesh.faces, color=(0.6, 0.9, 0.6))
                self._geom_specs.setdefault(body_idx, []).append((name, lpos, lquat))

    def update(self, body_pos_xy: np.ndarray, body_quat_xyzw: np.ndarray, world_offset: Optional[np.ndarray] = None):
        if not self._geom_specs:
            return
        if world_offset is None:
            world_offset = np.zeros(3, dtype=np.float32)

        for body_idx, geoms in self._geom_specs.items():
            if body_idx >= body_pos_xy.shape[0]:
                continue
            p_body = body_pos_xy[body_idx].astype(np.float32)
            q_body = body_quat_xyzw[body_idx].astype(np.float32)
            for name, local_pos, local_quat in geoms:
                q_world = _quat_mul_xyzw(q_body, local_quat)
                p_world = p_body + _quat_rotate_vec_xyzw(q_body, local_pos) + world_offset
                wxyz = np.array([q_world[3], q_world[0], q_world[1], q_world[2]], dtype=np.float32)
                self._viser.set_transform(name, p_world, wxyz)
