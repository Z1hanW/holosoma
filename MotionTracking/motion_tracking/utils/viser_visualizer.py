import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


class ViserHelper:
    """Thin wrapper over ``viser`` with helpers for meshes, lights, etc."""

    _global_servers: Dict[int, "ViserHelper"] = {}

    def __init__(self, port: int = 8080):
        self.port = int(port)
        self._server = None
        self._ok = False
        self._handles = {}
        self._init_server()

    @property
    def server(self):
        return self._server

    def ok(self) -> bool:
        return self._ok

    def _init_server(self):
        try:
            import viser  # type: ignore
        except Exception as exc:  # pragma: no cover
            print(f"[Viser] Not available ({exc})")
            return

        prev = ViserHelper._global_servers.get(self.port)
        if prev is not None and prev._server is not None:
            try:
                prev._server.stop()
            except Exception:
                pass
            ViserHelper._global_servers.pop(self.port, None)

        self._server = viser.ViserServer(port=self.port)
        ViserHelper._global_servers[self.port] = self
        self._ok = True

    def add_mesh_simple(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, float, float] = (0.6, 0.7, 0.9),
        side: str = "double",
        material: str = "standard",
        flat_shading: bool = False,
        cast_shadow: bool = True,
        receive_shadow: Union[bool, float] = True,
    ):
        if not self._ok:
            return
        handle = self._server.scene.add_mesh_simple(
            name,
            vertices.astype(np.float32),
            faces.astype(np.int32),
            color=color,
            side=side,
            material=material,
            flat_shading=flat_shading,
            cast_shadow=cast_shadow,
            receive_shadow=receive_shadow,
        )
        self._handles[name] = handle

    def replace_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, float, float],
        **kwargs,
    ):
        if not self._ok:
            return
        handle = self._handles.pop(name, None)
        if handle is not None and hasattr(handle, "remove"):
            try:
                handle.remove()
            except Exception:
                pass
        self.add_mesh_simple(name, vertices, faces, color=color, **kwargs)

    def set_transform(self, name: str, position: np.ndarray, wxyz: np.ndarray):
        if not self._ok:
            return
        handle = self._handles.get(name)
        if handle is None:
            return
        handle.position = position.astype(np.float32)
        handle.wxyz = wxyz.astype(np.float32)

    def update_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_size: float = 0.02,
    ):
        if not self._ok:
            return
        pts = points.astype(np.float32)
        handle = self._handles.get(name)
        if handle is None:
            color_arr = None
            if colors is not None:
                color_arr = np.asarray(colors, dtype=np.float32)
            try:
                handle = self._server.scene.add_point_cloud(
                    name,
                    points=pts,
                    colors=color_arr,
                    point_size=point_size,
                    precision="float32",
                )
            except TypeError:
                handle = self._server.scene.add_point_cloud(
                    name,
                    points=pts,
                    colors=color_arr,
                    point_size=point_size,
                )
            self._handles[name] = handle
        else:
            handle.points = pts
            if colors is not None:
                handle.colors = np.asarray(colors, dtype=np.float32)
            handle.point_size = point_size

    def update_line_segments(
        self,
        name: str,
        segments: np.ndarray,
        colors: Optional[np.ndarray] = None,
        line_width: float = 1.5,
    ):
        if not self._ok:
            return
        pts = segments.astype(np.float32)
        if name not in self._handles:
            handle = self._server.scene.add_line_segments(
                name,
                points=pts,
                colors=colors,
                line_width=line_width,
            )
            self._handles[name] = handle
        else:
            handle = self._handles[name]
            handle.points = pts
            if colors is not None:
                handle.colors = colors
            handle.line_width = line_width

    def add_line_segments(
        self,
        name: str,
        segments: np.ndarray,
        color: Tuple[float, float, float] = (0.6, 0.6, 0.6),
        line_width: float = 1.0,
    ):
        if not self._ok:
            return
        pts = segments.astype(np.float32)
        col = np.array(color, dtype=np.float32)
        colors = np.tile(col.reshape(1, 1, 3), (pts.shape[0], 2, 1))
        handle = self._server.scene.add_line_segments(
            name,
            points=pts,
            colors=colors,
            line_width=line_width,
        )
        self._handles[name] = handle

    def add_ambient_light(
        self,
        name: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        intensity: float = 0.3,
    ):
        if not self._ok:
            return
        self._server.scene.add_light_ambient(name, color=color, intensity=float(intensity))

    def add_directional_light(
        self,
        name: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        intensity: float = 1.0,
        cast_shadow: bool = True,
        wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        if not self._ok:
            return
        self._server.scene.add_light_directional(
            name,
            color=color,
            intensity=float(intensity),
            cast_shadow=cast_shadow,
            wxyz=wxyz,
        )

    def set_camera(self, position: np.ndarray, lookat: np.ndarray):
        if not self._ok or self._server is None:
            return
        for _, client in self._server.get_clients().items():
            client.camera.position = position.astype(np.float32)
            client.camera.look_at = lookat.astype(np.float32)


def add_ground_grid(
    viser: ViserHelper,
    width: float = 40.0,
    depth: float = 40.0,
    spacing: float = 1.0,
    height: float = 0.0,
    plane_color: Tuple[float, float, float] = (0.34, 0.34, 0.37),
    grid_color: Tuple[float, float, float] = (0.62, 0.62, 0.65),
) -> None:
    """Add a double-sided plane plus grid overlay in viser."""
    if not viser.ok():
        return

    half_w = float(width) * 0.5
    half_d = float(depth) * 0.5
    plane_verts = np.array(
        [
            [-half_w, -half_d, height],
            [half_w, -half_d, height],
            [half_w, half_d, height],
            [-half_w, half_d, height],
        ],
        dtype=np.float32,
    )
    plane_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    viser.add_mesh_simple(
        "/ground/plane",
        plane_verts,
        plane_faces,
        color=plane_color,
        side="double",
        flat_shading=True,
        cast_shadow=False,
        receive_shadow=True,
    )

    num_x = int(np.floor(width / max(spacing, 1e-4))) + 1
    num_y = int(np.floor(depth / max(spacing, 1e-4))) + 1
    segments = []
    z = height + 1e-4
    for ix in range(num_x):
        x = -half_w + ix * spacing
        segments.append([[x, -half_d, z], [x, half_d, z]])
    for iy in range(num_y):
        y = -half_d + iy * spacing
        segments.append([[-half_w, y, z], [half_w, y, z]])

    viser.add_line_segments(
        "/ground/grid",
        np.array(segments, dtype=np.float32),
        color=grid_color,
        line_width=1.0,
    )


def load_static_urdf(
    viser: ViserHelper,
    urdf_path: str,
    prefix: str = "/scene",
    offset: Optional[np.ndarray] = None,
) -> None:
    """Load URDF visuals once as static viser meshes."""
    if not viser.ok():
        return
    path = Path(os.path.expanduser(urdf_path))
    if not path.exists():
        print(f"[ViserScene] URDF not found: {path}")
        return

    try:
        import trimesh  # type: ignore
    except Exception as exc:
        print(f"[ViserScene] trimesh unavailable, skipping URDF scene: {exc}")
        return

    try:
        from scipy.spatial.transform import Rotation as sRot  # type: ignore
    except Exception as exc:
        print(f"[ViserScene] scipy unavailable, cannot parse URDF rotations: {exc}")
        return

    offset_vec = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)

    tree = ET.parse(str(path))
    root = tree.getroot()

    material_colors: Dict[str, Tuple[float, float, float]] = {}
    for material in root.findall("material"):
        mat_name = material.attrib.get("name")
        if not mat_name:
            continue
        color = _parse_rgba(material.attrib.get("rgba"))
        if color is not None:
            material_colors[mat_name] = color
            continue
        color_tag = material.find("color")
        if color_tag is not None:
            color = _parse_rgba(color_tag.attrib.get("rgba"))
            if color is not None:
                material_colors[mat_name] = color

    visuals = root.findall(".//visual")
    for idx, visual in enumerate(visuals):
        origin = visual.find("origin")
        xyz = np.zeros(3, dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if origin is not None:
            xyz_attr = origin.attrib.get("xyz")
            if xyz_attr:
                xyz = np.array([float(v) for v in xyz_attr.split()], dtype=np.float32)
            rpy_attr = origin.attrib.get("rpy")
            if rpy_attr:
                roll, pitch, yaw = [float(v) for v in rpy_attr.split()]
                quat_scipy = sRot.from_euler("xyz", [roll, pitch, yaw]).as_quat()
                quat = np.array(quat_scipy, dtype=np.float32)
        xyz = xyz + offset_vec

        color = (0.72, 0.72, 0.75)
        mat = visual.find("material")
        if mat is not None:
            color = _extract_material_color(mat, material_colors, default=color)

        geometry = visual.find("geometry")
        if geometry is None:
            continue
        mesh = _build_urdf_geometry_mesh(geometry, path.parent, trimesh)
        if mesh is None:
            continue

        name = f"{prefix}/object_{idx}"
        viser.add_mesh_simple(
            name,
            mesh.vertices,
            mesh.faces,
            color=color,
            flat_shading=True,
            cast_shadow=True,
            receive_shadow=True,
        )
        wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
        viser.set_transform(name, xyz, wxyz)


def _parse_rgba(text: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not text:
        return None
    vals = [float(v) for v in text.split()]
    if len(vals) >= 3:
        return tuple(vals[:3])
    return None


def _extract_material_color(
    material_node: ET.Element,
    material_map: Dict[str, Tuple[float, float, float]],
    default: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    rgba = _parse_rgba(material_node.attrib.get("rgba"))
    if rgba is not None:
        return rgba
    color_tag = material_node.find("color")
    if color_tag is not None:
        rgba = _parse_rgba(color_tag.attrib.get("rgba"))
        if rgba is not None:
            return rgba
    mat_name = material_node.attrib.get("name")
    if mat_name and mat_name in material_map:
        return material_map[mat_name]
    return default


def _build_urdf_geometry_mesh(geometry: ET.Element, asset_dir: Path, trimesh_module):
    if geometry.find("box") is not None:
        dims = np.array([float(v) for v in geometry.find("box").attrib.get("size", "").split()], dtype=np.float32)
        return trimesh_module.creation.box(extents=dims)
    if geometry.find("cylinder") is not None:
        cyl = geometry.find("cylinder")
        radius = float(cyl.attrib.get("radius", 0.5))
        length = float(cyl.attrib.get("length", 0.5))
        return trimesh_module.creation.cylinder(radius=radius, height=length, sections=48)
    if geometry.find("sphere") is not None:
        radius = float(geometry.find("sphere").attrib.get("radius", 0.5))
        return trimesh_module.creation.icosphere(subdivisions=2, radius=radius)
    mesh_tag = geometry.find("mesh")
    if mesh_tag is not None:
        filename = mesh_tag.attrib.get("filename")
        if not filename:
            return None
        scale_attr = mesh_tag.attrib.get("scale")
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if scale_attr:
            scale = np.array([float(v) for v in scale_attr.split()], dtype=np.float32)
        mesh_path = (asset_dir / filename).expanduser()
        if not mesh_path.exists():
            return None
        try:
            mesh = trimesh_module.load(str(mesh_path), force="mesh")
        except Exception:
            return None
        if mesh.is_empty:
            return None
        mesh.apply_scale(scale)
        return mesh
    return None
