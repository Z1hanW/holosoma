from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import trimesh

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[3]
VISER_SRC = REPO_ROOT / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.insert(0, str(VISER_SRC))

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_types.terrain import MeshType  # noqa: E402
from holosoma.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.camera_utils import resolve_camera_intrinsics  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.rotations import (  # noqa: E402
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_batched,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(cfg: ExperimentConfig) -> str:
    asset_root = _resolve_data_path(cfg.robot.asset.asset_root)
    urdf_path = os.path.join(asset_root, cfg.robot.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _depth_to_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=far, posinf=far, neginf=near)
    depth = np.clip(depth, near, far)
    denom = max(far - near, 1.0e-6)
    norm = (depth - near) / denom
    inv = 1.0 - norm
    img = (inv * 255.0).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2)


def _parse_vec3(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    parts = text.split()
    parts += ["0.0"] * max(0, 3 - len(parts))
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _load_terrain_mesh(cfg: ExperimentConfig) -> trimesh.Trimesh | None:
    terrain_cfg = cfg.terrain.terrain_term
    mesh_type = terrain_cfg.mesh_type
    mesh_type_value = mesh_type.value if isinstance(mesh_type, MeshType) else str(mesh_type)
    if mesh_type_value != "load_obj":
        return None
    if not terrain_cfg.obj_file_path:
        return None

    terrain_path = _resolve_data_path(terrain_cfg.obj_file_path)
    base = trimesh.load(terrain_path, process=False)
    if isinstance(base, trimesh.Scene):
        base = base.dump(concatenate=True)
    if not isinstance(base, trimesh.Trimesh):
        raise ValueError(f"Loaded terrain is not a trimesh: {type(base)}")

    num_rows = max(1, int(terrain_cfg.num_rows * terrain_cfg.scale_factor))
    num_cols = max(1, int(terrain_cfg.num_cols * terrain_cfg.scale_factor))
    if num_rows * num_cols <= 1:
        return base

    gap = 1e-4
    stride = (base.bounds[1] - base.bounds[0]) + gap
    tiles = []
    for r in range(num_rows):
        for c in range(num_cols):
            tile = base.copy()
            tile.apply_translation([c * stride[0], r * stride[1], 0.0])
            tiles.append(tile)
    return trimesh.util.concatenate(tiles)


def _resolve_camera_resolution(cfg) -> tuple[int, int]:
    width = cfg.camera_width if cfg.camera_width is not None else cfg.grid_size
    height = cfg.camera_height if cfg.camera_height is not None else cfg.grid_size
    return int(width), int(height)


def _build_default_joint_pos(robot_config: RobotConfig) -> tuple[np.ndarray, dict[str, float]]:
    defaults = robot_config.init_state.default_joint_angles
    joint_pos = np.zeros(len(robot_config.dof_names), dtype=np.float32)
    joint_dict: dict[str, float] = {}
    for idx, name in enumerate(robot_config.dof_names):
        val = float(defaults.get(name, 0.0))
        joint_pos[idx] = val
        joint_dict[name] = val
    return joint_pos, joint_dict


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def _parse_urdf_joints(urdf_path: str) -> tuple[str, dict[str, JointInfo]]:
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name") for link in root.findall("link") if link.get("name")}
    child_links: set[str] = set()
    child_to_joint: dict[str, JointInfo] = {}

    for joint in root.findall("joint"):
        joint_type = joint.get("type", "fixed")
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if parent_node is None or child_node is None:
            continue
        parent_link = parent_node.get("link")
        child_link = child_node.get("link")
        if not parent_link or not child_link:
            continue
        origin = joint.find("origin")
        xyz = _parse_vec3(origin.get("xyz") if origin is not None else None)
        rpy = _parse_vec3(origin.get("rpy") if origin is not None else None)
        axis_node = joint.find("axis")
        axis = _parse_vec3(axis_node.get("xyz") if axis_node is not None else None)

        child_links.add(child_link)
        child_to_joint[child_link] = JointInfo(
            name=joint.get("name", ""),
            joint_type=joint_type,
            parent=parent_link,
            child=child_link,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
        )

    roots = sorted(link for link in links if link not in child_links)
    if not roots:
        raise ValueError(f"No root link found in URDF: {urdf_path}")
    return roots[0], child_to_joint


class UrdfKinematics:
    def __init__(self, urdf_path: str) -> None:
        root_link, child_to_joint = _parse_urdf_joints(urdf_path)
        self.root_link = root_link
        self.child_to_joint = child_to_joint

    def compute_link_pose(
        self,
        link_name: str,
        joint_positions: dict[str, float],
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if link_name != self.root_link and link_name not in self.child_to_joint:
            raise ValueError(f"Link '{link_name}' not found in URDF tree rooted at {self.root_link}.")

        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def _pose(link: str) -> tuple[torch.Tensor, torch.Tensor]:
            if link == self.root_link:
                return root_pos, root_quat
            if link in cache:
                return cache[link]

            joint = self.child_to_joint[link]
            parent_pos, parent_quat = _pose(joint.parent)

            origin_pos = torch.tensor(joint.origin_xyz, dtype=torch.float32)
            origin_rpy = torch.tensor(joint.origin_rpy, dtype=torch.float32)
            origin_quat = quat_from_euler_xyz(origin_rpy[0], origin_rpy[1], origin_rpy[2])

            rel_pos = origin_pos
            rel_quat = origin_quat
            if joint.joint_type in {"revolute", "continuous"}:
                angle = torch.tensor(float(joint_positions.get(joint.name, 0.0)), dtype=torch.float32)
                axis = torch.tensor(joint.axis, dtype=torch.float32)
                if torch.linalg.norm(axis) < 1.0e-6:
                    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
                axis = axis / torch.linalg.norm(axis)
                motion_quat = quat_from_angle_axis(angle, axis, w_last=True)
                rel_quat = quat_mul(origin_quat.unsqueeze(0), motion_quat.unsqueeze(0), w_last=True).squeeze(0)
            elif joint.joint_type == "prismatic":
                displacement = float(joint_positions.get(joint.name, 0.0))
                axis = torch.tensor(joint.axis, dtype=torch.float32)
                if torch.linalg.norm(axis) < 1.0e-6:
                    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
                axis = axis / torch.linalg.norm(axis)
                disp_vec = axis * displacement
                rel_pos = origin_pos + quat_apply(
                    origin_quat.unsqueeze(0), disp_vec.unsqueeze(0), w_last=True
                ).squeeze(0)

            child_pos = parent_pos + quat_apply(parent_quat.unsqueeze(0), rel_pos.unsqueeze(0), w_last=True).squeeze(0)
            child_quat = quat_mul(parent_quat.unsqueeze(0), rel_quat.unsqueeze(0), w_last=True).squeeze(0)
            cache[link] = (child_pos, child_quat)
            return cache[link]

        return _pose(link_name)


def _build_camera_rays(
    width: int,
    height: int,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pitch_deg: float,
) -> torch.Tensor:
    u_coords = torch.arange(width, dtype=torch.float32)
    v_coords = torch.arange(height, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

    x = (u_grid - cx) / fx
    y = (v_grid - cy) / fy

    dirs_cam = torch.stack((torch.ones_like(x), -x, y), dim=-1)
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True).clamp(min=1.0e-6)
    dirs_cam = dirs_cam.view(-1, 3)

    pitch_rad = torch.deg2rad(torch.tensor(pitch_deg))
    pitch_quat = quat_from_euler_xyz(torch.tensor(0.0), pitch_rad, torch.tensor(0.0))
    pitch_quat = pitch_quat.unsqueeze(0).expand(dirs_cam.shape[0], -1)
    dirs_base = quat_rotate_inverse(pitch_quat, dirs_cam, w_last=True)
    return dirs_base / torch.norm(dirs_base, dim=-1, keepdim=True).clamp(min=1.0e-6)


def _raycast_depth(
    mesh: trimesh.Trimesh | None,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    depth = np.full(ray_origins.shape[0], max_distance, dtype=np.float32)
    if mesh is None:
        return depth

    try:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    locations, index_ray, _ = intersector.intersects_location(ray_origins, ray_dirs, multiple_hits=False)
    if len(index_ray) == 0:
        return depth

    delta = locations - ray_origins[index_ray]
    distances = np.einsum("ij,ij->i", delta, ray_dirs[index_ray])
    valid = distances > 0.0
    if np.any(valid):
        depth[index_ray[valid]] = distances[valid]
    return np.clip(depth, 0.0, max_distance)


def replay_perception(cfg: ExperimentConfig) -> None:
    if cfg.perception is None or not cfg.perception.enabled or cfg.perception.output_mode != "camera_depth":
        raise RuntimeError("Perception camera_depth is required for viser_perception.")

    width, height = _resolve_camera_resolution(cfg.perception)
    fx, fy, cx, cy, _, _ = resolve_camera_intrinsics(
        width,
        height,
        vfov_deg=cfg.perception.camera_vfov_deg,
        hfov_deg=cfg.perception.camera_hfov_deg,
        fx=cfg.perception.camera_fx,
        fy=cfg.perception.camera_fy,
        cx=cfg.perception.camera_cx,
        cy=cfg.perception.camera_cy,
    )

    camera_rays_base = _build_camera_rays(
        width,
        height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        pitch_deg=cfg.perception.camera_pitch_deg,
    )

    server = viser.ViserServer(port=int(os.environ.get("HOLOSOMA_VISER_PORT", "6060")))
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    robot_urdf_path = _resolve_robot_urdf_path(cfg)
    vr = ViserUrdf(server, urdf_or_path=Path(robot_urdf_path), root_node_name="/robot")

    camera_frame = server.scene.add_frame(
        "/robot/d435i",
        show_axes=True,
        axes_length=0.12,
        axes_radius=0.006,
        origin_color=(0, 200, 255),
    )

    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))

    terrain_mesh = _load_terrain_mesh(cfg)
    terrain_handle = None
    if terrain_mesh is not None:
        terrain_handle = server.scene.add_mesh_trimesh("/terrain", terrain_mesh)

    viser_joint_names = list(vr.get_actuated_joint_names())
    joint_pos, joint_dict = _build_default_joint_pos(cfg.robot)
    name_to_robot_idx = {name: idx for idx, name in enumerate(cfg.robot.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")

    viser_joints = np.array([joint_pos[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32)
    vr.update_cfg(viser_joints)

    root_pos = torch.tensor(cfg.robot.init_state.pos, dtype=torch.float32)
    rot_vals = cfg.robot.init_state.rot
    if len(rot_vals) == 4:
        root_quat = torch.tensor(rot_vals, dtype=torch.float32)
    elif len(rot_vals) == 3:
        root_quat = quat_from_euler_xyz(
            torch.tensor(rot_vals[0]),
            torch.tensor(rot_vals[1]),
            torch.tensor(rot_vals[2]),
        )
    else:
        raise ValueError("robot.init_state.rot must have 3 (rpy) or 4 (xyzw) values.")

    root_quat_np = root_quat.detach().cpu().numpy()
    robot_root.position = root_pos.detach().cpu().numpy()
    robot_root.wxyz = root_quat_np[[3, 0, 1, 2]]

    kinematics = UrdfKinematics(robot_urdf_path)
    camera_body = cfg.perception.camera_body_name or kinematics.root_link
    body_pos, body_quat = kinematics.compute_link_pose(camera_body, joint_dict, root_pos, root_quat)

    sensor_offset = torch.tensor(cfg.perception.sensor_offset, dtype=torch.float32)
    offset_world = quat_apply(body_quat.unsqueeze(0), sensor_offset.unsqueeze(0), w_last=True).squeeze(0)
    cam_pos = body_pos + offset_world

    pitch_rad = torch.deg2rad(torch.tensor(cfg.perception.camera_pitch_deg, dtype=torch.float32))
    pitch_quat = quat_from_euler_xyz(torch.tensor(0.0), pitch_rad, torch.tensor(0.0))
    cam_quat = quat_mul(body_quat.unsqueeze(0), pitch_quat.unsqueeze(0), w_last=True).squeeze(0)

    cam_pos_np = cam_pos.detach().cpu().numpy()
    cam_quat_np = cam_quat.detach().cpu().numpy()
    camera_frame.position = cam_pos_np
    camera_frame.wxyz = cam_quat_np[[3, 0, 1, 2]]

    ray_dirs_world = quat_rotate_batched(body_quat.unsqueeze(0), camera_rays_base.unsqueeze(0)).view(-1, 3)
    ray_starts = cam_pos.unsqueeze(0).expand(ray_dirs_world.shape[0], -1)
    depth = _raycast_depth(
        terrain_mesh,
        ray_starts.detach().cpu().numpy(),
        ray_dirs_world.detach().cpu().numpy(),
        max_distance=cfg.perception.max_distance,
    )
    depth_map = depth.reshape(height, width)

    depth_img = _depth_to_rgb(depth_map, cfg.perception.camera_near, cfg.perception.max_distance)
    depth_handle = server.gui.add_image(depth_img, label="D435i Depth")

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=True)
        show_terrain_cb = server.gui.add_checkbox("Show terrain", initial_value=terrain_handle is not None)

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)

    @show_terrain_cb.on_update
    def _(_evt) -> None:
        if terrain_handle is not None:
            terrain_handle.visible = bool(show_terrain_cb.value)

    depth_handle.image = depth_img

    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    while True:
        time.sleep(1.0)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_perception(tyro_cfg)


if __name__ == "__main__":
    main()
