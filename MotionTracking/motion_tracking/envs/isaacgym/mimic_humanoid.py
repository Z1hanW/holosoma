from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import xml.etree.ElementTree as ET

from isaac_utils import torch_utils
from motion_tracking.envs.isaacgym.disc_humanoid import DiscHumanoid
from motion_tracking.envs.common.common_mimic import BaseMimic
from motion_tracking.utils.motion_lib import MotionLib
from typing import Optional
from scipy.spatial.transform import Rotation as sRot
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
import time
from motion_tracking.utils.calculate_voxel import read_urdf
from motion_tracking.utils.urdfpoints import save_points
import os
from motion_tracking.utils.viser_visualizer import ViserHelper, add_ground_grid
from motion_tracking.utils.robot_viser import RobotMjcfViser

class MimicHumanoid(BaseMimic, DiscHumanoid):
    def __init__(
        self, config, device: torch.device, motion_lib: Optional[MotionLib] = None
    ):
        self.scene=True
        if config.scene_file == 'abc':
            self.scene=False
        self.pointcloud = None
        self._scene_asset_loaded = False
        self._urdf_vis_vertices = None
        self._urdf_vis_triangles = None
        self._urdf_vis_added = False
        self._urdf_point_samples = None
        
        self.save_state_path = config["save_state_path"]
        
        self._use_triangle_mesh = False
        self._use_scene_point_obs = getattr(config, "force_point_cloud_obs", False) or getattr(config, "seperate_point_goal", False)
        if "mesh" in config.scene_file:
            self._use_triangle_mesh = True
            print("Using triangle mesh tiling for scene")
        # Viser state (initialize early so robot can be visualized even without triangle mesh)
        self._viser = None
        self._viser_ok = False
        self._tm_viz_built = False
        self._robot_vis = None
        if self.save_state_path is not None:
            self.pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
            self._body_names_orig = SMPL_MUJOCO_NAMES
            self.mujoco_2_smpl = [self._body_names_orig.index(q) for q in SMPL_BONE_ORDER_NAMES if q in self._body_names_orig]
            self.save_imitation_result = {"pose_aa":[], "trans":[]}
            self.start_record_pose = False

        super().__init__(config, device, motion_lib=motion_lib)
        # Start viser server upfront if requested (override base-class default when testing single env)
        single_env = getattr(config, "num_envs", 0)
        if single_env == 1:
            self.use_viser = True
        if getattr(self, "use_viser", False) and self._viser is None:
            try:
                self._viser = ViserHelper(port=8080)
                self._viser_ok = self._viser.ok()
                if self._viser_ok:
                    print(f"[Viser] Server started (port=8080), triangle_mesh={self._use_triangle_mesh}")
                    add_ground_grid(self._viser)
            except Exception as _e:
                print(f"[Viser] Failed to start server: {_e}")

        if not self.headless and self.config.provide_future_states and not self.config.sync_motion and self.config.visualize_markers:
            self._build_marker_state_tensors()

        self.num_closest_point = self.config.num_obs_num_point
        if self.scene and self._use_scene_point_obs:
            self._prepare_scene_observations(device)
            # import pdb;pdb.set_trace()
        

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless and self.config.provide_future_states and not self.config.sync_motion and self.config.visualize_markers:
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()
        if self.scene:
            self._scene_handles = []
            self._load_scene_asset()

        super().create_envs(num_envs, spacing, num_per_row)

        # When using triangle mesh tiling, override env_offsets to align
        # each env with its assigned mesh tile so global conversions work.
        if self.scene and self._use_triangle_mesh:
            # Ensure tile layout params exist (set in _load_scene_asset)
            if not hasattr(self, "_tm_rows"):
                # Fallback defaults
                self._tm_rows = 2
                self._tm_cols = 2
                self._tm_spacing_scale = 1.1
                # Safe extent if not computed
                self._tm_extent_xy = np.array([spacing, spacing], dtype=np.float32)

            step_x = float(self._tm_extent_xy[0] * self._tm_spacing_scale)
            step_y = float(self._tm_extent_xy[1] * self._tm_spacing_scale)

            # Precompute tile origins (global positions)
            tile_origins = []
            for r in range(self._tm_rows):
                for c in range(self._tm_cols):
                    tile_origins.append([c * step_x, r * step_y, 0.0])
            tile_origins = torch.tensor(tile_origins, dtype=torch.float, device=self.device)
            num_tiles = tile_origins.shape[0]

            # Assign envs to tiles in round-robin to distribute evenly
            env_ids = torch.arange(self.num_envs, device=self.device)
            tile_ids = torch.remainder(env_ids, num_tiles)
            assigned_tile_offsets = tile_origins[tile_ids]

            # Capture Isaac Gym env origins
            env_origins = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                env_origins.append([origin.x, origin.y, origin.z])
            env_origins = torch.tensor(env_origins, dtype=torch.float, device=self.device)

            # Store origin and per-env tile offsets separately
            # - env_offsets: IsaacGym env origins (for local<->global)
            # - _tm_env_tile_offsets: additional global shift to align with a tile
            self.env_offsets[:, :] = 0.0
            self.env_offsets[:, :3] = env_origins
            self._tm_env_tile_offsets = assigned_tile_offsets

    
    def _load_marker_asset(self):
        asset_root = "motion_tracking/data/assets/mjcf/"
        asset_file = "traj_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)

        if not self.headless and self.config.provide_future_states and not self.config.sync_motion and self.config.visualize_markers:
            self._build_marker(env_id, env_ptr)
        if self.scene:
            self._build_scene(env_id, env_ptr)
    
    def _load_scene_asset(self):
        """Load either URDF asset or OBJ triangle mesh for tiling.

        - If self._use_triangle_mesh is False: keep URDF asset behavior.
        - If True: load OBJ once and precompute bounds for grid duplication.
        """
        
        if not self._use_triangle_mesh: 
            asset_root = "motion_tracking/data/assets/urdf/"
            asset_file = self.config.scene_file + ".urdf"
            # import pdb;pdb.set_trace()
            asset_options = gymapi.AssetOptions()
            # if any(k in asset_file for k in ["emdb", "chair", "nksr", "coacd", "vdb", "ours"]):
                
            asset_options.vhacd_enabled = False
            # asset_options.vhacd_params.resolution = 1000000
            # asset_options.vhacd_params.max_convex_hulls = 200
            # asset_options.vhacd_params.concavity = 0.002
            asset_options.vhacd_params.resolution = 500000  # Increase for more detail
            asset_options.vhacd_params.max_convex_hulls = 256  # More hulls for complex shapes
            asset_options.vhacd_params.max_num_vertices_per_ch = 128  # More vertices per hull
                # asset_options.vhacd_params.concavity = 0.001  # Tighter fit to mesh
                # asset_options.vhacd_params.min_volume_per_ch = 0.00001  # Allow smaller hulls
                # asset_options.vhacd_params.plane_downsampling = 2  # Finer clipping plane search
                # asset_options.vhacd_params.convex_hull_downsampling = 2  # More precise hulls

                # asset_options.vhacd_params.alpha = 0.01  # Less symmetry bias
                # asset_options.vhacd_params.beta = 0.01  # Less revolution axis bias
                # asset_options.vhacd_params.pca = 1  # Normalize mesh
                # asset_options.vhacd_params.convex_hull_approximation = False  # More accurate hulls
                
            # asset_options.max_angular_velocity = 100.0
            # asset_options.angular_damping = 0.01
            # asset_options.linear_damping = 0.01
            # 
            asset_options.density = 1000
            asset_options.fix_base_link = True
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            self._scene_asset = self.gym.load_asset(self.sim, asset_root,
                                                    asset_file, asset_options)
            urdf_path = os.path.join(asset_root, asset_file)
            self._build_urdf_vis_mesh(urdf_path)
            self._scene_asset_loaded = True
        
        else:
            def load_obj_file(filepath):
                """Parse OBJ file and extract vertices and faces"""
                vertices = []
                faces = []
                
                with open(filepath, 'r') as file:
                    for line in file:
                        if line.startswith('v '):  # Vertex
                            parts = line.strip().split()
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            vertices.append(vertex)
                        elif line.startswith('f '):  # Face
                            parts = line.strip().split()
                            face = []
                            for i in range(1, len(parts)):
                                vertex_idx = int(parts[i].split('/')[0]) - 1
                                face.append(vertex_idx)
                            # Triangulate if needed (for quads)
                            if len(face) == 4:
                                faces.append([face[0], face[1], face[2]])
                                faces.append([face[0], face[2], face[3]])
                            else:
                                faces.append(face[:3])
                
                return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)
            
            # Resolve OBJ path from scene_file first, fallback to legacy hardcoded path
            candidate = f"motion_tracking/data/assets/urdf/{self.config.scene_file}.obj"
            
            if not os.path.exists(candidate):
                candidate = f"motion_tracking/data/assets/urdf/0911/36_outdoor_long_walk/scene_mesh_nksr_rot.obj"
            self._scene_vertices, self._scene_triangles = load_obj_file(candidate)
            if self._scene_vertices.size > 0 and self._scene_triangles.size > 0:
                try:
                    self._viser.add_mesh_simple(
                        "/scene_triangle_mesh",
                        self._scene_vertices.astype(np.float32),
                        self._scene_triangles.astype(np.int32),
                        color=(0.6, 0.7, 0.9),
                        side="double",
                    )
                    self._urdf_vis_added = True
                except Exception as _e_mesh:
                    print(f"[Viser] triangle mesh viz skipped: {_e_mesh}")
            self._scene_asset_loaded = True

            # Precompute XY bounds for spacing
            if self._scene_vertices.size > 0:
                verts_np = self._scene_vertices.reshape(-1, 3)
                min_xy = verts_np[:, :2].min(axis=0)
                max_xy = verts_np[:, :2].max(axis=0)
                self._tm_extent_xy = (max_xy - min_xy).astype(np.float32)
            else:
                self._tm_extent_xy = np.array([1.0, 1.0], dtype=np.float32)

            # Grid defaults (can be overridden via config)
            if not self.headless:
                self._tm_rows = int(getattr(self.config, "triangle_mesh_rows", 1))
                self._tm_cols = int(getattr(self.config, "triangle_mesh_cols", 1))
                self._tm_spacing_scale = float(getattr(self.config, "triangle_mesh_spacing_scale", 1.1))
            else:
                self._tm_rows = int(getattr(self.config, "triangle_mesh_rows", 8))
                self._tm_cols = int(getattr(self.config, "triangle_mesh_cols", 8))
                self._tm_spacing_scale = float(getattr(self.config, "triangle_mesh_spacing_scale", 1.1))
            self._tm_built_once = False
        
        return

    def _build_scene(self, env_id, env_ptr):
        """Modified version to add triangle mesh instead of actor"""
        
        
        if self._use_triangle_mesh:
            # Build tiled meshes only once, on the first env
            if env_id == 0 and not getattr(self, "_tm_built_once", False):
                step_x = float(self._tm_extent_xy[0] * self._tm_spacing_scale)
                step_y = float(self._tm_extent_xy[1] * self._tm_spacing_scale)

                triangles_uint32 = self._scene_triangles.astype(np.uint32)

                for r in range(self._tm_rows):
                    for c in range(self._tm_cols):
                        tm_params = gymapi.TriangleMeshParams()
                        tm_params.nb_vertices = len(self._scene_vertices)
                        tm_params.nb_triangles = len(triangles_uint32)
                        tm_params.transform = gymapi.Transform()
                        tm_params.transform.p = gymapi.Vec3(c * step_x, r * step_y, 0.0)
                        tm_params.transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

                        self.gym.add_triangle_mesh(
                            self.sim,
                            self._scene_vertices.flatten(),
                            triangles_uint32.flatten(),
                            tm_params
                        )

                # Cache tile origins for later use in create_envs
                self._tm_step_x = step_x
                self._tm_step_y = step_y
                self._tm_built_once = True
                self._scene_handles.append(None)

                # Also visualize a single triangle mesh tile in Viser if enabled and available
                if getattr(self, "use_viser", False):
                    try:
                        if self._viser is None:
                            self._viser = ViserHelper(port=8080)
                            self._viser_ok = self._viser.ok()
                        if self._viser_ok and not self._tm_viz_built:
                            base_verts = self._scene_vertices.astype(np.float32)
                            faces = triangles_uint32.astype(np.int32)
                            off = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            verts = (base_verts + off[None, :]).astype(np.float32)
                            self._viser.add_mesh_simple("/scene_tile_0_0", verts, faces, color=(0.6, 0.7, 0.9), side="double")
                            self._tm_viz_built = True
                            # Initialize robot MJCF viser if available
                            mjcf_path = os.path.join(self.config.asset.asset_root, self.config.asset.asset_file_name)
                            try:
                                body_names = getattr(self, 'body_names', None)
                            except Exception:
                                body_names = None
                            self._robot_vis = RobotMjcfViser(self._viser, mjcf_path, body_names)
                    except Exception as _e:
                        print(f"[Viser] Skipping mesh viz: {_e}")
            
        else:
            # Original Method 1: Create actor from asset
            # print(f"Building scene for env {env_id} using asset")
            
            col_group = env_id
            col_filter = 2
            segmentation_id = 1
            default_pose = gymapi.Transform()

            scene_handle = self.gym.create_actor(
                env_ptr,
                self._scene_asset,
                default_pose,
                "scene",
                col_group,
                col_filter,
                segmentation_id,
            )

            self.gym.set_rigid_body_color(env_ptr, scene_handle, 0,
                                        gymapi.MESH_VISUAL,
                                        gymapi.Vec3(0.25, 0.41, 0.88))
            self._scene_handles.append(scene_handle)
        
        return

    def _prepare_scene_observations(self, device: torch.device):
        """Build per-environment tensors for point-cloud and voxel observations."""
        if not self._scene_asset_loaded and not self._use_triangle_mesh:
            print(f"[MimicHumanoid] Scene asset for {self.config.scene_file} not loaded; skipping point cloud init.")
            return

        asset_root = "motion_tracking/data/assets/urdf/"
        base_name = self.config.scene_file
        npy_path = os.path.join(asset_root, f"{base_name}.npy")
        urdf_path = os.path.join(asset_root, f"{base_name}.urdf")
        if self._urdf_vis_vertices is None and os.path.isfile(urdf_path):
            self._build_urdf_vis_mesh(urdf_path)
        required_points = int(self.config.num_obs_num_point)

        points = None
        if os.path.isfile(npy_path):
            try:
                points = np.load(npy_path)
            except Exception as err:
                print(f"[MimicHumanoid] Failed to load cached point cloud {npy_path}: {err}")

        if points is None and not self._use_triangle_mesh and os.path.isfile(urdf_path):
            try:
                points = save_points(urdf_path, "", save=False)
            except Exception as err:
                print(f"[MimicHumanoid] Failed to sample surface points from {urdf_path}: {err}")

        if points is None and self._use_triangle_mesh:
            verts = getattr(self, "_scene_vertices", None)
            faces = getattr(self, "_scene_triangles", None)
            if verts is not None and faces is not None and verts.size > 0 and faces.size > 0:
                num_samples = int(getattr(self.config, "max_pointcloud_points", 4096))
                sampled = self._sample_points_from_mesh(verts.astype(np.float32), faces.astype(np.int32), num_samples)
                if sampled is not None and sampled.shape[0] > 0:
                    points = sampled
                else:
                    points = verts.astype(np.float32)
            elif verts is not None and verts.size > 0:
                points = verts.astype(np.float32)
        if points is None and self._urdf_point_samples is not None:
            points = self._urdf_point_samples.copy()

        if points is not None:
            max_pts = int(getattr(self.config, "max_pointcloud_points", 4096))
            if points.shape[0] > max_pts:
                rng = np.random.default_rng(seed=0)
                indices = rng.choice(points.shape[0], size=max_pts, replace=False)
                points = points[indices]
            if points.shape[0] == 0:
                points = np.zeros((required_points, 3), dtype=np.float32)
            points_tensor = torch.from_numpy(points.astype(np.float32)).float().to(device)
            num_scene_points = points_tensor.shape[0]
            self.num_closest_point = min(required_points, num_scene_points)
            self.pointcloud = points_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1)
        else:
            print(f"[MimicHumanoid] Unable to initialize surface points for scene_file={base_name}.")
            if self.pointcloud is None:
                self.pointcloud = torch.zeros(self.num_envs, required_points, 3, device=device)
                self.num_closest_point = required_points

        if self.voxel and os.path.isfile(urdf_path):
            try:
                boxes = read_urdf(urdf_path)
                boxes_tensor = torch.tensor(boxes).float().to(device)
                self.boxes_tensor = boxes_tensor.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
            except Exception as err:
                print(f"[MimicHumanoid] Failed to prepare voxel boxes from {urdf_path}: {err}")
        elif self.voxel:
            print(f"[MimicHumanoid] Voxel observation requested but URDF {urdf_path} is missing; disabling voxel features.")
            self.voxel = False

    # Coordinate transforms: add tile offset on top of env origin
    def convert_to_global_coords(self, humanoid_root_states, env_offsets):
        base = super().convert_to_global_coords(humanoid_root_states, env_offsets)
        if self.scene and self._use_triangle_mesh and hasattr(self, "_tm_env_tile_offsets"):
            toff = self._tm_env_tile_offsets
            # Repeat per-env tile offsets if base is flattened across future steps etc.
            if base.shape[0] % toff.shape[0] == 0 and base.shape[0] != toff.shape[0]:
                rep = base.shape[0] // toff.shape[0]
                toff = toff.repeat_interleave(rep, dim=0)
            # Match last dim (2D or 3D inputs)
            if base.shape[-1] == 2:
                toff = toff[..., :2]
            # Expand for broadcasting across extra dims (e.g., N x 1 x 2)
            while toff.dim() < base.dim():
                toff = toff.unsqueeze(1)
            base = base + toff
        return base

    def transfer_to_env_coordinates(self, root_pos, env_ids):
        # When placing actors in physics, add tile offset so they sit on the tile
        if self.scene and self._use_triangle_mesh and hasattr(self, "_tm_env_tile_offsets"):
            toff = self._tm_env_tile_offsets[env_ids]
            # If root_pos is flattened across future steps, repeat offsets
            if root_pos.shape[0] % toff.shape[0] == 0 and root_pos.shape[0] != toff.shape[0]:
                rep = root_pos.shape[0] // toff.shape[0]
                toff = toff.repeat_interleave(rep, dim=0)
            # Match last dim (2D or 3D)
            if root_pos.shape[-1] == 2:
                toff = toff[..., :2]
            # Expand for broadcasting across extra dims (e.g., N x B x 3)
            while toff.dim() < root_pos.dim():
                toff = toff.unsqueeze(1)
            return root_pos + toff
        else:
            return root_pos
    
    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        num_markers_per_env = len(self.config.key_bodies) + 1

        for i in range(num_markers_per_env):
            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 0, 0)
            color = gymapi.Vec3(0.8, 0.0, 0.0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color)
            self._marker_handles[env_id].append(marker_handle)

        if self.terrain is not None:
            for i in range(self.num_height_points):
                marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 0, 0)
                color = gymapi.Vec3(0.0, 0.8, 0.0)
                self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color)
                self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_markers_per_env = len(self.config.key_bodies) + 1  # for root
        if self.terrain is not None:
            num_markers_per_env += self.num_height_points

        num_actors = self.get_num_actors_per_env()
        if self.total_num_objects > 0:
            self._marker_states = self.root_states[:-self.total_num_objects].view(self.num_envs, num_actors, self.root_states.shape[-1])[..., 1:(1 + num_markers_per_env), :]
        else:
            self._marker_states = self.root_states.view(self.num_envs, num_actors, self.root_states.shape[-1])[..., 1:(1 + num_markers_per_env), :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self.humanoid_actor_ids.unsqueeze(-1) + torch_utils.to_torch(self._marker_handles, dtype=torch.int32, device=self.device)
        self._marker_actor_ids = self._marker_actor_ids.flatten()

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        (target_pos, _, _, _, _, _) = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        target_pos += self.respawn_offsets.clone().view(self.num_envs, 1, 3)
        current_pos, current_rot, _, _ = self.get_bodies_state()

        env_global_positions = self.convert_to_global_coords(target_pos[:, 0, :2], self.env_offsets[..., :2])
        target_pos[..., -1:] += self.get_ground_heights_below_base(env_global_positions).view(self.num_envs, 1, 1)

        body_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=self.device),  # root
                self.key_body_ids
            ], dim=0
        )
        target_pos = target_pos[:, body_ids, :].view(-1, len(body_ids), 3)

        self._marker_pos[:, :len(body_ids)] = target_pos

        if self.terrain is not None:
            height_maps = self.get_heights(None, return_all_dims=True)
            height_maps = height_maps.view(self.num_envs, -1, 3)
            height_maps[..., :2] -= self.env_offsets[..., :2].view(self.num_envs, 1, 2)
            self._marker_pos[:, len(body_ids):] = height_maps

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids),
                                                     len(self._marker_actor_ids))

    def draw_mimic_markers(self):
        self._update_marker()

    def render(self):
        super().render()
        

        if self.viewer and self.config.provide_future_states and not self.config.sync_motion and self.config.visualize_markers:
            self.draw_mimic_markers()

        # Update Viser robot visualization (env 0) as geometry/points aligned to scene tile
        if getattr(self, "use_viser", False) and self._viser is not None and self._viser_ok:
            try:
                # Lazily create robot visualizer once asset path is known (works for both triangle-mesh and URDF scenes)
                if self._robot_vis is None and hasattr(self, "asset_mjcf_path") and self.asset_mjcf_path is not None:
                    try:
                        body_names = getattr(self, 'body_names', None)
                        self._robot_vis = RobotMjcfViser(self._viser, self.asset_mjcf_path, body_names)
                    except Exception as _e_init:
                        print(f"[Viser] Robot visualizer init failed: {_e_init}")
                rb0 = self.rigid_body_pos[0].detach().cpu().numpy()
                rq0 = self.rigid_body_rot[0].detach().cpu().numpy()
                world_off = np.zeros(3, dtype=np.float32)
                if self.scene and self._use_triangle_mesh and hasattr(self, "_tm_env_tile_offsets"):
                    world_off = self._tm_env_tile_offsets[0].detach().cpu().numpy()
                # Update robot MJCF meshes if available; otherwise show points
                if self._robot_vis is not None:
                    self._robot_vis.update(rb0, rq0, world_off)
                else:
                    pts = rb0 + world_off
                    self._viser.update_point_cloud("/robot_points", pts, point_size=0.02)

                # Set a simple camera following the root once
                if not hasattr(self, "_viser_cam_set"):
                    root = rb0[0] + world_off
                    cam = root + np.array([0.0, -2.0, 1.5], dtype=np.float32)
                    look = root + np.array([0.0, 0.0, 0.4], dtype=np.float32)
                    self._viser.set_camera(cam, look)
                    self._viser_cam_set = True
                self._ensure_urdf_scene_visual()
                self._update_pointnet_viser_points()
            except Exception as _e:
                print(f"[Viser] robot update skipped: {_e}")
        if self.save_state_path is not None and getattr(self, "control_i", 0) == 0:
            if self.start_record_pose and self.motion_times[0] == 0.0:
                self.save_record_pose = True
            if not self.start_record_pose and self.motion_times[0] == 0.0:
                self.start_record_pose = True
                self.save_record_pose = False
            if self.save_record_pose:
                poses = torch.cat(self.save_imitation_result["pose_aa"])[1:].float().numpy()
                transl = torch.cat(self.save_imitation_result["trans"])[1:].float().numpy()
                np.savez(f"{self.save_state_path}.npz", poses=poses, transl=transl)
                print(f"save to {self.save_state_path}.npz")
                import time, sys
                time.sleep(2)
                sys.exit()
            if self.start_record_pose:
                body_quat = self.rigid_body_rot.clone()
                root_trans = self.rigid_body_pos[:, 0, :].cpu().clone()
                N = body_quat.shape[0]
                pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).cpu().numpy()) * self.pre_rot).as_quat().reshape(N, -1, 4)
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    self.sk_tree, torch.from_numpy(pose_quat), root_trans, is_local=False
                )
                local_rot = new_sk_state.local_rotation
                pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).cpu().numpy()).as_rotvec().reshape(N, -1, 3)
                pose_aa = torch.from_numpy(pose_aa[:, self.mujoco_2_smpl, :].reshape(N, -1))
                self.save_imitation_result["pose_aa"].append(pose_aa[0:1])
                self.save_imitation_result["trans"].append(root_trans[0:1])


    def _update_pointnet_viser_points(self):
        if not getattr(self, "_use_scene_point_obs", False):
            return
        if not getattr(self, "use_viser", False) or self._viser is None or not self._viser_ok:
            return
        sample_points = getattr(self, "_latest_sampled_scene_points", None)
        if sample_points is None:
            try:
                env_tensor = torch.tensor([0], device=self.device, dtype=torch.long)
                _ = self.compute_point_obs(env_tensor)
                sample_points = getattr(self, "_latest_sampled_scene_points", None)
            except Exception as _e_sample:
                print(f"[Viser] Failed to refresh point samples: {_e_sample}")
                return
            if sample_points is None:
                return
        env_id = 0
        if env_id >= sample_points.shape[0]:
            return
        points = sample_points[env_id]
        if points.numel() == 0:
            return

        k = int(min(self.num_closest_point, points.shape[0]))
        if k <= 0:
            return

        offset = torch.zeros(3, device=points.device)
        if hasattr(self, "env_offsets"):
            offset = offset + self.env_offsets[env_id, :3]
        if self.scene and self._use_triangle_mesh and hasattr(self, "_tm_env_tile_offsets"):
            offset = offset + self._tm_env_tile_offsets[env_id]

        pts_np = (points[:k] + offset).detach().cpu().numpy()
        try:
            color_np = np.tile(np.array([[1.0, 0.4, 0.1]], dtype=np.float32), (pts_np.shape[0], 1))
            self._viser.update_point_cloud(
                "/policy_scene_points",
                pts_np,
                point_size=0.015,
                color=color_np,
            )
        except Exception as _e_pc:
            print(f"[Viser] point cloud update skipped: {_e_pc}")

    def _ensure_urdf_scene_visual(self):
        if not getattr(self, "use_viser", False) or self._viser is None or not self._viser_ok:
            return
        if self._use_triangle_mesh:
            verts = getattr(self, "_scene_vertices", None)
            faces = getattr(self, "_scene_triangles", None)
            if verts is None or faces is None or verts.size == 0 or faces.size == 0:
                return
            try:
                self._viser.add_mesh_simple(
                    "/scene_triangle_mesh",
                    verts.astype(np.float32),
                    faces.astype(np.int32),
                    color=(0.6, 0.7, 0.9),
                    side="double"
                )
                self._urdf_vis_added = True
            except Exception as _e_mesh:
                print(f"[Viser] triangle mesh viz skipped: {_e_mesh}")
            return
        if self._urdf_vis_added:
            return
        if self._urdf_vis_vertices is None or self._urdf_vis_triangles is None:
            return
        try:
            self._viser.add_mesh_simple(
                "/scene_urdf_mesh",
                self._urdf_vis_vertices,
                self._urdf_vis_triangles,
                color=(0.6, 0.7, 0.9),
                side="double"
            )
            self._urdf_vis_added = True
        except Exception as _e_mesh:
            print(f"[Viser] URDF mesh viz skipped: {_e_mesh}")

    def _build_urdf_vis_mesh(self, urdf_path: str):
        verts, faces = self._extract_urdf_mesh(urdf_path)
        if verts is not None and faces is not None and verts.size > 0 and faces.size > 0:
            self._urdf_vis_vertices = verts.astype(np.float32)
            self._urdf_vis_triangles = faces.astype(np.int32)
            num_samples = int(getattr(self.config, "max_pointcloud_points", 4096))
            sampled = self._sample_points_from_mesh(self._urdf_vis_vertices, self._urdf_vis_triangles, num_samples)
            if sampled is not None:
                self._urdf_point_samples = sampled

    def _extract_urdf_mesh(self, urdf_path: str):
        if not os.path.isfile(urdf_path):
            return None, None
        try:
            tree = ET.parse(urdf_path)
        except Exception as err:
            print(f"[Viser] Failed to parse URDF for viz: {err}")
            return None, None
        root = tree.getroot()
        base_dir = os.path.dirname(urdf_path)
        vert_blocks = []
        face_blocks = []

        for visual in root.findall(".//visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            origin = visual.find("origin")
            pos = np.zeros(3, dtype=np.float32)
            rpy = np.zeros(3, dtype=np.float32)
            if origin is not None:
                if origin.get("xyz"):
                    pos = np.array(list(map(float, origin.get("xyz").split())), dtype=np.float32)
                if origin.get("rpy"):
                    rpy = np.array(list(map(float, origin.get("rpy").split())), dtype=np.float32)

            verts = None
            faces = None

            mesh_tag = geom.find("mesh")
            box_tag = geom.find("box")
            if mesh_tag is not None:
                filename = mesh_tag.get("filename")
                if filename is None:
                    continue
                scale_attr = mesh_tag.get("scale")
                if scale_attr is not None:
                    scale_vals = list(map(float, scale_attr.split()))
                    if len(scale_vals) == 1:
                        scale_vec = np.array([scale_vals[0]] * 3, dtype=np.float32)
                    elif len(scale_vals) == 3:
                        scale_vec = np.array(scale_vals, dtype=np.float32)
                    else:
                        scale_vec = np.ones(3, dtype=np.float32)
                else:
                    scale_vec = np.ones(3, dtype=np.float32)
                loaded = self._load_scene_mesh_file(filename, base_dir)
                if loaded is None:
                    continue
                verts_raw, faces_raw = loaded
                verts = verts_raw * scale_vec
                faces = faces_raw.astype(np.int32)
            elif box_tag is not None:
                size_attr = box_tag.get("size")
                if size_attr is None:
                    continue
                size_vals = np.array(list(map(float, size_attr.split())), dtype=np.float32)
                verts, faces = self._create_box_mesh(size_vals)
            else:
                continue

            if verts is None or faces is None:
                continue
            R = self._rpy_to_matrix(rpy)
            transformed = (verts @ R.T) + pos
            offset = sum(block.shape[0] for block in vert_blocks)
            vert_blocks.append(transformed)
            face_blocks.append(faces + offset)

        if not vert_blocks or not face_blocks:
            return None, None
        verts_all = np.concatenate(vert_blocks, axis=0)
        faces_all = np.concatenate(face_blocks, axis=0)
        return verts_all, faces_all

    def _sample_points_from_mesh(self, verts: np.ndarray, faces: np.ndarray, num_samples: int):
        if verts is None or faces is None or len(faces) == 0 or num_samples <= 0:
            return None
        tri_vertices = verts[faces]
        vec0 = tri_vertices[:, 1] - tri_vertices[:, 0]
        vec1 = tri_vertices[:, 2] - tri_vertices[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(vec0, vec1), axis=1)
        total_area = np.sum(areas)
        if total_area <= 1e-8:
            return tri_vertices.reshape(-1, 3)
        probs = areas / total_area
        rng = np.random.default_rng(seed=0)
        sample_idx = rng.choice(len(faces), size=num_samples, p=probs)
        chosen = tri_vertices[sample_idx]
        u = rng.random((num_samples, 1))
        v = rng.random((num_samples, 1))
        mask = (u + v) > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        samples = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])
        return samples.astype(np.float32)

    def _create_box_mesh(self, size: np.ndarray):
        hx, hy, hz = size / 2.0
        vertices = np.array([
            [ hx,  hy,  hz],
            [ hx, -hy,  hz],
            [-hx, -hy,  hz],
            [-hx,  hy,  hz],
            [ hx,  hy, -hz],
            [ hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx,  hy, -hz],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ], dtype=np.int32)
        return vertices, faces

    def _load_scene_mesh_file(self, filename: str, base_dir: str):
        mesh_path = self._resolve_mesh_path(filename, base_dir)
        if mesh_path is None:
            print(f"[Viser] Unable to resolve mesh path: {filename}")
            return None
        try:
            if mesh_path.lower().endswith(".obj"):
                return self._simple_load_obj(mesh_path)
            else:
                import trimesh  # type: ignore
                mesh = trimesh.load(mesh_path, force="mesh")
                return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int32)
        except Exception as err:
            print(f"[Viser] Failed to load mesh {mesh_path}: {err}")
            return None

    def _simple_load_obj(self, filepath: str):
        vertices = []
        faces = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith("f "):
                        parts = line.strip().split()[1:]
                        idxs = [int(part.split("/")[0]) - 1 for part in parts]
                        if len(idxs) == 3:
                            faces.append(idxs)
                        elif len(idxs) == 4:
                            faces.append([idxs[0], idxs[1], idxs[2]])
                            faces.append([idxs[0], idxs[2], idxs[3]])
        except Exception as err:
            print(f"[Viser] Failed to read obj {filepath}: {err}")
            return None
        if not vertices or not faces:
            return None
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

    def _resolve_mesh_path(self, mesh_path: str, base_dir: str):
        path = mesh_path
        if mesh_path.startswith("package://"):
            path = mesh_path[len("package://"):]
            path = os.path.join("motion_tracking", path)
        elif mesh_path.startswith("file://"):
            path = mesh_path[len("file://"):]
        elif not os.path.isabs(mesh_path):
            path = os.path.join(base_dir, mesh_path)
        path = os.path.normpath(path)
        if os.path.isfile(path):
            return path
        return None

    def _rpy_to_matrix(self, rpy: np.ndarray):
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [   -sp,                 cp * sr,                 cp * cr],
        ], dtype=np.float32)


        
