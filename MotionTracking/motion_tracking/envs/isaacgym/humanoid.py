# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import os.path as osp
import glob

from isaacgym import gymtorch, gymapi, gymutil

from isaac_utils import torch_utils, rotations

from motion_tracking.envs.common.utils import (
    build_pd_action_offset_scale
)
from motion_tracking.envs.isaacgym.base_interface import GymBaseInterface
from motion_tracking.envs.common.common_humanoid import BaseHumanoid
from motion_tracking.data.assets.skeleton_configs import isaacgym_asset_file_to_stats
from motion_tracking.utils.file_utils import load_yaml

import torch
from torch import Tensor
from poselib.skeleton.skeleton3d import SkeletonTree
from typing import Optional
import xml.etree.ElementTree as ET

try:
    # Prefer SMPLSim if available (pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master)
    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalSmplRobot
except Exception:
    try:
        # Fallback to repo-local implementation
        from smpllib.smpllib.smpl_local_robot import Robot as LocalSmplRobot
    except Exception:
        LocalSmplRobot = None


class Humanoid(BaseHumanoid, GymBaseInterface):
    def __init__(self, config, device: torch.device):
        self.w_last = True  # quaternion structure in isaacgym
        self.config = config
        self.device = device
        # Enable Viser-based visualization by default (non-blocking if not installed)
        if config.headless:
            self.use_viser = False
        else:
            self.use_viser = True
        self.sim_params = self.parse_sim_params()
        self.physics_engine = gymapi.SIM_PHYSX

        self.debug_viz = self.config.enable_debug_vis
        self.plane_static_friction = self.config.plane.static_friction
        self.plane_dynamic_friction = self.config.plane.dynamic_friction
        self.plane_restitution = self.config.plane.restitution

        super().__init__(config, device)
        assert self.dof_offsets[-1] == self.num_dof

        self.dt: float = self.control_freq_inv * self.sim_params.dt

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor: Tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, sensors_per_env * 6
        )

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor: Tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_dof
        )

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states: Tensor = gymtorch.wrap_tensor(actor_root_state)

        # self.object_root_states = self.root_states[-self.total_num_objects:]

        num_actors = self.get_num_actors_per_env()

        if self.total_num_objects == 0:
            self.humanoid_root_states = self.root_states.view(
                self.num_envs, num_actors, actor_root_state.shape[-1]
            )[..., 0, :]
        else:
            self.humanoid_root_states = self.root_states[:-self.total_num_objects].view(
                self.num_envs, num_actors, actor_root_state.shape[-1]
            )[..., 0, :]


        self.initial_humanoid_root_states = self.humanoid_root_states.clone()
        self.initial_humanoid_root_states[:, 7:13] = 0

        self.humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        # create some wrapper tensors for different slices
        self.dof_state: Tensor = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self.dof_state.shape[0] // self.num_envs
        self.dof_pos = self.dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 0
        ]
        self.dof_vel = self.dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., : self.num_dof, 1
        ]

        self.initial_dof_pos = torch.zeros_like(
            self.dof_pos, device=self.device, dtype=torch.float
        )
        self.initial_dof_vel = torch.zeros_like(
            self.dof_vel, device=self.device, dtype=torch.float
        )

        if self.total_num_objects == 0:
            self.rigid_body_state: Tensor = gymtorch.wrap_tensor(rigid_body_state)
        else:
            self.rigid_body_state: Tensor = gymtorch.wrap_tensor(rigid_body_state)[:-self.total_num_objects]

        bodies_per_env = self.rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self.rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )

        self.rigid_body_pos = rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self.rigid_body_rot = rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self.rigid_body_vel = rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self.rigid_body_ang_vel = rigid_body_state_reshaped[
            ..., : self.num_bodies, 10:13
        ]

        if self.total_num_objects == 0:
            contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        else:
            contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)[:-self.total_num_objects]
        self.contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., : self.num_bodies, :]

        self.build_termination_heights()

        self.key_body_ids = self.build_body_ids_tensor(self.config.key_bodies)
        self.contact_body_ids = self.build_body_ids_tensor(self.config.contact_bodies)

        if self.viewer is not None:
            self.init_camera()

        self.export_video: bool = self.config.export_video

        if self.export_video:
            self.setup_cameras()

        self.export_motion: bool = self.config.export_motion

        if self.export_motion:
            self.motion_recording = {}
        # Allows the agent to disable resets temporarily.
        self.disable_reset = False

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def parse_sim_params(self):
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / self.config.sim_fps
        sim_params.num_client_threads = 0

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024
        sim_params.use_gpu_pipeline = True

        gymutil.parse_sim_config(self.config.sim, sim_params)
        return sim_params

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, "z")
        super().create_sim()

        self.create_ground_plane()
        self.create_envs(
            self.num_envs, self.config.env_spacing, int(np.sqrt(self.num_envs))
        )

    def create_ground_plane(self):
        if self.config.terrain is None:
            self.add_default_ground()
        else:
            self.add_terrain()

    def add_default_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def add_terrain(self):
        self.terrain_offset = -self.terrain.border_size

        if self.config.dummy_terrain:
            self.add_default_ground()
        else:
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = self.terrain.vertices.shape[0]
            tm_params.nb_triangles = self.terrain.triangles.shape[0]
            tm_params.transform.p.x = self.terrain_offset
            tm_params.transform.p.y = self.terrain_offset
            tm_params.transform.p.z = 0.0
            tm_params.static_friction = self.config.terrain.terrain_config.static_friction
            tm_params.dynamic_friction = self.config.terrain.terrain_config.dynamic_friction
            tm_params.restitution = self.config.terrain.terrain_config.restitution
            self.gym.add_triangle_mesh(
                self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params
            )

    def create_envs(self, num_envs, spacing, num_per_row):
        # lower = gymapi.Vec3(0., 0., 0.)
        # upper = gymapi.Vec3(spacing, spacing, spacing)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.config.asset.asset_root
        asset_file = self.config.asset.asset_file_name

        # Optionally override with a dynamically generated SMPL MJCF built from shape params
        dynamic_shape_cfg: Optional[object] = getattr(self.config.asset, "dynamic_shape", None)
        dyn_enabled = False
        dyn_asset_path: Optional[str] = None
        # 
        if dynamic_shape_cfg is not None and getattr(dynamic_shape_cfg, "enabled", False):
            if LocalSmplRobot is None:
                print("[Humanoid] SMPL dynamic shape requested but SMPL generator not available. Falling back to static asset.")
            else:
                try:

                    robot_cfg = {'mesh': False, 'replace_feet': True, 
                                 'rel_joint_lm': False, 'remove_toe': False, 
                                 'freeze_hand': False, 'real_weight_porpotion_capsules': True, 
                                 'real_weight_porpotion_boxes': True, 'real_weight': True, 
                                 'master_range': 30, 'big_ankle': True, 'box_body': True, 
                                 'masterfoot': False, 'upright_start': True, 
                                 'model': 'smpl', 'create_vel_sensors': False, 
                                 'body_params': {}, 'joint_params': {}, 'geom_params': {}, 
                                 'actuator_params': {}, 'gender': 'neutral'}
                    
                    data_dir = getattr(dynamic_shape_cfg, "smpl_data_dir", None)
                    smpl_data_dir = data_dir if data_dir else "data/smpl"

                    smpl_robot = None
                    smplsim_err = None
                    local_err = None
                    # Prefer SMPLSim if available; fall back to local generator on any error
                    try:
                        from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as _Robot
                        smpl_robot = _Robot(robot_cfg, data_dir=smpl_data_dir)
                    except Exception as e1:
                        smplsim_err = e1
                        try:
                            from smpllib.smpllib.smpl_local_robot import Robot as _Robot
                            smpl_robot = _Robot(robot_cfg, data_dir=smpl_data_dir)
                        except Exception as e2:
                            local_err = e2
                            raise

                    # Optionally override betas from motion sidecar (.npz with key 'betas' or 'beta')
                    # import pdb;pdb.set_trace()
                    betas_list = list(getattr(dynamic_shape_cfg, "betas", []))
                    try:
                        use_motion_betas = bool(getattr(dynamic_shape_cfg, "use_motion_betas", False))
                    except Exception:
                        use_motion_betas = False

                    if use_motion_betas:
                        try:
                            motion_file_path = getattr(self.config, "motion_lib", None)
                            # motion_lib can be an object or a config; support both
                            if motion_file_path is not None and hasattr(motion_file_path, "motion_file"):
                                motion_file_path = motion_file_path.motion_file
                            # If we still didn't get a path, try top-level config
                            if motion_file_path is None and hasattr(self.config, "motion_file"):
                                motion_file_path = self.config.motion_file

                            def _extract_betas_from_data(data_obj):
                                keys = ("betas", "beta")
                                if data_obj is None:
                                    return None
                                try:
                                    if isinstance(data_obj, dict):
                                        for key in keys:
                                            if key in data_obj:
                                                return data_obj[key]
                                    # Structured arrays
                                    if isinstance(data_obj, np.ndarray):
                                        if data_obj.dtype.names:
                                            for key in keys:
                                                if key in data_obj.dtype.names:
                                                    return data_obj[key]
                                        if data_obj.dtype == object or data_obj.shape == ():
                                            try:
                                                item = data_obj.item()
                                            except Exception:
                                                item = None
                                            if isinstance(item, dict):
                                                for key in keys:
                                                    if key in item:
                                                        return item[key]
                                    # np.load on .npz returns an NpzFile with `.files`
                                    files = getattr(data_obj, "files", None)
                                    if files is not None:
                                        for key in keys:
                                            if key in files:
                                                return data_obj[key]
                                except Exception:
                                    return None
                                return None

                            if motion_file_path is not None:
                                motion_file_str = str(motion_file_path)
                                betas_arr = None

                                if os.path.exists(motion_file_str):
                                    try:
                                        npy_data = np.load(motion_file_str, allow_pickle=True)
                                        betas_arr = _extract_betas_from_data(npy_data)
                                        if betas_arr is not None:
                                            betas_row = betas_arr[0]
                                            betas_list = [float(x) for x in np.asarray(betas_row).ravel().tolist()]
                                            print(f"[Humanoid] Loaded betas directly from motion file: {motion_file_str}")
                                        else:
                                            print(f"[Humanoid] No 'betas' or 'beta' key in {motion_file_str}; using configured betas.")
                                    except Exception as _e_motion:
                                        print(f"[Humanoid] Failed to load betas from motion file {motion_file_str}: {_e_motion}")
                                else:
                                    print(f"[Humanoid] Motion file not found at {motion_file_str}; using configured betas.")
                        except Exception as _e_betas:
                            print(f"[Humanoid] Failed to load betas from motion file: {_e_betas}; using configured betas.")
                    # import pdb;pdb.set_trace()
                    # Normalize betas length for SMPL (10); pad or trim as needed
                    if len(betas_list) < 10:
                        betas_list = betas_list + [0.0] * (10 - len(betas_list))
                    elif len(betas_list) > 10:
                        betas_list = betas_list[:10]
                    # import pdb;pdb.set_trace()
                    gender_val = int(getattr(dynamic_shape_cfg, "gender", 0))  # 0 neutral, 1 male, 2 female
                    betas_t = torch.tensor(betas_list, dtype=torch.float32).view(1, -1) 
                    gender_t = torch.tensor([gender_val], dtype=torch.long)

                    smpl_robot.load_from_skeleton(betas=betas_t, gender=gender_t, objs_info=None)

                    tmp_dir = "/tmp/smpl"
                    os.makedirs(tmp_dir, exist_ok=True)
                    dyn_asset_path = os.path.join(tmp_dir, "smpl_humanoid_custom.xml")
                    smpl_robot.write_xml(dyn_asset_path)

                    # Merge only pose/size from dynamic XML into base XML to keep physics identical
                    try:
                        base_asset_path = os.path.join(self.config.asset.asset_root, self.config.asset.asset_file_name)

                        def _merge_shape_into_base(base_path: str, shape_path: str, out_path: str):
                            base_tree = ET.parse(base_path)
                            shape_tree = ET.parse(shape_path)
                            broot = base_tree.getroot()
                            sroot = shape_tree.getroot()

                            # Map bodies by name
                            for bbody in broot.findall(".//body"):
                                name = bbody.attrib.get("name")
                                if not name:
                                    continue
                                sbody = sroot.find(f".//body[@name='{name}']")
                                if sbody is None:
                                    continue
                                # Update body placement
                                if "pos" in sbody.attrib:
                                    bbody.attrib["pos"] = sbody.attrib["pos"]
                                if "quat" in sbody.attrib:
                                    bbody.attrib["quat"] = sbody.attrib["quat"]

                                # Preserve joints exactly as in base (do not modify joint pos/params)

                                # Update geoms by order within body
                                bgeoms = [g for g in bbody.findall("geom")]
                                sgeoms = [g for g in sbody.findall("geom")]
                                n = min(len(bgeoms), len(sgeoms))
                                for i in range(n):
                                    bg = bgeoms[i]
                                    sg = sgeoms[i]
                                    gtype = bg.attrib.get("type")
                                    if gtype == "capsule":
                                        if "fromto" in sg.attrib:
                                            bg.attrib["fromto"] = sg.attrib["fromto"]
                                        if "size" in sg.attrib:
                                            bg.attrib["size"] = sg.attrib["size"]
                                    elif gtype in ("box", "sphere"):
                                        if "pos" in sg.attrib:
                                            bg.attrib["pos"] = sg.attrib["pos"]
                                        if "quat" in sg.attrib:
                                            bg.attrib["quat"] = sg.attrib["quat"]
                                        if "size" in sg.attrib:
                                            bg.attrib["size"] = sg.attrib["size"]
                                    elif gtype == "mesh":
                                        # Keep mesh geometry mostly intact; optionally update placement
                                        if "pos" in sg.attrib:
                                            bg.attrib["pos"] = sg.attrib["pos"]
                                        if "quat" in sg.attrib:
                                            bg.attrib["quat"] = sg.attrib["quat"]

                            base_tree.write(out_path)

                        merged_asset_path = os.path.join(tmp_dir, "smpl_humanoid_custom_merged.xml")
                        _merge_shape_into_base(base_asset_path, dyn_asset_path, merged_asset_path)
                        dyn_asset_path = merged_asset_path
                    except Exception as _merge_e:
                        print(f"[Humanoid] Shape merge skipped due to error: {_merge_e}")

                    # Use the dynamically generated asset instead of the static one
                    dyn_enabled = True
                    asset_root = os.path.dirname(dyn_asset_path)
                    asset_file = os.path.basename(dyn_asset_path)
                except Exception as e:
                    msg = f"[Humanoid] Failed to generate SMPL asset from dynamic shape: {e}."
                    if 'smplsim_err' in locals() and smplsim_err is not None:
                        msg += f" SMPLSim error: {smplsim_err}."
                    if 'local_err' in locals() and local_err is not None:
                        msg += f" Local generator error: {local_err}."
                    print(msg + " Falling back to static asset.")

        if not dyn_enabled:
            asset_path = os.path.join(asset_root, asset_file)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self.humanoid_asset = humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # Cache rigid body names for visualization mapping
        try:
            self.body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        except Exception:
            self.body_names = None
        # Build skeleton tree from the same MJCF file used to load the asset
        skel_src_path = dyn_asset_path if dyn_enabled else os.path.join(asset_root, asset_file)
        # Cache the exact MJCF path used (for exporting/visualization later)
        self.asset_mjcf_path = skel_src_path
        self.sk_tree = SkeletonTree.from_mjcf(skel_src_path)

        if "smpl" in asset_file:
            # create force sensors at the feet
            right_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "L_Ankle"
            )
            left_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "R_Ankle"
            )
        else:
            # create force sensors at the feet
            right_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "right_foot"
            )
            left_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "left_foot"
            )
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = torch_utils.to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.object_handles = []
        self.envs = []
        self.object_envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        self.load_object_assets()

        self.env_offsets = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.build_env(i, env_ptr, humanoid_asset)
            self.envs.append(env_ptr)
            env_origin = self.gym.get_env_origin(env_ptr)
            #need check
            # self.env_offsets[i][0] = env_origin.x
            # self.env_offsets[i][1] = env_origin.y
            # self.env_offsets[i][2] = env_origin.z

        if self.total_num_objects > 0:
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.build_object_env(env_ptr)
            self.object_envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(
            self.envs[0], self.humanoid_handles[0]
        )
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = torch_utils.to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = torch_utils.to_torch(self.dof_limits_upper, device=self.device)

        if self.pd_control:
            self._pd_action_offset, self._pd_action_scale = build_pd_action_offset_scale(
                self.dof_offsets,
                self.dof_limits_lower,
                self.dof_limits_upper,
                self.device,
                self.gym.get_asset_dof_names(humanoid_asset),
                "smpl" in asset_file and self.config.fix_pd_offsets,
                specific_pd_fixes=self.config.specific_pd_fixes,
            )

    def load_object_assets(self):
        self.total_num_objects = 0
        if self.config.max_num_objects > 0:
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.fix_base_link = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params.resolution = 1000000

            self.object_assets = {}
            self.object_target_positions = {}
            self.object_names = {}

            total_num_objects = 0
            for object_action_type in self.object_action_types:
                for object_category in self.object_action_type_to_object_category[object_action_type]:
                    self.object_assets[object_action_type + "_" + object_category] = []
                    self.object_target_positions[object_action_type + "_" + object_category] = []
                    self.object_action_type_to_object[object_action_type + "_" + object_category] = []

                    category_root = osp.join(self.config.object_asset_root, object_category)

                    # filter all urdf files that also have a corresponding obj file
                    if self.config.object_name is not None:
                        filenames = [self.config.object_name]
                    else:
                        filenames = [osp.splitext(osp.basename(name))[0] for name in
                                     sorted(glob.glob(osp.join(category_root, '*.urdf')))]
                        filenames = [name for name in filenames if osp.exists(osp.join(category_root, name + '.obj'))]
                    self.object_names[object_action_type + "_" + object_category] = [name for name in filenames]

                    n = min(len(self.object_names[object_action_type + "_" + object_category]),
                            self.config.max_objects_per_category)

                    print(f"Loading {n} objects from {object_category} category")
                    self.object_names[object_action_type + "_" + object_category] = self.object_names[object_action_type + "_" + object_category][:n]
                    for object_name in self.object_names[object_action_type + "_" + object_category]:
                        if object_action_type + "_" + object_category not in self.object_action_type_keys:
                            self.object_action_type_keys.append(object_action_type + "_" + object_category)
                        # Load Asset
                        self.object_assets[object_action_type + "_" + object_category].append(
                            self.gym.load_asset(self.sim, category_root, "{}.urdf".format(object_name),
                                                object_asset_options)
                        )

                        # Load Joint Target Positions
                        target_position = load_yaml(osp.join(category_root, "{}.yaml".format(object_name)))["hip"]
                        self.object_target_positions[object_action_type + "_" + object_category].append(
                            torch.tensor(target_position, device=self.device, dtype=torch.float).view(-1)
                        )
                        total_num_objects += self.config.object_replications

            self.total_num_objects = min(total_num_objects, self.config.max_num_objects)
            print("=========== Total number of loaded objects is ", total_num_objects)

    def build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.config.asset.asset_file_name
        if (
            asset_file == "mjcf/ov_humanoid.xml"
            or asset_file == "mjcf/ov_humanoid_sword_shield.xml"
        ):
            char_h = 0.927
        else:
            char_h = 0.89

        start_pose.p = gymapi.Vec3(*torch_utils.get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(
            env_ptr,
            humanoid_asset,
            start_pose,
            "humanoid",
            col_group,
            col_filter,
            segmentation_id,
        )

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        humanoid_mass = np.sum(
            [prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)]
        )

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(
                env_ptr,
                humanoid_handle,
                j,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.54, 0.85, 0.2),
            )

        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        if self.pd_control:
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            if self.config.pd_scale:
                pd_scale = humanoid_mass / self.config.default_humanoid_mass
                dof_prop["stiffness"] *= pd_scale
                dof_prop["damping"] *= pd_scale
        else:
            dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT

        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        if "smpl" in self.config.asset.robot_type:
            if self.config.asset.robot_type == "smpl":
                filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif self.config.asset.robot_type in ["smplh", "smplx"]:
                filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)

            assert (len(filter_ints) == len(props))
            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]

            self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self.humanoid_handles.append(humanoid_handle)

    def build_object_env(self, env_ptr):
        import trimesh
        # from collections import defaultdict
        # from xml.etree import cElementTree as ET
        #
        # def etree_to_dict(t):
        #     d = {t.tag: {} if t.attrib else None}
        #     children = list(t)
        #     if children:
        #         dd = defaultdict(list)
        #         for dc in map(etree_to_dict, children):
        #             for k, v in dc.items():
        #                 dd[k].append(v)
        #         d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        #     if t.attrib:
        #         d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
        #     if t.text:
        #         text = t.text.strip()
        #         if children or t.attrib:
        #             if text:
        #                 d[t.tag]['#text'] = text
        #         else:
        #             d[t.tag] = text
        #     return d

        from motion_tracking.envs.utils.object_utils import get_object_heightmap, compute_bounding_box, as_mesh

        env_origin = self.gym.get_env_origin(env_ptr)

        map_length = self.terrain.terrain_config.map_length * self.terrain.terrain_config.num_terrains
        num_objects_per_row = int(map_length / self.terrain.spacing_between_objects)

        # TODO: fix this, very hacky way for constructing an iterative list of objects to spawn
        objects_to_spawn = []
        action_types = [elem for elem in self.object_action_types]
        object_action_type_to_object_category = {}
        for action_type in action_types:
            object_action_type_to_object_category[action_type] = [elem for elem in self.object_action_type_to_object_category[action_type]]
        all_objects = {}
        object_names = {}
        object_target_positions = {}
        for key in self.object_assets.keys():
            all_objects[key] = []
            object_names[key] = []
            object_target_positions[key] = []
            for _ in range(self.config.object_replications):
                for elem in self.object_assets[key]:
                    all_objects[key].append(elem)
                for elem in self.object_names[key]:
                    object_names[key].append(elem)
                for elem in self.object_target_positions[key]:
                    object_target_positions[key].append(elem)

        while len(objects_to_spawn) < self.total_num_objects:
            action_type = action_types[0]
            object_category = object_action_type_to_object_category[action_type][0]
            key = action_type + "_" + object_category
            if len(all_objects[key]) > 0:
                objects_to_spawn.append([all_objects[key].pop(0), object_names[key].pop(0), object_target_positions[key].pop(0), action_type, object_category])
            action_types = action_types[1:] + [action_types[0]]
            object_action_type_to_object_category[action_type] = object_action_type_to_object_category[action_type][1:] + object_action_type_to_object_category[action_type][:1]

        for object_idx, (object_asset, object_name, object_target_position, action_type, object_category) in enumerate(objects_to_spawn):
            if object_idx >= self.config.max_num_objects:
                break

            x_offset = (object_idx % num_objects_per_row) * self.terrain.spacing_between_objects
            y_offset = (object_idx // num_objects_per_row) * self.terrain.spacing_between_objects + (self.terrain.tot_cols - self.terrain.border - self.terrain.object_playground_cols ) * self.terrain.horizontal_scale

            object_offset = self.object_offsets[object_category]

            object_pose = gymapi.Transform()
            object_pose.r = gymapi.Quat(0, 0, 0, 1)
            object_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            object_pose.p.x = x_offset - env_origin.x + object_offset[0]
            object_pose.p.y = y_offset - env_origin.y + object_offset[1]
            object_pose.p.z = self.get_ground_heights_below_base(torch.tensor([[x_offset + object_offset[0], y_offset + object_offset[1]]], device=self.device, dtype=torch.float)).item()

            self.object_action_type_to_object[action_type + "_" + object_category].append(torch.tensor([x_offset, y_offset], device=self.device, dtype=torch.float))
            self.object_id_to_object_position.append(torch.tensor([x_offset, y_offset, object_pose.p.z], device=self.device, dtype=torch.float))

            object_offset_3d = torch.tensor([x_offset + object_offset[0], y_offset + object_offset[1], 0], device=self.device, dtype=torch.float).view(-1)

            self.object_id_to_target_position.append(object_target_position + object_offset_3d)
            self.spawned_object_names.append(object_category + "_" + object_name)

            object_handle = self.gym.create_actor(env_ptr, object_asset, object_pose, object_name, -1, 0)
            self.object_handles.append(object_handle)

            category_root = osp.join(self.config.object_asset_root, object_category)

            obj_path = osp.join(category_root, "{}.obj".format(object_name))
            mesh = as_mesh(trimesh.load_mesh(obj_path))

            # with open(category_root + "/{}.urdf".format(object_name)) as f:
            #     parsed_urdf = etree_to_dict(ET.fromstring(f.read().strip()))
            #
            # obj_filename = parsed_urdf["robot"]["link"]["visual"]["geometry"]["mesh"]["@filename"]
            #
            # obj_path = osp.join(category_root, "{}".format(obj_filename))
            # mesh = as_mesh(trimesh.load_mesh(obj_path))
            #
            # rotation = parsed_urdf["robot"]["link"]["visual"]["origin"]["@rpy"].split(' ')
            # rpy = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
            # rotation_matrix = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes='sxyz')
            # mesh.apply_transform(rotation_matrix)
            #
            # # apply translation
            # translation = parsed_urdf["robot"]["link"]["visual"]["origin"]["@xyz"].split(' ')
            # translation = [float(translation[0]), float(translation[1]), float(translation[2])]
            # mesh.apply_translation(translation)
            #
            # # apply scale if in the urdf
            # if "@scale" in parsed_urdf["robot"]["link"]["visual"]["geometry"]["mesh"]:
            #     scale = parsed_urdf["robot"]["link"]["visual"]["geometry"]["mesh"]["@scale"].split(' ')
            #     scale = [float(scale[0]), float(scale[1]), float(scale[2])]
            #     mesh.apply_scale(scale)

            w_x, w_y, w_z, m_x, m_y, m_z = compute_bounding_box(mesh)

            # compute the 8 coords for the bounding box, taking into account the offsets
            min_x = x_offset + object_offset[0] + m_x
            max_x = min_x + w_x
            min_y = y_offset + object_offset[1] + m_y
            max_y = min_y + w_y
            min_z = m_z
            # min_z = max(m_z, 0)
            max_z = min_z + w_z
            object_P0 = np.array([min_x, min_y, min_z])
            object_P1 = np.array([min_x, max_y, min_z])
            object_P2 = np.array([max_x, max_y, min_z])
            object_P3 = np.array([max_x, min_y, min_z])

            object_P4 = np.array([min_x, min_y, max_z])
            object_P5 = np.array([min_x, max_y, max_z])
            object_P6 = np.array([max_x, max_y, max_z])
            object_P7 = np.array([max_x, min_y, max_z])

            self.object_id_to_object_bounding_box.append(
                torch.tensor(
                    np.concatenate([
                        object_P0, object_P1, object_P2, object_P3,
                        object_P4, object_P5, object_P6, object_P7,
                    ]), device=self.device, dtype=torch.float
                )
            )

            # TODO: gym automatically creates this vector for each object, but it's aligned to the local frame
            # TODO: (cont.) the rotations are also not correct, objects are rotated 90 degrees
            vec_dir = torch.tensor([0.0, -1.0, 0.0], device=self.device, dtype=torch.float).view(1, 3)
            angle = rotations.vec_to_heading(vec_dir)
            neg = angle < 0
            angle[neg] += 2 * torch.pi
            quat_dir = rotations.heading_to_quat(angle, w_last=self.w_last).view(-1)
            self.object_root_states.append(
                torch.cat([torch.tensor(
                    [
                        x_offset + object_offset[0], y_offset + object_offset[1], m_z,
                    ], device=self.device, dtype=torch.float
                ), quat_dir, torch.tensor([self.object_types.index(object_category)], device=self.device, dtype=torch.float)])
            )

            scale = 2.0
            heightmap_path = osp.join(category_root, "{}_{}_{}.pt".format(object_name, scale, self.terrain.terrain_config.horizontal_scale))
            if osp.exists(heightmap_path):
                heightmap = torch.load(heightmap_path)
            else:
                print("Creating object heightmap for object {} at scale {}".format(object_name, scale))
                heightmap = torch.tensor(get_object_heightmap(mesh, dim_x=int(np.ceil(w_x / (self.terrain.terrain_config.horizontal_scale / scale))), dim_y=int(np.ceil(w_y / (self.terrain.terrain_config.horizontal_scale / scale)))), dtype=torch.float)
                torch.save(heightmap, heightmap_path)

            heightmap = heightmap.to(self.device)

            # 1. Create a grid for the object in global coordinates --> each cell has the global coordinates of the center of that cell.
            # 2. Do the same for the heightmap.
            # 3. Go cell by cell in the heightmap, where the object resides.
            # 3.1. Find the appropriate cells in the object grid, and perform bilinear interpolation to get the height at that point.

            object_min_coords = [x_offset + object_offset[0] + m_x - self.terrain_offset, y_offset + object_offset[1] + m_y - self.terrain_offset]
            object_max_coords = [object_min_coords[0] + w_x, object_min_coords[1] + w_y]
            object_min_cell_idx = [int(np.floor(coord / self.terrain.terrain_config.horizontal_scale)) for coord in object_min_coords]
            object_max_cell_idx = [int(np.ceil(coord / self.terrain.terrain_config.horizontal_scale)) for coord in object_max_coords]

            for x_idx, x in enumerate(range(object_min_cell_idx[0] - 1, object_max_cell_idx[0] + 1)):
                for y_idx, y in enumerate(range(object_min_cell_idx[1] - 1, object_max_cell_idx[1] + 1)):
                    # get coordinates in object-relative frame, remove object offset
                    object_coords = [x * self.terrain.terrain_config.horizontal_scale, y * self.terrain.terrain_config.horizontal_scale]
                    object_coords = [object_coords[0] - (x_offset + object_offset[0] - self.terrain_offset), object_coords[1] - (y_offset + object_offset[1] - self.terrain_offset)]
                    object_coords = [object_coords[0] - m_x, object_coords[1] - m_y]

                    object_floor_idx = [int(np.floor(object_coords[0] / (self.terrain.terrain_config.horizontal_scale / scale))), int(np.floor(object_coords[1] / (self.terrain.terrain_config.horizontal_scale / scale)))]

                    # TODO: For now, pick max height since there's some issue with billinear due to discretization size

                    # perform billinear interpolation, if out of bounds interpolate with 0
                    x1 = object_floor_idx[0]
                    x2 = x1 + 1
                    y1 = object_floor_idx[1]
                    y2 = y1 + 1
                    xm = object_coords[0] / (self.terrain.terrain_config.horizontal_scale / scale)
                    ym = object_coords[1] / (self.terrain.terrain_config.horizontal_scale / scale)

                    x1y1 = heightmap[x1, y1] if 0 <= x1 < heightmap.shape[0] and 0 <= y1 < heightmap.shape[1] else 0
                    x2y1 = heightmap[x2, y1] if 0 <= x2 < heightmap.shape[0] and 0 <= y1 < heightmap.shape[1] else 0
                    x1y2 = heightmap[x1, y2] if 0 <= x1 < heightmap.shape[0] and 0 <= y2 < heightmap.shape[1] else 0
                    x2y2 = heightmap[x2, y2] if 0 <= x2 < heightmap.shape[0] and 0 <= y2 < heightmap.shape[1] else 0

                    # height_point = (x2 - xm) * (y2 - ym) * x1y1 + (xm - x1) * (y2 - ym) * x2y1 + (x2 - xm) * (ym - y1) * x1y2 + (xm - x1) * (ym - y1) * x2y2
                    height_point = max(x1y1, x2y1, x1y2, x2y2)

                    self.height_samples[x, y] += height_point

    ###############################################################
    # Getters
    ###############################################################
    def get_humanoid_root_states(self):
        return self.humanoid_root_states[..., :7].clone()

    def get_num_actors_per_env(self):
        num_actors = (self.root_states.shape[0] - self.total_num_objects) // self.num_envs
        return num_actors

    def get_body_id(self, body_name):
        return self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], body_name)

    def get_body_positions(self):
        return self.rigid_body_pos.clone()

    def get_bodies_contact_buf(self):
        return self.contact_forces.clone()

    def get_dof_offsets(self):
        return self.dof_offsets

    def get_bodies_state(self):
        return self.rigid_body_pos.clone(), self.rigid_body_rot.clone(), self.rigid_body_vel.clone(), self.rigid_body_ang_vel.clone()

    def get_dof_state(self):
        return self.dof_pos.clone(), self.dof_vel.clone()

    def get_humanoid_root_velocities(self):
        return self.humanoid_root_states[:, 7:10].clone()

    ###############################################################
    # Environment step logic
    ###############################################################
    def apply_pd_control(self):
        pd_tar = self.action_to_pd_targets(self.actions)
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

    def apply_motor_forces(self):
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.reset_happened:
            env_ids = self.reset_ref_env_ids
            self.humanoid_root_states[env_ids, 0:3] = self.reset_states["root_pos"]
            self.humanoid_root_states[env_ids, 3:7] = self.reset_states["root_rot"]
            self.humanoid_root_states[env_ids, 7:10] = self.reset_states["root_vel"]
            self.humanoid_root_states[env_ids, 10:13] = self.reset_states["root_ang_vel"]

            self.dof_pos[env_ids] = self.reset_states["dof_pos"]
            self.dof_vel[env_ids] = self.reset_states["dof_vel"]

            self.rigid_body_pos[env_ids] = self.reset_states["rb_pos"]
            self.rigid_body_rot[env_ids] = self.reset_states["rb_rot"]
            self.rigid_body_vel[env_ids] = self.reset_states["rb_vel"]
            self.rigid_body_ang_vel[env_ids] = self.reset_states["rb_ang_vel"]
            self.reset_happened = False

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def post_physics_step(self):
        self.refresh_sim_tensors()

        super().post_physics_step()

        # debug viz
        if self.viewer and self.debug_viz:
            self.update_debug_viz()

        if self.export_video:
            self.camera_step()

        if self.export_motion:
            self.store_motion_data()

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self.reset_actors(env_ids)
            self.reset_env_tensors(env_ids)
            self.refresh_sim_tensors()
            self.compute_observations(env_ids)

    def reset_env_tensors(self, env_ids):
        super().reset_env_tensors(env_ids)

        env_ids_int32 = self.humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_default(self, env_ids):
        target_root_states = self.initial_humanoid_root_states[env_ids].clone()
        target_root_states[:, :2] = 0
        target_root_states[:, :3] += self.get_envs_respawn_position(env_ids)
        self.humanoid_root_states[env_ids] = target_root_states
        self.dof_pos[env_ids] = self.initial_dof_pos[env_ids]
        self.dof_vel[env_ids] = self.initial_dof_vel[env_ids]

    ###############################################################
    # Helpers
    ###############################################################
    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]),
                )

    def setup_character_props(self):
        key_bodies = self.config.key_bodies
        asset_file = self.config.asset.asset_file_name
        num_key_bodies = len(key_bodies)

        (
            self.dof_body_ids,
            self.dof_offsets,
            self.dof_obs_size,
            self.num_obs,
            self.num_act,
        ) = isaacgym_asset_file_to_stats(
            asset_file, num_key_bodies, self.config.use_max_coords_obs
        )

    def render(self):
        if self.viewer:
            self.update_camera()
            self.gym.clear_lines(self.viewer)
            self.draw_obj_bbox()

        super().render()

        

    def draw_obj_bbox(self):
        if len(self.object_id_to_object_position) == 0:
            return

        # draw lines of the bbox
        cols = np.zeros((12, 3), dtype=np.float32) # 12 lines
        cols[:] = [1.0, 0.0, 0.0] # red

        dir_col = np.zeros((1, 3), dtype=np.float32)
        dir_col[:] = [0.0, 1.0, 0.0]

        bbs = self.object_id_to_object_bounding_box

        verts = torch.cat([
            bbs[:, 0, :], bbs[:, 1, :],
            bbs[:, 1, :], bbs[:, 2, :],
            bbs[:, 2, :], bbs[:, 3, :],
            bbs[:, 3, :], bbs[:, 0, :],

            bbs[:, 4, :], bbs[:, 5, :],
            bbs[:, 5, :], bbs[:, 6, :],
            bbs[:, 6, :], bbs[:, 7, :],
            bbs[:, 7, :], bbs[:, 4, :],

            bbs[:, 0, :], bbs[:, 4, :],
            bbs[:, 1, :], bbs[:, 5, :],
            bbs[:, 2, :], bbs[:, 6, :],
            bbs[:, 3, :], bbs[:, 7, :],
        ], dim=-1).cpu().numpy()


        env_ptr = self.envs[0]
        for i in range(verts.shape[0]):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([12, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)


        object_pos = self.object_root_states[..., :3].clone()
        object_pos[..., -1] = 0.5
        object_rot = self.object_root_states[..., 3:7].clone()
        object_dir_verts = torch.zeros_like(self.object_root_states[..., :3])
        object_dir_verts[..., 0] = 1
        object_dir_verts = torch_utils.quat_rotate(object_rot, object_dir_verts, self.w_last)

        object_dir_verts += object_pos

        dir_verts = torch.cat(
            [object_pos, object_dir_verts], dim=-1
        ).cpu().numpy()
        dir_verts.reshape(-1, 6)

        self.gym.add_lines(self.viewer, env_ptr, dir_verts.shape[0], dir_verts, dir_col)

    def build_body_ids_tensor(self, body_names):
        body_ids = []

        for body_name in body_names:
            body_id = self.body_name_to_index(body_name)
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def body_name_to_index(self, body_name):
        return self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], body_name)

    ###############################################################
    # Camera logic
    ###############################################################
    def init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.cam_prev_char_pos = self.humanoid_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(
            self.cam_prev_char_pos[0], self.cam_prev_char_pos[1] - 3.0, self.cam_prev_char_pos[2] + 0.4
        )
        cam_target = gymapi.Vec3(
            self.cam_prev_char_pos[0], self.cam_prev_char_pos[1], self.cam_prev_char_pos[2] + 0.2
        )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self.cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2] + 0.2)
        new_cam_pos = gymapi.Vec3(
            char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], char_root_pos[2] + cam_delta[2]
        )

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self.cam_prev_char_pos[:] = char_root_pos

    def update_debug_viz(self):
        self.gym.clear_lines(self.viewer)

    def setup_cameras(self):
        self.cameras = []

        camera_config = self.config.camera

        camera_props = gymapi.CameraProperties()
        camera_props.width = camera_config.width
        camera_props.height = camera_config.height
        camera_offset = gymapi.Vec3(*camera_config.pos)
        camera_props.enable_tensors = True
        # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(*camera_config["axis"]), np.deg2rad(camera_config["angle"]))

        pitch = np.deg2rad(camera_config.pitch_deg)
        roll = np.deg2rad(camera_config.roll_deg)
        yaw = np.deg2rad(camera_config.yaw_deg)
        camera_rotation = gymapi.Quat.from_euler_zyx(roll, pitch, yaw)
        transform = gymapi.Transform(camera_offset, camera_rotation)

        # 

        # cam_pos = gymapi.Vec3(-11,-0, 6)
        # cam_target = gymapi.Vec3(0, -8, 0)

        follow = (
            gymapi.FOLLOW_TRANSFORM
            if camera_config["rotate_with_agent"]
            else gymapi.FOLLOW_POSITION
        )

        for env, han in zip(self.envs, self.humanoid_handles):
            body_handle = self.gym.get_actor_rigid_body_handle(env, han, 0)
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.attach_camera_to_body(
                camera_handle, env, body_handle, transform, follow
            )
            # self.gym.set_camera_location(camera_handle, env, cam_pos, cam_target)
            self.cameras.append(camera_handle)

        self.frames = [[] for _ in range(len(self.cameras))]
        self.cpu_frames = [[] for _ in range(len(self.cameras))]
        self.max_gpu_frames = 1000

    def camera_step(self):
        if self.config.record_viewer:
            viewer_record_dir = self.config.viewer_record_dir
            if not os.path.exists(viewer_record_dir):
                os.makedirs(viewer_record_dir)
            self.gym.write_viewer_image_to_file(self.viewer, viewer_record_dir + "/%04d.png" % len(self.frames[0]))

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i, han in enumerate(self.cameras):
            camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], han, gymapi.IMAGE_COLOR
            )
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            torch_camera_tensor = torch_camera_tensor[:, :, :3]
            self.frames[i].append(torch_camera_tensor.clone().cpu().numpy())

        self.gym.end_access_image_tensors(self.sim)

    def store_motion_data(self):
        root_pos = self.humanoid_root_states[..., 0:3].clone()
        body_rot = self.rigid_body_rot.clone()
        rot_mask = body_rot[..., -1] < 0
        body_rot[rot_mask] = -body_rot[rot_mask]
        motion_data = {
            "root_pos": root_pos.cpu(),
            "global_rot": body_rot.cpu(),
            "rigid_body_rot": self.rigid_body_rot.clone().cpu(),
            "rigid_body_pos": self.rigid_body_pos.clone().cpu(),
        }

        for key, item in motion_data.items():
            if key not in self.motion_recording:
                self.motion_recording[key] = []
            self.motion_recording[key].append(item)

    def convert_to_global_coords(self, humanoid_root_states, env_offsets):
        return humanoid_root_states + env_offsets
