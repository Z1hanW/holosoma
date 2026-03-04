from abc import ABC

import numpy as np
import torch
import warp as wp
import trimesh
import os
from .sensor_utils import (
    parse_urdf_meshes,
    convert_to_warp_mesh,
)

class BaseSensor(ABC):
    """Base class and shared utilities for Warp-based sensors."""
    def __init__(
        self, 
        num_envs, 
        config, 
        terrain, 
        device="cuda:0",
    ):
        self.num_envs = num_envs
        self.device = device
        self.ray_cast_bodies = list(config.ray_cast_bodies.keys())
        
        # Offpath obstacle bodies
        self.add_offpath_obstacle = config.add_offpath_obstacle
        if self.add_offpath_obstacle:
            self.ray_cast_bodies.extend(list(config.offpath_obstacle_bodies.keys()))
            
        self.is_dyna_mesh = config.dynamic_meshes
        self.init_warp(device)
        self.build_warp_meshes(
            terrain, 
            config, 
        )
        self.init_ray_cast_body_poses_and_quats()
    
    def init_ray_cast_body_poses_and_quats(self):
        self.ray_cast_body_poses_tensor = torch.zeros(
            self.num_envs, len(self.ray_cast_bodies), 3, device=self.device
        )
        self.ray_cast_body_quats_tensor = torch.zeros(
            self.num_envs, len(self.ray_cast_bodies), 4, device=self.device
        )
        self.ray_cast_body_quats_tensor[..., 3] = 1.0
        self.ray_cast_body_poses = wp.from_torch(
            self.ray_cast_body_poses_tensor.view(self.num_envs, len(self.ray_cast_bodies), 3), dtype=wp.vec3
        )
        self.ray_cast_body_quats = wp.from_torch(
            self.ray_cast_body_quats_tensor.view(self.num_envs, len(self.ray_cast_bodies), 4), dtype=wp.quat
        )

    def init_warp(self, device):
        """Initialize Warp and set device safely."""
        wp.init()
        try:
            wp.set_device(device)
        except Exception as e:
            print(e)
            # Older warp versions may not require explicit set_device
            raise Exception("Warp version is too old. Please update warp to the latest version.")

    def build_warp_meshes(
        self, 
        terrain, 
        config, 
    ):
        """Build a Warp mesh from terrain (plane/heightfield/trimesh) and return per-env mesh ids.

        Returns (wp_mesh, mesh_ids_wp_array)
        """
        # Initialize robot body meshes
        self.robot_body_meshes = []
        self.num_robot_bodies = 0
        # Parse URDF to get mesh filenames for ray_cast_bodies on the robot
        # body_meshes_dict = parse_urdf_meshes(robot_config)
        # Directly get the body meshes from the robot config
        body_meshes_dict = config.ray_cast_bodies
        asset_meshes_root = config.asset_meshes_root
        # Load and combine robot body meshes
        try:
            for mesh_file in body_meshes_dict.values():
                body_mesh = trimesh.load(os.path.join(asset_meshes_root, mesh_file))
                self.robot_body_meshes.append(
                    convert_to_warp_mesh(body_mesh.vertices, body_mesh.faces, device=self.device)
                )
                self.num_robot_bodies += 1
        except Exception as e:
            print(f"Error loading mesh {mesh_file}: {e}")
            raise Exception(f"Error loading mesh {mesh_file}: {e}")

        def add_obstacle_meshes(obstacle_meshes_root, obstacle_meshes_dict):
            try:
                for mesh_file in obstacle_meshes_dict.values():
                    obstacle_mesh = trimesh.load(os.path.join(obstacle_meshes_root, mesh_file))
                    self.robot_body_meshes.append(
                        convert_to_warp_mesh(obstacle_mesh.vertices, obstacle_mesh.faces, device=self.device)
                    )
                    self.num_robot_bodies += 1
            except Exception as e:
                print(f"Error loading offpath obstacle mesh {mesh_file}: {e}")
                raise Exception(f"Error loading offpath obstacle mesh {mesh_file}: {e}")

        # Add offpath obstacle meshes
        if self.add_offpath_obstacle:
            add_obstacle_meshes(config.offpath_obstacle_meshes_root, config.offpath_obstacle_bodies)

        # Convert to warp mesh ids
        self.robot_mesh_ids = wp.array(
            [self.robot_body_meshes[i].id for i in range(len(self.robot_body_meshes))], 
            dtype=wp.uint64, device=self.device
        ) 

        # Initialize terrain mesh
        try:
            # Add terrain mesh to the list of meshes to combine
            self.terrain_mesh = convert_to_warp_mesh(terrain.vertices, terrain.faces, device=self.device)
            self.terrain_mesh_id = self.terrain_mesh.id
        except Exception as e:
            raise Exception(f"Failed to load terrain mesh: {str(e)}")
