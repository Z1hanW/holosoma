# import nvtx
import warp as wp
import math
import torch
from .sensor_utils import (
    quat_from_euler_xyz_tensor,
    torch_rand_float_tensor
)
from .camera_kernels_warp import (
    DepthCameraWarpKernels,
)
from .base_sensor import BaseSensor

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

class CameraSensor(BaseSensor):
    def __init__(
        self, num_envs, config, terrain, device="cuda:0",
    ):
        super().__init__(num_envs, config, terrain, device)
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        all_camera_names = list(self.cfg.base_link_frame.keys())
        self.camera_names = []
        for i in range(self.num_sensors):
            self.camera_names.append(all_camera_names[i])

        self.width = self.cfg.width
        self.height = self.cfg.height

        self.horizontal_fov = math.radians(self.cfg.horizontal_fov_deg)
        self.far_plane = self.cfg.max_range
        self.calculate_depth = self.cfg.calculate_depth
        self.device = device

        self.camera_position_array = None
        self.camera_orientation_array = None
        self.graph = None

        # Initialize camera matrices
        self.initialize_camera_matrices()
        # Initialize camera tensors
        self.create_warp_camera_tensors()
        self._synchronize_cuda_initialization()

    def _synchronize_cuda_initialization(self):
        """Publish Warp mesh/BVH and Torch storage before cross-stream capture.

        Warp may build meshes on its own CUDA stream while camera tensors are
        initialized on PyTorch's current stream.  Later captures deliberately
        run on the caller's Torch stream, which is not implicitly ordered
        after either setup stream.  A one-time setup barrier makes every mesh,
        mapping and pose allocation visible before the sensor is returned;
        collection itself remains fully asynchronous.
        """

        torch_device = torch.device(self.device)
        if torch_device.type != "cuda":
            return
        wp.synchronize_device(self.device)
        torch.cuda.synchronize(device=torch_device)

    def initialize_camera_matrices(self):
        # Calculate camera params
        W = self.width
        H = self.height
        (u_0, v_0) = (W / 2, H / 2)
        f = W / 2 * 1 / math.tan(self.horizontal_fov / 2)

        vertical_fov = 2 * math.atan(H / (2 * f))
        alpha_u = u_0 / math.tan(self.horizontal_fov / 2)
        alpha_v = v_0 / math.tan(vertical_fov / 2)

        # simple pinhole model
        self.K = wp.mat44(
            alpha_u,
            0.0,
            u_0,
            0.0,
            0.0,
            alpha_v,
            v_0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.K_inv = wp.inverse(self.K)

        self.c_x = int(u_0)
        self.c_y = int(v_0)
    
    def create_warp_camera_tensors(self):
        """Create all camera-related tensors and attach them to the CameraSensor.

        Returns a dict with created tensors and base link indices.
        """
        # Create camera image/segmentation tensors
        self.depth_tensors = torch.zeros(
            (
                self.num_envs,
                self.num_sensors,
                self.cfg.height,
                self.cfg.width,
            ),
            device=self.device,
            requires_grad=False,
        )
        self.segmentation_tensors = torch.zeros(
            (
                self.num_envs,
                self.num_sensors,
                self.cfg.height,
                self.cfg.width,
            ),
            device=self.device,
            requires_grad=False,
        )
        # Kernels index simulator state and the full output tensor through this
        # launch-row -> global-environment mapping.  The captured full-batch
        # graph uses the immutable identity mapping.  Reset-only captures use a
        # compact K-row mapping, so they launch K * cameras * pixels rays rather
        # than launching the full grid and returning early behind a mask.
        self.full_capture_env_ids_tensor = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.int32,
            requires_grad=False,
        )
        self.full_capture_env_ids = wp.from_torch(
            self.full_capture_env_ids_tensor, dtype=wp.int32
        )
        # Keep the compact tensor/Warp view alive until a later capture.  Warp
        # borrows Torch storage and its launch is asynchronous.
        self._capture_env_ids_tensor = self.full_capture_env_ids_tensor
        self._capture_env_ids = self.full_capture_env_ids
        # Create camera sensor pose tensors
        euler_sensor_frame_rot = self.cfg.offset_rot_base
        sensor_frame_rot_rad = torch.deg2rad(
            torch.tensor(euler_sensor_frame_rot, device=self.device, requires_grad=False)
        )
        sensor_quat = quat_from_euler_xyz_tensor(sensor_frame_rot_rad)
        # This tensor is updated later by the perception manager, so it must own
        # its storage instead of being an expanded view with overlapping memory.
        self.camera_sensor_data_frame_quat = sensor_quat.expand(self.num_envs, self.num_sensors, -1).clone()
        self.camera_sensor_local_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.camera_sensor_local_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.camera_sensor_local_orientation[..., 3] = 1.0
        # Define min and max translation and rotation
        self.camera_sensor_translation = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.camera_sensor_rotation = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        for cam_id, cam_name in enumerate(self.camera_names):
            camera_sensor_translation = self.cfg.offset[cam_name]['offset_pos']
            camera_sensor_rotation = torch.tensor(
                self.cfg.offset[cam_name]['offset_rot'], device=self.device, requires_grad=False
            )
            self.camera_sensor_translation[:, cam_id, :] = torch.tensor(camera_sensor_translation, device=self.device, requires_grad=False)
            self.camera_sensor_rotation[:, cam_id, :] = torch.tensor(camera_sensor_rotation, device=self.device, requires_grad=False)
        
        # Randomize placement of the sensor
        if self.cfg.randomize_placement == True:
            camera_sensor_min_translation = torch.zeros_like(self.camera_sensor_local_position)
            camera_sensor_max_translation = torch.zeros_like(self.camera_sensor_local_position)
            # rpy and xyz both have 3 dimensions anyways :D
            camera_sensor_min_rotation = torch.zeros_like(self.camera_sensor_local_position)
            camera_sensor_max_rotation = torch.zeros_like(self.camera_sensor_local_position)

            for cam_id, cam_name in enumerate(self.camera_names):
                camera_sensor_min_translation[:, cam_id, :] = torch.tensor(
                    self.cfg.min_translation[cam_name], device=self.device, requires_grad=False
                ) + self.camera_sensor_translation[:, cam_id, :]
                camera_sensor_max_translation[:, cam_id, :] = torch.tensor(
                    self.cfg.max_translation[cam_name], device=self.device, requires_grad=False
                ) + self.camera_sensor_translation[:, cam_id, :]
                camera_sensor_min_rotation[:, cam_id, :] = torch.deg2rad(
                    torch.tensor(
                        self.cfg.min_euler_rotation_deg[cam_name], device=self.device, requires_grad=False
                    ) + self.camera_sensor_rotation[:, cam_id, :]
                )
                camera_sensor_max_rotation[:, cam_id, :] = torch.deg2rad(
                    torch.tensor(
                        self.cfg.max_euler_rotation_deg[cam_name], device=self.device, requires_grad=False
                    ) + self.camera_sensor_rotation[:, cam_id, :]
                )
            # sample local position from min and max translations
            self.camera_sensor_local_position[:] = torch_rand_float_tensor(
                camera_sensor_min_translation[:],
                camera_sensor_max_translation[:],
            )
            # sample local orientation from min and max rotations
            local_euler_rotation = torch_rand_float_tensor(
                camera_sensor_min_rotation[:], camera_sensor_max_rotation[:]
            )
            self.camera_sensor_local_orientation[:] = quat_from_euler_xyz(
                local_euler_rotation[..., 0],
                local_euler_rotation[..., 1],
                local_euler_rotation[..., 2],
            )
        else:
            self.camera_sensor_local_position[:] = self.camera_sensor_translation
            camera_sensor_local_orientation = torch.deg2rad(self.camera_sensor_rotation)
            self.camera_sensor_local_orientation[:] = quat_from_euler_xyz(
                camera_sensor_local_orientation[..., 0],
                camera_sensor_local_orientation[..., 1],
                camera_sensor_local_orientation[..., 2],
            )
        
        # Initialize GLOBAL camera sensor position and orientation
        self.camera_sensor_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.camera_sensor_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.camera_sensor_orientation[..., 3] = 1.0
        self.set_pose_tensor(
            positions=self.camera_sensor_position, orientations=self.camera_sensor_orientation
        )
        # The base simulator creates pixels with shape (num_envs, num_sensors, height, width)
        # The warp kernel expects (num_envs, num_sensors, height, width) for depth mode
        # So we can pass the tensor directly since num_sensors == num_sensors
        self.set_image_tensors(
            pixels=self.depth_tensors, segmentation_pixels=self.segmentation_tensors
        )
        # Set No Hit Value
        self.no_hit = float(self.far_plane) + 1.0

    def create_render_graph_pointcloud(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:
            raise ValueError("Segmentation camera is not supported for pointcloud")
        else:
            self._launch_pointcloud(
                self.full_capture_env_ids,
                self.num_envs,
            )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def create_render_graph_depth_range(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        # with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:
            raise ValueError("Segmentation camera is not supported for depth range")
        else:
            if not self.is_dyna_mesh:
                self._launch_depth_range(
                    self.full_capture_env_ids,
                    self.num_envs,
                )
            else:
                # Dynamic mesh support
                self._launch_depth_range(
                    self.full_capture_env_ids,
                    self.num_envs,
                )
                # # 1) refill with no_hit
                # wp.launch(
                #     DepthCameraWarpKernels.memset_pixels4,
                #     dim=(self.num_envs, self.num_sensors, self.width, self.height),
                #     inputs=[self.pixels, self.no_hit],
                #     device=self.device,
                # )

                # # 2) single pass (4D)
                # wp.launch(
                #     DepthCameraWarpKernels.draw_optimized_kernel_depth_range_dynamic_singlepass_4d,
                #     dim=(self.num_envs, self.num_sensors * (self.num_robot_bodies + 1), self.width, self.height),
                #     inputs=[
                #         self.terrain_mesh_id,
                #         self.robot_mesh_ids,
                #         self.ray_cast_body_poses,
                #         self.ray_cast_body_quats,
                #         self.camera_position_array,
                #         self.camera_orientation_array,
                #         self.K_inv,
                #         self.far_plane,
                #         self.pixels_flat,        # 1D view used by kernel for atomic_min
                #         self.num_sensors,        # num_cams (only pass real camera count!)
                #         self.width,
                #         self.height,
                #         self.c_x,
                #         self.c_y,
                #         self.calculate_depth,
                #         self.no_hit,                  # same as memset
                #     ],
                #     device=self.device,
                # )
        if not debug:
            print(f"finishing capture of render graph")
            self.graph = wp.capture_end(device=self.device)

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        # Convert to warp tensors for processing
        if self.cfg.return_pointcloud:
            self.pixels = wp.from_torch(pixels, dtype=wp.vec3)
            self.pointcloud_in_world_frame = self.cfg.pointcloud_in_world_frame
        else:
            self.pixels = wp.from_torch(pixels, dtype=wp.float32)
            self.pixels_flat = wp.from_torch(pixels.view(-1), dtype=wp.float32)
        self.pixels_tensors = torch.zeros_like(pixels)
        self.pixels_flat_tensors = torch.zeros_like(pixels.view(-1))

        if self.cfg.segmentation_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels, dtype=wp.int32)
        else:
            self.segmentation_pixels = segmentation_pixels
        
    def set_pose_tensor(self, positions, orientations):
        self.camera_position_array = wp.from_torch(positions, dtype=wp.vec3)
        self.camera_orientation_array = wp.from_torch(orientations, dtype=wp.quat)

    def set_capture_env_ids(self, env_ids=None):
        """Prepare a launch-row -> global-env mapping and return its row count.

        ``index_select`` both preserves caller order/duplicates and validates
        indices before Warp consumes them.  Keeping the selected tensor on the
        sensor is required because ``wp.from_torch`` is a zero-copy view and
        launches are asynchronous.
        """
        if env_ids is None:
            self._capture_env_ids_tensor = self.full_capture_env_ids_tensor
            self._capture_env_ids = self.full_capture_env_ids
            return self.num_envs
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        self._capture_env_ids_tensor = torch.index_select(
            self.full_capture_env_ids_tensor,
            0,
            env_ids,
        )
        self._capture_env_ids = wp.from_torch(
            self._capture_env_ids_tensor, dtype=wp.int32
        )
        return env_ids.numel()

    def _current_warp_stream(self):
        torch_device = torch.device(self.device)
        if torch_device.type != "cuda":
            return None
        torch_stream = torch.cuda.current_stream(device=torch_device)
        return wp.stream_from_torch(torch_stream)

    def _launch_pointcloud(self, launch_env_ids, launch_env_count, stream=None):
        wp.launch(
            kernel=DepthCameraWarpKernels.draw_optimized_kernel_pointcloud,
            dim=(launch_env_count, self.num_sensors, self.width, self.height),
            inputs=[
                self.terrain_mesh_id,
                launch_env_ids,
                self.camera_position_array,
                self.camera_orientation_array,
                self.K_inv,
                self.far_plane,
                self.pixels,
                self.c_x,
                self.c_y,
                self.pointcloud_in_world_frame,
            ],
            device=self.device,
            stream=stream,
        )

    def _launch_depth_range(self, launch_env_ids, launch_env_count, stream=None):
        if not self.is_dyna_mesh:
            kernel = DepthCameraWarpKernels.draw_optimized_kernel_depth_range
            inputs = [
                self.terrain_mesh_id,
                launch_env_ids,
                self.camera_position_array,
                self.camera_orientation_array,
                self.K_inv,
                self.far_plane,
                self.pixels,
                self.c_x,
                self.c_y,
                self.calculate_depth,
            ]
        else:
            kernel = DepthCameraWarpKernels.draw_optimized_kernel_depth_range_dynamic
            inputs = [
                self.terrain_mesh_id,
                launch_env_ids,
                self.robot_mesh_ids,
                self.primitive_body_active,
                self.primitive_body_half_extents,
                self.ray_cast_body_poses,
                self.ray_cast_body_quats,
                self.primitive_body_poses,
                self.primitive_body_quats,
                self.camera_position_array,
                self.camera_orientation_array,
                self.K_inv,
                self.far_plane,
                self.pixels,
                self.c_x,
                self.c_y,
                self.calculate_depth,
                self.num_robot_bodies,
                len(self.primitive_bodies),
            ]
        wp.launch(
            kernel=kernel,
            dim=(launch_env_count, self.num_sensors, self.width, self.height),
            inputs=inputs,
            device=self.device,
            stream=stream,
        )

    # @nvtx.annotate()
    def capture(self, debug=False, active_env_ids=None):
        if active_env_ids is not None:
            launch_env_count = self.set_capture_env_ids(active_env_ids)
            if launch_env_count:
                stream = self._current_warp_stream()
                if self.cfg.return_pointcloud:
                    self._launch_pointcloud(
                        self._capture_env_ids,
                        launch_env_count,
                        stream=stream,
                    )
                else:
                    self._launch_depth_range(
                        self._capture_env_ids,
                        launch_env_count,
                        stream=stream,
                    )
            self.pixels_tensors = wp.to_torch(self.pixels)
            return self.pixels_tensors

        if self.graph is None:
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud(debug=debug)
            else:
                self.create_render_graph_depth_range(debug=debug)
        if self.graph is not None:
            # Pose/mapping tensors are produced by PyTorch immediately before
            # capture, and the returned depth is consumed by PyTorch
            # immediately afterwards.  Launch on the active Torch stream so
            # this ordering remains correct even when the caller uses a
            # non-default CUDA stream.
            wp.capture_launch(self.graph, stream=self._current_warp_stream())

        self.pixels_tensors = wp.to_torch(self.pixels)
        # Apply noise
        # self.apply_noise_vanilla()
        return self.pixels_tensors
    
    # def apply_noise_vanilla(self): # TODO: do it in _process_depth_images()
    #     if self.cfg.sensor_noise.enable_sensor_noise == True:
    #         # logger.debug("Applying sensor noise")
    #         self.pixels_tensors[:] = torch.normal(
    #             mean=self.pixels_tensors, std=self.cfg.sensor_noise.pixel_std_dev_multiplier * self.pixels_tensors
    #         )
    #         # self.pixels_tensors[
    #         #     torch.bernoulli(torch.ones_like(self.pixels_tensors) * self.cfg.sensor_noise.pixel_dropout_prob) > 0
    #         # ] = self.cfg.near_out_of_range_value
    
    # def apply_noise(self):
    #     if self.cfg.sensor_noise.enable_sensor_noise == True:
    #         # logger.debug("Applying sensor noise")
    #         sensor_noise_params = self.cfg.sensor_noise
    #         std_a = sensor_noise_params.std_a
    #         std_b = sensor_noise_params.std_b
    #         std_c = sensor_noise_params.std_c
    #         mean_offset = sensor_noise_params.mean_offset
    #         std_val = std_a * self.pixels_tensors**2 + std_b * self.pixels_tensors + std_c
    #         self.pixels_tensors[:] = torch.normal(
    #             mean= (self.pixels_tensors - mean_offset),
    #             std=std_val
    #         )
    #         # self.pixels_tensors[
    #         #     torch.bernoulli(
    #         #         torch.ones_like(self.pixels_tensors) * self.cfg.sensor_noise.pixel_dropout_prob
    #         #     )
    #         #     > 0
    #         # ] = self.cfg.near_out_of_range_value
