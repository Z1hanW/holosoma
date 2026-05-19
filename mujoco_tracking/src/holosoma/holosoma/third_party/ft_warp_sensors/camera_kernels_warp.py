import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)
RAY_EPS = wp.constant(1.0e-6)


@wp.func
def _intersect_axis_aligned_box(
    ro: wp.vec3,
    rd: wp.vec3,
    half_extents: wp.vec3,
    max_dist: float,
) -> float:
    t_min = wp.float32(0.0)
    t_max = wp.float32(max_dist)

    for axis in range(3):
        origin = ro[axis]
        direction = rd[axis]
        extent = half_extents[axis]
        if extent <= wp.float32(0.0):
            return wp.float32(NO_HIT_RAY_VAL)

        if wp.abs(direction) < wp.float32(RAY_EPS):
            if origin < -extent or origin > extent:
                return wp.float32(NO_HIT_RAY_VAL)
            continue

        inv_direction = wp.float32(1.0) / direction
        t1 = (-extent - origin) * inv_direction
        t2 = (extent - origin) * inv_direction
        t_near = wp.min(t1, t2)
        t_far = wp.max(t1, t2)
        t_min = wp.max(t_min, t_near)
        t_max = wp.min(t_max, t_far)
        if t_max < t_min:
            return wp.float32(NO_HIT_RAY_VAL)

    if t_max < wp.float32(0.0):
        return wp.float32(NO_HIT_RAY_VAL)
    if t_min >= wp.float32(0.0):
        return t_min
    return t_max


class DepthCameraWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        terrain_mesh_id: wp.uint64,
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        c_x: int,
        c_y: int,
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = terrain_mesh_id
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.normalize(wp.transform_vector(K_inv, cam_coords))
        uv_principal = wp.normalize(
            wp.transform_vector(K_inv, cam_coords_principal)
        )  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
            dist = t
        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = ro + dist * rd
        else:
            pixels[env_id, cam_id, y, x] = dist * uv

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range(
        terrain_mesh_id: wp.uint64,
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),
        cam_quats: wp.array(dtype=wp.quat, ndim=2),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float, ndim=4),
        c_x: int,
        c_y: int,
        calculate_depth: bool,
    ):

        env_id, cam_id, x, y = wp.tid()
        mesh = terrain_mesh_id
        cam_pos = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]
        cam_coords = wp.vec3(
            float(x), float(y), 1.0
        )  # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
        cam_coords_principal = wp.vec3(
            float(c_x), float(c_y), 1.0
        )  # get the vector of principal axis
        # transform to uv [-1,1]
        uv = wp.transform_vector(K_inv, cam_coords)
        uv_principal = wp.transform_vector(K_inv, cam_coords_principal)  # uv for principal axis
        # compute camera ray
        # cam origin in world space
        ro = cam_pos
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_principal = wp.normalize(
            wp.quat_rotate(cam_quat, uv_principal)
        )  # ray direction of principal axis
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)
        multiplier = 1.0
        if calculate_depth:
            multiplier = wp.dot(
                rd, rd_principal
            )  # multiplier to project each ray on principal axis for depth instead of range
        dist = NO_HIT_RAY_VAL
        if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
            dist = multiplier * t

        pixels[env_id, cam_id, y, x] = dist

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range_dynamic(
        # --- static terrain ---
        terrain_id: wp.uint64,
        # --- robot bodies: canonical meshes (one per body, shared across envs) ---
        robot_ids: wp.array(dtype=wp.uint64),            # [num_bodies]
        primitive_active: wp.array(dtype=wp.int32, ndim=2),     # [num_envs, num_primitive_bodies]
        primitive_half_extents: wp.array(dtype=wp.vec3, ndim=2),  # [num_envs, num_primitive_bodies]
        # --- per-env body poses in world ---
        body_poss: wp.array(dtype=wp.vec3, ndim=2),      # [num_envs, num_bodies]
        body_quats: wp.array(dtype=wp.quat,  ndim=2),    # [num_envs, num_bodies]
        primitive_poss: wp.array(dtype=wp.vec3, ndim=2),   # [num_envs, num_primitive_bodies]
        primitive_quats: wp.array(dtype=wp.quat, ndim=2),  # [num_envs, num_primitive_bodies]
        # --- cameras ---
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),       # [num_envs, num_cams]
        cam_quats: wp.array(dtype=wp.quat,  ndim=2),     # [num_envs, num_cams]
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.float32, ndim=4),      # [env, cam, H, W]
        c_x: int,
        c_y: int,
        calculate_depth: bool,
        num_bodies: int,
        num_primitive_bodies: int,
    ):
        env_id, cam_id, x, y = wp.tid()

        # camera ray in world
        cam_pos  = cam_poss[env_id, cam_id]
        cam_quat = cam_quats[env_id, cam_id]

        uv   = wp.transform_vector(K_inv, wp.vec3(float(x),   float(y),   1.0))
        uv_c = wp.transform_vector(K_inv, wp.vec3(float(c_x), float(c_y), 1.0))

        ro   = cam_pos
        rd   = wp.normalize(wp.quat_rotate(cam_quat, uv))
        rd_c = wp.normalize(wp.quat_rotate(cam_quat, uv_c))   # principal axis

        # depth-vs-range multiplier (clamped)
        mul = wp.float32(1.0)
        if calculate_depth:
            d   = wp.dot(rd, rd_c)
            eps = wp.float32(1.0e-6)
            d = wp.max(d, eps)
            d = wp.min(d, wp.float32(1.0))
            mul = d

        best = wp.float32(NO_HIT_RAY_VAL)
        far_bound_world = wp.float32(far_plane)

        # ---------- 1) robot bodies (transform ray to each body local) ----------
        for b in range(num_bodies):
            qb = body_quats[env_id, b]
            tb = body_poss[env_id,  b]

            # world -> body local
            ro_l = wp.quat_rotate_inv(qb, ro - tb)
            rd_l = wp.quat_rotate_inv(qb, rd)

            # fresh writable outputs for each call
            t = float(0.0)
            u = float(0.0)
            v = float(0.0)
            sign = float(0.0)
            n = wp.vec3()
            f = int(0)

            if wp.mesh_query_ray(robot_ids[b], ro_l, rd_l, far_bound_world / mul, t, u, v, sign, n, f):
                d = mul * t
                if (best == NO_HIT_RAY_VAL) or (d < best):
                    best = d
                    far_bound_world = best

        # ---------- 2) analytic primitive boxes (per-env pose + half extents) ----------
        for b in range(num_primitive_bodies):
            if primitive_active[env_id, b] == 0:
                continue

            qb = primitive_quats[env_id, b]
            tb = primitive_poss[env_id, b]
            half_extents = primitive_half_extents[env_id, b]

            ro_l = wp.quat_rotate_inv(qb, ro - tb)
            rd_l = wp.quat_rotate_inv(qb, rd)
            t_box = _intersect_axis_aligned_box(ro_l, rd_l, half_extents, far_bound_world / mul)
            if t_box != wp.float32(NO_HIT_RAY_VAL):
                d = mul * t_box
                if (best == NO_HIT_RAY_VAL) or (d < best):
                    best = d
                    far_bound_world = best

        # ---------- 2) terrain (static, world frame) ----------
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)

        if wp.mesh_query_ray(terrain_id, ro, rd, far_bound_world / mul, t, u, v, sign, n, f):
            best = mul * t
            far_bound_world = best

        pixels[env_id, cam_id, y, x] = best
    
    # @staticmethod
    # @wp.kernel
    # def memset_pixels4(
    #     pixels: wp.array(dtype=wp.float32, ndim=4),  # [env, cam, H, W]
    #     value: float,
    # ):
    #     env_id, cam_id, x, y = wp.tid()
    #     pixels[env_id, cam_id, y, x] = value

    # @staticmethod
    # @wp.kernel
    # def draw_optimized_kernel_depth_range_dynamic_singlepass_4d(
    #     # --- static terrain ---
    #     terrain_id: wp.uint64,
    #     # --- robot canonical meshes (shared) ---
    #     robot_ids: wp.array(dtype=wp.uint64),            # [num_bodies]
    #     # --- per-env body poses in world ---
    #     body_poss:  wp.array(dtype=wp.vec3, ndim=2),     # [num_envs, num_bodies]
    #     body_quats: wp.array(dtype=wp.quat,  ndim=2),    # [num_envs, num_bodies]
    #     # --- cameras ---
    #     cam_poss:   wp.array(dtype=wp.vec3, ndim=2),     # [num_envs, num_cams]
    #     cam_quats:  wp.array(dtype=wp.quat,  ndim=2),    # [num_envs, num_cams]
    #     K_inv: wp.mat44,
    #     far_plane: float,
    #     pixels_flat: wp.array(dtype=wp.float32),
    #     num_cams: int,
    #     width: int,
    #     height: int,
    #     c_x: int,
    #     c_y: int,
    #     calculate_depth: bool,
    #     no_hit_value: float,
    # ):
    #     env_id, camk_id, x, y = wp.tid()

    #     num_bodies = robot_ids.shape[0]
    #     span = num_bodies + 1                 
    #     cam_id = camk_id // span                  
    #     k      = camk_id - cam_id * span

    #     idx = (((env_id * num_cams) + cam_id) * height + y) * width + x

    #     cam_pos  = cam_poss[env_id, cam_id]
    #     cam_quat = cam_quats[env_id, cam_id]

    #     uv   = wp.transform_vector(K_inv, wp.vec3(float(x),   float(y),   1.0))
    #     uv_c = wp.transform_vector(K_inv, wp.vec3(float(c_x), float(c_y), 1.0))

    #     ro   = cam_pos
    #     rd   = wp.normalize(wp.quat_rotate(cam_quat, uv))
    #     rd_c = wp.normalize(wp.quat_rotate(cam_quat, uv_c))

    #     mul = wp.float32(1.0)
    #     if calculate_depth:
    #         d = wp.dot(rd, rd_c)
    #         eps = wp.float32(1.0e-6)
    #         d = wp.max(d, eps)
    #         d = wp.min(d, wp.float32(1.0))
    #         mul = d

    #     t = float(0.0)
    #     u = float(0.0)
    #     v = float(0.0)
    #     sign = float(0.0)
    #     n = wp.vec3()
    #     f = int(0)

    #     dist = wp.float32(no_hit_value)
    #     is_terrain = (k == num_bodies)

    #     if is_terrain:
    #         if wp.mesh_query_ray(terrain_id, ro, rd, wp.float32(far_plane)/mul, t, u, v, sign, n, f):
    #             dist = mul * t
    #     else:
    #         qb = body_quats[env_id, k]
    #         tb = body_poss[env_id,  k]
    #         ro_l = wp.quat_rotate_inv(qb, ro - tb)
    #         rd_l = wp.quat_rotate_inv(qb, rd)
    #         if wp.mesh_query_ray(robot_ids[k], ro_l, rd_l, wp.float32(far_plane)/mul, t, u, v, sign, n, f):
    #             dist = mul * t

    #     wp.atomic_min(pixels_flat, idx, dist)
