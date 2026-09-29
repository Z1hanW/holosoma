import torch
import warp as wp
import numpy as np
import xml.etree.ElementTree as ET
import os

@torch.jit.script
def quat_apply_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to a vector.

    Parameters
    ----------
    a : torch.Tensor
        Quaternion of shape (..., 4) in xyzw order
    b : torch.Tensor
        Vector to rotate of shape (..., 3)

    Returns
    -------
    torch.Tensor
        Rotated vector of shape (..., 3)
    """
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def tf_apply_xyzw(q: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply a transformation to a vector.

    Parameters
    ----------
    q : torch.Tensor
        Quaternion of shape (..., 4) in xyzw order
    t : torch.Tensor
        Translation of shape (..., 3)
    v : torch.Tensor
        Vector to transform of shape (..., 3)

    Returns
    -------
    torch.Tensor
        Transformed vector of shape (..., 3)
    """
    return quat_apply_xyzw(q, v) + t

def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.

    Parameters
    ----------
    a : torch.Tensor
        First quaternion of shape (..., 4) in xyzw order
    b : torch.Tensor
        Second quaternion of shape (..., 4) in xyzw order

    Returns
    -------
    torch.Tensor
        Resulting quaternion of shape (..., 4) in xyzw order
    """
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([x, y, z, w], dim=-1).view(shape)

@torch.jit.script
def torch_rand_float_tensor(lower, upper):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    return (upper - lower) * torch.rand_like(upper) + lower

@torch.jit.script
def quat_from_euler_xyz_tensor(euler_xyz_tensor: torch.Tensor) -> torch.Tensor:
    roll = euler_xyz_tensor[..., 0]
    pitch = euler_xyz_tensor[..., 1]
    yaw = euler_xyz_tensor[..., 2]
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

@torch.jit.script
def cart2sphere(cart):
    epsilon = 1e-9
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    r = torch.norm(cart, dim=1)
    theta = torch.atan2(y, x)
    phi = torch.asin(z / (r + epsilon))
    return torch.stack((r, theta, phi), dim=-1)


def farthest_point_sampling(point_cloud, sample_size):
    """
    Sample points using the farthest point sampling algorithm
    Args:
        point_cloud: Tensor of shape (num_envs, 1, num_points,1, 3)
        sample_size: Number of points to sample
    Returns:
        Downsampled point cloud of shape (num_envs, 1, sample_size, 3)
    """
    num_envs, _, num_points, _ = point_cloud.shape
    device = point_cloud.device
    result = []

    for env_idx in range(num_envs):
        points = point_cloud[env_idx, 0]  # (num_points, 3)

        # Initialize with a random point
        sampled_indices = torch.zeros(sample_size, dtype=torch.long, device=device)
        sampled_indices[0] = torch.randint(0, num_points, (1,), device=device)

        # Calculate distances
        distances = torch.norm(points - points[sampled_indices[0]], dim=1)

        # Iteratively select farthest points
        for i in range(1, sample_size):
            # Select the farthest point
            sampled_indices[i] = torch.argmax(distances)

            # Update distances
            if i < sample_size - 1:
                new_distances = torch.norm(points - points[sampled_indices[i]], dim=1)
                distances = torch.min(distances, new_distances)

        # Get the sampled points
        sampled_points = points[sampled_indices]
        result.append(sampled_points.unsqueeze(0))  # Add sensor dimension back

    return torch.stack(result)

def downsample_spherical_points_vectorized(sphere_points, num_theta_bins=10, num_phi_bins=10):
    """
    Downsample points in spherical coordinates by binning theta and phi values.

    Args:
        sphere_points: Tensor of shape (num_envs, num_points, 3) where dim 2 is (r, theta, phi)
        num_theta_bins: Number of bins for theta range (-3.14, 3.14)
        num_phi_bins: Number of bins for phi range (-0.12, 0.907)

    Returns:
        Downsampled points tensor of shape (num_envs, num_theta_bins*num_phi_bins, 3)
    """
    num_envs = sphere_points.shape[0]
    num_points = sphere_points.shape[1]
    device = sphere_points.device
    num_bins = num_theta_bins * num_phi_bins

    # Define bin ranges
    theta_min, theta_max = -3.14, 3.14
    phi_min, phi_max = -0.12, 0.907

    # Extract r, theta, phi for all environments
    r = sphere_points[:, :, 0]       # [num_envs, num_points]
    theta = sphere_points[:, :, 1]   # [num_envs, num_points]
    phi = sphere_points[:, :, 2]     # [num_envs, num_points]

    # Compute bin indices for theta and phi
    theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
    phi_bin = ((phi - phi_min) / (phi_max - phi_min) * num_phi_bins).long()

    # Clamp to valid bin indices
    theta_bin = torch.clamp(theta_bin, 0, num_theta_bins - 1)
    phi_bin = torch.clamp(phi_bin, 0, num_phi_bins - 1)

    # Compute linear bin index (flatten 2D bin indices to 1D)
    bin_indices = theta_bin * num_phi_bins + phi_bin  # [num_envs, num_points]

    # Create an environment index tensor to handle multiple environments
    env_indices = torch.arange(num_envs, device=device).view(-1, 1).expand(-1, num_points)

    # Flatten tensors for scatter operation
    flat_bin_indices = bin_indices.view(-1)            # [num_envs * num_points]
    flat_env_indices = env_indices.view(-1)            # [num_envs * num_points]
    flat_r = r.view(-1)                               # [num_envs * num_points]

    # Create 2D indices for scatter operation (env_idx, bin_idx)
    scatter_indices = torch.stack([flat_env_indices, flat_bin_indices], dim=1)  # [num_envs * num_points, 2]

    # Prepare tensors for scatter operations
    r_sum = torch.zeros(num_envs, num_bins, device=device)
    bin_count = torch.zeros(num_envs, num_bins, device=device)

    # Use scatter_add_ to compute sum and count for each bin
    r_sum.scatter_add_(1, bin_indices, r)
    ones = torch.ones_like(r)
    bin_count.scatter_add_(1, bin_indices, ones)

    # Avoid division by zero for empty bins
    bin_count = torch.clamp(bin_count, min=1.0)

    # Compute average r per bin
    avg_r = r_sum / bin_count  # [num_envs, num_bins]

    # Create bin centers for theta and phi
    theta_centers = torch.linspace(
        theta_min + (theta_max - theta_min) / (2 * num_theta_bins),
        theta_max - (theta_max - theta_min) / (2 * num_theta_bins),
        num_theta_bins, device=device
    )

    phi_centers = torch.linspace(
        phi_min + (phi_max - phi_min) / (2 * num_phi_bins),
        phi_max - (phi_max - phi_min) / (2 * num_phi_bins),
        num_phi_bins, device=device
    )

    # Create meshgrid of bin centers
    theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
    theta_centers_flat = theta_grid.reshape(-1)  # [num_bins]
    phi_centers_flat = phi_grid.reshape(-1)      # [num_bins]

    # Create final output tensor
    downsampled = torch.zeros(num_envs, num_bins, 3, device=device)
    downsampled[:, :, 0] = avg_r                              # r values
    downsampled[:, :, 1] = theta_centers_flat.unsqueeze(0)    # theta values
    downsampled[:, :, 2] = phi_centers_flat.unsqueeze(0)      # phi values

    return downsampled

def parse_urdf_meshes(robot_config):
        """Parse URDF file to extract mesh filenames for ray_cast_bodies."""
        urdf_path = os.path.join(robot_config.asset.asset_root, robot_config.asset.urdf_file)

        if not os.path.exists(urdf_path):
            print(f"URDF file not found: {urdf_path}")
            return {}

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            body_meshes = {}

            # Find all link elements
            for link in root.findall('link'):
                link_name = link.get('name')

                # Check if this link is in ray_cast_bodies
                if link_name in robot_config.ray_cast_bodies:
                    # Look for visual geometry with mesh
                    visual = link.find('visual')
                    if visual is not None:
                        geometry = visual.find('geometry')
                        if geometry is not None:
                            mesh = geometry.find('mesh')
                            if mesh is not None:
                                mesh_filename = mesh.get('filename')
                                if mesh_filename:
                                    # Remove 'meshes/' prefix if present
                                    if mesh_filename.startswith('meshes/'):
                                        mesh_filename = mesh_filename[7:]
                                    body_meshes[link_name] = mesh_filename
                                    print(f"Found mesh for {link_name}: {mesh_filename}")

            return body_meshes

        except ET.ParseError as e:
            print(f"Error parsing URDF file {urdf_path}: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error parsing URDF file {urdf_path}: {e}")
            return {}

def convert_to_warp_mesh(points: np.ndarray, indices: np.ndarray, device: str) -> wp.Mesh:
    """Create a warp mesh object with a mesh defined from vertices and triangles.

    Args:
        points: The vertices of the mesh. Shape is (N, 3), where N is the number of vertices.
        indices: The triangles of the mesh as references to vertices for each triangle.
            Shape is (M, 3), where M is the number of triangles / faces.
        device: The device to use for the mesh.

    Returns:
        The warp mesh object.
    """
    return wp.Mesh(
        points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(indices.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )
