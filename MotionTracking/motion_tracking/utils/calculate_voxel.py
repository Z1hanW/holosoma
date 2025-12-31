import torch
import numpy as np
import xml.etree.ElementTree as ET

def RPY_to_rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [   -sp,          cp * sr,          cp * cr]
    ])

def compute_box_corners(size, origin):
    x, y, z = map(float, size.split())
    ox, oy, oz = map(float, origin['xyz'].split())
    rpy = list(map(float, origin['rpy'].split())) if 'rpy' in origin else (0.0, 0.0, 0.0)
    R = RPY_to_rotation_matrix(*rpy)
    dx, dy, dz = x / 2, y / 2, z / 2
    corners = np.array([
        [dx, dy, dz], [dx, dy, -dz], [dx, -dy, dz], [dx, -dy, -dz],
        [-dx, dy, dz], [-dx, dy, -dz], [-dx, -dy, dz], [-dx, -dy, -dz]
    ])
    return np.dot(R, corners.T).T + [ox, oy, oz]

def read_urdf(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []
    for visual in root.findall(".//visual"):
        geometry = visual.find('geometry')
        box = geometry.find('box')
        if box is not None:
            size = box.get('size')
            origin = visual.find('origin').attrib
            corners = compute_box_corners(size, origin)
            xmin, ymin, zmin = corners.min(axis=0)
            xmax, ymax, zmax = corners.max(axis=0)
            boxes.append(((xmin, ymin, zmin), (xmax, ymax, zmax)))
    return boxes

def compute_voxel_grid_batch(root_positions, urdf_boxes, voxel_size=0.2, grid_dim=10):
    batch_size = root_positions.shape[0]
    num_boxes = urdf_boxes.shape[1]
    grids = torch.zeros((batch_size, grid_dim, grid_dim, grid_dim), dtype=torch.uint8, device=root_positions.device)
    grid_center_offset = torch.tensor([1.0, 0, 0], device=root_positions.device)
    root_positions = root_positions + grid_center_offset
    indices = torch.arange(grid_dim, device=root_positions.device).float()
    offsets = (indices - grid_dim / 2 + 0.5) * voxel_size
    voxel_centers = root_positions[:, None, None, None, :] + torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing='ij'), dim=-1)[None, :, :, :, :]
    box_mins = urdf_boxes[:, :, 0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    box_maxs = urdf_boxes[:, :, 1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    inside_box = (voxel_centers.unsqueeze(-2) >= box_mins) & (voxel_centers.unsqueeze(-2) <= box_maxs)
    inside_box = inside_box.all(dim=-1).any(dim=-1)
    grids[inside_box] = 1
    return grids

