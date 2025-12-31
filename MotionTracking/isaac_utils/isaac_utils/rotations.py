import torch
from torch import Tensor
from isaac_utils.maths import *
from typing import Tuple


def rad2deg(radian_value: Tensor, device=None) -> Tensor:
    """_summary_

    Args:
        radian_value (torch.Tensor): _description_
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    return torch.rad2deg(radian_value).float().to(device)


def deg2rad(degree_value: float, device=None) -> Tensor:
    """_summary_

    Args:
        degree_value (torch.Tensor): _description_
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    return torch.deg2rad(degree_value).float().to(device)


@torch.jit.script
def quat_mul(a, b, w_last: bool):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_apply(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if w_last:
        xyz = a[:, :3]
        w = a[:, 3:]
    else:
        xyz = a[:, 1:]
        w = a[:, :1]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_mul_norm(x: Tensor, y: Tensor, w_last: bool) -> Tensor:
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_unit(quat_mul(x, y, w_last))


@torch.jit.script
def quat_angle_axis(x: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    if w_last:
        w = x[..., -1]
        axis = x[..., :3]
    else:
        w = x[..., 0]
        axis = x[..., 1:]
    s = 2 * (w ** 2) - 1
    angle = s.clamp(-1, 1).arccos()  # just to be safe
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis


@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def vec_to_heading(h_vec):
    h_theta = torch.atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta


@torch.jit.script
def heading_to_quat(h_theta, w_last: bool):
    axis = torch.zeros(h_theta.shape + [3,], device=h_theta.device)
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(h_theta, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_axis(q: Tensor, axis: int, w_last: bool) -> Tensor:
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec, w_last)


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def get_basis_vector(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    return quat_rotate(q, v, w_last)


@torch.jit.script
def get_euler_xyz(q: Tensor, w_last: bool) -> Tuple[Tensor, Tensor, Tensor]:
    if w_last:
        qx, qy, qz, qw = 0, 1, 2, 3
    else:
        qw, qx, qy, qz = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll: Tensor, pitch: Tensor, yaw: Tensor, w_last: bool) -> Tensor:
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

    if w_last:
        return torch.stack([qx, qy, qz, qw], dim=-1)
    else:
        return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def quat_diff_rad(a: Tensor, b: Tensor, w_last: bool) -> Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b, w_last)
    mul = quat_mul(a, b_conj, w_last)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(torch.clamp(torch.norm(mul[:, 1:], p=2, dim=-1), max=1.0))


# NB: do not make this function jit, since it is passed around as an argument.
def normalise_quat_in_pose(pose):
    """Takes a pose and normalises the quaternion portion of it.

    Args:
        pose: shape N, 7
    Returns:
        Pose with normalised quat. Shape N, 7
    """
    pos = pose[:, 0:3]
    quat = pose[:, 3:7]
    quat /= torch.norm(quat, dim=-1, p=2).reshape(-1, 1)
    return torch.cat([pos, quat], dim=-1)


@torch.jit.script
def quat_apply_yaw(quat: Tensor, vec: Tensor, w_last: bool) -> Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec, w_last)
