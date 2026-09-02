import os
import numpy as np
import random
import torch

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

@torch.jit.script
def euler_from_quat(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]
    y = quat_angle[:,1]
    z = quat_angle[:,2]
    w = quat_angle[:,3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

@torch.jit.script
def sphere2cart(sphere_coords):
    # type: (Tensor) -> Tensor
    """ Convert spherical coordinates to cartesian coordinates
    Args:
        sphere_coords (torch.Tensor): Spherical coordinates (l, pitch, yaw)
    Returns:
        cart_coords (torch.Tensor): Cartesian coordinates (x, y, z)
    """
    l = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    cart_coords = torch.zeros_like(sphere_coords)
    cart_coords[:, 0] = l * torch.cos(pitch) * torch.cos(yaw)
    cart_coords[:, 1] = l * torch.cos(pitch) * torch.sin(yaw)
    cart_coords[:, 2] = l * torch.sin(pitch)
    return cart_coords

@torch.jit.script
def cart2sphere(cart_coords):
    # type: (Tensor) -> Tensor
    """ Convert cartesian coordinates to spherical coordinates
    Args:
        cart_coords (torch.Tensor): Cartesian coordinates (x, y, z)
    Returns:
        sphere_coords (torch.Tensor): Spherical coordinates (l, pitch, yaw)
    """
    sphere_coords = torch.zeros_like(cart_coords)
    xy_len = torch.norm(cart_coords[:, :2], dim=1)
    sphere_coords[:, 0] = torch.norm(cart_coords, dim=1)
    sphere_coords[:, 1] = torch.atan2(cart_coords[:, 2], xy_len)
    sphere_coords[:, 2] = torch.atan2(cart_coords[:, 1], cart_coords[:, 0])
    return sphere_coords