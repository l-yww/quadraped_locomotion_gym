# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, quat_conjugate, quat_mul, get_euler_xyz
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def get_scale_shift(range):
    scale = 2. / (range[1] - range[0])
    shift = (range[1] + range[0]) / 2.
    return scale, shift

def get_ee_euler(base_link_w,ee_state_w):
    """
    Get the Euler angles of the End-Effector relative to the base link.
    """
    baselink_to_world = quat_conjugate(base_link_w)
    baselink_to_ee = quat_mul(baselink_to_world,ee_state_w)
    roll, pitch, yaw = get_euler_xyz(baselink_to_ee)
    return roll, pitch, yaw

def euler_to_rotation_matrix(roll, pitch, yaw, device, order='xyz'):
    """
    Convert Euler angles to a rotation matrix
    parameters:
        roll: x
        pitch: y
        yaw: z
        order: euler default order, 'xyz'
    return:
        3x3 rotation matrix
    """
    # Calculate the trigonometric function values
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    if order == 'xyz':
        R = torch.stack([
            torch.stack([cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],dim=1),
            torch.stack([sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],dim=1),
            torch.stack([-sp,     cp*sr,            cp*cr          ],dim=1)
        ],dim=1)
    elif order == 'zyx':
           R = torch.stack([
               torch.stack([cy*cp,  -sy*cr + cy*sp*sr,   sy*sr + cy*sp*cr],dim=1),
               torch.stack([sy*cp,   cy*cr + sy*sp*sr,  -cy*sr + sy*sp*cr],dim=1),
               torch.stack([-sp,     cp*sr,              cp*cr          ],dim=1)
        ],dim=1)
    else:
        raise ValueError("Please use 'xyz' or 'zyx'")

    return R


def quaternion_to_rotation_matrix(quaternions):
    """
    Converts a quaternion to a rotation matrix.
        Args:
            quaternions : A tensor of shape (n, 4) containing quaternions
                in the (x, y, z, w) format.
        Returns:
            rot_matrices: A tensor of shape (n, 3, 3) representing the
                rotation matrices.
    """
    quaternions = quaternions.float()

    x = quaternions[:, 0]
    y = quaternions[:, 1]
    z = quaternions[:, 2]
    w = quaternions[:, 3]

    norm = torch.sqrt(x**2 + y**2 + z**2 + w**2)
    x = x / norm
    y = y / norm
    z = z / norm
    w = w / norm

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    n = quaternions.shape[0]
    rot_matrices = torch.zeros((n, 3, 3), device=quaternions.device)

    rot_matrices[:, 0, 0] = 1 - 2 * (y2 + z2)
    rot_matrices[:, 0, 1] = 2 * (xy - zw)
    rot_matrices[:, 0, 2] = 2 * (xz + yw)
    rot_matrices[:, 1, 0] = 2 * (xy + zw)
    rot_matrices[:, 1, 1] = 1 - 2 * (x2 + z2)
    rot_matrices[:, 1, 2] = 2 * (yz - xw)
    rot_matrices[:, 2, 0] = 2 * (xz - yw)
    rot_matrices[:, 2, 1] = 2 * (yz + xw)
    rot_matrices[:, 2, 2] = 1 - 2 * (x2 + y2)

    return rot_matrices

# @torch.jit.script
def quat_slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # q: [..., 4] (x, y, z, w)
    # t: [0, 1]

    # standardize: ensure short path (dot > 0)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)

    # clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # linear interpolation for very small angles to avoid division by zero
    mask = (sin_theta_0 < 1e-6).flatten()

    # lerp
    res_lerp = (1.0 - t) * q0 + t * q1

    # slerp
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    s0 = torch.cos(theta) - dot * sin_theta / (sin_theta_0 + 1e-9)
    s1 = sin_theta / (sin_theta_0 + 1e-9)
    res_slerp = s0 * q0 + s1 * q1

    res = torch.where(mask[..., None], res_lerp, res_slerp)
    return normalize(res)

# @torch.jit.script
def standardize_quaternion(q):
    # type: (Tensor) -> Tensor
    # Ensure w is positive
    return torch.where(q[..., 3:4] < 0, -q, q)

def quat_rotate_inverse_np(q, v):
    '''
    Rotate vector v by the inverse of quaternion q.
    q: shape (..., 4)
    v: shape (..., 3)'''
    shape = q.shape
    assert v.shape == shape[:-1] + (3,), f"Shape mismatch: q {shape}, v {v.shape}"
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * (2.0 * q_w)[..., None]
    c = q_vec * (2.0 * np.sum(q_vec * v, axis=-1, keepdims=True))
    return a - b + c