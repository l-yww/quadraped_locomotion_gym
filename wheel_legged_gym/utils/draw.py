
import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from wheel_legged_gym.utils.math import get_ee_euler,euler_to_rotation_matrix, quaternion_to_rotation_matrix
from wheel_legged_gym.utils.isaacgym_utils import sphere2cart
import torch



def draw_point(envs, gym, viewer, num_envs, commands, base_link_w, ee_pos, ee_state_w):

        sphere_geom_cmd = gymutil.WireframeSphereGeometry(0.01, 32, 32, None, color=(1, 0, 0))
        sphere_geom_eepos = gymutil.WireframeSphereGeometry(0.01, 32, 32, None, color=(0, 1, 0))

        for id in range(num_envs):
            pose_cmd = gymapi.Transform(gymapi.Vec3(commands[id,0]+base_link_w[id,0], commands[id,1]+base_link_w[id,1], commands[id,2]+base_link_w[id,2]), r=None)
            gymutil.draw_lines(sphere_geom_cmd, gym, viewer, envs[id], pose_cmd)
            pose_ee = gymapi.Transform(gymapi.Vec3(ee_pos[id,0]+base_link_w[id,0], ee_pos[id,1]+base_link_w[id,1], ee_pos[id,2]+base_link_w[id,2]), r=None)
            gymutil.draw_lines(sphere_geom_eepos, gym, viewer, envs[id], pose_ee)



def draw_axis(envs, gym, viewer, num_envs, commands, base_link_w, ee_pos, ee_state_w,device):

        axisx = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(1, 0, 0))
        axisy = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(0, 1, 0))
        axisz = gymutil.WireframeSphereGeometry(0.01, 16, 16, None, color=(0, 0, 1))

        R_cmd = euler_to_rotation_matrix(commands[:,3],commands[:,4],commands[:,5],device,'xyz')
        roll, pitch, yaw = get_ee_euler(base_link_w[:,3:7].clone(),ee_state_w[:,3:7].clone())
        R_gt = euler_to_rotation_matrix(roll,pitch,yaw,device,'xyz')
        # baselink_to_world = quat_conjugate(base_link_w[:,3:7].clone())
        # baselink_to_ee = quat_mul(baselink_to_world,ee_state_w[:,3:7])
        # R_gt = quaternion_to_rotation_matrix(baselink_to_ee)
        poseA = torch.tensor([0.1,0.0,0.0],device=device)
        poseB = torch.tensor([0.0,0.1,0.0],device=device)
        poseC = torch.tensor([0.0,0.0,0.1],device=device)
        poseA_local = torch.matmul(R_gt,poseA)
        poseB_local = torch.matmul(R_gt,poseB)
        poseC_local = torch.matmul(R_gt,poseC)

        poseA_cmd = torch.matmul(R_cmd,poseA)
        poseB_cmd = torch.matmul(R_cmd,poseB)
        poseC_cmd = torch.matmul(R_cmd,poseC)

        for id in range(num_envs):
            for i in range(10):
                pose_ee = gymapi.Transform(gymapi.Vec3(ee_pos[id,0]+0.1*(i+1)*poseA_local[id,0]+base_link_w[id,0], ee_pos[id,1]+0.1*(i+1)*poseA_local[id,1]+base_link_w[id,1], ee_pos[id,2]+0.1*(i+1)*poseA_local[id,2]+base_link_w[id,2]), r=None)
                gymutil.draw_lines(axisx, gym, viewer, envs[id], pose_ee)
                pose_ee = gymapi.Transform(gymapi.Vec3(ee_pos[id,0]+0.1*(i+1)*poseB_local[id,0]+base_link_w[id,0], ee_pos[id,1]+0.1*(i+1)*poseB_local[id,1]+base_link_w[id,1], ee_pos[id,2]+0.1*(i+1)*poseB_local[id,2]+base_link_w[id,2]), r=None)
                gymutil.draw_lines(axisy, gym, viewer, envs[id], pose_ee)
                pose_ee = gymapi.Transform(gymapi.Vec3(ee_pos[id,0]+0.1*(i+1)*poseC_local[id,0]+base_link_w[id,0], ee_pos[id,1]+0.15*(i+1)*poseC_local[id,1]+base_link_w[id,1], ee_pos[id,2]+0.15*(i+1)*poseC_local[id,2]+base_link_w[id,2]), r=None)
                gymutil.draw_lines(axisz, gym, viewer, envs[id], pose_ee)

                pose_ee = gymapi.Transform(gymapi.Vec3(commands[id,0]+0.1*(i+1)*poseA_cmd[id,0]+base_link_w[id,0], commands[id,1]+0.1*(i+1)*poseA_cmd[id,1]+base_link_w[id,1], commands[id,2]+0.1*(i+1)*poseA_cmd[id,2]+base_link_w[id,2]), r=None)
                gymutil.draw_lines(axisx, gym, viewer, envs[id], pose_ee)
                pose_ee = gymapi.Transform(gymapi.Vec3(commands[id,0]+0.1*(i+1)*poseB_cmd[id,0]+base_link_w[id,0], commands[id,1]+0.1*(i+1)*poseB_cmd[id,1]+base_link_w[id,1], commands[id,2]+0.1*(i+1)*poseB_cmd[id,2]+base_link_w[id,2]), r=None)
                gymutil.draw_lines(axisy, gym, viewer, envs[id], pose_ee)
                pose_ee = gymapi.Transform(gymapi.Vec3(commands[id,0]+0.1*(i+1)*poseC_cmd[id,0]+base_link_w[id,0], commands[id,1]+0.15*(i+1)*poseC_cmd[id,1]+base_link_w[id,1], commands[id,2]+0.15*(i+1)*poseC_cmd[id,2]+base_link_w[id,2]), r=None)
                gymutil.draw_lines(axisz, gym, viewer, envs[id], pose_ee)


def draw_ee_pos_key_points(envs, gym, viewer, num_envs, base_link_w, ee_pos, ee_state_w,device):
        joint_sphere = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 1, 0) 
        )
        joint_axes = gymutil.AxesGeometry(scale=0.8)  # 坐标系几何体

        # 2. 批量将 rpy 转换为四元数（适配 Isaac Gym 的 Transform 要求）
        batch_quats = ee_state_w[:,3:7]

        # 3. 遍历所有环境绘制
        for env_id in range(num_envs):
            # 获取当前环境的关节数据（Tensor -> 标量）
            x = ee_state_w[env_id, 0].item()
            y = ee_state_w[env_id, 1].item()
            z = ee_state_w[env_id, 2].item()
            quat = batch_quats[env_id]  # (x, y, z, w)

            # 4. 构建 Transform（位置+旋转）
            transform = gymapi.Transform(
                p=gymapi.Vec3(x, y, z),  # 关节世界位置
                r=gymapi.Quat(quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item())  # 关节姿态（四元数）
            )

            # 5. 绘制关节位置（绿色小球）和姿态（坐标系）
            gymutil.draw_lines(joint_sphere, gym, viewer, envs[env_id], transform)
            # gymutil.draw_lines(joint_axes, gym, viewer, envs[env_id], transform)
    

            

def euler_angles_to_quaternion(roll, pitch, yaw, device, order='xyz'):
    """
    将欧拉角 (roll, pitch, yaw) 转换为四元数 (x, y, z, w)。
    :param roll: 绕 x 轴旋转 (张量, shape: [N])
    :param pitch: 绕 y 轴旋转 (张量, shape: [N])
    :param yaw: 绕 z 轴旋转 (张量, shape: [N])
    :param device: 计算设备
    :param order: 旋转顺序, 默认为 'xyz'
    :return: 四元数张量，形状为 (N, 4)
    """
    roll = roll.to(device)
    pitch = pitch.to(device)
    yaw = yaw.to(device)

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    if order == 'xyz':
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
    else:
        raise ValueError("Only 'xyz' order is supported.")

    return torch.stack([qx, qy, qz, qw], dim=1)

def draw_command_key_points(envs, gym, viewer, num_envs, commands, commands_quat, base_link_w, device):

    sphere_coords = commands[:, :3] + base_link_w
    # sphere_euler = commands[:,3:6]
    # sphere_quat = quat_from_euler_xyz(sphere_euler[:,0],sphere_euler[:,1],sphere_euler[:,2])
    sphere_quat = commands_quat


    # 2. 将球坐标转换为笛卡尔坐标 (x, y, z)
    # 这个坐标是相对于机器人底座中心的
    ee_local_cart = sphere_coords
    axes_geom = gymutil.AxesGeometry(scale=0.3)
    ee_global_cart = ee_local_cart

    target_sphere = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(1, 1, 0)) # 黄色球体，更醒目

    for env_id in range(commands.shape[0]):
        # 关键修正：取当前环境的四元数（[4] 张量），再逐个取标量

        # env_id = 0
        current_quat = sphere_quat[env_id]  # [4]

        # 获取当前环境的世界坐标（[3] 张量）
        target_pos_world = ee_global_cart[env_id]  # [3]

        # 构建变换（位置+姿态）
        transform = gymapi.Transform(
            # 位置：逐个取标量
            p=gymapi.Vec3(
                target_pos_world[0].item(),
                target_pos_world[1].item(),
                target_pos_world[2].item()
            ),
            # 姿态：取当前环境的四元数标量（核心修正点）
            r=gymapi.Quat(
                current_quat[0].item(),
                current_quat[1].item(),
                current_quat[2].item(),
                current_quat[3].item()
            )
        )

        # 在指定环境中绘制这个球体
        gymutil.draw_lines(target_sphere, gym, viewer, envs[env_id], transform)
        gymutil.draw_lines(axes_geom, gym, viewer, envs[env_id], transform)




