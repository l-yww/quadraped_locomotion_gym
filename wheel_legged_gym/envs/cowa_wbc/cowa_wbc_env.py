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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from wheel_legged_gym.envs.cowa_wbc.cowa_wbc_config import CowaCfg, CowaCfgPPO

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from collections import deque
import torch
import random
from wheel_legged_gym.envs.cowa_wbc.legged_robot import LeggedRobot

from wheel_legged_gym.utils.terrain import  Terrain
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift,quaternion_to_rotation_matrix

import numpy as np
from scipy.interpolate import make_interp_spline
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from collections import deque
from torch import Tensor
from typing import Tuple, Dict

from wheel_legged_gym.utils import class_to_dict
from wheel_legged_gym.utils.isaacgym_utils import euler_from_quat, sphere2cart, cart2sphere


import matplotlib.pyplot as plt

class CowaEnv(LeggedRobot):

    def __init__(self, cfg: CowaCfg, train_cfg: CowaCfgPPO, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.wheel_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        self.last_feet_z = 0.12

        self.feet_contact_safety = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
        self.feet_contact_ratio = torch.zeros((self.num_envs, 2), device=self.device)

        self.start_curriculum = 0
        self.goal_ee_ranges = class_to_dict(self.cfg.commands.ranges)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

        self.feet_contact_safety[env_ids] = False
        self.feet_contact_ratio[env_ids] = 0.0

        self._resample_arm_commands(env_ids)
        self.commands_force[env_ids] = 0
        self.forces[env_ids, self.gripper_idx, :3] = 0
    
        self.selected_env_ids_gripper_cmd[env_ids] = 0
        self.selected_env_ids_gripper_ext[env_ids] = 0
        self.force_target_gripper_ext[env_ids, :3] = 0.
        self.current_Fxyz_gripper_cmd[env_ids, :3] = 0.

    # TODO
    def get_ee_goal_spherical_center(self):  # 球坐标系的原点
        return self.rigid_state[:,1,:3]

    def _init_buffers(self):
        super()._init_buffers()        
        self.noised_leg_q = torch.zeros((self.num_envs, self.num_dof - self.wheel_nums), device=self.device)
        self.noised_dq = torch.zeros_like(self.dof_vel)
        self.noised_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.noised_euler_rpy = torch.zeros_like(self.base_euler_rpy)

    def _get_phase(self):
        cycle_time = self.cfg.control.cycle_time
        offset = self.cfg.control.offset
        self.phase = (self.episode_length_buf * self.dt) % cycle_time / cycle_time
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
    
    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        self._get_phase()
        self.sin_pos = torch.sin(2 * torch.pi * self.phase)
        self.cos_pos = torch.cos(2 * torch.pi * self.phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = self.sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = self.sin_pos < 0

        return stance_mask
    
    def compute_ref_state(self):
        """ 参考轨迹v0,但参考轨迹的foot是始终在中线上的,如果权重太大,会导致迈不开腿
        """
        self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        scale_1 = -0.15 #-0.4
        scale_2 = 10
        scale_3 = -0.3
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l < 0] = 0

        # self.ref_dof_pos[:, 0] += sin_pos_l * scale_3  #hip roll
        self.ref_dof_pos[:, 1] -= sin_pos_l * scale_1  #hip pitch
        self.ref_dof_pos[:, 2] -= 2 * sin_pos_l * scale_1  #knee
        self.ref_dof_pos[:, 3] -= sin_pos_l * scale_1  #foot
        # self.ref_dof_vel[:, 4] += sin_pos_l * scale_2  #wheel

        # self.ref_dof_pos[:, 5] += sin_pos_l * scale_3
        # self.ref_dof_pos[:, 6] += sin_pos_l * scale_1
        # self.ref_dof_pos[:, 7] += sin_pos_l * scale_1
        # self.ref_dof_pos[:, 8] += sin_pos_l * scale_1
        # self.ref_dof_vel[:, 9] += sin_pos_l * scale_2
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        self.ref_dof_pos[:, 6] += sin_pos_r * scale_1
        self.ref_dof_pos[:, 7] += 2 * sin_pos_r * scale_1
        self.ref_dof_pos[:, 8] += sin_pos_r * scale_1

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        orientation_dim = 3 if self.cfg.env.projected_gravity else 2

        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales

        noise_vec[:17] = 0
        noise_vec[17:17+self.num_actions-2] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[17+self.num_actions-2:17+2*self.num_actions-2] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[17+2*self.num_actions-2:17+3*self.num_actions-2] = 0 
        noise_vec[17+3*self.num_actions-2:17+3*self.num_actions-2+3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[17+3*self.num_actions-2+3:17+3*self.num_actions-2+3+3] = 0
        noise_vec[17+3*self.num_actions-2+3+3:17+3*self.num_actions-2+3+3+orientation_dim] = noise_scales.gravity  * self.obs_scales.gravity
        noise_vec[17+3*self.num_actions-2+3+3+orientation_dim:] = 0
        return noise_vec

    def step(self, actions):
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay

        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions

        return super().step(actions)

    def _check_feet_collision(self):
        foot_forces_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=-1)
        collision_danger = foot_forces_xy > self.cfg.rewards.max_collision_xy_force_threshold 
        return collision_danger

    def get_volume_sample_points_terrain_height(self):
        """
        计算volume_sample_points对应位置的地形高度
        
        Returns:
            torch.Tensor: 形状为(num_envs, num_sample_points)的地形高度值
        """
        self.refresh_volume_sample_points()
        
        xy_coords = self.volume_sample_points[..., :2] 
        
        x = xy_coords[..., 0].flatten()  
        y = xy_coords[..., 1].flatten()  
        
        x = x + self.terrain.cfg.border_size
        y = y + self.terrain.cfg.border_size
        
        px = (x / self.terrain.cfg.horizontal_scale).long()
        py = (y / self.terrain.cfg.horizontal_scale).long()
        
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        
        h00 = self.height_samples[px, py]
        h10 = self.height_samples[px+1, py]
        h01 = self.height_samples[px, py+1]
        h11 = self.height_samples[px+1, py+1]
        
        dx = (x / self.terrain.cfg.horizontal_scale) - px.float()
        dy = (y / self.terrain.cfg.horizontal_scale) - py.float()
        
        height = (1-dx)*(1-dy)*h00 + dx*(1-dy)*h10 + (1-dx)*dy*h01 + dx*dy*h11
        
        height = height * self.terrain.cfg.vertical_scale
        
        terrain_heights = height.view(self.num_envs, -1)
        
        return terrain_heights

    def _get_feet_heights(self):
        """
        计算每只脚的接触安全性
        通过统计每只脚采样点中距离地面小于0.01的点数比例来判断
        如果比例大于0.8，则认为落脚点安全
        
        Returns:
            feet_contact_safety: 形状为(num_envs, 2)的布尔张量，表示每只脚是否安全
            feet_contact_ratio: 形状为(num_envs, 2)的张量，表示每只脚接触地面的点数比例
        """
        self.refresh_volume_sample_points()
        
        sample_z = self.volume_sample_points[..., 2]

        terrain_heights = self.get_volume_sample_points_terrain_height()
        
        distances = sample_z - terrain_heights
        
        num_points_per_foot = distances.shape[1] // 2

        close_points_mask = distances < 0.01

        left_foot_mask = close_points_mask[:, :num_points_per_foot]
        right_foot_mask = close_points_mask[:, num_points_per_foot:]

        left_foot_ratio = left_foot_mask.sum(dim=1) / num_points_per_foot
        right_foot_ratio = right_foot_mask.sum(dim=1) / num_points_per_foot
        
        self.feet_contact_ratio[:, 0] = left_foot_ratio
        self.feet_contact_ratio[:, 1] = right_foot_ratio

        left_foot_distances = distances[:, :num_points_per_foot]
        right_foot_distances = distances[:, num_points_per_foot:]
        
        left_foot_heights = left_foot_distances.mean(dim=1)
        right_foot_heights = right_foot_distances.mean(dim=1)
        
        feet_heights = torch.cat((left_foot_heights.unsqueeze(1), right_foot_heights.unsqueeze(1)), dim=1)
        
        left_foot_heights_var = left_foot_distances.var(dim=1)
        right_foot_heights_var = right_foot_distances.var(dim=1)
        
        feet_heights_var = torch.cat((left_foot_heights_var.unsqueeze(1), right_foot_heights_var.unsqueeze(1)), dim=1)

        return torch.clip(feet_heights, min=0.), feet_heights_var

    def compute_observations(self):

        # forces_apply_gripper = self.forces[:, self.gripper_idx, 0:3]
        # force_sum = self.current_Fxyz_gripper_cmd +forces_apply_gripper
        ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
        # ee_goal = force_sum / self.gripper_force_kps + ee_local_cart
        # ee_goal_sphere = cart2sphere(ee_goal) #考虑外力影响的末端执行器目标位置

        # Spherical to cartesian coordinates in the arm base frame 
        radius = torch.norm(ee_local_cart, dim=1).view(self.num_envs,1)
        pitch = torch.asin(ee_local_cart[:,2].view(self.num_envs,1)/radius).view(self.num_envs,1)
        yaw = torch.atan2(ee_local_cart[:,1].view(self.num_envs,1), ee_local_cart[:,0].view(self.num_envs,1)).view(self.num_envs,1)
        self.ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(self.num_envs,3)  #末端执行器的实际位置

        self.compute_ref_state()
        self._get_gait_phase()

        if self.cfg.terrain.mesh_type == 'terrain':
            self._get_feet_heights()
        
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        self.contact_force_left = self.contact_forces[:, self.feet_indices[0], :]
        self.contact_force_right = self.contact_forces[:, self.feet_indices[1], :]

        sin_pos = self.sin_pos.unsqueeze(1) 
        cos_pos = self.cos_pos.unsqueeze(1) 

        self.privileged_obs_buf = torch.cat(( 
            sin_pos, # 1
            cos_pos, # 1
            self.commands[:, :3] * self.commands_scale,  # 3
            self.arm_commands[:,:3] * self.arm_commands_scale,  # 3
            self.arm_commands[:,3:] * self.obs_scales.ee_orn_quat, # 9
            # self.commands_force * self.obs_scales.ee_force, # 3
            self.leg_pos * self.obs_scales.dof_pos, # 10 - 2 + 6
            self.dof_vel * self.obs_scales.dof_vel, # 10 + 6
            self.actions, # 10 + 6
            self.base_ang_vel * self.obs_scales.ang_vel, # 3
            # self.base_euler_rpy[:,:] * self.obs_scales.quat,
            self.projected_gravity, # 重力投影  # 3
            # ee_goal_sphere[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd,  # 3 #考虑了外力补偿后的目标球坐标位置,目标位置会根据受到的外力进行偏移，以实现柔顺控制或抗干扰
            # ee_goal_sphere[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
            # ee_goal_sphere[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd,
            self.ee_quat * self.obs_scales.ee_orn_quat,  # 4
        ),dim=-1)

        if self.cfg.env.priv_observe_friction:
             # 1
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (self.friction_coeffs[:, 0].unsqueeze(1)
                                                    - friction_coeffs_shift) * friction_coeffs_scale),dim=1)
        
        if self.cfg.env.priv_observe_restitution:
            # 1
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(1) 
                                                 - restitutions_shift) * restitutions_scale),dim=1)

        if self.cfg.env.priv_observe_payloads:
            # 1
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),dim=1)

        if self.cfg.env.priv_observe_inertia:
            # 1
            inertia_scale, inertia_shift = get_scale_shift(self.cfg.domain_rand.randomize_inertia_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.inertia_scale.unsqueeze(1) - inertia_shift) * inertia_scale),dim=1)

        if self.cfg.env.priv_observe_motor_strength:
            # 10
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            # 10
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1) 
        if self.cfg.env.priv_observe_com_displacement:
            # 3
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.domain_rand.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.com_displacements[:, :3] - com_displacements_shift) * com_displacements_scale), dim=1)
        # 始终将需要预测的值量放在最后（若有EST网络）
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                            self.base_height_obs *self.obs_scales.height_measurements,  # 1
                                            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                                            self.ee_pos_sphe_arm[:, 0:1] * self.obs_scales.ee_sphe_radius_cmd, # 3
                                            self.ee_pos_sphe_arm[:, 1:2] * self.obs_scales.ee_sphe_pitch_cmd,
                                            self.ee_pos_sphe_arm[:, 2:3] * self.obs_scales.ee_sphe_yaw_cmd,
                                            # forces_apply_gripper * self.obs_scales.ee_force,  # 3
                                            ), dim=1)

        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        if self.cfg.domain_rand.add_imu_lag:
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            if self.cfg.env.projected_gravity == True:
                self.lagged_projected_gravity = self.lagged_imu[:,-3:].clone()
            else:
                self.lagged_base_euler_rpy = self.lagged_imu[:,-3:].clone()
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_rpy = self.base_euler_rpy
            self.lagged_projected_gravity = self.projected_gravity

        lagged_q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        lagged_dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        # 剔除 wheel 的 dof_pos 观测
        mask = torch.ones(lagged_q.shape[1], dtype=torch.bool)
        mask[self.wheel_indices-1] = False
        lagged_leg_q = lagged_q[:, mask]

        obs_buf = torch.cat((
            sin_pos,  # 1
            cos_pos,  # 1
            self.commands[:, :3]  * self.commands_scale,  # 3
            self.arm_commands[:,:3] * self.arm_commands_scale,  # 3
            self.arm_commands[:,3:] * self.obs_scales.ee_orn_quat, # 9
            lagged_leg_q, # 10 - 2 + 6
            lagged_dq, # 10 + 6
            self.actions, # 10 + 6
            self.lagged_base_ang_vel * self.obs_scales.ang_vel, # 3
            # self.commands_force * self.obs_scales.ee_force, # 3
        ), dim=-1)
        if self.cfg.env.projected_gravity:
            obs_buf = torch.cat((
                obs_buf,
                self.lagged_projected_gravity * self.obs_scales.quat, # 3
            ),dim=-1)
        else:
            obs_buf = torch.cat((
                obs_buf,
                # self.lagged_base_euler_rpy * self.obs_scales.quat,
                self.lagged_base_euler_rpy[:,:2] * self.obs_scales.quat, # 2
            ),dim=-1)

        obs_now = obs_buf.clone()

        if self.add_noise:  
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def _push_gripper(self, env_ids_all):
        """ Randomly pushes the gripper stators. Emulates an impulse by setting a randomized gripper stator velocity.
        """
        
        if self.cfg.commands.push_gripper_stators:
            # print("start")
            # cmd force
            # FORCE CONTROLLED ENVS
            min_force_cmd = self.cfg.commands.max_push_force_xyz_gripper_cmd[0]
            max_force_cmd = self.cfg.commands.max_push_force_xyz_gripper_cmd[1]
            if min_force_cmd < -self.cfg.commands.max_force_cmd_range:
                min_force_cmd = -self.cfg.commands.max_force_cmd_range
            if max_force_cmd > self.cfg.commands.max_force_cmd_range:
                max_force_cmd = self.cfg.commands.max_force_cmd_range
            min_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[0]
            max_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[1]
            if min_force_ext < -self.cfg.commands.max_force_ext_range:
                min_force_ext = -self.cfg.commands.max_force_ext_range
            if max_force_ext > self.cfg.commands.max_force_ext_range:
                max_force_ext = self.cfg.commands.max_force_ext_range

            new_selected_env_ids_cmd = env_ids_all[(self.episode_length_buf % self.push_interval_gripper_cmd[:, 0]) == 0]
            
            # Define force and duration for the push 
            if new_selected_env_ids_cmd.nelement() > 0:
                
                self.freed_envs_gripper_cmd[new_selected_env_ids_cmd] = torch.rand(len(new_selected_env_ids_cmd), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.gripper_forced_prob_cmd


                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 0] = torch_rand_float(min_force_cmd*0.3, max_force_cmd*0.3, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 1] = torch_rand_float(min_force_cmd*0.3, max_force_cmd*0.3, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                self.force_target_gripper_cmd[new_selected_env_ids_cmd, 2] = torch_rand_float(min_force_cmd, max_force_cmd, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd))
                push_duration_gripper_cmd = torch_rand_float(self.push_duration_gripper_cmd_min, self.push_duration_gripper_cmd_max, (len(new_selected_env_ids_cmd), 1), device=self.device).view(len(new_selected_env_ids_cmd)) # 4.0/self.dt
                push_duration_gripper_cmd = torch.clip(push_duration_gripper_cmd, max=(self.push_interval_gripper_cmd[new_selected_env_ids_cmd, 0] - self.settling_time_force_gripper)/2).to(self.device)
                self.push_end_time_gripper_cmd[new_selected_env_ids_cmd] = self.episode_length_buf[new_selected_env_ids_cmd] + push_duration_gripper_cmd
                self.push_duration_gripper_cmd[new_selected_env_ids_cmd] = push_duration_gripper_cmd
                
                self.selected_env_ids_gripper_cmd[new_selected_env_ids_cmd] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_gripper_cmd == 1]

                # Step 1: apply force from 0 to force_target_gripper_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] < (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_cmd[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, :3] = (self.force_target_gripper_cmd[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_gripper_cmd[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                    
                    self.commands_force[env_ids_apply_push_step1, 0] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 0] #torch.norm(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, :2], dim=1)
                    self.commands_force[env_ids_apply_push_step1, 1] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 1] #torch.atan2(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 1], self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 0])
                    self.commands_force[env_ids_apply_push_step1, 2] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step1, 2]
 
                # Step 2: apply force from force_target_gripper_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] > (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1] + self.settling_time_force_gripper).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_cmd[env_ids_apply_push_step2].unsqueeze(-1)
                    self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, :3] = self.force_target_gripper_cmd[env_ids_apply_push_step2, :3] - (self.force_target_gripper_cmd[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_gripper_cmd[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    # World frame
                    self.commands_force[env_ids_apply_push_step2, 0] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 0] #torch.norm(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, :2], dim=1)
                    self.commands_force[env_ids_apply_push_step2, 1] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 1] #torch.atan2(self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 1], self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 0])
                    self.commands_force[env_ids_apply_push_step2, 2] = self.current_Fxyz_gripper_cmd[env_ids_apply_push_step2, 2]
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_cmd == 1] >= (self.push_end_time_gripper_cmd[self.selected_env_ids_gripper_cmd == 1] + self.settling_time_force_gripper + self.push_duration_gripper_cmd[self.selected_env_ids_gripper_cmd == 1]).type(torch.int32)]
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_gripper_cmd[env_ids_to_reset] = 0
                    self.force_target_gripper_cmd[env_ids_to_reset, :3] = 0.
                    self.current_Fxyz_gripper_cmd[env_ids_to_reset, :3] = 0.
                    self.push_end_time_gripper_cmd[env_ids_to_reset] = 0.
                    self.push_duration_gripper_cmd[env_ids_to_reset] = 0.
                    self.commands_force[env_ids_to_reset, 0] = 0.0
                    self.commands_force[env_ids_to_reset, 1] = 0.0
                    self.commands_force[env_ids_to_reset, 2] = 0.0
                    self.push_interval_gripper_cmd[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_gripper_cmd_min), int(self.push_interval_gripper_cmd_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_gripper_cmd[self.freed_envs_gripper_cmd] = 0
            self.force_target_gripper_cmd[self.freed_envs_gripper_cmd, :3] = 0.
            self.current_Fxyz_gripper_cmd[self.freed_envs_gripper_cmd, :3] = 0.
            self.push_end_time_gripper_cmd[self.freed_envs_gripper_cmd] = 0.
            self.push_duration_gripper_cmd[self.freed_envs_gripper_cmd] = 0. 
            self.commands_force[self.freed_envs_gripper_cmd, 0] = 0.0
            self.commands_force[self.freed_envs_gripper_cmd, 1] = 0.0
            self.commands_force[self.freed_envs_gripper_cmd, 2] = 0.0


            # ext force
            # FORCE CONTROLLED ENVS
            new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_gripper_ext[:, 0]) == 0]
                
            # Define force and duration for the push 
            if new_selected_env_ids_ext.nelement() > 0:
                
                self.freed_envs_gripper_ext[new_selected_env_ids_ext] = torch.rand(len(new_selected_env_ids_ext), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.gripper_forced_prob_ext
            

                self.force_target_gripper_ext[new_selected_env_ids_ext, 0] = torch_rand_float(min_force_ext*0.3, max_force_ext*0.3, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_gripper_ext[new_selected_env_ids_ext, 1] = torch_rand_float(min_force_ext*0.3, max_force_ext*0.3, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_gripper_ext[new_selected_env_ids_ext, 2] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                push_duration_gripper_ext = torch_rand_float(self.push_duration_gripper_ext_min, self.push_duration_gripper_ext_max, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext)) # 4.0/self.dt
                push_duration_gripper_ext = torch.clip(push_duration_gripper_ext, max=(self.push_interval_gripper_ext[new_selected_env_ids_ext, 0] - self.settling_time_force_gripper)/2).to(self.device)
                self.push_end_time_gripper_ext[new_selected_env_ids_ext] = self.episode_length_buf[new_selected_env_ids_ext] + push_duration_gripper_ext
                self.push_duration_gripper_ext[new_selected_env_ids_ext] = push_duration_gripper_ext
                
                self.selected_env_ids_gripper_ext[new_selected_env_ids_ext] = 1
                
            # Get ids of all envs to apply a force to 
            if self.episode_length_buf[self.selected_env_ids_gripper_ext == 1].nelement() > 0:
                subset_env_ids_selected = env_ids_all[self.selected_env_ids_gripper_ext == 1]

                # Step 1: apply force from 0 to force_target_gripper_cmd
                env_ids_apply_push_step1 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] < (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1]).type(torch.int32)]
                # print(env_ids_apply_push_step1)
                if env_ids_apply_push_step1.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_ext[env_ids_apply_push_step1].unsqueeze(-1)
                    
                    self.forces[env_ids_apply_push_step1, self.gripper_idx, :3] = (self.force_target_gripper_ext[env_ids_apply_push_step1, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step1].unsqueeze(-1) - (self.push_end_time_gripper_ext[env_ids_apply_push_step1].unsqueeze(-1)-push_duration_reshaped), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                  
                # Step 2: apply force from force_target_gripper_cmd back to 0
                env_ids_apply_push_step2 = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] > (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1] + self.settling_time_force_gripper).type(torch.int32)]
                if env_ids_apply_push_step2.nelement() > 0:
                    push_duration_reshaped = self.push_duration_gripper_ext[env_ids_apply_push_step2].unsqueeze(-1)
                    
                    # world frame
                    self.forces[env_ids_apply_push_step2, self.gripper_idx, :3] = self.force_target_gripper_ext[env_ids_apply_push_step2, :3] - (self.force_target_gripper_ext[env_ids_apply_push_step2, :3]/push_duration_reshaped)*(torch.clamp(self.episode_length_buf[env_ids_apply_push_step2].unsqueeze(-1) - (self.push_end_time_gripper_ext[env_ids_apply_push_step2].unsqueeze(-1)+self.settling_time_force_gripper), torch.zeros_like(push_duration_reshaped), push_duration_reshaped))
                
                    
                # Reset the tensors
                env_ids_to_reset = subset_env_ids_selected[self.episode_length_buf[self.selected_env_ids_gripper_ext == 1] >= (self.push_end_time_gripper_ext[self.selected_env_ids_gripper_ext == 1] + self.settling_time_force_gripper + self.push_duration_gripper_ext[self.selected_env_ids_gripper_ext == 1]).type(torch.int32)]                                        
                if env_ids_to_reset.nelement() > 0:
                    self.selected_env_ids_gripper_ext[env_ids_to_reset] = 0
                    self.force_target_gripper_ext[env_ids_to_reset, :3] = 0.
                    self.push_end_time_gripper_ext[env_ids_to_reset] = 0.
                    self.push_duration_gripper_ext[env_ids_to_reset] = 0.
                    self.push_interval_gripper_ext[env_ids_to_reset, 0] = torch.randint(int(self.push_interval_gripper_ext_min), int(self.push_interval_gripper_ext_max), (len(env_ids_to_reset), 1), device=self.device)[:, 0]
                    
            self.selected_env_ids_gripper_ext[self.freed_envs_gripper_ext] = 0
            self.force_target_gripper_ext[self.freed_envs_gripper_ext, :3] = 0.
            self.push_end_time_gripper_ext[self.freed_envs_gripper_ext] = 0.
            self.push_duration_gripper_ext[self.freed_envs_gripper_ext] = 0. 
            
            self.forces[self.freed_envs_gripper_ext, self.gripper_idx, :3] = 0


# ==================================================================================================================== #
# ================================================ Custom Rewards Function ================================================== #
# ==================================================================================================================== #
    
    def _reward_wheel_distance(self):
        """
        Calculates the reward based on the distance between the wheel. Penalize wheel get close to each other or too far away.
        """
        wheel_pos = self.rigid_state[:, self.wheel_indices, :2]
        wheel_dist = torch.norm(wheel_pos[:, 0, :] - wheel_pos[:, 1, :], dim=1)
        self.wheel_dist = wheel_dist.unsqueeze(1)
        fd = self.cfg.rewards.min_wheel_dist
        max_fd = self.cfg.rewards.max_wheel_dist
        d_min = torch.clamp(wheel_dist - fd, -0.5, 0.)
        d_max = torch.clamp(wheel_dist - max_fd, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        self.foot_dist = foot_dist.unsqueeze(1)
        fd = self.cfg.rewards.min_feet_dist
        max_fd = self.cfg.rewards.max_feet_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_fd, 0, 0.5)
        # return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
        return (torch.exp(torch.abs(d_min)) - 1) + (torch.exp(torch.abs(d_max)) - 1)

    def _reward_wheel_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the wheel.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.wheel_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        base_lin_acc = torch.norm(self.last_root_vel[:,0:3] - self.root_states[:, 7:10], dim=1) / self.cfg.sim.dt
        base_ang_acc = torch.norm(self.last_root_vel[:,3:6] - self.root_states[:, 10:13], dim=1) / self.cfg.sim.dt
        rew = base_lin_acc + 0.02 * base_ang_acc
        return rew
 
    # ------------------------------------------------------------------------------# 
    # ------------------------- stand still rewards --------------------------------# 
    # ------------------------------------------------------------------------------# 
    def _reward_stand_base_vel_penality(self):
        """当命令很小时，机器人不应该有各个方向的速度"""
        # Penalize motion at zero commands
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_wheel_vel_penality(self):
        """当命令很小时,机器人不应该有关节速度"""
        # Penalize motion at zero commands
        return torch.sum((torch.abs(self.wheel_vel)), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.exp(2 * abs(self.leg_pos[:,6:]))-1, dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    
    # ------------------------------------------------------------------------------# 
    # ------------------------- termination rewards --------------------------------# 
    # ------------------------------------------------------------------------------# 
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    # ------------------------------------------------------------------------------# 
    # --------------------------- tracking rewards ---------------------------------# 
    # ------------------------------------------------------------------------------# 
    def _reward_tracking_lin_vel(self):
        commands_lin_vel = self.commands[:,0]
        lin_vel_error = torch.square(commands_lin_vel - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error * (self.cfg.rewards.tracking_sigma_lin_vel))
    
    def _reward_tracking_lin_vel_enhance(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error *  self.cfg.rewards.tracking_sigma_lin_vel / 10) - 1

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_ang_vel)
    
    def _reward_tracking_ee_combine(self):
        tracking_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        threshold_base = self.cfg.commands.tracking_ee_start_step * self.train_cfg.runner.num_steps_per_env
        max_curriculum_steps = threshold_base * 5.0 
        
        if self.global_steps > max_curriculum_steps * 1.0:
            curriculum_progress = 1.0
        elif self.global_steps > max_curriculum_steps * 0.8:
            curriculum_progress = 0.8
        elif self.global_steps > max_curriculum_steps * 0.6:
            curriculum_progress = 0.6
        elif self.global_steps > max_curriculum_steps * 0.4:
            curriculum_progress = 0.4
        elif self.global_steps > max_curriculum_steps * 0.2:
            curriculum_progress = 0.2
        else:
            curriculum_progress = 0.0
        

        if self.global_steps > max_curriculum_steps * 3.0:
            alpha = 0.1
        elif self.global_steps > max_curriculum_steps * 2.4:
            alpha = 0.2
        elif self.global_steps > max_curriculum_steps * 1.9:
            alpha = 0.4
        elif self.global_steps > max_curriculum_steps * 1.5:
            alpha = 0.6
        elif self.global_steps > max_curriculum_steps * 1.2:
            alpha = 0.8
        else:
            alpha = 1

        if curriculum_progress == 0.0:

            initial_pos_l = 0.7
            initial_pos_p = 0.7
            initial_pos_y = 0.0
            initial_roll = 0.0
            initial_pitch = 0.5 * torch.pi
            initial_yaw = 0.0
            
            initial_pos = torch.tensor([initial_pos_l, initial_pos_p, initial_pos_y], 
                                    dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            initial_quat = quat_from_euler_xyz(
                torch.tensor([initial_roll], dtype=torch.float, device=self.device).repeat(self.num_envs),
                torch.tensor([initial_pitch], dtype=torch.float, device=self.device).repeat(self.num_envs),
                torch.tensor([initial_yaw], dtype=torch.float, device=self.device).repeat(self.num_envs)
            )
            initial_rot_mat = quaternion_to_rotation_matrix(initial_quat)
            target_pos = initial_pos
            target_rot = initial_rot_mat
        else:
            target_pos = self.arm_commands[:, :3]
            target_rot = self.arm_commands[:, 3:].reshape(self.num_envs, 3, 3)
        
        ee_local_cart = quat_rotate_inverse(self.base_yaw_quat, self.ee_pos - self.get_ee_goal_spherical_center())
        ee_pos_sphe_arm = cart2sphere(ee_local_cart)
        norm_force_offset = (ee_pos_sphe_arm - target_pos) / self.commands_arm_lpy_range
        
        rot_mat_truly = quaternion_to_rotation_matrix(self.ee_quat)
        rot_err_mat = target_rot @ rot_mat_truly.transpose(1, 2)
        trace = torch.diagonal(rot_err_mat, dim1=-2, dim2=-1).sum(dim=-1)
        trace = torch.clamp(trace, min=-1 + 1e-8, max=3 - 1e-8)
        rotation_magnitude = torch.arccos((trace - 1) / 2)
        rotation_magnitude = rotation_magnitude % (2 * np.pi)
        rotation_magnitude = torch.min(
            rotation_magnitude,
            2 * np.pi - rotation_magnitude,
        )
        rpy_error = rotation_magnitude
        lpy_error = torch.norm(norm_force_offset, dim=1, p=1)

        self.mean_lpy_error = torch.mean(lpy_error)
        self.mean_rpy_error = torch.mean(rpy_error)
    
        tracking_reward = torch.exp(-(lpy_error/(self.cfg.rewards.tracking_ee_pos_sigma *alpha) + 
                                                    rpy_error/(self.cfg.rewards.tracking_ee_orn_sigma*alpha)))

        return tracking_reward


    def _reward_wheel_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.wheel_vel), dim=1)
    
    def _reward_wheel_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.wheel_acc), dim=1)

    def _reward_default_joint_pos(self):
        rew = torch.norm(self.leg_pos[:,6:], dim=1)
        if self.reward_scales["default_joint_pos"] < 0:
            return rew
        else:
            return torch.exp(-20 * rew)

    def _reward_same_wheel_x_position(self):
        reward = 0
        wheel_positions_base = self.wheel_positions - \
                            (self.base_pos).unsqueeze(1).repeat(1, len(self.wheel_indices), 1)
        for i in range(len(self.wheel_indices)):
            wheel_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, wheel_positions_base[:, i, :] )
        wheel_x_position_err = wheel_positions_base[:,0,0] - wheel_positions_base[:,1,0]
        # reward = torch.exp(-(wheel_x_position_err ** 2)/ self.cfg.rewards.wheel_x_position_sigma)
        reward = torch.abs(wheel_x_position_err)
        return reward

    def _reward_same_foot_x_position(self):
        reward = 0
        feet_positions_base = self.feet_positions - \
                            (self.base_pos).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            feet_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, feet_positions_base[:, i, :] )
        foot_x_position_err = feet_positions_base[:,0,0] - feet_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return reward
        
    # ------------------------------------------------------------------------------# 
    # ---------------------- common regularization rewards -------------------------# 
    # ------------------------------------------------------------------------------#     
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        scale = self.reward_scales.get("base_height", None)
        if scale is None:
            return torch.zeros_like(self.base_height, device=self.device)
        
        if scale < 0:
            return torch.abs(self.base_height - self.commands[:, 2])
        else:
            base_height_error = torch.square(self.base_height - self.commands[:, 2])
            return torch.exp(-200 * base_height_error)   
        
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_action(self):
        # Penalize actions
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        return torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    # def _reward_dof_pos_limits(self):
    #     lower_violation = torch.clamp(self.dof_pos_limits[:, 0] * self.cfg.rewards.soft_dof_pos_limit - self.dof_pos, min=0.0)
    #     upper_violation = torch.clamp(self.dof_pos - self.dof_pos_limits[:, 1] * self.cfg.rewards.soft_dof_pos_limit, min=0.0)
    #     lower_violation[:, self.wheel_indices-1] = 0.0
    #     upper_violation[:, self.wheel_indices-1] = 0.0
    #     any_violation = torch.any((lower_violation > 0) | (upper_violation > 0), dim=1)
    #     return torch.where(any_violation, torch.tensor(10.0, device=self.device), torch.tensor(0.0, device=self.device))

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0] * self.cfg.rewards.soft_dof_pos_limit).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1] * self.cfg.rewards.soft_dof_pos_limit).clip(min=0.0)  # upper limit
        out_of_limits[:, self.wheel_indices-1] = 0.0
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limits).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_power(self):
        # Penalize torques
        return torch.sum(self.power, dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.abs(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_acc = self.dof_acc_500hz
        return torch.sum(torch.abs(dof_acc), dim=1)

    # def _reward_feet_air_time(self):
    #     """
    #     Calculates the reward for feet air time, promoting longer steps. This is achieved by
    #     checking the first contact with the ground after being in the air. The air time is
    #     limited to a maximum value for reward calculation.
    #     """
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    #     stance_mask = self._get_gait_phase()
    #     self.contact_filt_feet = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts_feet)
    #     self.last_contacts_feet = contact
    #     first_contact = (self.feet_air_time > 0.) * self.contact_filt_feet
    #     self.feet_air_time += self.dt
    #     air_time = self.feet_air_time.clamp(0, 0.4) * first_contact
    #     self.feet_air_time *= ~self.contact_filt_feet
    #     return air_time.sum(dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum((self.feet_air_time - 0.4) * self.first_contacts_feet, dim=1) # reward only on first contact with the ground
        # print("self.feet_air_time", self.feet_air_time)
        # print("self.first_contacts_feet", self.first_contacts_feet)
        # print("rew_airTime", rew_airTime)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 # no reward for zero command
        return rew_airTime

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -1.3) 
        return torch.mean(reward, dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_contact(self):
        self._get_phase()
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.feet_indices)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 5
            is_consistent = ~(contact ^ is_stance)
            res += torch.where(is_consistent, 
                               torch.tensor(1.0, device=self.device), 
                               torch.tensor(-1.3, device=self.device))
        return res * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_feet_clearance(self):
        cur_feetvel_translated = self.feet_velocities - self.root_states[:, 7:10].unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_feetvel_translated[:, i, :])
        feet_height, feet_height_var = self._get_feet_heights()
        # print("feet_height", feet_height)
        height_error = torch.square(feet_height - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * feet_leteral_vel, dim=1)
    
    def _reward_default_hip_roll(self):

        return torch.sum(torch.abs(self.dof_pos[:, [6,11]] - self.default_dof_pos[:, [6,11]]), dim=1)
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_velocities[:,:,:2], dim=2) * contact, dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             3 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_contact_safety(self):
        """
        惩罚当足端接触地面时接触比例小于0.8的情况
        只在足端接触地面时计算奖励
        """
        contact_forces = self.contact_forces[:, self.feet_indices, 2]  
        is_left_contact = contact_forces[:, 0] > 1.0 
        is_right_contact = contact_forces[:, 1] > 1.0  
        
        left_ratio_diff = torch.clamp(0.8 - self.feet_contact_ratio[:, 0], min=0.0)
        right_ratio_diff = torch.clamp(0.8 - self.feet_contact_ratio[:, 1], min=0.0)
        
        left_penalty = left_ratio_diff * is_left_contact.float()
        right_penalty = right_ratio_diff * is_right_contact.float()

        return (left_penalty + right_penalty)

    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3* self.dt) * self.contact_filt_feet
        return torch.sum(feet_heights_var * continue_contact, dim=1)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, :2], dim=1) > 0.1)) # full reward for zero command
        return rew_no_fly

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0