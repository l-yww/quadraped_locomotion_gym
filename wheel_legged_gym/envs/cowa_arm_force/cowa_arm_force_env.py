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


from wheel_legged_gym.envs.cowa_arm_force.cowa_arm_force_config import CowaCfg, CowaCfgPPO
import pdb
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from collections import deque
import torch
import random
from wheel_legged_gym.envs.cowa_arm_force.legged_robot import LeggedRobot

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
        self.forces[env_ids,self.gripper_idx, :3] = 0
        self.selected_env_ids_gripper_ext[env_ids] = 0
        self.force_target_gripper_ext[env_ids, :3] = 0.

    # TODO
    def get_ee_goal_spherical_center(self):  # 球坐标系的原点
        return self.rigid_state[:,1,:3]

    def _init_buffers(self):
        super()._init_buffers()        
        self.noised_q = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.noised_dq = torch.zeros_like(self.dof_vel)
        self.noised_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.noised_euler_rpy = torch.zeros_like(self.base_euler_rpy)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales

        noise_vec[:12] = 0
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0 
        noise_vec[12+3*self.num_actions:] = 0
        return noise_vec

    def command_set_sample(self, dof_pos):
        """修复版：采样关节位姿并检测碰撞/位置越界"""
        # 1. 克隆输入关节位置，避免原张量被修改
        target_dof_pos = dof_pos.clone()
        # 2. 更新DOF状态（修复维度匹配问题）
        self.dof_state[:, 0] = target_dof_pos.view(-1)  # 展平为一维，匹配dof_state维度
        # 3. 生成环境索引，设置DOF状态
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids),
            len(env_ids)
        )
        
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        self.render()
        pos = quat_rotate_inverse(self.base_quat, self.rigid_state[:, 6, :3] - self.rigid_state[:, 0, :3])
        pos_2 = quat_rotate_inverse(self.base_quat, self.rigid_state[:, 6, :3] - self.rigid_state[:, 1, :3])
       
        quat = self.ee_quat  
        ls_collision = False
        for env_idx in range(self.num_envs):
            x, y, z = pos[env_idx, 0], pos[env_idx, 1], pos[env_idx, 2]
            x_2, y_2, z_2 = pos_2[env_idx, 0], pos_2[env_idx, 1], pos_2[env_idx, 2]
            if ((-0.6 <= x <= 0.4) and (-0.4 <= y <= 0.4) and (z <= 0.4)) or ((-0.2 <= x_2 <= 0.2) and (-0.2 <= y_2 <= 0.2) and (0<= z_2 <= 0.5)):
                ls_collision = True
                break
        pos = quat_rotate_inverse(self.base_quat, self.rigid_state[:, 6, :3] - self.rigid_state[:, 1, :3])
        return pos[0], quat[0], ls_collision



    def compute_observations(self):

        forces_apply_gripper = self.forces[:, self.gripper_idx, 0:3]
        if self.cfg.commands.test_chip :
            ee_goal = self.commands[:,:3]
        else:
            ee_goal = self.commands[:,:3] - forces_apply_gripper * 0.05
        # ee_goal = self.commands[:,:3] 
        self.privileged_obs_buf = torch.cat(( 
            self.arm_commands[:,:3] * self.arm_commands_scale,  # 3
            self.arm_commands[:,3:] * self.obs_scales.ee_orn_quat, # 9
            self.dof_pos * self.obs_scales.dof_pos, # 6
            self.dof_vel * self.obs_scales.dof_vel, # 6
            self.last_actions, # 6
            forces_apply_gripper *self.obs_scales.ee_force,
            ee_goal

        ),dim=-1)

        # 始终将需要预测的值量放在最后（若有EST网络）
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                            self.rigid_state[:,6,:3] - self.rigid_state[:,1,:3]
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

        obs_buf = torch.cat((
            self.arm_commands[:,:3] * self.arm_commands_scale,  # 3
            self.arm_commands[:,3:] * self.obs_scales.ee_orn_quat, # 9
            lagged_q, # 6
            lagged_dq, # 6
            self.last_actions, # 6
            ee_goal
        ), dim=-1)

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
            # FORCE CONTROLLED ENVS
            if self.global_steps > self.cfg.commands.force_start_step * self.train_cfg.runner.num_steps_per_env * 3:
                self.start_curriculum = 2
            elif self.global_steps > self.cfg.commands.force_start_step * self.train_cfg.runner.num_steps_per_env * 2:
                self.start_curriculum = 1
            else:
                self.start_curriculum = 0

            min_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[0] - 10 * self.start_curriculum
            max_force_ext = self.cfg.commands.max_push_force_xyz_gripper_ext[1] + 10 * self.start_curriculum
            if min_force_ext < -self.cfg.commands.max_force_ext_range:
                min_force_ext = -self.cfg.commands.max_force_ext_range
            if max_force_ext > self.cfg.commands.max_force_ext_range:
                max_force_ext = self.cfg.commands.max_force_ext_range
            # print(max_force_ext)

            # ext force
            new_selected_env_ids_ext = env_ids_all[(self.episode_length_buf % self.push_interval_gripper_ext[:, 0]) == 0]
                
            
            # Define force and duration for the push 
            if new_selected_env_ids_ext.nelement() > 0:
                
                self.freed_envs_gripper_ext[new_selected_env_ids_ext] = torch.rand(len(new_selected_env_ids_ext), dtype=torch.float, device=self.device, requires_grad=False) > self.cfg.commands.gripper_forced_prob_ext
            

                self.force_target_gripper_ext[new_selected_env_ids_ext, 0] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
                self.force_target_gripper_ext[new_selected_env_ids_ext, 1] = torch_rand_float(min_force_ext, max_force_ext, (len(new_selected_env_ids_ext), 1), device=self.device).view(len(new_selected_env_ids_ext))
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

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_tracking_ee_combine(self):

        norm_offset = self.rigid_state[:,6,:3] - self.get_ee_goal_spherical_center() - self.arm_commands[:,:3]

        #ee_orentation
        rot_mat_truly = quaternion_to_rotation_matrix(self.ee_quat)
        rot_mat_target = self.arm_commands[:,3:].reshape(self.num_envs,3,3)
        rot_err_mat = rot_mat_target @ rot_mat_truly.transpose(1, 2)
        trace = torch.diagonal(rot_err_mat, dim1=-2, dim2=-1).sum(dim=-1)
        # to prevent numerical instability, clip the trace to [-1, 3]
        trace = torch.clamp(trace, min=-1 + 1e-8, max=3 - 1e-8)
        rotation_magnitude = torch.arccos((trace - 1) / 2)
        # account for symmetry
        rotation_magnitude = rotation_magnitude % (2 * np.pi)
        rotation_magnitude = torch.min(
            rotation_magnitude,
            2 * np.pi - rotation_magnitude,
        )
        rpy_error = rotation_magnitude
        lpy_error = torch.norm(norm_offset, dim=1,p=1)

        self.mean_lpy_error = torch.mean(lpy_error)
        self.mean_rpy_error = torch.mean(rpy_error)

        return torch.exp(-(lpy_error/(self.cfg.rewards.tracking_ee_pos_sigma) + rpy_error/(self.cfg.rewards.tracking_ee_orn_sigma)))

        
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