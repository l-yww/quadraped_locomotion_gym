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


from wheel_legged_gym.envs.cowa_10dof.cowa_10dof_config import CowaCfg, CowaCfgPPO

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import random
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot

from wheel_legged_gym.utils.terrain import  Terrain
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift


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

        # 添加mode状态，0表示轮子滚动模式，1表示驻足行走模式
        self.mode = torch.ones(self.num_envs, dtype=torch.float, device=self.device)  # 初始值为1
        self.last_mode = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # 初始值为0
        
        # 添加foot关节的目标位置
        self.foot_lift = -2.5  # 从1到0切换时foot的目标位置 delta
        self.foot_down = 0  # 从0到1切换时foot的目标位置
        
        # 记录模式切换时的起始模式
        self.mode_switch_start_mode = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.iter_count = 0

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
            
    def _init_buffers(self):
        super()._init_buffers()        
        self.noised_leg_q = torch.zeros((self.num_envs, self.num_dof - self.wheel_nums), device=self.device)
        self.noised_dq = torch.zeros_like(self.dof_vel)
        self.noised_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.noised_euler_rpy = torch.zeros_like(self.base_euler_rpy)
        self.episode_length_init = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.transition_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def  _get_phase(self):
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

    # def compute_ref_state(self):
    #     """
    #     Computes the reference state for the robot based on the current mode and mode transitions.
    #     - Wheeled mode (1): No reference trajectory, keep wheels on ground.
    #     - Legged mode (0): Stand or step in place.
    #     - Mode transition: Special reference trajectories for smooth transitions (only one cycle).
    #     """
    #     swing_duration = self.cfg.feedforward.swing_duration  
    #     phase_offset = self.cfg.feedforward.phase_offset      
    #     transition_duration = phase_offset + swing_duration

    #     self._get_phase()
    #     sin_pos = torch.sin(2 * torch.pi * self.phase)
    #     sin_pos_l = sin_pos.clone()
    #     sin_pos_r = sin_pos.clone()
    #     self.ref_dof_vel = torch.zeros_like(self.dof_pos)
    #     self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        
    #     # 检测模式切换
    #     mode_changed = (self.mode != self.last_mode)
        
    #     # 记录模式切换
    #     if torch.any(mode_changed):
    #         self.mode_switch_start_mode[mode_changed] = self.last_mode[mode_changed]
    #         # 只在第一次模式改变时记录episode_length_init
    #         # 使用torch.where确保只在mode_changed为True且episode_length_init为0时更新
    #         self.episode_length_init = torch.where(mode_changed & (self.episode_length_init == 0), 
    #                                             self.episode_length_buf.clone(), 
    #                                             self.episode_length_init)
    #     # 计算从模式切换开始的相位（只执行一个周期）
    #     phase_transition = (self.episode_length_buf - self.episode_length_init) * self.dt
    #     # 确保相位不超过一个周期
    #     phase_transition = torch.clamp(phase_transition, 0, transition_duration)
        
    #     self.transition_completed = phase_transition >= transition_duration

    #     # 情况1: 从0(轮式)切换到1(足式)
    #     switch_0_to_1 = (self.mode_switch_start_mode == 0) & (self.mode == 1) & (~self.transition_completed)

    #     if torch.any(switch_0_to_1):
    #         # 计算sin_pos_l和sin_pos_r基于过渡相位
    #         sin_pos_l = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]
    #         sin_pos_r = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]
            
    #         scale_1 = -0.2 #-0.4
    #         # 左脚运动（前半周期）
    #         sin_pos_l_mask = sin_pos_l > 0  # 只在前半周期运动
    #         self.ref_dof_pos[:, 1] -= sin_pos_l * sin_pos_l_mask.float() * scale_1  # hip pitch
    #         self.ref_dof_pos[:, 2] -= 2 * sin_pos_l * sin_pos_l_mask.float() * scale_1  # knee
    #         if sin_pos_l_mask:
    #             self.ref_dof_pos[:, 3] -= 0  # foot保持不变
            
    #         # 右脚运动（后半周期）
    #         sin_pos_r_mask = sin_pos_r < 0  # 只在后半周期运动
    #         self.ref_dof_pos[:, 6] += sin_pos_r * sin_pos_r_mask.float() * scale_1
    #         self.ref_dof_pos[:, 7] += 2 * sin_pos_r * sin_pos_r_mask.float() * scale_1
    #         if sin_pos_r_mask:
    #             self.ref_dof_pos[:, 8] += 0  # foot保持不变
        
    #     # 情况2: 从1(足式)切换到0(轮式)
    #     switch_1_to_0 = (self.mode_switch_start_mode == 1) & (self.mode == 0) & (~self.transition_completed)
        
    #     if torch.any(switch_1_to_0):
    #         # 计算sin_pos_l和sin_pos_r基于过渡相位
    #         sin_pos_l = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]
    #         sin_pos_r = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]

    #         scale_1 = -0.2 #-0.4
    #         # 左脚运动（前半周期）
    #         sin_pos_l_mask = sin_pos_l > 0  # 只在前半周期运动
    #         self.ref_dof_pos[:, 1] -= sin_pos_l * sin_pos_l_mask.float() * scale_1  # hip pitch
    #         self.ref_dof_pos[:, 2] -= 2 * sin_pos_l * sin_pos_l_mask.float() * scale_1  # knee
    #         if sin_pos_l_mask:
    #             self.ref_dof_pos[:, 3] -= 2.5 * sin_pos_l_mask.float()  # foot在前半周期移动
    #         else:
    #             self.ref_dof_pos[:, 3] = -2.5
            
    #         # 右脚运动（后半周期）
    #         sin_pos_r_mask = sin_pos_r < 0  # 只在后半周期运动
    #         self.ref_dof_pos[:, 6] += sin_pos_r * sin_pos_r_mask.float() * scale_1
    #         self.ref_dof_pos[:, 7] += 2 * sin_pos_r * sin_pos_r_mask.float() * scale_1
    #         self.ref_dof_pos[:, 8] += -2.5 * sin_pos_r_mask.float()  # foot在后半周期移动
    
    #     # 模式切换完成后，保持脚的位置
    #     if self.transition_completed.any():
    #         # 从0切换到1完成后，脚的位置保持为0
    #         completed_0_to_1 = (self.mode_switch_start_mode == 0) & (self.mode == 1) & self.transition_completed
    #         if completed_0_to_1.any():
    #             self.ref_dof_pos[completed_0_to_1, 3] = 0  # 左脚foot
    #             self.ref_dof_pos[completed_0_to_1, 8] = 0  # 右脚foot
                
    #         # 从1切换到0完成后，脚的位置保持为-2.5
    #         completed_1_to_0 = (self.mode_switch_start_mode == 1) & (self.mode == 0) & self.transition_completed
    #         if completed_1_to_0.any():
    #             self.ref_dof_pos[completed_1_to_0, 3] = -2.5  # 左脚foot
    #             self.ref_dof_pos[completed_1_to_0, 8] = -2.5  # 右脚foot

    #     return self.ref_dof_pos, self.ref_dof_vel

    def compute_ref_state(self):
        """
        Computes the reference state for the robot based on the current mode and mode transitions.
        - Wheeled mode (0): Foot position target is -2.5
        - Legged mode (1): Foot position target is 0
        - Mode transition: Special reference trajectories for smooth transitions (only one cycle).
        """
        swing_duration = self.cfg.feedforward.swing_duration  
        phase_offset = self.cfg.feedforward.phase_offset
        scale = - self.cfg.feedforward.scale   
        transition_duration = phase_offset + swing_duration

        self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        
        # 检测模式切换
        mode_changed = (self.mode != self.last_mode)
        
        # 记录模式切换
        if torch.any(mode_changed):
            self.mode_switch_start_mode[mode_changed] = self.last_mode[mode_changed]
            # 只在第一次模式改变时记录episode_length_init
            # 使用torch.where确保只在mode_changed为True且episode_length_init为0时更新
            self.episode_length_init = torch.where(mode_changed & (self.episode_length_init == 0), 
                                                self.episode_length_buf.clone(), 
                                                self.episode_length_init)
        # 计算从模式切换开始的相位（只执行一个周期）
        phase_transition = (self.episode_length_buf - self.episode_length_init) * self.dt
        # 确保相位不超过一个周期
        phase_transition = torch.clamp(phase_transition, 0, transition_duration)
        
        self.transition_completed = phase_transition >= transition_duration

        # 情况1: 从0(轮式)切换到1(足式)
        switch_0_to_1 = (self.mode_switch_start_mode == 0) & (self.mode == 1) & (~self.transition_completed)

        if torch.any(switch_0_to_1):
            # 计算sin_pos_l和sin_pos_r基于过渡相位
            sin_pos_l = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]
            sin_pos_r = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]
            
            # 左脚运动（前半周期）
            sin_pos_l_mask = sin_pos_l > 0  # 只在前半周期运动
            self.ref_dof_pos[:, 1] -= sin_pos_l * sin_pos_l_mask.float() * scale  # hip pitch
            self.ref_dof_pos[:, 2] -= 2 * sin_pos_l * sin_pos_l_mask.float() * scale  # knee

            # 前半周期(0-0.3s)完成翻转
            foot_l_flip_mask = phase_transition < (transition_duration / 4)
            if foot_l_flip_mask.any():
                # 计算从-2.5到0的插值
                foot_l_progress = (phase_transition / (transition_duration / 4)).clamp(0, 1)  # 只在前半周期变化
                self.ref_dof_pos[:, 3] = -2.5 + 2.5 * foot_l_progress  # 从-2.5移动到0
            else:
                # 后半周期保持0
                self.ref_dof_pos[:, 3] = 0
            
            # 右脚运动（后半周期）
            sin_pos_r_mask = sin_pos_r < 0  # 只在后半周期运动
            self.ref_dof_pos[:, 6] += sin_pos_r * sin_pos_r_mask.float() * scale
            self.ref_dof_pos[:, 7] += 2 * sin_pos_r * sin_pos_r_mask.float() * scale
            
            # 前半周期(0.6-0.9s)完成翻转
            foot_r_flip_mask = phase_transition >= (transition_duration / 2)
            if foot_r_flip_mask.any():
                # 计算从-2.5到0的插值
                foot_r_progress = ((phase_transition - transition_duration/2) / (transition_duration / 4)).clamp(0, 1)  # 只在后半周期变化
                self.ref_dof_pos[:, 8] = -2.5 + 2.5 * foot_r_progress  # 从-2.5移动到0
            else:
                # 前半周期保持-2.5
                self.ref_dof_pos[:, 8] = -2.5
        
        # 情况2: 从1(足式)切换到0(轮式)
        switch_1_to_0 = (self.mode_switch_start_mode == 1) & (self.mode == 0) & (~self.transition_completed)
        
        if torch.any(switch_1_to_0):
            # 计算sin_pos_l和sin_pos_r基于过渡相位
            sin_pos_l = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]
            sin_pos_r = torch.sin(2 * torch.pi * phase_transition / transition_duration)  # 归一化到[0, 1]

            # 左脚运动（前半周期）
            sin_pos_l_mask = sin_pos_l > 0  # 只在前半周期运动
            self.ref_dof_pos[:, 1] -= sin_pos_l * sin_pos_l_mask.float() * scale  # hip pitch
            self.ref_dof_pos[:, 2] -= 2 * sin_pos_l * sin_pos_l_mask.float() * scale  # knee
            
            # 左脚foot在前半周期保持0，在后半周期从0移动到-2.5
            # 后半周期(0.3-0.6s)完成翻转
            foot_l_flip_mask = phase_transition >= (transition_duration / 4)
            if foot_l_flip_mask.any():
                # 计算从0到-2.5的插值
                foot_l_progress = ((phase_transition - transition_duration/4) / (transition_duration / 4)).clamp(0, 1)  # 只在后半周期变化
                self.ref_dof_pos[:, 3] = 0 - 2.5 * foot_l_progress  # 从0移动到-2.5

            # 右脚运动（后半周期）
            sin_pos_r_mask = sin_pos_r < 0  # 只在后半周期运动
            self.ref_dof_pos[:, 6] += sin_pos_r * sin_pos_r_mask.float() * scale
            self.ref_dof_pos[:, 7] += 2 * sin_pos_r * sin_pos_r_mask.float() * scale
            
            # 后半周期(0.9-1.2s)完成翻转
            foot_r_flip_mask = phase_transition > (3 * transition_duration / 4)
            if foot_r_flip_mask.any():
                # 计算从0到-2.5的插值
                foot_r_progress = ((phase_transition - 3 * transition_duration / 4) / (transition_duration / 4)).clamp(0, 1)  # 只在前半周期变化
                self.ref_dof_pos[:, 8] = 0 - 2.5 * foot_r_progress  # 从0移动到-2.5


        # 模式切换完成后，根据当前模式保持脚的位置
        if self.transition_completed.any():
            # 从0切换到1完成后，脚的位置保持为0
            completed_0_to_1 = (self.mode_switch_start_mode == 0) & (self.mode == 1) & self.transition_completed
            if completed_0_to_1.any():
                self.ref_dof_pos[completed_0_to_1, 3] = 0  # 左脚foot
                self.ref_dof_pos[completed_0_to_1, 8] = 0  # 右脚foot
                
            # 从1切换到0完成后，脚的位置保持为-2.5
            completed_1_to_0 = (self.mode_switch_start_mode == 1) & (self.mode == 0) & self.transition_completed
            if completed_1_to_0.any():
                self.ref_dof_pos[completed_1_to_0, 3] = -2.5  # 左脚foot
                self.ref_dof_pos[completed_1_to_0, 8] = -2.5  # 右脚foot
        
        # 根据当前模式设置脚的目标位置
        # 对于没有进行模式切换的环境，根据当前模式设置脚的位置
        no_transition = ~mode_changed
        if no_transition.any():
            # 当前模式为1(足式模式)，脚的位置保持为0
            mode_1 = (self.mode == 1) & no_transition
            if mode_1.any():
                self.ref_dof_pos[mode_1, 3] = 0  # 左脚foot
                self.ref_dof_pos[mode_1, 8] = 0  # 右脚foot
                
            # 当前模式为0(轮式模式)，脚的位置保持为-2.5
            mode_0 = (self.mode == 0) & no_transition
            if mode_0.any():
                self.ref_dof_pos[mode_0, 3] = -2.5  # 左脚foot
                self.ref_dof_pos[mode_0, 8] = -2.5  # 右脚foot

        return self.ref_dof_pos, self.ref_dof_vel


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

        noise_vec[:5] = 0
        noise_vec[5:5+self.num_actions-2] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[5+self.num_actions-2:5+2*self.num_actions-2] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[5+2*self.num_actions-2:5+3*self.num_actions-2] = 0 
        noise_vec[5+3*self.num_actions-2:5+3*self.num_actions-2+3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[5+3*self.num_actions-2+3:5+3*self.num_actions-2+3+orientation_dim] = noise_scales.gravity  * self.obs_scales.gravity
        noise_vec[5+3*self.num_actions-2+3+orientation_dim:] = 0
        return noise_vec

    def step(self, actions):

        self.iter_count += 1
        iter = int(self.iter_count / self.train_cfg.runner.num_steps_per_env) # num_steps_per_env

        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay

        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions

        ## add forward feedbacks
        if self.cfg.feedforward.use_feedforward:
            kb_values =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            kb = torch.tensor(kb_values[:self.num_dof], device=self.device, dtype=torch.float32)
            final_actions = kb * actions + self.alpha * self.ref_dof_pos

            if self.cfg.feedforward.use_annealing:
                if iter > self.cfg.feedforward.start_iter:
                    self.alpha = max(0.0, 1.0 - (iter - self.cfg.feedforward.start_iter) / self.cfg.feedforward.duration)
        else:
            kb_values =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            kb = torch.tensor(kb_values[:self.num_dof], device=self.device, dtype=torch.float32)
            final_actions = kb * actions

        return super().step(final_actions)

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

    def _resample_mode(self, env_ids):
        """ Randomly sample modes for some environments

        Args:
            env_ids (List[int]): Environments ids for which new modes are needed
        """
        if len(env_ids) == 0:
            return
        
        # Randomly sample new modes (0 or 1) for these environments
        new_modes = torch.randint(0, 2, (len(env_ids),), device=self.device).float()
        
        # Update modes
        self.last_mode[env_ids] = self.mode[env_ids].clone()
        self.mode[env_ids] = new_modes

        # print("last_mode", self.last_mode[env_ids])
        # print("mode", self.mode[env_ids])

        self.episode_length_init[env_ids] = 0
        self.mode_switch_start_mode[env_ids] = self.mode[env_ids].clone()


    def compute_observations(self):
        self.compute_ref_state()
        self._get_gait_phase()
        self._get_feet_heights()

        # self.is_collision = self._check_feet_collision().float()
        
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
        
        # step_width_val = torch.tensor(self.terrain.step_width, device=self.device)
        # step_width_tensor = step_width_val.unsqueeze(0).unsqueeze(1).expand(self.num_envs, -1) * self.obs_scales.height_measurements

        self.privileged_obs_buf = torch.cat(( # 
            sin_pos,
            cos_pos,
            self.commands[:, :3] * self.commands_scale,  # 3
            self.leg_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            # self.base_euler_rpy[:,:] * self.obs_scales.quat,
            self.projected_gravity, # 重力投影
            # heights,
            # step_width_tensor,
            self.mode.unsqueeze(1),
        ), dim=-1)

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
                                             self.base_height_obs *self.obs_scales.height_measurements,
                                            self.base_lin_vel * self.obs_scales.lin_vel, 
                                            # self.is_collision, # 2
                                            # self.feet_contact_ratio, # 2
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

        lagged_q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        lagged_dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        # 剔除 wheel 的 dof_pos 观测
        mask = torch.ones(lagged_q.shape[1], dtype=torch.bool)
        mask[self.wheel_indices-1] = False
        lagged_leg_q = lagged_q[:, mask]

        obs_buf = torch.cat((
            sin_pos,
            cos_pos,
            self.commands[:, :3]  * self.commands_scale,
            lagged_leg_q,
            lagged_dq,
            self.actions,
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,
        ), dim=-1)
        if self.cfg.env.projected_gravity:
            obs_buf = torch.cat((
                obs_buf,
                self.lagged_projected_gravity * self.obs_scales.quat,
            ),dim=-1)
        else:
            obs_buf = torch.cat((
                obs_buf,
                # self.lagged_base_euler_rpy * self.obs_scales.quat,
                self.lagged_base_euler_rpy[:,:2] * self.obs_scales.quat,
            ),dim=-1)

        # obs_buf = torch.cat((
        #     obs_buf,  
        #     heights,
        # ), dim=-1)

        obs_buf = torch.cat((
            obs_buf,  
            self.mode.unsqueeze(1),  # 只保留当前模式
        ), dim=-1)

        obs_now = obs_buf.clone()

        if self.add_noise:  
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise
        self.noised_leg_q = obs_now[:, 3:3+self.cfg.env.num_actions-2] / self.obs_scales.dof_pos
        self.noised_dq = obs_now[:, 3+self.cfg.env.num_actions-2 : 3+2*self.cfg.env.num_actions-2] / self.obs_scales.dof_vel
        self.noised_ang_vel = obs_now[: 3+3*self.cfg.env.num_actions-2:3+3*self.cfg.env.num_actions-2+3] / self.obs_scales.ang_vel
        self.noised_euler_rpy = obs_now[:, 3+3*self.cfg.env.num_actions-2+3:3+3*self.cfg.env.num_actions-2+5] / self.obs_scales.quat

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        
        mode_env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time_mode / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_mode(mode_env_ids)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

        self.feet_contact_safety[env_ids] = False
        self.feet_contact_ratio[env_ids] = 0.0

        self._resample_mode(env_ids)

        # 重置mode为1（行走模式）
        self.mode[env_ids] = 1.0
        self.last_mode[env_ids] = 1.0
        self.mode_switch_start_mode[env_ids] = 1.0
        self.episode_length_init[env_ids] = 0

# ==================================================================================================================== #
# ================================================ Custom Rewards Function ================================================== #
# ==================================================================================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        The reward is only computed during mode transitions (when transition_completed is False).
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        
        # Check if mode transition is ongoing (not completed)
        transition_ongoing = ~ self.transition_completed.bool()
        
        # Calculate joint position difference
        diff = joint_pos[:,[1,2,6,7]] - pos_target[:,[1,2,6,7]]
        
        # Compute reward
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        
        # Apply reward only during mode transitions
        return r * transition_ongoing
    
    def _reward_foot_pos(self):
        """
        Calculates the reward based on the difference between the current foot positions and the target foot positions.
        The reward is computed during mode transitions (0→1 or 1→0) and after transitions are completed.
        No reward is calculated when there is no mode change.
        """
        foot_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        
        # 检测是否发生了模式切换（mode != last_mode）或切换正在进行中
        mode_changed = (self.mode != self.last_mode) | (~self.transition_completed.bool())
        
        # Calculate foot position difference (left foot: index 3, right foot: index 8)
        diff = foot_pos[:,[3,8]] - pos_target[:,[3,8]]
        # print("foot_pos", foot_pos[:,[3,8]])
        # print("pos_target", pos_target[:,[3,8]])
        # print(diff)
        
        # Compute reward using exponential decay function
        r = torch.exp(-2 * torch.norm(diff, dim=1))
        
        # 只在发生模式切换或切换进行中时应用奖励
        return r * mode_changed
    
    def _reward_foot_pos_mode(self):
        """
        根据当前模式奖励脚的位置保持在正确的位置：
        - mode=1 (足式模式): 脚的位置应保持在 0
        - mode=0 (轮式模式): 脚的位置应保持在 -2.5
        
        Returns:
            torch.Tensor: 奖励值，范围在[0, 1]之间，1表示完美匹配
        """
        # 获取当前脚的位置
        foot_pos = self.dof_pos.clone()
        # 左脚foot: index 3, 右脚foot: index 8
        current_foot_pos = foot_pos[:, [3, 8]]
        
        # 根据模式确定目标位置
        target_foot_pos = torch.zeros_like(current_foot_pos)
        
        # mode=1 (足式模式)时，目标位置为0
        mode_1_mask = self.mode == 1
        if mode_1_mask.any():
            target_foot_pos[mode_1_mask] = 0
        
        # mode=0 (轮式模式)时，目标位置为-2.5
        mode_0_mask = self.mode == 0
        if mode_0_mask.any():
            target_foot_pos[mode_0_mask] = -2.5
        
        # 计算当前位置与目标位置的差异
        diff = current_foot_pos - target_foot_pos

        reward = torch.sum(torch.square(diff), dim=1)
        
        return reward


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

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.8 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)        


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

    def _reward_nominal_wheel_position(self):
        #1. calculate foot postion wrt base in base frame 
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.wheel_radius)
        wheel_positions_base = self.wheel_positions - \
                            (self.base_pos).unsqueeze(1).repeat(1, len(self.wheel_indices), 1)
        reward = 0
        for i in range(len(self.wheel_indices)):
            wheel_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, wheel_positions_base[:, i, :] )
            height_error = nominal_base_height - wheel_positions_base[:, i, 2]   # 腿长的约束
            reward += torch.exp(-500*(height_error ** 2)) / 2.0
        wheel_center_positions_base = (wheel_positions_base[:, 0, :] + wheel_positions_base[:, 1, :]) / 2.0    # 质心落点的约束
        x_error = wheel_center_positions_base[:, 0] - self.base_com[:, 0]
        reward += torch.exp(-20000*(x_error ** 2))
        return reward / 2.0 * torch.exp(- 4 * torch.sum(torch.square(self.commands[:, :2]),dim=1))

    def _reward_stability(self):
        """当命令很小时,惩罚机器人速度、角速度、关节扭矩，以保持机器人静止时较为稳定"""
        velocity_error = torch.sum(torch.abs(self.base_lin_vel[:, :3]), dim=1)
        energy_cost = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1) * 0.01  # 能量惩罚系数
        stability_penalty = torch.sum(torch.abs(self.base_ang_vel[:, :3]), dim=1) * 0.2  # 身体角速度惩罚
        reward = (1.0 / (1.0 + velocity_error)) - energy_cost - stability_penalty
        return reward * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
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
    
    def _reward_wheel_vel(self):
        # Penalize dof velocities only when mode=1 (legged mode)
        # When mode=0 (wheeled mode), do not penalize wheel velocity
        return torch.sum(torch.square(self.wheel_vel), dim=1)
    
    def _reward_wheel_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.wheel_acc), dim=1)

    def _reward_default_joint_pos(self):
        rew = torch.norm(self.leg_pos, dim=1)
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
    
    def _reward_opposite_wheel_vel(self):
            opposite_vel_l = self.commands[:, 0] * self.wheel_vel[:, 0]
            opposite_vel_r = self.commands[:, 0] * self.wheel_vel[:, 1]
            opposite_vel_l_ind = torch.where(opposite_vel_l>0)
            opposite_vel_r_ind = torch.where(opposite_vel_r>0)
            opposite_vel_l[opposite_vel_l_ind] = 0
            opposite_vel_r[opposite_vel_r_ind] = 0
            return torch.abs(opposite_vel_l) + torch.abs(opposite_vel_r)
    
    def _reward_opposite_vel(self):
            opposite_vel = self.commands[:, 0] * self.base_lin_vel[:, 0]
            opposite_vel_ind = torch.where(opposite_vel>0)
            opposite_vel[opposite_vel_ind] = 0
            return torch.abs(opposite_vel)
        
    # ------------------------------------------------------------------------------# 
    # ---------------------- common regularization rewards -------------------------# 
    # ------------------------------------------------------------------------------#     
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
        c_update = (lin_mismatch + ang_mismatch) / 2.
        return c_update

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_positive(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_rpy[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

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
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.exp(2 * abs(self.leg_pos))-1, dim=1)
    
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
        return torch.sum(height_error * feet_leteral_vel, dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_default_hip_roll(self):

        return torch.sum(torch.abs(self.dof_pos[:, [0,5]] - self.default_dof_pos[:, [0,5]]), dim=1)
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_velocities[:,:,:2], dim=2) * contact, dim=1)

    def _reward_feet_pitch_level(self):
        feet_pitch = self.feet_euler_rpy[:, 1, :] 
        # contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        pitch_penalty = torch.sum(torch.square(feet_pitch), dim=1)
        return pitch_penalty

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