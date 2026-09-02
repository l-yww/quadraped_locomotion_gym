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
from collections import deque
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
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
            
    def _init_buffers(self):
        super()._init_buffers()        
        self.noised_leg_q = torch.zeros((self.num_envs, self.num_dof - self.wheel_nums), device=self.device)
        self.noised_dq = torch.zeros_like(self.dof_vel)
        self.noised_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.noised_euler_rpy = torch.zeros_like(self.base_euler_rpy)

        self.contact_force_history = deque(maxlen=self.cfg.env.contact_force_frame)
        wheel_contact_forces = self.contact_forces[:, self.wheel_indices, :]
        for _ in range(self.cfg.env.contact_force_frame):
            self.contact_force_history.append(wheel_contact_forces)
        self.avg_contact_force = torch.mean(torch.stack([self.contact_force_history[i] for i in range(self.cfg.env.contact_force_frame)]),dim=0)
        self.episode_length_init = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.leading_leg = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.mask_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.mask_right = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_wheel_z = torch.zeros(self.num_envs, len(self.wheel_indices), device=self.device, dtype=torch.float)
        self.start = torch.ones_like(self.mask_right)
        self.force_history_len = self.cfg.feedforward.trigger_len 
        self.left_contact_history = torch.zeros(self.num_envs, self.force_history_len, dtype=torch.bool, device=self.device)
        self.right_contact_history = torch.zeros(self.num_envs, self.force_history_len, dtype=torch.bool, device=self.device)
        self.mask = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    
    def _get_gait_phase(self):

        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = self.mask_left
        # right foot stance
        stance_mask[:, 1] = self.mask_right

        return stance_mask
    
    def compute_ref_state(self):
        swing_duration = self.cfg.feedforward.swing_duration  # e.g. 0.6
        phase_offset = self.cfg.feedforward.phase_offset      # e.g. 0.6

        phase = (self.episode_length_buf - self.episode_length_init) * self.dt
        phase = phase * (1 - self.start)  # 如果 start=1，phase=0，跳过 phase 控制

        reset_index = torch.where(phase >= (phase_offset + swing_duration - self.dt))[0]
        self.start[reset_index] = 1
        self.mask[reset_index] = 0

        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()

        scale_1 = self.cfg.feedforward.scale
        scale_2 = 2.0 * scale_1

        # ======================
        # 左腿 swing 控制
        # ======================

        # 构造 left_start / left_end，区分 leading_leg 情况（仅对 mask==3 有效）
        left_start = torch.where(
            self.leading_leg == 0,
            0.01,
            0.01 + phase_offset
        )
        left_end = left_start + swing_duration

        phase_left = (phase - left_start) / swing_duration
        mask_left = (phase < left_start) | (phase >= left_end)
        phase_left[mask_left] = 0.0
        phase_cos_l = (-torch.cos(2 * torch.pi * phase_left) + 1) / 2
        phase_cos_l[mask_left] = 0.0

        left_valid = (self.mask == 1) | (self.mask == 3)
        self.ref_dof_pos[left_valid, 1] += phase_cos_l[left_valid] * scale_1
        self.ref_dof_pos[left_valid, 2] += phase_cos_l[left_valid] * scale_2
        # ======================
        # 右腿 swing 控制
        # ======================

        right_start = torch.where(
            self.leading_leg == 1,
            0.01,
            0.01 + phase_offset
        )
        right_end = right_start + swing_duration

        phase_right = (phase - right_start) / swing_duration
        mask_right = (phase < right_start) | (phase >= right_end)
        phase_right[mask_right] = 0.0
        phase_cos_r = (-torch.cos(2 * torch.pi * phase_right) + 1) / 2
        phase_cos_r[mask_right] = 0.0

        right_valid = (self.mask == 2) | (self.mask == 3)
        self.ref_dof_pos[right_valid, 5] += phase_cos_r[right_valid] * scale_1
        self.ref_dof_pos[right_valid, 6] += phase_cos_r[right_valid] * scale_2

        self.mask_left = mask_left.float()  # 1 = stance, 0 = swing
        self.mask_right = mask_right.float()

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

        noise_vec[:3] = 0
        noise_vec[3:3+self.num_actions-2] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[3+self.num_actions-2:3+2*self.num_actions-2] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[3+2*self.num_actions-2:3+3*self.num_actions-2] = 0 
        noise_vec[3+3*self.num_actions-2:3+3*self.num_actions-2+3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[3+3*self.num_actions-2+3:3+3*self.num_actions-2+3+orientation_dim] = noise_scales.gravity  * self.obs_scales.gravity
        noise_vec[3+3*self.num_actions-2+3+orientation_dim:] = 0
        return noise_vec

    def step(self, actions):
        self.trigger_state_update()
        self.compute_ref_state()
        if self.cfg.feedforward.use_ref_action_only:
            # 在调试模式下，完全模拟 mask==3 的情况，并实现周期性运动
            swing_duration = self.cfg.feedforward.swing_duration  
            phase_offset = self.cfg.feedforward.phase_offset
            full_cycle_duration = phase_offset + swing_duration  # 完整周期时长
            
            # 使用周期性的 phase
            phase = self.episode_length_buf * self.dt
            cycle_phase = phase % full_cycle_duration  # 使用模运算实现周期性
            
            # 初始化 ref 动作
            self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
            scale_1 = self.cfg.feedforward.scale
            scale_2 = 2.0 * scale_1

            # 模拟 mask==3 的情况，双腿都运动
            # 左腿运动（leading_leg=0）
            left_start = 0.01
            left_end = left_start + swing_duration
            
            # 计算左腿相位
            phase_left = (cycle_phase - left_start) / swing_duration
            mask_left = (cycle_phase < left_start) | (cycle_phase >= left_end)
            phase_left[mask_left] = 0.0
            phase_cos_l = torch.sin(torch.pi * phase_left)
            # phase_cos_l = (-torch.cos(2 * torch.pi * phase_left) + 1) / 2
            phase_cos_l[mask_left] = 0.0

            # 右腿运动（leading_leg=1）
            right_start = 0.01 + phase_offset
            right_end = right_start + swing_duration
            
            # 计算右腿相位
            phase_right = (cycle_phase - right_start) / swing_duration
            mask_right = (cycle_phase < right_start) | (cycle_phase >= right_end)
            phase_right[mask_right] = 0.0
            phase_cos_r = torch.sin(torch.pi * phase_right)
            # phase_cos_r = (-torch.cos(2 * torch.pi * phase_right) + 1) / 2
            phase_cos_r[mask_right] = 0.0

            # 应用运动（与 compute_ref_state 中 mask==3 的情况一致）
            self.ref_dof_pos[:, 1] += phase_cos_l * scale_1
            self.ref_dof_pos[:, 2] += phase_cos_l * scale_2
            self.ref_dof_pos[:, 5] += phase_cos_r * scale_1
            self.ref_dof_pos[:, 6] += phase_cos_r * scale_2

            # 更新 mask 状态用于观察
            self.mask_left = mask_left.float()  # 1 = stance, 0 = swing
            self.mask_right = mask_right.float()
            
            final_actions = self.alpha * self.ref_dof_pos
        else:
            ## add forward feedbacks
            if self.cfg.feedforward.use_feedforward:
                kb_values =[0.8, 1., 1., 1., 0.8, 1., 1., 1.]
                # kb_values =[1, 1, 1, 1, 1, 1, 1, 1]
                kb = torch.tensor(kb_values[:self.num_dof], device=self.device, dtype=torch.float32)
                final_actions = kb * actions + self.alpha * self.ref_dof_pos

                if self.cfg.feedforward.use_annealing:
                    if iter > self.cfg.feedforward.start_iter:
                        self.alpha = max(0.0, 1.0 - (iter - self.cfg.feedforward.start_iter) / self.cfg.feedforward.duration)
            else:
                kb_values =[0.8, 1., 1., 1., 0.8, 1., 1., 1.]
                # kb_values =[1, 1, 1, 1., 1, 1, 1, 1.]
                kb = torch.tensor(kb_values[:self.num_dof], device=self.device, dtype=torch.float32)
                final_actions = kb * actions

        self.actions = final_actions

        return super().step(self.actions)

    def trigger_state_update(self):
        # 当前帧满足条件的 mask
        # 当前帧是否接触（单帧判定）
        left_xy_force = self.avg_contact_force[:, 0, :2]
        right_xy_force = self.avg_contact_force[:, 1, :2]
        left_force_sum = torch.sum(torch.abs(left_xy_force), dim=1)
        right_force_sum = torch.sum(torch.abs(right_xy_force), dim=1)
        left_contact_now = left_force_sum > 20.
        right_contact_now = right_force_sum > 20.

        # 更新左脚接触历史
        self.left_contact_history[:, :-1] = self.left_contact_history[:, 1:].clone()
        self.left_contact_history[:, -1] = left_contact_now

        # 更新右脚接触历史
        self.right_contact_history[:, :-1] = self.right_contact_history[:, 1:].clone()
        self.right_contact_history[:, -1] = right_contact_now

        # 连续多帧都满足才认为“确实接触”
        left_contact_stable = torch.all(self.left_contact_history, dim=1)
        right_contact_stable = torch.all(self.right_contact_history, dim=1)

        # 判断是否有过接触
        left_any = torch.any(self.left_contact_history, dim=1)
        right_any = torch.any(self.right_contact_history, dim=1)

        #! 判断三种情况(new)
        only_left = left_contact_stable & ~right_any
        only_right = ~left_any & right_contact_stable
        both_stable = (left_contact_stable & right_any) | (right_contact_stable & left_any) | (left_contact_stable & right_contact_stable)

        # 初始化 mask
        mask = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        mask[only_left] = 1
        mask[only_right] = 2
        mask[both_stable] = 3

        init_mask = (mask > 0) & (self.start == 1)
        init_indices = torch.where(init_mask)[0]
        self.mask[init_indices] = mask[init_indices]
        # 左腿先抬
        left_update_mask = (mask == 1) & (self.start ==1)
        self.leading_leg[left_update_mask] = 0

        # 右腿先抬
        right_update_mask = (mask == 2) & (self.start ==1)
        self.leading_leg[right_update_mask] = 1

        #! 双腿都接触，比较力大小决定谁先抬(new)
        both_indices = torch.where(self.mask == 3)[0]
        if both_indices.numel() > 0:
            # 提取相关状态
            left_stable_b = left_contact_stable[both_indices]
            right_stable_b = right_contact_stable[both_indices]
            left_any_b = left_any[both_indices]
            right_any_b = right_any[both_indices]

            # 初始化 leading_leg 为 -1（表示未定）
            leading_leg = torch.full((both_indices.shape[0],), -1, dtype=torch.long, device=self.device)

            # 条件1：左stable & 右only any → 抬左腿（leading_leg = 0）
            cond_left_lead = left_stable_b & right_any_b & ~right_stable_b
            leading_leg[cond_left_lead] = 0

            # 条件2：右stable & 左only any → 抬右腿（leading_leg = 1）
            cond_right_lead = right_stable_b & left_any_b & ~left_stable_b
            leading_leg[cond_right_lead] = 1

            # 条件3：两腿都 stable → 力大的先抬
            cond_both_stable = left_stable_b & right_stable_b
            if cond_both_stable.any():
                indices = torch.where(cond_both_stable)[0]
                actual_indices = both_indices[indices]

                left_forces = torch.abs(self.avg_contact_force[actual_indices, 0, 0])
                right_forces = torch.abs(self.avg_contact_force[actual_indices, 1, 0])

                lead_leg_force_based = torch.where(
                    right_forces > left_forces,
                    torch.ones_like(right_forces, dtype=torch.long, device=self.device),  # right
                    torch.zeros_like(left_forces, dtype=torch.long, device=self.device)   # left
                )
                leading_leg[indices] = lead_leg_force_based

            # 更新 leading_leg
            self.leading_leg[both_indices] = leading_leg

        # 更新状态
        self.episode_length_init[init_indices] = self.episode_length_buf[init_indices]
        self.start[init_indices] = 0

    def compute_observations(self):
        self.compute_ref_state()
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )

        self.contact_force_left = self.contact_forces[:, self.wheel_indices[0], :]
        self.contact_force_right = self.contact_forces[:, self.wheel_indices[1], :]
        
        self.privileged_obs_buf = torch.cat(( # 
            self.commands[:, :3] * self.commands_scale,  # 3
            self.leg_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            # self.base_euler_rpy[:,:] * self.obs_scales.quat,
            self.projected_gravity, # 重力投影
            heights,
        ), dim=-1)

        if self.cfg.env.priv_avg_contact_force:
             # 6
            self.avg_contact_force_left = self.avg_contact_force[:,0,:]
            self.avg_contact_force_right = self.avg_contact_force[:,1,:]
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                self.avg_contact_force_left * self.obs_scales.forces,  # 3
                                                self.avg_contact_force_right * self.obs_scales.forces,  # 3
            ),dim=1)

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
                                            # self.wheel_height * self.obs_scales.height_measurements,   # 2
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

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

        # trigger相关
        self.start[env_ids] = 1 
        self.episode_length_init[env_ids] = 0
        self.left_contact_history[env_ids] = False
        self.right_contact_history[env_ids] = False
        self.mask[env_ids] = 0


    def post_physics_step(self):
        super().post_physics_step()
        # contact force
        self.contact_force_history.append(self.contact_forces[:, self.wheel_indices, :].clone())
        self.avg_contact_force = torch.mean(torch.stack([self.contact_force_history[i] for i in range(self.cfg.env.contact_force_frame)]),dim=0)

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
        # return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
        return (torch.exp(torch.abs(d_min)) - 1) + (torch.exp(torch.abs(d_max)) - 1)


    def _reward_wheel_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the wheel.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.wheel_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)


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
        # Penalize dof velocities
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
        lower_violation = torch.clamp(self.dof_pos_limits[:, 0] * self.cfg.rewards.soft_dof_pos_limit - self.dof_pos, min=0.0)
        upper_violation = torch.clamp(self.dof_pos - self.dof_pos_limits[:, 1] * self.cfg.rewards.soft_dof_pos_limit, min=0.0)
        lower_violation[:, self.wheel_indices-1] = 0.0
        upper_violation[:, self.wheel_indices-1] = 0.0
        any_violation = torch.any((lower_violation > 0) | (upper_violation > 0), dim=1)
        return torch.where(any_violation, torch.tensor(10.0, device=self.device), torch.tensor(0.0, device=self.device))

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

    def _reward_wheel_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum((self.wheel_air_time - 0.5) * self.first_contacts, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 # no reward for zero command
        return rew_airTime * (1 - self.start)

    def _reward_wheel_clearance(self):
        contact = self.contact_forces[:, self.wheel_indices, 2] > 5.
        wheel_z = self.rigid_state[:, self.wheel_indices, 2] - self.cfg.asset.wheel_radius
        delta_z = wheel_z - self.last_wheel_z
        self.wheel_height += delta_z
        # print("self.wheel_height ", self.wheel_height)
        self.last_wheel_z = wheel_z
        swing_mask = 1 - self._get_gait_phase()
        rew_pos = (self.wheel_height > self.cfg.rewards.target_wheel_height_min) * (self.wheel_height < self.cfg.rewards.target_wheel_height_max)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.wheel_height *= ~contact
        return rew_pos * (1 - self.start)
    
    def _reward_default_hip_roll(self):
        return torch.sum(torch.abs(self.dof_pos[:, [0,4]] - self.default_dof_pos[:, [0,4]]), dim=1)
    
    def _reward_wheel_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.wheel_indices, :2], dim=2) >\
              torch.abs(self.contact_forces[:, self.wheel_indices, 2]), dim=1)

    def _reward_wheel_spin(self):
        # penalize wheel slip
        wheel_indices = [3, 7]
        wheel_des_linear_vel = torch.abs(self.dof_vel[:, wheel_indices] * self.cfg.asset.wheel_radius)
        wheel_lin_vel_err = 0.8 * wheel_des_linear_vel - self.wheel_velocities[: ,:, :3].norm(dim=-1) - 0.1 ##容许些许打滑
        return torch.sum((wheel_lin_vel_err >= 0.0) * wheel_lin_vel_err, dim=1) 

    def _reward_wheel_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.wheel_indices, 2] > 5.
        stance_mask = self._get_gait_phase()

        reward = torch.where(contact == stance_mask, 1, -1.3) 
        return torch.mean(reward, dim=1) * (1 - self.start)

    def _reward_wheel_angular_vel_limit(self):
        wheel_indices = [3, 7]
        wheel_ang_vel = torch.abs(self.dof_vel[:, wheel_indices])
        excess_vel = torch.clamp(wheel_ang_vel - 10, min=0.0)
        penalty = torch.sum(excess_vel ** 2, dim=1)
        return penalty 

    def _reward_wheel_zero_velocity(self):
        wheel_indices = [3, 7] 
        wheel_angular_vel = self.dof_vel[:, wheel_indices]  
        stance_mask = self._get_gait_phase() 
        swing_mask = 1 - stance_mask 
        wheel_vel_squared = torch.sum(torch.square(wheel_angular_vel) * swing_mask, dim=-1)
        wheel_zero_reward = torch.exp(-wheel_vel_squared)
        return wheel_zero_reward

    def _reward_same_wheel_z_position(self):
        reward = 0
        wheel_positions_base = self.wheel_positions - \
                            (self.base_pos).unsqueeze(1).repeat(1, len(self.wheel_indices), 1)
        for i in range(len(self.wheel_indices)):
            wheel_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, wheel_positions_base[:, i, :] )
        wheel_z_position_err = wheel_positions_base[:,0,2] - wheel_positions_base[:,1,2]
        reward = torch.abs(wheel_z_position_err)
        return reward

    def _reward_opposite_wheel_vel(self):
        opposite_vel_l = self.commands[:, 0] * self.dof_vel[:, 3]
        opposite_vel_r = self.commands[:, 0] * self.dof_vel[:, 7]
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