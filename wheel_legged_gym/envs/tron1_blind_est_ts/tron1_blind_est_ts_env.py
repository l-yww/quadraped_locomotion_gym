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


from .legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import random
from .legged_robot import LeggedRobot
from .tron1_blind_est_ts_config import Tron1Cfg

from wheel_legged_gym.utils.terrain import  Terrain
# from collections import deque
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from IPython import embed; eee=embed

class Tron1FreeEnv(LeggedRobot):
    '''
    CowaFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: Tron1Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05   # 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()

    def _init_buffers(self):
        super()._init_buffers()
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.last_feet_z = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, dtype=torch.float)
        self.feet_height = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, dtype=torch.float)
        self.episode_length_init = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.leading_leg = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.mask_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.mask_right = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.start = torch.ones_like(self.mask_right)
        self.force_history_len = self.cfg.feedforward.trigger_len  # 连续帧数
        self.left_contact_history = torch.zeros(self.num_envs, self.force_history_len, dtype=torch.bool, device=self.device)
        self.right_contact_history = torch.zeros(self.num_envs, self.force_history_len, dtype=torch.bool, device=self.device)
        self.mask = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # return float mask 1 is stance, 0 is swing
    def _get_gait_phase(self):
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)

        # 左腿逻辑: 0.0–1.5s stance, 1.5–2.0s swing
        stance_mask[:, 0] = self.mask_left

        # 右腿逻辑: 0.0–1.1s stance, 1.1–1.6s swing, 1.6–2.0s stance
        # right_stance = (phase < 1.3) | (phase >= 1.8)
        stance_mask[:, 1] = self.mask_right
        return stance_mask
    
    def compute_ref_state(self):
        swing_duration = self.cfg.feedforward.swing_duration  # e.g. 0.6
        phase_offset = self.cfg.feedforward.phase_offset      # e.g. 0.6

        # 更新当前 phase
        phase = (self.episode_length_buf - self.episode_length_init) * self.dt
        phase = phase * (1 - self.start)  # 如果 start=1，phase=0，跳过 phase 控制

        # reset 标志位
        reset_index = torch.where(phase >= (phase_offset + swing_duration - self.dt))[0]
        self.start[reset_index] = 1
        self.mask[reset_index] = 0

        # 初始化 ref 动作
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()

        scale_1 = 1.5
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

        # phase normalized
        phase_left = (phase - left_start) / swing_duration
        mask_left = (phase < left_start) | (phase >= left_end)
        phase_left[mask_left] = 0.0
        phase_cos_l = torch.sin(torch.pi * phase_left) # phase_cos_l = (-tocrh.cos(2 * torch.pi * phase_left) + 1) / 2
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
        phase_cos_r = torch.sin(torch.pi * phase_right) # # phase_cos_r = (-tocrh.cos(2 * torch.pi * phase_right) + 1) / 2
        phase_cos_r[mask_right] = 0.0

        right_valid = (self.mask == 2) | (self.mask == 3)
        self.ref_dof_pos[right_valid, 5] -= phase_cos_r[right_valid] * scale_1
        self.ref_dof_pos[right_valid, 6] -= phase_cos_r[right_valid] * scale_2


        self.mask_left = mask_left.float()  # 1 = stance, 0 = swing
        self.mask_right = mask_right.float()
    # TODO
    def _get_noise_scale_vec(self, cfg):
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

        noise_vec[:6] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[6:14] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[14:22] = 0  # actions  
        noise_vec[22:25] = noise_scales.ang_vel  * self.obs_scales.ang_vel
        noise_vec[25:28] = noise_scales.gravity  * self.obs_scales.quat
        noise_vec[28:] = 0
        if not self.cfg.env.dagger_on and self.cfg.env.teacher:
            noise_vec[35:38] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[38:39] = noise_scales.height_measurements
            noise_vec[39:41] = noise_scales.probability

        return noise_vec


    def step(self, actions, iter):
        self.trigger_state_update()
        self.compute_ref_state()
        ## add forward feedbacks
        if self.cfg.feedforward.use_feedforward:
            kb_values =[0.8, 1.2, 1.2, 1.2, 0.8, 1.2, 1.2, 1.2]
            kb = torch.tensor(kb_values[:self.num_dof], device=self.device, dtype=torch.float32)
            final_actions = kb * actions + self.alpha * self.ref_dof_pos
            if self.cfg.feedforward.use_annealing:
                if iter > self.cfg.feedforward.start_iter:
                    self.alpha = max(0.0, 1.0 - (iter - self.cfg.feedforward.start_iter) / self.cfg.feedforward.duration)
        else:
            final_actions = actions
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * final_actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions

        self._action_clip(actions)
        return super().step(self.actions, iter)
    
    def _action_clip(self, actions):
        self.actions = actions

    def trigger_state_update(self):
        # 当前帧满足条件的 mask
        # 当前帧是否接触（单帧判定）
        left_contact_now = torch.abs(self.avg_contact_force[:, 0, 0]) > 30 #20.
        right_contact_now = torch.abs(self.avg_contact_force[:, 1, 0]) > 30 #20.

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

        #* 判断三种情况(old)
        # only_left = left_contact_stable & ~right_contact_stable
        # only_right = ~left_contact_stable & right_contact_stable
        # both_stable = left_contact_stable & right_contact_stable

        #! 判断三种情况(new)
        only_left = left_contact_stable & ~right_any
        only_right = ~left_any & right_contact_stable
        both_stable = (left_contact_stable & right_any) | (right_contact_stable & left_any) | (left_contact_stable & right_contact_stable)

        # 初始化 mask
        mask = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        mask[only_left] = 1
        mask[only_right] = 2
        mask[both_stable] = 3

        # 判断哪些环境满足初始化条件
        init_mask = (mask > 0) & (self.start == 1)
        init_indices = torch.where(init_mask)[0]
        self.mask[init_indices] = mask[init_indices]
        # 左腿先抬
        left_update_mask = (mask == 1) & (self.start ==1)
        self.leading_leg[left_update_mask] = 0

        # 右腿先抬
        right_update_mask = (mask == 2) & (self.start ==1)
        self.leading_leg[right_update_mask] = 1

        #* 双腿都接触，比较力大小决定谁先抬(old)
        # both_indices = torch.where(self.mask == 3)[0]
        # if both_indices.numel() > 0:
        #     left_forces = torch.abs(self.avg_contact_force[both_indices, 0, 0])
        #     right_forces = torch.abs(self.avg_contact_force[both_indices, 1, 0])
        #     leading_leg = torch.where(
        #         right_forces > left_forces,
        #         torch.ones_like(right_forces, dtype=torch.long, device=self.device),  # 1 = right
        #         torch.zeros_like(left_forces, dtype=torch.long, device=self.device)   # 0 = left
        #     )
        #     self.leading_leg[both_indices] = leading_leg


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
        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (  #torch.Size([4096, 121])
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.33 - self.measured_heights,  #4096*121
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )

        # zsy added
        self.avg_contact_force_left = self.avg_contact_force[:,0,:]
        self.avg_contact_force_right = self.avg_contact_force[:,1,:]

        dof_list = [0,1,2, 4,5,6]
        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,  # 3
            (self.dof_pos[:, dof_list] - self.default_dof_pos[:, dof_list]) * self.obs_scales.dof_pos,  # 4 
            self.dof_vel * self.obs_scales.dof_vel,  # 8
            self.actions,  # 8
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity * self.obs_scales.quat,  # 3
            # self.avg_contact_force_left * self.obs_scales.forces,
            # self.avg_contact_force_right * self.obs_scales.forces,
        ), dim=-1)

        ## NOTE： EST only estimates the below params ---zsy
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                            self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                            # self.base_height_obs, # 1
                                            # contact_mask, # 2
                                            ), dim=1)

        # agibot
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
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
            self.lagged_projected_gravity = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz
            self.lagged_projected_gravity = self.projected_gravity
        
        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel
        if not self.cfg.env.dagger_on and self.cfg.env.teacher:
            obs_buf = self.privileged_obs_buf.clone()
        else:
            obs_buf = torch.cat((
                q[:, dof_list], # 6
                dq,  # 8
                self.actions,   # 8
                self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.lagged_projected_gravity * self.obs_scales.quat,  # 3
                self.commands[:, :3]  * self.commands_scale,   # 3 
            ), dim=-1)
        obs_now = obs_buf.clone()

        if self.add_noise:  
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

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

# ================================================ Rewards Humanoid Gym ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        diff[:,hip_roll_yaw_pitch_indices] *= 3
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

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
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_pos = self.rigid_state[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        self.knee_dist = knee_dist.unsqueeze(1)
        fd = self.cfg.rewards.min_knee_dist
        max_df = self.cfg.rewards.max_knee_dist # / 1.5
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2 

    def _reward_feet_xy_contact_forces(self):
        xy_contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=-1) 
        return torch.sum((xy_contact_forces - 50).clip(0, 40), dim=1)/20


    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.04, 0, 50)
        return torch.exp(-yaw_roll * 100)


    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_base_lin_acc(self):
        # Penalize base linear accelerations
        base_lin_acc = torch.sum(torch.square((self.last_base_lin_vel - self.base_lin_vel[:, 0:1]) / self.dt), dim=1)
        return base_lin_acc

    def _reward_base_lin_acc_limit(self):
        # Penalize base linear accelerations which beyond the limit
        base_lin_acc = torch.abs((self.last_base_lin_vel - self.base_lin_vel[:, 0:1]) / self.sim_params.dt)
        base_lin_acc_limit = torch.clamp(base_lin_acc - self.cfg.rewards.base_lin_acc_limit, 0, 0.5)
        return torch.sum(torch.square(base_lin_acc_limit), dim=1)

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
    
    def _reward_feet_height_smoothness(self):
        """
        
        """
        r = torch.sum(torch.square(self.feet_height - self.last_feet_z), dim=1)
        return r

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
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)        
    
    def _reward_action_smooth(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions + self.last_last_actions), dim=1)

    def _reward_stand_still_vel_penality(self):
        # Penalize motion at zero commands
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_base_pos_penality(self):

        # Penalize motion at zero commands
        diff_x = 5 * torch.square(self.base_pos[:, 0] - self.base_pos_init[:, 0])
        diff_y_z = torch.sum(torch.square(self.base_pos[:, 1:3] - self.base_pos_init[:, 1:3]), dim=1)
        return (diff_x + diff_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_base_pos(self):

        # Penalize motion at zero commands
        diff = torch.norm(self.base_pos[:, :2] - self.base_pos_init[:, :2], dim=1)
        # return () * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return torch.exp(-100 * diff) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_wheel_vel_penality(self):

        # Penalize motion at zero commands 

        # if self.reward_scales["stand_still_wheel_vel_penality"]  < 0:   
        return torch.sum((torch.square(self.dof_vel[:,[2,5]], dim=1)), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)    
    
    def _reward_stand_still_wheel_vel(self):
        wheel_vel_error = torch.sum(torch.square(self.dof_vel[:,[2,5]]), dim=1)
        return torch.exp(-400 * wheel_vel_error)
       
             
    # ------------------------- Rewards Unitree Gym --------------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_lin_vel_yz(self):
        # Penalize yz axis base linear velocity
        return torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # print(self.commands[0, 2], self.base_height[0])
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        if self.reward_scales["base_height"] < 0:
            return torch.abs(base_height - self.commands[:, 2])
        else:
            base_height_error = torch.square(base_height - self.commands[:, 2])
            return torch.exp(-200 * base_height_error)   
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1, dim=1)     
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, :2] - self.dof_pos_limits[:2, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, :2] - self.dof_pos_limits[:2, 1]).clip(
            min=0.0
        )
        out_of_limits += -(self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, 3:5] - self.dof_pos_limits[3:5, 1]).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        vel_max = 0.4372 * self.dof_pos[:,[1,4]]**3 - 1.7520 * self.dof_pos[:,[1,4]]**2 - 1.9508 * self.dof_pos[:,[1,4]] + 4.3146
        return torch.sum((torch.abs(self.dof_vel[:,[1,4]]) - vel_max).clip(min=0.),dim = 1)
        # return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*0.8).clip(min=0.), dim=1) ## 0.8zhangchenyang�p

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        ### y = -111.01802348599728 x^2 + -78.63874503900921x + 224.35358459222218 ------pos_torque_max
        # tau_max = 10.0186 * self.dof_pos[:,[1,4]]**3 - 40.1493 * self.dof_pos[:,[1,4]]**2 - 44.7066 * self.dof_pos[:,[1,4]] + 98.8757 # NOTE: old ones
        tau_max = -111.01802348599728 * self.dof_pos[:,[1,4]]**2 - 78.63874503900921 * self.dof_pos[:,[1,4]] + 224.35358459222218 # NOTE: new ones
        return torch.sum((torch.abs(self.torques[:, [1,4]]) - tau_max).clip(min=0.), dim=1)
        # return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error = torch.sum(torch.abs(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_x_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_x_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_y_error = torch.abs(0 - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_lin_vel_x_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel_x() - self.rwd_linVelXTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_tracking_lin_vel_y_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel_y() - self.rwd_linVelYTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (
            self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev
        )
        # return ang_vel_error
        return delta_phi

    def _reward_stability(self):
        velocity_error = torch.sum((self.base_lin_vel[:, :3])**2, dim=1)
        energy_cost = torch.sum(torch.abs(self.torques), dim=1) * 0.01  # 
        stability_penalty = torch.sum(self.base_ang_vel[:, :3]**2, dim=1) * 0.2  # 

        reward = (1.0 / (1.0 + velocity_error)) - energy_cost - stability_penalty

        return reward
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # -------------------------------------------------- wl
    def _reward_nominal_state(self):
        # return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        if self.reward_scales["nominal_state"] < 0:
            return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        else:
            ang_diff = torch.square(self.theta0[:, 0] - self.theta0[:, 1])
            return torch.exp(-ang_diff / 0.1)

    def _reward_power(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    # zsy add
    def _reward_wheel_power(self):
        # Penalize wheel torques
        return torch.sum(torch.abs(self.torques[:,[2,5]] * self.dof_vel[:,[2,5]]), dim=1)
    
    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_pos).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        reward = 0
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        vel_cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    

    def _reward_same_foot_z_position(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :3]
        foot_z_dist = foot_pos[:, 0, 2] - foot_pos[:, 1, 2]
        foot_dist = foot_z_dist
        fd = self.cfg.rewards.min_feet_z_dist
        max_fd = self.cfg.rewards.max_feet_z_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_fd, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            self.base_pos.unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

    def _reward_same_foot_y_position(self):

        foot_positions_base = self.foot_positions - \
                            self.base_pos.unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :])
        foot_y_position_err = torch.abs(
            foot_positions_base[:, 0, 1] - foot_positions_base[:, 1, 1]
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - foot_y_position_err, 0, 1) + \
                torch.clip(foot_y_position_err - self.cfg.rewards.max_feet_distance, 0, 1)
        
        return reward

    def _reward_wheel_adjustment(self):
        
        incline_x = self.projected_gravity[:, 0]
        # mean velocity
        wheel_x_mean = (self.foot_velocities[:, 0, 0] + self.foot_velocities[:, 1, 0]) / 2
        
        wheel_x_invalid = (self.foot_velocities[:, 0, 0] * self.foot_velocities[:, 1, 0]) < 0
        wheel_x_mean[wheel_x_invalid] = 0.0
        wheel_x_mean = wheel_x_mean.reshape(-1)
        reward = incline_x * wheel_x_mean > 0
        return reward
    
    def _reward_wheel_vel(self):
        # Penalize dof velocities
        # left_wheel_vel = self.commands[:,0]/2 - self.commands[:,1]
        # right_wheel_vel = self.commands[:,0]/2 + self.commands[:,1]
        # return torch.sum(torch.square(self.dof_vel[:, 2] - left_wheel_vel) + torch.square(self.dof_vel[:, 5]) - right_wheel_vel)
        return torch.sum(torch.square(self.dof_vel[:, [3, 7]]), dim=1)
    
    def _reward_wheel_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel[:, [3,7]] - self.dof_vel[:, [3,7]]) / self.dt), dim=1)
    
    def _reward_base_lin_acc(self):
        # Penalize base linear accelerations
        base_lin_acc = torch.sum(torch.square((self.last_base_lin_vel - self.base_lin_vel[:, 0:1]) / self.dt), dim=1)
        return base_lin_acc

    def _reward_base_lin_acc_limit(self):
        # Penalize base linear accelerations which beyond the limit
        base_lin_acc = torch.abs((self.last_base_lin_vel - self.base_lin_vel[:, 0:1]) / self.sim_params.dt)
        base_lin_acc_limit = torch.clamp(base_lin_acc - self.cfg.rewards.base_lin_acc_limit, 0, 0.5)
        return torch.sum(torch.square(base_lin_acc_limit), dim=1)

    def _reward_nominal_state(self):
        # return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        if self.reward_scales["nominal_state"] < 0:
            return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        else:
            ang_diff = torch.square(self.theta0[:, 0] - self.theta0[:, 1])
            return torch.exp(-ang_diff / 0.1)


    def _reward_dof_vel(self):
        # Penalize dof velocity
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_ref_hip_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[0,3]] - pos_target[:,[0,3]]) * 1
        # hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        # diff[:,hip_roll_yaw_pitch_indices] *= 3
        # r = torch.exp(-50 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.8)
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r
    
    def _reward_ref_knee_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[1,4]] - pos_target[:,[1,4]]) * 1
        # hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        # diff[:,hip_roll_yaw_pitch_indices] *= 3
        # r = torch.exp(-50 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.8)
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r
    
    def _reward_ref_wheel_vel(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_vel = self.dof_vel.clone()
        vel_target = self.ref_dof_vel.clone()
        diff = (joint_vel[:,[2,5]] - vel_target[:,[2,5]]) * 0.5
        # hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        # diff[:,hip_roll_yaw_pitch_indices] *= 3
        # r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r = torch.exp(-0.5 * torch.sum(torch.abs(diff), dim=1))
        return r

    def _reward_default_joint_pos(self):
        pos_offset = torch.tensor([0.1152,0.0117,0,0.1152,0.0117,0], dtype=torch.float, device=self.device,
                                         requires_grad=False) 
        joint_diff = self.dof_pos - self.default_motor_offset - self.default_dof_pos - pos_offset
        rew = torch.norm(joint_diff[:,[0,1,3,4]], dim=1)
        # print(f"joint_diff: =================")
        # print(joint_diff[:,[0,1,3,4]])
        return torch.exp(-20 * rew)
    
    def _reward_feet_air_time(self):
        """
        鼓励迈大步
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        # import ipdb;ipdb.set_trace()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1) * (1 - self.start)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        # reward = torch.where(contact == stance_mask, 1, -0.3) # orgin
        # reward = torch.where(contact == stance_mask, 1, -0.7) # wh debug1
        # 等于的地方是1,不等于的地方是-1.3
        reward = torch.where(contact == stance_mask, 1, -1.3) # wh debug2   (惩罚不能大到2)
        return torch.mean(reward, dim=1) * (1 - self.start)

    def _reward_feet_contact_number_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_feet_contact_number() - self.rwd_FeetContactNumPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_feet_clearance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.asset.foot_radius
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        swing_mask = 1 - self._get_gait_phase()
        rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height) * (self.feet_height < self.cfg.rewards.target_feet_height_max)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos * (1 - self.start)

    def _reward_feet_clearance_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_feet_clearance() - self.rwd_FeetClearancePrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_feet_height_limit(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.asset.foot_radius
        swing_mask = 1 - self._get_gait_phase()

        penalty_mask = swing_mask * (~contact).float()

        # 超过高度限制,小幅度惩罚
        height_excess = torch.clamp(feet_z - 0.15, min=0.0)
        penalty_high = torch.sum(torch.abs(height_excess) * penalty_mask, dim=1)

        # 没抬脚,大幅度惩罚
        too_low = (feet_z < 0.001).float()
        penalty_low = torch.sum(too_low * penalty_mask, dim=1)
        height_penalty = penalty_high + penalty_low
        
        return height_penalty

    ## 模仿光省加的, 避免轮子打滑
    def _reward_wheel_spin(self):
        # penalize wheel slip
        wheel_indices = [3, 7]
        wheel_radius = 0.127
        wheel_des_linear_vel = torch.abs(self.dof_vel[:, wheel_indices] * wheel_radius)
        wheel_lin_vel_err = 0.8 * wheel_des_linear_vel - self.feet_vel[: ,:, :3].norm(dim=-1) - 0.1 ##容许些许打滑
        return torch.sum((wheel_lin_vel_err >= 0.0) * wheel_lin_vel_err, dim=1) 

    def _reward_hip_roll_default_pose(self):
        """
        奖励hip roll关节接近默认姿态，值越小奖励越大
        Reward hip roll joints staying close to default posture
        """
        hip_roll_indices = [0, 4]  
        
        hip_roll_current = self.dof_pos[:, hip_roll_indices]
        hip_roll_default = self.default_dof_pos[:, hip_roll_indices]
        
        # 计算与默认姿态的偏差
        hip_roll_error = torch.sum(torch.abs(hip_roll_current - hip_roll_default), dim=1)
        
        # hip_roll_reward = torch.exp(-hip_roll_error)
        # return hip_roll_reward

        return hip_roll_error
    
    def _reward_feet_z_contact_forces(self):
        """
        限制z方向接触力不超过200, 超出时给予惩罚
        """
        # 难
        # avg_z_contact_forces = torch.norm(self.avg_contact_force[:, :, 2], dim=1)
        # 0722 简单
        avg_z_contact_forces = torch.mean(self.avg_contact_force[:, :, 2], dim=1)
        return (avg_z_contact_forces - self.cfg.rewards.max_contact_force).clip(0, 50) / 50  

    # Tracking the sine curves
    def _reward_tracking_target_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        hip_knee_list = [1,2, 5,6]
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone() * 0.2
        diff = joint_pos - pos_target
        # r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        # r = torch.exp(-2 * torch.norm(diff, dim=1) ** 2) 
        if self.cfg.rewards.scales.tracking_target_joint_pos < 0:
            r = torch.sum(torch.square(diff[:, hip_knee_list]), dim=1)
        else:
            r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
    
    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        rew = torch.sum(torch.square(self.last_root_vel - self.root_states[:, 7:13]), dim=1)
        return rew
    
    # def _reward_stand_still_vel_penality(self):
    #     # Penalize motion at zero commands
    #     term_x = 5 * torch.square(self.base_lin_vel[:, 0])
    #     term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
    #     return (term_x + term_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_wheel_vel_penality(self):
        # Penalize motion at zero commands 
        return torch.sum((torch.square(self.dof_vel[:,[3,7]])), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)    
    
    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

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