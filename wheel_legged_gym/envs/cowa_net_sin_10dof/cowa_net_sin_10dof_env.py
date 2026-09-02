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

from wheel_legged_gym.envs.cowa_net_sin_10dof.cowa_net_sin_10dof_config import CowaCfg_Net_Sin
from wheel_legged_gym.envs.cowa_10dof.cowa_10dof_env import CowaEnv

import torch
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

class CowaEnv_net_sin(CowaEnv):
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
    def __init__(self, cfg: CowaCfg_Net_Sin, train_cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

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

        noise_vec[:8] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[8:18] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[18:28] = 0 
        noise_vec[28:] = 0
        return noise_vec
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.domain_rand.add_action_lag:
            self.lag_buffer[:,:,1:] = self.lag_buffer[:,:,:self.cfg.domain_rand.lag_timesteps_range[1]].clone()
            self.lag_buffer[:,:,0] = actions.clone()
            if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                self.lag_timestep = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.lag_timestep > self.last_lag_timestep + 1
                self.lag_timestep[cond] = self.last_lag_timestep[cond] + 1
                self.last_lag_timestep = self.lag_timestep.clone()
            self.lagged_actions_scaled = self.lag_buffer[torch.arange(self.num_envs),:,self.lag_timestep.long()]
        else:
            self.lagged_actions_scaled = actions

        pos_ref = self.lagged_actions_scaled * self.cfg.control.pos_action_scale
        vel_ref = self.lagged_actions_scaled * self.cfg.control.vel_action_scale

        pos_ref[:, 4] *= 0
        pos_ref[:, 9] *= 0
        vel_ref[:, :4] *= 0
        vel_ref[:, 5:8] *= 0

        if not self.cfg.mode.use_net:
            pos_ref = self.ref_dof_pos
            vel_ref = self.ref_dof_vel

        # pd controller
        self.joint_pos_target = pos_ref + self.default_dof_pos

        if self.cfg.domain_rand.randomize_PD_factor:
            p_gains = self.Kp_factors * self.p_gains
            d_gains = self.Kd_factors * self.d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        torques = p_gains * (self.joint_pos_target - self.dof_pos + self.motor_offsets) + d_gains * (vel_ref - self.dof_vel)
        
        if self.cfg.domain_rand.randomize_motor_strength:
            torques *= self.motor_strengths

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def  _get_phase(self):
        cycle_time = self.cfg.control.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def compute_ref_state(self):
        """ 参考轨迹v0,但参考轨迹的foot是始终在中线上的,如果权重太大,会导致迈不开腿
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        self.sin_pos=sin_pos
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        scale_1 = -0.4
        scale_2 = 10
        scale_3 = -0.3
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l < 0] = 0

        self.ref_dof_pos[:, 0] += sin_pos_l * scale_3  #hip roll
        self.ref_dof_pos[:, 1] += sin_pos_l * scale_1  #hip pitch
        self.ref_dof_pos[:, 2] += sin_pos_l * scale_1  #knee
        self.ref_dof_pos[:, 3] += sin_pos_l * scale_1  #foot
        self.ref_dof_vel[:, 4] += sin_pos_l * scale_2  #wheel

        self.ref_dof_pos[:, 5] += sin_pos_l * scale_3
        self.ref_dof_pos[:, 6] += sin_pos_l * scale_1
        self.ref_dof_pos[:, 7] += sin_pos_l * scale_1
        self.ref_dof_pos[:, 8] += sin_pos_l * scale_1
        self.ref_dof_vel[:, 9] += sin_pos_l * scale_2
        # right foot stance phase set to default joint pos
        # sin_pos_r[sin_pos_r < 0] = 0
        # self.ref_dof_pos[:, 6] -= sin_pos_r * scale_1
        # self.ref_dof_pos[:, 7] -= sin_pos_r * scale_1
        # self.ref_dof_vel[:, 3] += sin_pos_l * scale_2
        # self.ref_dof_pos[:, 9] += sin_pos_r * scale_2
        # self.ref_dof_pos[:, 3] += sin_pos_r * scale_1

    def compute_observations(self):
        self.compute_ref_state()
        phase = self._get_phase()
        sin_pos_obs = torch.sin(2*torch.pi*phase).unsqueeze(1)
        self.base_height_obs = self.base_height.unsqueeze(1)

        pos_error = (self.dof_pos[:, [0,1,2,3,5,6,7,8]] - self.ref_dof_pos[:, [0,1,2,3,5,6,7,8]]) * self.obs_scales.dof_pos
        vel_error = (self.dof_vel[:, [4,9]] - self.ref_dof_vel[:, [4,9]]) * self.obs_scales.dof_vel
        self.privileged_obs_buf = torch.cat((#4+6+4+2+6+1
            (self.dof_pos[:, [0,1,2,3,5,6,7,8]] - self.default_dof_pos[:, [0,1,2,3,5,6,7,8]]) * self.obs_scales.dof_pos,  # 8 
            self.dof_vel * self.obs_scales.dof_vel,  # 10
            pos_error, # 8
            vel_error, # 2
            self.actions,  # 10
            sin_pos_obs, #1
        ), dim=-1)

        # ---------------- delay 模拟 --------------- #
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

        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel
        
        obs_buf = torch.cat((
            q[:, [0,1,2,3,5,6,7,8]], #8
            dq,  # 10
            self.actions,   # 10
            sin_pos_obs,#1
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

    # --------------------------跟踪sin的奖励函数-------------------
    def _reward_ref_hip_roll_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[0,5]] - pos_target[:,[0,5]])
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r 
    
    def _reward_ref_hip_pitch_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[1,6]] - pos_target[:,[1,6]])
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r 
    
    def _reward_ref_knee_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[2,7]] - pos_target[:,[2,7]])
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r

    def _reward_ref_foot_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[3,8]] - pos_target[:,[3,8]])
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r
    
    def _reward_ref_wheel_vel(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_vel = self.dof_vel.clone()
        vel_target = self.ref_dof_vel.clone()
        diff = (joint_vel[:,[4,9]] - vel_target[:,[4,9]])
        r = torch.exp(-torch.sum(torch.abs(diff), dim=1))
        return r
