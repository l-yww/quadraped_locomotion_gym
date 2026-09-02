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
from .quadwheel_config import QuadwheelCfg, QuadwheelCfgPPO

from isaacgym.torch_utils import *

import torch
from wheel_legged_gym.envs.quadwheel.legged_robot import LeggedRobot
from wheel_legged_gym.utils.math import wrap_to_pi, get_scale_shift

class QuadwheelEnv(LeggedRobot):

    def __init__(self, cfg: QuadwheelCfg, train_cfg: QuadwheelCfgPPO, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        
    def _create_envs(self):
        super()._create_envs()

        # create wheel indices for wheel joint handling
        wheel_names = []
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        print("###wheel_names:", wheel_names)

        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])


    def _init_buffers(self):
        super()._init_buffers()        
        self.noised_q = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.noised_dq = torch.zeros_like(self.dof_vel)
        self.noised_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.noised_euler_rpy = torch.zeros_like(self.base_euler_rpy)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([
            self.obs_scales.lin_vel,
            self.obs_scales.lin_vel,
            self.obs_scales.ang_vel,
        ], device=self.device, requires_grad=False)

    def _resample_commands(self, env_ids):
        """Resample commands (M20-style: lin_vel_x, lin_vel_y, heading)"""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # ------- resample commands ------- #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # ------- 获取高程图 ------- #
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

        # ------- push robot ------- #
        if self.cfg.domain_rand.push_robots:
            env_ids = (self.envs_steps_buf % int(self.cfg.domain_rand.push_interval_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
            self._push_robots(env_ids)

        # ------- randomize motor params 【 电机能力 & 电机属性 】, 以及 base_link 的质量质心 ------- #
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._randomize_dof_props(env_ids)

        # 形体参数随机化
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids)
            self.refresh_actor_rigid_shape_props(env_ids)
            self.refresh_actor_rigid_body_props(env_ids)

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

        num_commands = self.cfg.env.num_commands
        noise_vec[:num_commands] = 0
        noise_vec[num_commands : num_commands + self.num_actions] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[num_commands + self.num_actions : num_commands + 2 * self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[num_commands + 2 * self.num_actions : num_commands + 3 * self.num_actions] = 0
        noise_vec[num_commands + 3 * self.num_actions : num_commands + 3 * self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        if self.cfg.env.projected_gravity:
            noise_vec[num_commands + 3 * self.num_actions + 3 : num_commands + 3 * self.num_actions + 3 + 3] = noise_scales.gravity * self.obs_scales.gravity
        else:
            noise_vec[num_commands + 3 * self.num_actions + 3 : num_commands + 3 * self.num_actions + 3 + 2] = noise_scales.quat * self.obs_scales.quat

        if self.cfg.env.observe_heights:
            height_start = num_commands + 3 * self.num_actions
            if self.cfg.env.projected_gravity:
                height_start += 3 + 3
            else:
                height_start += 3 + 2
            height_dim = 13 * 7
            noise_vec[height_start : height_start + height_dim] = noise_scales.height_measurements * self.obs_scales.height_measurements

        return noise_vec

    def compute_observations(self):
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
        )
        self.privileged_obs_buf = torch.cat(( 
            self.commands[:, :3] * self.commands_scale,
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity, # 重力投影
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
            # 6
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_offset:
            # 6
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1) 
        if self.cfg.env.priv_observe_com_displacement:
            # 3
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.domain_rand.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.com_displacements[:, :3] - com_displacements_shift) * com_displacements_scale), dim=1)
        if self.cfg.env.priv_observe_heightmap:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 heights * self.obs_scales.height_measurements), dim=1)        
        # 始终将需要预测的值量放在最后（若有EST网络）
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                             self.base_height_obs *self.obs_scales.height_measurements,
                                            self.base_lin_vel * self.obs_scales.lin_vel,
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
            self.lagged_projected_gravity = self.projected_gravity
            self.lagged_base_euler_rpy = self.base_euler_rpy

        lagged_q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        lagged_dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            lagged_q,
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
                self.lagged_base_euler_rpy[:,:2] * self.obs_scales.quat,
            ),dim=-1)

        if self.cfg.env.observe_heights:
            obs_buf = torch.cat((
                obs_buf,
                heights * self.obs_scales.height_measurements,
            ), dim=-1)
        obs_now = obs_buf.clone()

        if self.add_noise:
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise
    
        self.noised_q = obs_now[:, self.cfg.env.num_commands:self.cfg.env.num_commands+self.cfg.env.num_actions] / self.obs_scales.dof_pos
        self.noised_dq = obs_now[:, self.cfg.env.num_commands+self.cfg.env.num_actions : self.cfg.env.num_commands+2*self.cfg.env.num_actions] / self.obs_scales.dof_vel
        self.noised_ang_vel = obs_now[: self.cfg.env.num_commands+3*self.cfg.env.num_actions:self.cfg.env.num_commands+3*self.cfg.env.num_actions+3] / self.obs_scales.ang_vel
        self.noised_euler_rpy = obs_now[:, self.cfg.env.num_commands+3*self.cfg.env.num_actions+3:self.cfg.env.num_commands+3*self.cfg.env.num_actions+5] / self.obs_scales.quat

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

        # print(self.obs_buf)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # log terrain_level per terrain type name
        if hasattr(self.cfg.terrain, 'terrain_type_names') and self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.extras["episode"]['terrain_level_all'] = torch.mean(self.terrain_levels.float())
            type_names = self.cfg.terrain.terrain_type_names
            name2cols = {}
            for col_idx, name in enumerate(type_names):
                if name not in name2cols:
                    name2cols[name] = []
                name2cols[name].append(col_idx)
            for name, cols in name2cols.items():
                cols_tensor = torch.tensor(cols, device=self.device)
                mask = torch.isin(self.terrain_types, cols_tensor)
                self.extras["episode"]['terrain_level_' + name] = torch.mean(self.terrain_levels[mask].float())
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ==================================================================================================================== #
# ================================================ Custom Rewards Function ================================================== #
# ==================================================================================================================== #
    

    # ------------------------------------------------------------------------------#
    # ------------------------- termination rewards --------------------------------#
    # ------------------------------------------------------------------------------# 
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    

    # ------------------------------------------------------------------------------# 
    # --------------------------- tracking rewards ---------------------------------# 
    # ------------------------------------------------------------------------------# 
    
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return -2.5 * lin_vel_x_error / (lin_vel_x_error + 0.2) ** 0.5
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        lin_vel_y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

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
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
        
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    

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
        out_of_limits[:,self.wheel_indices]=0.
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities, exclude wheel joints to encourage wheel rolling
        vel = self.dof_vel.clone()
        vel[:,self.wheel_indices] = 0
        return torch.sum(torch.square(vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel_50hz - self.dof_vel) / self.dt), dim=1)
    
    def _reward_hip_default(self):
        hip_err = torch.sum((self.dof_pos[:, [0, 4, 8, 12]] - self.default_dof_pos[:, [0, 4, 8, 12]]) ** 2, dim = 1)
        return hip_err
    
    def _reward_run_still(self):
        # Penalize motion at running commands        
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands        
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_wheel_spin(self):
        # penalize wheel slip
        wheel_des_linear_vel = torch.abs(self.dof_vel[:, self.wheel_indices] * self.cfg.asset.wheel_radius)
        wheel_lin_vel_err = 0.8 * wheel_des_linear_vel - self.wheel_velocities[: ,:, :3].norm(dim=-1) - 0.1 ##容许些许打滑
        return torch.sum((wheel_lin_vel_err >= 0.0) * wheel_lin_vel_err, dim=1) 

    def _reward_joint_power(self):
        # penalize torque * vel
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1) 
    
    def _reward_joint_reference(self):
        # Penalize motion at zero commands        
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0
        return torch.sum(torch.square(dof_err), dim=1)