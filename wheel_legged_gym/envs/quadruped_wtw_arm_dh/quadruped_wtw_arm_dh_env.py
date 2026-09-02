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


from wheel_legged_gym.envs.quadruped_wtw_arm.quadruped_wtw_arm_config import QuadWtwCfg, QuadWtwCfgPPO

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import random
from wheel_legged_gym.envs.quadruped_wtw_arm.legged_robot import LeggedRobot, get_euler_rpy_tensor

from wheel_legged_gym.utils.terrain import  Terrain
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift

class QuadWtwEnv(LeggedRobot):

    def __init__(self, cfg: QuadWtwCfg, train_cfg: QuadWtwCfgPPO, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        
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
            self.obs_scales.body_height_cmd,
            self.obs_scales.gait_freq_cmd,
            self.obs_scales.gait_phase_cmd,
            self.obs_scales.gait_phase_cmd,
            self.obs_scales.gait_phase_cmd,
            self.obs_scales.gait_duration_cmd,
            self.obs_scales.footswing_height_cmd,
            self.obs_scales.body_pitch_cmd,
            self.obs_scales.body_roll_cmd
        ], device=self.device, requires_grad=False)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices_tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_foot_z = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_height = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.common_step_counter = 0

    def _resample_commands(self, env_ids):
        """Resample all 12 commands"""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_ranges["body_height_cmd"][0], self.command_ranges["body_height_cmd"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 4] = torch_rand_float(
            self.command_ranges["gait_frequency_cmd_range"][0], self.command_ranges["gait_frequency_cmd_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 5] = torch_rand_float(
            self.command_ranges["gait_phase_cmd_range"][0], self.command_ranges["gait_phase_cmd_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 6] = torch_rand_float(
            self.command_ranges["gait_offset_cmd_range"][0], self.command_ranges["gait_offset_cmd_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 7] = torch_rand_float(
            self.command_ranges["gait_bound_cmd_range"][0], self.command_ranges["gait_bound_cmd_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 8] = torch_rand_float(
            self.command_ranges["gait_duration_cmd_range"][0], self.command_ranges["gait_duration_cmd_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 9] = torch_rand_float(
            self.command_ranges["footswing_height_range"][0], self.command_ranges["footswing_height_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 10] = torch_rand_float(
            self.command_ranges["body_pitch_range"][0], self.command_ranges["body_pitch_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 11] = torch_rand_float(
            self.command_ranges["body_roll_range"][0], self.command_ranges["body_roll_range"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        # 小命令清零
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        # 10% 概率生成站立命令（零速度），增加静止训练样本
        # stand_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        # self.commands[env_ids[stand_mask], 0:3] = 0.0

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # ------- resample commands ------- #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 1] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

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


    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]

            # Detect zero-velocity commands: all feet on ground (stand still)
            zero_cmd_mask = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & \
                            (torch.abs(self.commands[:, 2]) < 0.1)

            gait_increment = self.dt * frequencies * (~zero_cmd_mask).float()
            self.gait_indices = torch.remainder(self.gait_indices + gait_increment, 1.0)
            
            # foot_indices = [
            #     self.gait_indices + phases + offsets + bounds,
            #     self.gait_indices + offsets,
            #     self.gait_indices + bounds,
            #     self.gait_indices + phases
            # ]

            foot_indices = [
                self.gait_indices + phases + offsets + bounds,    # FL
                self.gait_indices + bounds,                       # FR
                self.gait_indices + offsets,                      # RL
                self.gait_indices + phases                        # RR
            ]
            
            self.foot_indices_tensor = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (0.5 / (1 - durations[swing_idxs]))

            # Zero clock inputs for stand-still envs to avoid oscillating signals
            clock_vals = torch.stack([torch.sin(2 * np.pi * foot_indices[i]) for i in range(4)], dim=1)
            self.clock_inputs = clock_vals * (~zero_cmd_mask).unsqueeze(1).float()

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
            for i in range(4):
                foot_phase = torch.remainder(foot_indices[i], 1.0)
                desired_contact = (
                    smoothing_cdf_start(foot_phase) *
                    (1 - smoothing_cdf_start(foot_phase - 0.5)) +
                    smoothing_cdf_start(foot_phase - 1) *
                    (1 - smoothing_cdf_start(foot_phase - 0.5 - 1))
                )
                # When velocity commands are near zero, force all feet to stance
                self.desired_contact_states[:, i] = torch.where(
                    zero_cmd_mask, torch.ones_like(desired_contact), desired_contact
                )

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

        num_commands = self.cfg.commands.num_commands
        noise_vec[:num_commands] = 0
        noise_vec[num_commands : num_commands + self.num_dof] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[num_commands + self.num_dof : num_commands + 2 * self.num_dof] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[num_commands + 2 * self.num_dof : num_commands + 2 * self.num_dof + self.num_actions] = 0
        noise_vec[num_commands + 2 * self.num_dof + self.num_actions : num_commands + 2 * self.num_dof + self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        if self.cfg.env.projected_gravity:
            noise_vec[num_commands + 2 * self.num_dof + self.num_actions + 3 : num_commands + 2 * self.num_dof + self.num_actions + 3 + 3] = noise_scales.gravity * self.obs_scales.gravity
        else:
            noise_vec[num_commands + 2 * self.num_dof + self.num_actions + 3 : num_commands + 2 * self.num_dof + self.num_actions + 3 + 2] = noise_scales.quat * self.obs_scales.quat
        return noise_vec

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        self._resample_commands(env_ids)
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)
            
        self._randomize_dof_props(env_ids)
        self.randomize_lag_props(env_ids)
        self._refresh_actor_dof_props(env_ids)  # refresh joint damping/friction/aramture
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids)
            self.refresh_actor_rigid_shape_props(env_ids)
            self.refresh_actor_rigid_body_props(env_ids)
        
        # reset buffers
        self.dof_vel[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel_50hz[env_ids] = 0.
        self.last_dof_vel_200hz[env_ids] = 0.
        self.dof_acc_50hz[env_ids] = 0.
        self.dof_acc_200hz[env_ids] = 0.

        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if 'P3O' in self.train_cfg.runner_class_name:
            for key in self.cost_episode_sums.keys():
                self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
                self.cost_episode_sums[key][env_ids] = 0. 
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # fix reset gravity bug
        self.base_pos_init[env_ids] = self.root_states[env_ids, 0:3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_rpy = get_euler_rpy_tensor(self.feet_quat)
        self.gait_indices[env_ids]=0
        self.last_foot_z[env_ids] = 0.
        self.foot_height[env_ids] = 0.

        if self.cfg.env.action_curriculum:
            self.random_upper_actions[env_ids] = 0.
            self.current_upper_actions[env_ids] = 0.
            self.delta_upper_actions[env_ids] = 0.


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.global_counter += 1

        # --- Upper body curriculum: generate random position targets for arm joints ---
        if self.cfg.env.action_curriculum:
            if (self.common_step_counter % self.cfg.domain_rand.upper_interval == 0):
                self.random_upper_ratio = min(self.action_curriculum_ratio, 1.0)
                uu = torch.rand(self.num_envs, self.upper_num_dof, device=self.device)
                self.random_upper_ratio = -1.0 / (20 * (1 - self.random_upper_ratio * 0.99)) * torch.log(1 - uu + uu * np.exp(-20 * (1 - self.random_upper_ratio * 0.99)))
                self.random_joint_ratio = self.random_upper_ratio * torch.rand(self.num_envs, self.upper_num_dof, device=self.device)
                rand_pos = torch.rand(self.num_envs, self.upper_num_dof, device=self.device) - 0.5
                self.random_upper_actions = ((self.action_min[:, self.num_lower_dof:] * (rand_pos >= 0)) + (self.action_max[:, self.num_lower_dof:] * (rand_pos < 0))) * self.random_joint_ratio
                self.delta_upper_actions = (self.random_upper_actions - self.current_upper_actions) / (self.cfg.domain_rand.upper_interval)
            self.current_upper_actions += self.delta_upper_actions

            actions = torch.cat((actions, self.current_upper_actions), dim=-1)

        actions = torch.clamp(actions, min = -self.cfg.normalization.clip_actions, max = self.cfg.normalization.clip_actions)

        self.actions = actions
        if self.cfg.control.action_smoothness:
            ratio = self.cfg.control.ratio
            self.actions = ratio * self.actions + (1 - ratio) * self.last_actions

        self.render()
        
        self.pre_physics_step()
           
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # ---------------- 随机 电机编码器 延迟 --------------- #
            if self.cfg.domain_rand.add_dof_lag:
                q = self.dof_pos
                self.dof_lag_buffer[:,:,1:] = self.dof_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_lag_buffer[:,:,0] = q.clone()
                dq = self.dof_vel
                self.dof_vel_lag_buffer[:,:,1:] = self.dof_vel_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_vel_lag_buffer[:,:,0] = dq.clone()
            # ---------------- 随机 IMU 延迟 --------------- #
            if self.cfg.domain_rand.add_imu_lag:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
                self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                self.imu_lag_buffer[:,:,1:] = self.imu_lag_buffer[:,:,:self.cfg.domain_rand.imu_lag_timesteps_range[1]].clone()
                if self.cfg.env.projected_gravity == True:
                    self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.projected_gravity ), 1).clone()
                else:
                    self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.base_euler_rpy ), 1).clone()   
            # ---------- 下发 Torque, 仿真器步进一步 ----------- #
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.control.control_type == "M":
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        if ('HIM' or 'VAE') not in self.train_cfg.runner_class_name:
            self.post_physics_step()
        else:
            # 获取termination_privileged_obs(终止环境被刷新前的特权观测)
            termination_ids, termination_priveleged_obs = self.post_physics_step()
            
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
            # interpolation = torch.rand((self.cfg.depth.camera_num_envs, 1, 1), device=self.device)
            # self.extras["depth"] = self.depth_buffer[:, -1] * interpolation + self.depth_buffer[:, -2] * (1-interpolation)
        else:
            self.extras["depth"] = None
        
        if 'HIM' in self.train_cfg.runner_class_name:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs
        elif 'P3O' in self.train_cfg.runner_class_name:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, self.extras
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # feet相关更新
        self._compute_feet_states()
    
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        
        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
        self.dof_acc_50hz = (self.last_dof_vel_50hz - self.dof_vel) / self.dt
        self.power = torch.abs(self.torques * self.dof_vel)
        self.foot_velocities = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        if 'P3O' in self.train_cfg.runner_class_name:
            self.compute_cost() ## NOTE: costs

        # reset environments
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if ('HIM' or 'VAE') in self.train_cfg.runner_class_name:
            termination_privileged_obs = self.compute_privileged_observations(env_ids)
        self.reset_idx(env_ids)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        # 老版 0812
        self.last_dof_vel_50hz[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_feet_positions[:] = self.feet_positions[:]
        self.last_base_pos[:] = self.base_pos[:]

        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                # cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                if self.num_envs == 1:
                    cv2.imshow("Depth Image", self.forward_depth_output[self.lookat_id].cpu().numpy() )
                else:
                    cv2.imshow("Depth Image", self.forward_depth_output[self.lookat_id, -1].cpu().numpy() )
                cv2.waitKey(1)
        
        if self.viewer and self.cfg.viewer.draw_base_com:
            self._draw_base_com_vis()
        
        if ('HIM' or 'VAE') in self.train_cfg.runner_class_name:
            return env_ids, termination_privileged_obs
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        fail_buf |= self.projected_gravity[:, 2] > -0.1
        self.fail_buf *= fail_buf
        self.fail_buf += fail_buf
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_pos[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_pos[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_pos[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_pos[:, 1] < self.terrain_y_min + 1
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )
    

    def compute_observations(self):
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights,
                -1,
                1.0,
            )
        )
        self.privileged_obs_buf = torch.cat(( # 
            self.commands * self.commands_scale,  # 12
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions[:, :self.num_actions],
            self.base_ang_vel * self.obs_scales.ang_vel,
            # self.base_euler_rpy[:,:] * self.obs_scales.quat,
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

        # Add timing signals to privileged obs (critic)
        if self.cfg.env.observe_timing_parameter:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)  # 1
        if self.cfg.env.observe_clock_inputs:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.clock_inputs), dim=-1)  # 4


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
            self.commands * self.commands_scale,
            lagged_q,
            lagged_dq,
            self.actions[:, :self.num_actions],
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
        # Add timing signals to actor obs
        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)  # 1
        if self.cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf, self.clock_inputs), dim=-1)  # 4
        obs_now = obs_buf.clone()

        if self.add_noise:
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise
    
        self.noised_q = obs_now[:, self.cfg.commands.num_commands:self.cfg.commands.num_commands+self.cfg.env.num_actions] / self.obs_scales.dof_pos
        self.noised_dq = obs_now[:, self.cfg.commands.num_commands+self.cfg.env.num_actions : self.cfg.commands.num_commands+2*self.cfg.env.num_actions] / self.obs_scales.dof_vel
        self.noised_ang_vel = obs_now[: self.cfg.commands.num_commands+3*self.cfg.env.num_actions:self.cfg.commands.num_commands+3*self.cfg.env.num_actions+3] / self.obs_scales.ang_vel
        self.noised_euler_rpy = obs_now[:, self.cfg.commands.num_commands+3*self.cfg.env.num_actions+3:self.cfg.commands.num_commands+3*self.cfg.env.num_actions+5] / self.obs_scales.quat

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

    def get_command(self):
        return self.commands * self.commands_scale,

# ==================================================================================================================== #
# ================================================ Custom Rewards Function ================================================== #
# ==================================================================================================================== #
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
    # def _reward_stand_base_vel_penality(self):
    #     """当命令很小时，机器人不应该有各个方向的速度"""
    #     # Penalize motion at zero commands
    #     term_x = 5 * torch.square(self.base_lin_vel[:, 0])
    #     term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
    #     return (term_x + term_y_z) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_stand_base_vel_penality(self):
        """【修复版】静止时惩罚机身线速度漂移，使用平滑掩码防止起步抽搐"""
        # 取消内部的 5 倍硬编码，直接计算三轴速度的平方和，放大倍数交给 config.py
        lin_vel_sq = torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1)
        
        # 换成软掩码，解决起步抽搐
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        is_standstill = torch.exp(-cmd_norm / 0.1)
        
        # 返回误差（注意 config.py 里对应权重必须是负数，比如 -3.0）
        return lin_vel_sq * is_standstill

    # def _reward_stand_stability(self):
    #     """当命令很小时,惩罚机器人速度、角速度、关节扭矩，以保持机器人静止时较为稳定"""
    #     velocity_error = torch.sum(torch.abs(self.base_lin_vel[:, :3]), dim=1)
    #     energy_cost = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1) * 0.01  # 能量惩罚系数
    #     stability_penalty = torch.sum(torch.abs(self.base_ang_vel[:, :3]), dim=1) * 0.2  # 身体角速度惩罚
    #     reward = (1.0 / (1.0 + velocity_error)) - energy_cost - stability_penalty
    #     return reward * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_stand_posture_positive(self):
        """
        【正向糖果版】鼓励机器狗在零指令时做“安静的美男子”。
        只要机身不晃、关节不抖，就发正向奖励。绝对不扣分！
        """
        # 1. 抓取“发抖”和“滑步”的元凶：机身线速度、机身角速度、关节转速
        lin_vel_sq = torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1)
        ang_vel_sq = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)
        dof_vel_sq = torch.sum(torch.square(self.dof_vel), dim=1)
        
        # 2. 核心：用 exp() 高斯核把“误差”变成“有上限的糖果”
        # 当速度为 0 时，拿到 1.0 的满分；
        # 动得越厉害，糖果越少，最少掉到 0 分。绝对不会出现负数扣分！
        rew_base_steady = torch.exp(-(lin_vel_sq + ang_vel_sq) / 0.05)
        rew_dof_steady = torch.exp(-dof_vel_sq / 1.0) 
        
        # 3. 只有当摇杆归零（静止指令）时，考官才开始发糖
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        is_standstill = torch.exp(-cmd_norm / 0.1)
        
        # 将躯干稳定糖果和关节镇定糖果相加，发放给网络
        return (rew_base_steady + rew_dof_steady) * is_standstill

    # def _reward_stand_all_feet_contact(self):
    #     """当命令很小时，四只脚必须全部着地。离散奖励：全着地=1，否则=0。"""
    #     contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
    #     all_on_ground = contact.all(dim=1).float()  # 四脚全着地=1，否则=0
    #     return all_on_ground * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_stand_all_feet_contact(self):
        """
        当命令很小时，鼓励四只脚着地。连续奖励：着地一只脚给 0.25 分，全着地给 1.0 分。
        """
        # 1. 接触力判定：最好用 Z 轴受力（垂直地面的力），防止脚在空中侧面蹭到东西骗分
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # 2. 核心修复：把 0/1 变成按脚的数量给分（提供平滑梯度）
        # 1只脚=0.25分, 2只脚=0.5分, 3只脚=0.75分, 4只脚=1.0分
        # 这样网络即使只有3只脚着地，也能拿到0.75分，它就会顺着梯度去把第4只脚也踩下去！
        contact_score = torch.sum(contact.float(), dim=1) / 4.0 
        # 3. 核心修复：软静音掩码 (Soft Mask)，防止起步抽搐
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        is_standstill = torch.exp(-cmd_norm / 0.1)
        return contact_score * is_standstill


    def _reward_stand_action_freeze(self):
        """
        【智能静止锁 - 终极版】
        平时锁死动作；但在以下两种情况下会立刻解开：
        1. 受到外部巨大物理扰动（被推飞）时，解开自救。
        2. 收到高度/Pitch等姿态指令，且实际姿态尚未到达目标位置时，解开允许调整。
        """
        # ---------------- 1. 基础动作变化惩罚 ----------------
        # 只要相邻两帧输出的 action 不一样，这个值就会变小（扣分）
        action_diff = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        rew_freeze = torch.exp(-action_diff / 0.05)
        
        # ---------------- 2. 摇杆静音与姿态追踪掩码 ----------------
        # 检查是否下达了底盘移动指令（假设前3维是 vx, vy, yaw_rate）
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        
        # 获取当前的【实际】高度和 Pitch
        actual_height = self.root_states[:, 2] # 通常 root_states 第 2 维是 Z 坐标
        actual_pitch = self.base_pitch         # 假设你在环境里已经计算了 self.base_pitch
        
        # 计算【指令目标】与【实际状态】之间的物理误差
        # (注：如果你定义的高度指令不是第3维，Pitch不是第10维，请自行修改索引)
        height_error = torch.abs(self.commands[:, 3] - actual_height)
        pitch_error = torch.abs(self.commands[:, 10] - actual_pitch)
        
        # 判断是否正在调整姿态：如果误差 > 0.05，说明还没蹲到位/抬到位，正在运动中
        is_posture_adjusting = (height_error > 0.05) | (pitch_error > 0.05)
        
        # 核心逻辑：只有在 (速度指令为空) 且 (姿态已经调整到位，误差<0.05) 时，才激活锁
        is_standstill = torch.exp(-cmd_norm / 0.1) * (~is_posture_adjusting).float()
        
        # ---------------- 3. 外部扰动监控 (防推飞) ----------------
        # 监控机身在 XY 平面的实际物理被动速度
        actual_lin_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        # 被推的速度超过 0.2m/s 时，解锁自救
        is_safe_to_freeze = torch.exp(-actual_lin_vel / 0.2) 
        
        # ---------------- 4. 终极结算 ----------------
        # 只有在安全、且指令要求静止、且姿态误差极小的情况下，才给这笔定身糖果
        return rew_freeze * is_standstill * is_safe_to_freeze
        

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
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_ang_vel)

    def _reward_default_joint_pos(self):
        rew = torch.norm(self.leg_pos, dim=1)
        if self.reward_scales["default_joint_pos"] < 0:
            return rew
        else:
            return torch.exp(-20 * rew)
    
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

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     scale = self.reward_scales.get("base_height", None)
    #     if scale is None:
    #         return torch.zeros_like(self.base_height, device=self.device)
        
    #     if scale < 0:
    #         return torch.abs(self.base_height - self.commands[:, 3])
    #     else:
    #         base_height_error = torch.square(self.base_height - self.commands[:, 3])
    #         return torch.exp(-200 * base_height_error)   

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        target_height = self.commands[:, 3]
        return torch.abs(base_height - target_height)
        
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques[:, :self.num_lower_dof]), dim=1)

    def _reward_action(self):
        return torch.sum(torch.square(self.actions[:, :self.num_lower_dof]), dim=1)
    
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions[:, :self.num_lower_dof] - self.actions[:, :self.num_lower_dof]), dim=1)
    def _reward_action_smoothness(self):
        return torch.sum(torch.square(
            self.actions[:, :self.num_lower_dof] + self.last_last_actions[:, :self.num_lower_dof] - 2 * self.last_actions[:, :self.num_lower_dof]), dim=1)
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0] * self.cfg.rewards.soft_dof_pos_limit).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1] * self.cfg.rewards.soft_dof_pos_limit).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limits)[:, :self.num_lower_dof].clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques[:, :self.num_lower_dof]) - self.torque_limits[:self.num_lower_dof] * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_power(self):
        # Penalize torques
        return torch.sum(self.power, dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, :self.num_lower_dof]), dim=1)
    
    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.dof_acc_200hz[:, :self.num_lower_dof]), dim=1)
        
    def _reward_orientation_control(self):
        # Penalize non flat base orientation using exponential penalty
        # Small deviations produce large penalties, similar to tracking_lin_vel
        tracking_sigma = self.cfg.rewards.tracking_sigma
        roll_pitch_commands = self.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1], torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0], torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)

        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
        return torch.exp(-orientation_error / tracking_sigma)
    
    def _reward_tracking_contacts_shaped_force(self):
        '''
            Penalize unexpected contact forces during swing phase.
        '''
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))

        return reward / 4


    def _reward_tracking_contacts_shaped_vel(self):
        '''
            Penalize foot sliding during stance phase.
        '''
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i] * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / 0.5)))

        return reward / 4
    
    def _reward_feet_clearance_cmd_linear(self):
        '''
            Guide foot height trajectory during swing phase.
        '''
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_z = self.rigid_state[:, self.feet_indices, 2] - 0.04
        delta_z = foot_z - self.last_foot_z
        self.foot_height += delta_z
        self.last_foot_z = foot_z

        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices_tensor * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.foot_height
        target_height = self.commands[:, 9].unsqueeze(1) * phases
        rew_foot_clearance = torch.abs(target_height - foot_height) * (1 - self.desired_contact_states)
        self.foot_height *= ~contact
        return torch.sum(rew_foot_clearance, dim=1) * (torch.norm(self.commands[:, :3], dim=1) > 0.1)

   

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_dof_pos_symmetry(self):
        """Penalize left-right joint position asymmetry.
        Hip joints should be equal and opposite (FL+FR, RL+RR).
        Thigh/calf joints should be equal between left and right.
        Amplified strongly at standstill for complete symmetry.
        """
        # Hip symmetry: FL_hip + FR_hip should be ~0 (opposite signs)
        rew_hip = torch.square(self.dof_pos[:, 0] + self.dof_pos[:, 3]) + \
                  torch.square(self.dof_pos[:, 6] + self.dof_pos[:, 9])
        # Thigh symmetry: FL_thigh == FR_thigh, RL_thigh == RR_thigh
        rew_thigh = torch.square(self.dof_pos[:, 1] - self.dof_pos[:, 4]) + \
                    torch.square(self.dof_pos[:, 7] - self.dof_pos[:, 10])
        # Calf symmetry: FL_calf == FR_calf, RL_calf == RR_calf
        rew_calf = torch.square(self.dof_pos[:, 2] - self.dof_pos[:, 5]) + \
                   torch.square(self.dof_pos[:, 8] - self.dof_pos[:, 11])
        rew = rew_hip + rew_thigh + rew_calf

        # Amplify at standstill
        is_standstill = (
            torch.abs(self.commands[:, 0]) < 0.1
        ) & (
            torch.abs(self.commands[:, 1]) < 0.1
        ) & (
            torch.abs(self.commands[:, 2]) < 0.1
        )
        amplifier = 0.00001 + 1.0 * is_standstill.float()

        return rew * amplifier
    
    def _reward_default_hip_pos(self):
        """Penalize hip deviation from zero. Amplified when velocity commands are near zero."""
        joint_diff = torch.abs(self.dof_pos[:, 0]) + torch.abs(self.dof_pos[:, 3]) + \
                     torch.abs(self.dof_pos[:, 6]) + torch.abs(self.dof_pos[:, 9])

        # Detect near-zero velocity commands (standstill)
        is_standstill = (
            torch.abs(self.commands[:, 0]) < 0.1
        ) & (
            torch.abs(self.commands[:, 1]) < 0.1
        ) & (
            torch.abs(self.commands[:, 2]) < 0.1
        )

        # Standstill: large multiplier, otherwise: scale=1 (base weight from config is ~-2)
        # Amplifier of ~50x for standstill (config scale -2 becomes -100 effectively)
        amplifier = 1 + 4.0 * is_standstill.float()
        
        return joint_diff * amplifier


    """ =======================lyw新增奖励函数============================ """

    def _reward_stand_posture(self):
        """站立时保持稳定姿势：零速度、零角速度、低功耗"""
        is_stand = (torch.norm(self.commands[:, :3], dim=1) < 0.1).float()
        
        # 身体不能动
        vel_pen = torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1)
        ang_pen = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)
        
        # 关节不能抖（所有12个腿关节）
        dof_pen = torch.sum(torch.square(self.dof_pos[:, :12]), dim=1)
        
        # 关节不能有速度
        dof_vel_pen = torch.sum(torch.square(self.dof_vel[:, :12]), dim=1)
        
        return -(vel_pen + ang_pen + 0.1 * dof_pen + 0.01 * dof_vel_pen) * is_stand


    def _reward_feet_clearance_cmd_exp(self):
        '''
            [修改版] Guide foot height trajectory during swing phase using Exponential Reward.
        '''
        # 1. 高度累加逻辑（保持不变）
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_z = self.rigid_state[:, self.feet_indices, 2] - 0.04
        delta_z = foot_z - self.last_foot_z
        self.foot_height += delta_z
        self.last_foot_z = foot_z

        # 2. 目标轨迹计算（保持不变）
        #phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices_tensor * 2.0) - 1.0, 0.0, 1.0) * 2.0)  #三角形函数
        swing_phase = torch.clip((self.foot_indices_tensor * 2.0) - 1.0, 0.0, 1.0)  # 第一步：提取摆动相的纯净时钟 (0~1)，支撑相会被 clip 截断死锁为 0
        phases = torch.sin(swing_phase * torch.pi)          # 第二步：将 0~1 的纯净时钟映射为 0~Pi 的正弦半周期

        foot_height = self.foot_height
        target_height = self.commands[:, 9].unsqueeze(1) * phases

        # 3. 核心修改：计算高度误差
        height_error = target_height - foot_height
        
        # 4. 核心修改：使用高斯核将误差映射为 0~1 的奖励
        # 这里的 0.01 是方差参数，决定了奖励的“宽容度”。
        # 误差为 0 时得 1 分；误差为 5cm 时得 0.77 分；误差为 10cm 时得 0.36 分。
        rew_foot_clearance = torch.exp(-torch.square(height_error) / 0.01)
        
        # 5. 掩码操作：只在摆动相（需要悬空时）给予这个奖励
        rew_foot_clearance *= (1 - self.desired_contact_states)
        
        # 6. 接触清零与指令过滤（保持不变）
        self.foot_height *= ~contact
        return torch.sum(rew_foot_clearance, dim=1) * (torch.norm(self.commands[:, :3], dim=1) > 0.1)