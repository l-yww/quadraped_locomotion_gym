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


from .quadruped_wtw_him_arm_fix_config import QuadWtwCfg_HIM, QuadWtwCfgPPO_HIM
from wheel_legged_gym.envs.quadruped.quadruped_env import QuadEnv

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import random
import numpy as np
from wheel_legged_gym.envs.quadruped.legged_robot import get_euler_rpy_tensor
from wheel_legged_gym.utils.math import wrap_to_pi, get_scale_shift


class QuadWtwEnv_HIM(QuadEnv):
    def __init__(self, cfg: QuadWtwCfg_HIM, train_cfg: QuadWtwCfgPPO_HIM, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()

    # ==================================================================================================================== #
    # ================================================ Buffer Init ======================================================= #
    # ==================================================================================================================== #
    def _init_buffers(self):
        super()._init_buffers()
        self.noised_q = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.noised_dq = torch.zeros_like(self.dof_vel)
        self.noised_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.noised_euler_rpy = torch.zeros_like(self.base_euler_rpy)
        
        # 原始突变指令
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
        
        # 【新增】：统一的所有缓冲指令，用于替换网络观测和奖励计算
        self.smoothed_commands = torch.zeros_like(self.commands)

    # ==================================================================================================================== #
    # ================================================ Noise ============================================================= #
    # ==================================================================================================================== #
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales

        num_commands = self.cfg.commands.num_commands
        noise_vec[:num_commands] = 0
        noise_vec[num_commands: num_commands + self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[num_commands + self.num_actions: num_commands + 2 * self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[num_commands + 2 * self.num_actions: num_commands + 3 * self.num_actions] = 0
        noise_vec[num_commands + 3 * self.num_actions: num_commands + 3 * self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        if self.cfg.env.projected_gravity:
            noise_vec[num_commands + 3 * self.num_actions + 3: num_commands + 3 * self.num_actions + 3 + 3] = noise_scales.gravity * self.obs_scales.gravity
        else:
            noise_vec[num_commands + 3 * self.num_actions + 3: num_commands + 3 * self.num_actions + 3 + 2] = noise_scales.quat * self.obs_scales.quat
        return noise_vec

    # ==================================================================================================================== #
    # ================================================ Commands ========================================================== #
    # ==================================================================================================================== #
    def _resample_commands(self, env_ids):
        """Resample all 12 commands (生成原始目标指令，保留使用 self.commands)"""
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
        # 【原】步频独立随机采样(与速度无关 → 会产生低频+高速的不可能组合)
        # self.commands[env_ids, 4] = torch_rand_float(
        #     self.command_ranges["gait_frequency_cmd_range"][0], self.command_ranges["gait_frequency_cmd_range"][1],
        #     (len(env_ids), 1), device=self.device).squeeze(1)
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
        stand_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        self.commands[env_ids[stand_mask], 0:3] = 0.0

        # 【步频-速度耦合】用最终 vx 算步频:速度越大步频越高,避免"低频+高速"的不可能组合。
        # 部署(deploy)时务必按同一公式给 freq,否则 sim2real 步频对不上。
        # vx=0→flo(2Hz), vx=|lin_vel_x[1]|→fhi(4Hz), 中间线性。
        _flo = self.command_ranges["gait_frequency_cmd_range"][0]
        _fhi = self.command_ranges["gait_frequency_cmd_range"][1]
        _slope = (_fhi - _flo) / self.command_ranges["lin_vel_x"][1]
        self.commands[env_ids, 4] = torch.clamp(_flo + self.commands[env_ids, 0].abs() * _slope, _flo, _fhi)

        # # 运动时锁死姿态：base 水平（pitch=0, roll=0）+ 高度 0.40，不响应姿态命令。
        # # 仅静止（速度≈0）环境保留随机 pitch/roll/height，允许调姿态。
        # moving_mask = (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1) | \
        #               (torch.abs(self.commands[env_ids, 2]) > 0.1)
        # moving_ids = env_ids[moving_mask]
        # self.commands[moving_ids, 10] = 0.0    # pitch=0（水平）
        # self.commands[moving_ids, 11] = 0.0    # roll=0
        # self.commands[moving_ids, 3] = 0.40    # body height=0.40

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            # 【修改】：步态时钟严格按照平滑后的指令运转，避免换挡抽搐
            frequencies = self.smoothed_commands[:, 4]
            phases = self.smoothed_commands[:, 5]
            offsets = self.smoothed_commands[:, 6]
            bounds = self.smoothed_commands[:, 7]
            durations = self.smoothed_commands[:, 8]

            # Detect zero-velocity commands: all feet on ground (stand still)
            zero_cmd_mask = (torch.norm(self.smoothed_commands[:, :2], dim=1) < 0.1) & \
                            (torch.abs(self.smoothed_commands[:, 2]) < 0.1)

            gait_increment = self.dt * frequencies * (~zero_cmd_mask).float()
            self.gait_indices = torch.remainder(self.gait_indices + gait_increment, 1.0)

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

    # ==================================================================================================================== #
    # ================================================ Post Physics Step ================================================= #
    # ==================================================================================================================== #
    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations"""
        # ------- resample commands ------- #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # 【命令平滑开关】：一阶低通滤波 smoothed = alpha*smoothed + (1-alpha)*commands
        # use_smoothed_commands=False 时直通（smoothed=commands），便于消融/对比
        if self.cfg.commands.use_smoothed_commands:
            alpha = self.cfg.commands.smoothed_commands_alpha
            self.smoothed_commands = alpha * self.smoothed_commands + (1.0 - alpha) * self.commands
        else:
            self.smoothed_commands = self.commands

        self._step_contact_targets()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 注意：这里针对原始命令更新航向，下一帧自动由滤波器平滑
            self.commands[:, 1] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # ------- 获取高程图 ------- #
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # 【原度量：patch 平均】在 trimesh 过渡/粗糙地形上有偏且抖动，导致高度奖励与门控信号不稳 → 姿态乱。
        # self.base_height = torch.mean(
        #     self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        # )
        # Use only feet currently supporting the body. Swing feet on a step must not move
        # the base-height target; fall back to the four-foot mean before first contact.
        foot_ground_heights = self._get_foot_heights()
        foot_in_contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        support_count = foot_in_contact.sum(dim=1).clamp(min=1)
        support_ground_height = (foot_ground_heights * foot_in_contact).sum(dim=1) / support_count
        mean_ground_height = foot_ground_heights.mean(dim=1)
        ground_height = torch.where(foot_in_contact.any(dim=1), support_ground_height, mean_ground_height)
        self.base_height = self.root_states[:, 2] - ground_height

        # ------- push robot ------- #
        if self.cfg.domain_rand.push_robots:
            env_ids = (self.envs_steps_buf % int(self.cfg.domain_rand.push_interval_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
            self._push_robots(env_ids)

        # ------- randomize motor params ------- #
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._randomize_dof_props(env_ids)

        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids)
            self.refresh_actor_rigid_shape_props(env_ids)
            self.refresh_actor_rigid_body_props(env_ids)

    # ==================================================================================================================== #
    # ================================================ Reset ============================================================= #
    # ==================================================================================================================== #
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._resample_commands(env_ids)
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self._randomize_dof_props(env_ids)
        self.randomize_lag_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]

        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids)
            self.refresh_actor_rigid_shape_props(env_ids)
            self.refresh_actor_rigid_body_props(env_ids)

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

        # 【修改】：重置回合时，平滑指令强制对齐原始指令，避免重置时带有上一回合的残余指令
        self.smoothed_commands[env_ids] = self.commands[env_ids]

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if 'P3O' in self.train_cfg.runner_class_name:
            for key in self.cost_episode_sums.keys():
                self.extras["episode"]['cost_' + key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
                self.cost_episode_sums[key][env_ids] = 0.
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            self.extras["episode"]["terrain_level_max"] = torch.max(self.terrain_levels).float()  # 最好 env 的等级=能力天花板
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # 静止环境上的奖励统计（排除运动环境对均值的稀释）
        if hasattr(self, 'episode_sums_standstill'):
            for key, val in self.episode_sums_standstill.items():
                if 'count' in key:
                    self.extras["episode"][key] = val.item()
                else:
                    count_key = key + '_count'
                    cnt = self.episode_sums_standstill.get(count_key, 1)
                    self.extras["episode"][key] = val.item() / max(cnt, 1)
                self.episode_sums_standstill[key] = 0

        self.base_pos_init[env_ids] = self.root_states[env_ids, 0:3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_rpy = get_euler_rpy_tensor(self.feet_quat)
        self.gait_indices[env_ids] = 0
        self.last_foot_z[env_ids] = 0.
        self.foot_height[env_ids] = 0.

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # ==================================================================================================================== #
    # ================================================ Step ============================================================== #
    # ==================================================================================================================== #
    def step(self, actions):
        self.global_counter += 1
        actions = torch.clamp(actions, min=-self.cfg.normalization.clip_actions, max=self.cfg.normalization.clip_actions)
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
                self.dof_lag_buffer[:, :, 1:] = self.dof_lag_buffer[:, :, :self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_lag_buffer[:, :, 0] = q.clone()
                dq = self.dof_vel
                self.dof_vel_lag_buffer[:, :, 1:] = self.dof_vel_lag_buffer[:, :, :self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_vel_lag_buffer[:, :, 0] = dq.clone()
            # ---------------- 随机 IMU 延迟 --------------- #
            if self.cfg.domain_rand.add_imu_lag:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
                self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                self.imu_lag_buffer[:, :, 1:] = self.imu_lag_buffer[:, :, :self.cfg.domain_rand.imu_lag_timesteps_range[1]].clone()
                if self.cfg.env.projected_gravity:
                    self.imu_lag_buffer[:, :, 0] = torch.cat((self.base_ang_vel, self.projected_gravity), 1).clone()
                else:
                    self.imu_lag_buffer[:, :, 0] = torch.cat((self.base_ang_vel, self.base_euler_rpy), 1).clone()
            # ---------- 下发 Torque, 仿真器步进一步 ----------- #
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()

        if 'HIM' not in self.train_cfg.runner_class_name:
            self.post_physics_step()
        else:
            termination_ids, termination_priveleged_obs = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]
        else:
            self.extras["depth"] = None

        if 'HIM' in self.train_cfg.runner_class_name:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs
        elif 'P3O' in self.train_cfg.runner_class_name:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, self.extras
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._compute_feet_states()

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt

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

        self.check_termination()
        self.compute_reward()
        if 'P3O' in self.train_cfg.runner_class_name:
            self.compute_cost()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if 'HIM' in self.train_cfg.runner_class_name:
            termination_privileged_obs = self.compute_privileged_observations(env_ids)
        self.reset_idx(env_ids)

        self.update_depth_buffer()
        self.compute_observations()

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel_50hz[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_feet_positions[:] = self.feet_positions[:]
        self.last_base_pos[:] = self.base_pos[:]

        self.feet_air_time *= ~self.contact_filt

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if self.num_envs == 1:
                    cv2.imshow("Depth Image", self.forward_depth_output[self.lookat_id].cpu().numpy())
                else:
                    cv2.imshow("Depth Image", self.forward_depth_output[self.lookat_id, -1].cpu().numpy())
                cv2.waitKey(1)

        if self.viewer and self.cfg.viewer.draw_base_com:
            self._draw_base_com_vis()

        if 'HIM' in self.train_cfg.runner_class_name:
            return env_ids, termination_privileged_obs

    def check_termination(self):
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
        )
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_pos[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_pos[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_pos[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_pos[:, 1] < self.terrain_y_min + 1
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )

    # ==================================================================================================================== #
    # ================================================ Observations ====================================================== #
    # ==================================================================================================================== #
    def compute_observations(self):
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights,
                -1,
                1.0,
            )
        )
        # 【修改】：喂给网络的观测全替换为平滑后的指令
        self.privileged_obs_buf = torch.cat((
            self.smoothed_commands * self.commands_scale,  # 12
            self.dof_pos * self.obs_scales.dof_pos, # 12
            self.dof_vel * self.obs_scales.dof_vel, # 12
            self.actions, #12
            self.base_ang_vel * self.obs_scales.ang_vel, # 3
            self.projected_gravity, # 3
        ), dim=-1)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(1)
                                                  - friction_coeffs_shift) * friction_coeffs_scale), dim=1)

        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(1)
                                                  - restitutions_shift) * restitutions_scale), dim=1)

        if self.cfg.env.priv_observe_payloads:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale), dim=1)

        if self.cfg.env.priv_observe_inertia:
            inertia_scale, inertia_shift = get_scale_shift(self.cfg.domain_rand.randomize_inertia_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.inertia_scale.unsqueeze(1) - inertia_shift) * inertia_scale), dim=1)

        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)

        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1)

        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.domain_rand.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.com_displacements[:, :3] - com_displacements_shift) * com_displacements_scale), dim=1)

        if self.cfg.env.priv_observe_heightmap:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 heights * self.obs_scales.height_measurements), dim=1)

        # Add timing signals to privileged obs (critic) — must come BEFORE estimation targets
        if self.cfg.env.observe_timing_parameter:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        if self.cfg.env.observe_clock_inputs:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.clock_inputs), dim=-1)

        # 始终将需要预测的值量放在最后（若有EST网络）
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                             self.base_height_obs * self.obs_scales.height_measurements,
                                             self.base_lin_vel * self.obs_scales.lin_vel,
                                             ), dim=1)

        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                      self.cfg.domain_rand.dof_lag_timesteps_range[1] + 1, (self.num_envs,), device=self.device)
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
                                                      self.cfg.domain_rand.imu_lag_timesteps_range[1] + 1, (self.num_envs,), device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:, :3].clone()
            if self.cfg.env.projected_gravity:
                self.lagged_projected_gravity = self.lagged_imu[:, -3:].clone()
            else:
                self.lagged_base_euler_rpy = self.lagged_imu[:, -3:].clone()
        else:
            self.lagged_base_ang_vel = self.base_ang_vel[:, :3]
            self.lagged_projected_gravity = self.projected_gravity
            self.lagged_base_euler_rpy = self.base_euler_rpy

        lagged_q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        lagged_dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        # 【修改】：Actor观测统一使用平滑指令
        obs_buf = torch.cat((
            self.smoothed_commands * self.commands_scale,
            lagged_q,
            lagged_dq,
            self.actions,
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,
        ), dim=-1)
        
        if self.cfg.env.projected_gravity:
            obs_buf = torch.cat((
                obs_buf,
                self.lagged_projected_gravity * self.obs_scales.quat,
            ), dim=-1)
        else:
            obs_buf = torch.cat((
                obs_buf,
                self.lagged_base_euler_rpy[:, :2] * self.obs_scales.quat,
            ), dim=-1)
            
        # Add timing signals to actor obs
        if self.cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        if self.cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf, self.clock_inputs), dim=-1)
        obs_now = obs_buf.clone()

        if self.add_noise:
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise

        self.noised_q = obs_now[:, self.cfg.commands.num_commands:self.cfg.commands.num_commands + self.cfg.env.num_actions] / self.obs_scales.dof_pos
        self.noised_dq = obs_now[:, self.cfg.commands.num_commands + self.cfg.env.num_actions: self.cfg.commands.num_commands + 2 * self.cfg.env.num_actions] / self.obs_scales.dof_vel
        self.noised_ang_vel = obs_now[:, self.cfg.commands.num_commands + 3 * self.cfg.env.num_actions:self.cfg.commands.num_commands + 3 * self.cfg.env.num_actions + 3] / self.obs_scales.ang_vel
        self.noised_euler_rpy = obs_now[:, self.cfg.commands.num_commands + 3 * self.cfg.env.num_actions + 3:self.cfg.commands.num_commands + 3 * self.cfg.env.num_actions + 5] / self.obs_scales.quat

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def get_command(self):
        return self.smoothed_commands * self.commands_scale,

    def compute_privileged_observations(self, env_ids):
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        privileged_obs_buf = torch.cat((
            self.smoothed_commands * self.commands_scale,  # 12
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
        ), dim=-1)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.friction_coeffs[:, 0].unsqueeze(1)
                                             - friction_coeffs_shift) * friction_coeffs_scale), dim=1)

        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.restitutions[:, 0].unsqueeze(1)
                                             - restitutions_shift) * restitutions_scale), dim=1)

        if self.cfg.env.priv_observe_payloads:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale), dim=1)

        if self.cfg.env.priv_observe_inertia:
            inertia_scale, inertia_shift = get_scale_shift(self.cfg.domain_rand.randomize_inertia_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.inertia_scale.unsqueeze(1) - inertia_shift) * inertia_scale), dim=1)

        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)

        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1)

        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.domain_rand.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.com_displacements[:, :3] - com_displacements_shift) * com_displacements_scale), dim=1)

        # Add timing signals (must match compute_observations privileged_obs_buf) — before estimation targets
        if self.cfg.env.observe_timing_parameter:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        if self.cfg.env.observe_clock_inputs:
            privileged_obs_buf = torch.cat((privileged_obs_buf, self.clock_inputs), dim=-1)

        # 始终将需要预测的值量放在最后（若有EST网络）
        privileged_obs_buf = torch.cat((privileged_obs_buf,
                                        self.base_height_obs * self.obs_scales.height_measurements,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        ), dim=1)

        return privileged_obs_buf[env_ids]

    # ==================================================================================================================== #
    # ================================================ Reward Functions ================================================== #
    # ==================================================================================================================== #

    def _get_standstill_weight(self, transition_speed=0.05):
        """Smooth gate for rewards intended only while standing."""
        cmd_norm = torch.norm(self.smoothed_commands[:, :3], dim=1)
        return torch.exp(-torch.square(cmd_norm / transition_speed))

    def _get_moving_weight(self, transition_speed=0.15):
        """平滑运动掩码: 1 - standstill_weight = cmd_norm/(transition_speed + cmd_norm)"""
        cmd_norm = torch.norm(self.smoothed_commands[:, :3], dim=1)
        return cmd_norm / (transition_speed + cmd_norm)

    def _get_height_achievement(self):
        """高度达成度: exp(-height_error/0.05)，误差 5cm→0.37, 10cm→0.14
        """
        # actual_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)  # 老: patch 平均(trimesh 脏)
        actual_height = self.base_height  # 统一用脚底支撑面度量(见 _post_physics_step_callback)
        height_error = torch.abs(self.smoothed_commands[:, 3] - actual_height)
        # return torch.exp(-height_error / 0.05)
        # return torch.exp(-height_error / 0.02) 
        return torch.exp(-height_error / 0.015) #加大误差惩罚

    def _get_steady_weight(self):
        """稳态门控: base 的 height 和 pitch 都已到位→1.0; 正在调整(误差大)→趋 0。
        height 5cm≈0.37、pitch 0.15rad≈0.37,二者相乘:偏离越大惩罚越放松。
        """
        # actual_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)  # 老: patch 平均(trimesh 脏)
        actual_height = self.base_height  # 统一用脚底支撑面度量(见 _post_physics_step_callback)
        height_err = torch.abs(self.smoothed_commands[:, 3] - actual_height)
        pitch_err = torch.abs(-self.smoothed_commands[:, 10] - self.base_euler_rpy[:, 1])
        return torch.exp(-height_err / 0.05) * torch.exp(-pitch_err / 0.15)

    def _get_foot_heights(self):
        """采样 4 只脚正下方的地形高度(世界系 z),供 base_height 用"支撑面"度量。
        plane→0(等价 base_height=root_z);trimesh→脚下四点 min 插值查 height_samples。
        rigid_state 已是世界系, 脚的 xy 无需再按 yaw 旋转。
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, 4, device=self.device)
        points = self.rigid_state[:, self.feet_indices, :2] + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1], 0, self.height_samples.shape[1] - 2)
        h1 = self.height_samples[px, py]
        h2 = self.height_samples[px + 1, py]
        h3 = self.height_samples[px, py + 1]
        h = torch.min(h1, torch.min(h2, h3))
        return h * self.terrain.cfg.vertical_scale

    # ------------------------------------------------------------------------------#
    # --------------------------- base / regularization ---------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_base_acc(self):
        base_lin_acc = torch.norm(self.last_root_vel[:, 0:3] - self.root_states[:, 7:10], dim=1) / self.dt
        base_ang_acc = torch.norm(self.last_root_vel[:, 3:6] - self.root_states[:, 10:13], dim=1) / self.dt
        rew = base_lin_acc + 0.02 * base_ang_acc
        return rew * self._get_steady_weight()  # 稳态门控: 调 height/pitch 时放松, 不抑制调整

    def _reward_low_speed(self):
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.smoothed_commands[:, 0])
        speed_too_low = absolute_speed < 0.8 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.smoothed_commands[:, 0])
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.smoothed_commands[:, 0].abs() > 0.1)

    # ------------------------------------------------------------------------------#
    # ------------------------- stand still rewards --------------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_stand_base_vel_penality(self):
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * (torch.norm(self.smoothed_commands[:, :3], dim=1) < 0.1)

    def _reward_stand_stability(self):
        velocity_error = torch.sum(torch.abs(self.base_lin_vel[:, :3]), dim=1)
        energy_cost = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1) * 0.01
        stability_penalty = torch.sum(torch.abs(self.base_ang_vel[:, :3]), dim=1) * 0.2
        reward = (1.0 / (1.0 + velocity_error)) - energy_cost - stability_penalty
        return reward * (torch.norm(self.smoothed_commands[:, :3], dim=1) < 0.1)

    def _reward_stand_all_feet_contact(self):
        """当命令很小时，四只脚必须全部着地。使用软掩码提供平滑梯度。"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_score = torch.sum(contact.float(), dim=1) / 4.0

        # actual_height = self.root_states[:, 2]
        # actual_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)  # 老: patch 平均(trimesh 脏)
        actual_height = self.base_height  # 统一用脚底支撑面度量(见 _post_physics_step_callback)
        actual_roll = self.base_euler_rpy[:, 0]
        actual_pitch = self.base_euler_rpy[:, 1]

        height_error = torch.abs(self.smoothed_commands[:, 3] - actual_height)
        roll_error = torch.abs(-self.smoothed_commands[:, 11] - actual_roll)
        pitch_error = torch.abs(-self.smoothed_commands[:, 10] - actual_pitch)

        max_posture_error = torch.max(torch.stack([height_error, roll_error, pitch_error], dim=1), dim=1)[0]
        posture_mask = torch.exp(-max_posture_error / 0.5)

        is_standstill = self._get_standstill_weight() * posture_mask

        return contact_score * is_standstill * self._get_height_achievement()

    def _reward_stand_posture_positive(self):
        lin_vel_sq = torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1)
        ang_vel_sq = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)
        dof_vel_sq = torch.sum(torch.square(self.dof_vel), dim=1)

        base_error = lin_vel_sq + ang_vel_sq
        rew_base_steady = 0.5 * (1.0 - torch.clamp(base_error / 0.5, 0.0, 1.0)) + \
                          0.5 * torch.exp(-base_error / 0.05)

        rew_dof_steady = 0.5 * (1.0 - torch.clamp(dof_vel_sq / 2.0, 0.0, 1.0)) + \
                         0.5 * torch.exp(-dof_vel_sq / 1.0)

        # actual_height = self.root_states[:, 2]
        # actual_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)  # 老: patch 平均(trimesh 脏)
        actual_height = self.base_height  # 统一用脚底支撑面度量(见 _post_physics_step_callback)
        actual_roll = self.base_euler_rpy[:, 0]
        actual_pitch = self.base_euler_rpy[:, 1]

        height_error = torch.abs(self.smoothed_commands[:, 3] - actual_height)
        roll_error = torch.abs(-self.smoothed_commands[:, 11] - actual_roll)
        pitch_error = torch.abs(-self.smoothed_commands[:, 10] - actual_pitch)

        attitude_error = roll_error ** 2 + pitch_error ** 2
        attitude_score = torch.exp(-attitude_error / 0.05)
        height_score = torch.exp(-height_error / 0.2)

        posture_mask = attitude_score * height_score

        is_standstill = self._get_standstill_weight() * posture_mask
        return (rew_base_steady + rew_dof_steady) * is_standstill

    def _reward_stand_action_freeze(self):
        """静止时鼓励冻结动作，减少不必要的关节微调。
        条件: cmd≈0 + 到达目标高度 + 身体确实静止 + 动作变化小
        """
        action_diff = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        rew_freeze = torch.exp(-action_diff / 0.02)  # 指数衰减，action变化>0.05→奖励<0.3

        is_standstill = self._get_standstill_weight()
        actual_lin_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        is_safe_to_freeze = torch.exp(-actual_lin_vel / 0.1)  # 0.1m/s→0.37, 0.2m/s→0.14

        return rew_freeze * is_standstill * is_safe_to_freeze * self._get_height_achievement()

    def _reward_stand_absolute_stable(self):
        is_standstill = self._get_standstill_weight()

        actual_pitch = self.base_euler_rpy[:, 1]
        #actual_height = self.root_states[:, 2]
        # actual_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)  # 老: patch 平均(trimesh 脏)
        actual_height = self.base_height  # 统一用脚底支撑面度量(见 _post_physics_step_callback)
        
        pitch_error = torch.abs(-self.smoothed_commands[:, 10] - actual_pitch)
        height_error = torch.abs(self.smoothed_commands[:, 3] - actual_height)

        pitch_gate = torch.clamp((pitch_error - 0.05) / 0.10, 0.0, 1.0)
        height_gate = torch.clamp((height_error - 0.05) / 0.10, 0.0, 1.0)

        is_adjusting = torch.max(pitch_gate, height_gate)

        base_vel_sq = torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1)
        base_ang_sq = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)
        dof_vel_sq = torch.sum(torch.square(self.dof_vel), dim=1)

        rew_stable = torch.exp(-(base_vel_sq + base_ang_sq + 0.05 * dof_vel_sq) / 0.2)
        rew_stable_gated = rew_stable * (1.0 - is_adjusting)

        rew = is_standstill * rew_stable_gated

        # 按静止/运动分别统计，训练时看 TensorBoard 的 stand_absolute_stable_standonly
        if not hasattr(self, 'episode_sums_standstill'):
            self.episode_sums_standstill = {}
        stand_mask = is_standstill > 0.5
        if stand_mask.any():
            self.episode_sums_standstill.setdefault('stand_absolute_stable_standonly', 0)
            self.episode_sums_standstill['stand_absolute_stable_standonly'] += rew[stand_mask].sum()
            self.episode_sums_standstill.setdefault('stand_absolute_stable_count', 0)
            self.episode_sums_standstill['stand_absolute_stable_count'] += stand_mask.sum()

        return rew * self._get_height_achievement()

    def _reward_stand_feet_slip_penalty(self):
        is_standstill = (torch.norm(self.smoothed_commands[:, :3], dim=1) < 0.05)
        foot_velocities = self.rigid_state[:, self.feet_indices, 7:10]
        
        # 计算脚底的绝对速度
        foot_speeds = torch.norm(foot_velocities, dim=2)
        # 允许脚底有不超过 0.02 m/s 的微小仿真滑动，超过这个值才算真正的“打滑”
        slip = torch.clamp(foot_speeds - 0.02, min=0.0)
        slip_penalty = torch.sum(torch.square(slip), dim=1)
        return slip_penalty * is_standstill.float()

    def _reward_stand_pitch_tracking(self):
        """静止时专属的 pitch 正向跟踪奖励。

        提供明确的 pitch 拉力，弥补 orientation_control / pitch_tracking_penalty 在静止、
        action 近冻结时梯度偏弱的不足。乘 standstill_weight ⇒ 运动时自动归零，不影响运动行为。
        sigma=0.1 比常规 0.05 更宽，保证 pitch 误差较大时仍有有效梯度（避免远端梯度消失）。
        """
        actual_pitch = self.base_euler_rpy[:, 1]
        pitch_error = torch.abs(-self.commands[:, 10] - actual_pitch)
        pitch_score = torch.exp(-pitch_error / 0.1)
        return pitch_score * self._get_standstill_weight()

    def _reward_stand_height_tracking(self):
        height_error = torch.abs(self.base_height - self.smoothed_commands[:, 3])
        rew = torch.exp(-height_error / 0.01)
        return rew * self._get_standstill_weight()

    def _reward_stand_torque_penalty(self):
        torque_deadband = 85.0   # Nm, 低于此不罚
        over = (torch.abs(self.torques) - torque_deadband).clip(min=0.0)   # 12 关节超出部分
        rew = torch.sum(torch.square(over), dim=1)
        return rew * self._get_standstill_weight()

    # ------------------------------------------------------------------------------#
    # ------------------------- termination rewards --------------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    # ------------------------------------------------------------------------------#
    # --------------------------- tracking rewards ---------------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.smoothed_commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma_lin_vel)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.smoothed_commands[:, 2] - self.base_ang_vel[:, 2])
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
        return torch.square(self.base_lin_vel[:, 2]) * self._get_steady_weight()  # 稳态门控: 变高时放松

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity (带有姿态调整豁免权)"""
        actual_pitch = self.base_euler_rpy[:, 1]
        pitch_error = torch.abs(-self.smoothed_commands[:, 10] - actual_pitch)
        is_pitching = (pitch_error > 0.05).float()

        pen_roll = torch.square(self.base_ang_vel[:, 0])
        pen_pitch = torch.square(self.base_ang_vel[:, 1])

        total_penalty = pen_roll + pen_pitch * (1.0 - is_pitching)
        return total_penalty

    def _reward_vel_mismatch_exp(self):
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
        c_update = (lin_mismatch + ang_mismatch) / 2.
        return c_update

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_positive(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_rpy[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_base_height(self):
        """Penalize height errors, with a stronger cost when the base is too low."""
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)  # 老: patch 平均(trimesh 脏)
        base_height = self.base_height  # 统一脚底支撑面度量
        target_height = self.smoothed_commands[:, 3]
        low_error = torch.clamp(target_height - base_height - 0.01, min=0.0)
        high_error = torch.clamp(base_height - target_height - 0.01, min=0.0)
        return low_error + 0.25 * high_error

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_action(self):
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        return torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0] * self.cfg.rewards.soft_dof_pos_limit).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1] * self.cfg.rewards.soft_dof_pos_limit).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limits).clip(min=0.), dim=1)

    # def _reward_torque_limits(self):
    #     return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_torque_limits(self):
        # Penalize torques above soft_torque_limit of the velocity-dependent torque envelope.
        lower_limit, upper_limit = self._torque_velocity_limits()
        soft_ratio = getattr(self.cfg.rewards, "soft_torque_limit", 0.8)
        soft_upper_limit = soft_ratio * upper_limit
        soft_lower_limit = soft_ratio * lower_limit
        upper_violation = (self.torques - soft_upper_limit).clip(min=0.)
        lower_violation = (soft_lower_limit - self.torques).clip(min=0.)
        return torch.sum(upper_violation + lower_violation, dim=1)

    def _reward_near_torque_clip(self):
        """力矩接近 60/180 限幅时惩罚（clip 之前的 PD 原始命令），逼策略留余量。"""
        if not hasattr(self, 'torques_cmd'):
            return torch.zeros(self.num_envs, device=self.device)
        hip_idx = [0, 1, 3, 4, 6, 7, 9, 10]
        calf_idx = [2, 5, 8, 11]
        # 40Nm/120Nm 开始罚，到 60/180 线性增到最大
        hip_over = torch.relu(torch.abs(self.torques_cmd[:, hip_idx]) - 40.0) / 20.0
        calf_over = torch.relu(torch.abs(self.torques_cmd[:, calf_idx]) - 120.0) / 60.0
        return torch.sum(hip_over, dim=1) + torch.sum(calf_over, dim=1)

    def _reward_power(self):
        return torch.sum(self.power, dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        dof_acc = self.dof_acc_200hz
        return torch.sum(torch.square(dof_acc), dim=1)

    # ------------------------------------------------------------------------------#
    # ---------------------- advanced tracking / gait rewards ----------------------#
    # ------------------------------------------------------------------------------#
    def _reward_orientation_control(self):
        """Penalize non flat base orientation using a Dual-Kernel exponential penalty."""
        tracking_sigma = self.cfg.rewards.tracking_sigma
        roll_pitch_commands = self.smoothed_commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1], torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0], torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))
        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)
        orientation_error_sq = torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
        reward_broad = torch.exp(-orientation_error_sq / (tracking_sigma * 2.0))
        reward_sharp = torch.exp(-orientation_error_sq / tracking_sigma)
        return 0.5 * reward_broad + 0.5 * reward_sharp

    def _reward_pitch_agility(self):
        """主动奖励快速切换 Pitch。"""
        target_pitch = -self.smoothed_commands[:, 10]
        actual_pitch = self.base_euler_rpy[:, 1]
        pitch_error = target_pitch - actual_pitch

        is_adjusting = (torch.abs(pitch_error) > 0.05).float()
        desired_direction = torch.sign(pitch_error)

        actual_pitch_vel = self.base_ang_vel[:, 1]
        effective_vel = desired_direction * actual_pitch_vel
        agility_reward = torch.clamp(effective_vel, min=0.0, max=1.5)

        return agility_reward * is_adjusting

    def _reward_pitch_tracking_penalty(self):
        target_pitch = -self.smoothed_commands[:, 10]
        actual_pitch = self.base_euler_rpy[:, 1]
        pitch_error = torch.abs(target_pitch - actual_pitch)
        error = torch.clamp(pitch_error - 0.02, min=0.0)
        return error

    def _reward_tracking_contacts_shaped_force(self):
        """Penalize unexpected contact forces during swing phase."""
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        """Penalize foot sliding during stance phase."""
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i] * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / 0.5)))
        return reward / 4

    def _reward_feet_clearance_cmd_linear(self):
        """Guide foot height trajectory during swing phase (linear version)."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_z = self.rigid_state[:, self.feet_indices, 2] - 0.04
        delta_z = foot_z - self.last_foot_z
        self.foot_height += delta_z
        self.last_foot_z = foot_z

        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices_tensor * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.foot_height
        target_height = self.smoothed_commands[:, 9].unsqueeze(1) * phases

        rew_foot_clearance = torch.abs(target_height - foot_height) * (1 - self.desired_contact_states)
        self.foot_height *= ~contact
        # 惩罚版用反向 height_achievement：蹲低时惩罚放大（2-h_a），逼策略抬高 base
        return torch.sum(rew_foot_clearance, dim=1) * (torch.norm(self.smoothed_commands[:, :3], dim=1) > 0.1) * (2.0 - self._get_height_achievement())

    def _reward_feet_clearance_cmd_exp(self):
        """Guide foot height trajectory during swing phase using Exponential Reward."""
        swing_phase = torch.clip((self.foot_indices_tensor * 2.0) - 1.0, 0.0, 1.0)
        # sin 轨迹曲线
        phases = torch.sin(swing_phase * torch.pi)

        # 非对称 sin 轨迹曲线
        # phases_rise = torch.sin(swing_phase * torch.pi)
        # phases_fall = torch.sin((1.0 - swing_phase) * torch.pi) ** 1.5
        # phases = torch.where(swing_phase < 0.5, phases_rise, phases_fall)

        # cycloid 摆动线轨迹曲线
        # phases = swing_phase - torch.sin(2 * torch.pi * swing_phase) / (2 * torch.pi)

        # t_rise = swing_phase * 2.0                                   # 上升段归一化 0→1
        # t_fall = (swing_phase - 0.5) * 2.0                           # 下降段归一化 0→1
        # phases_rise = torch.sin(t_rise * torch.pi * 0.5)             # 0→1 的四分之一正弦（快抬）
        # phases_fall = 1.0 - (3.0 * t_fall**2 - 2.0 * t_fall**3)      # smoothstep 1→0（两端平，落地斜率0）
        # phases = torch.where(swing_phase < 0.5, phases_rise, phases_fall)

        # linear 版三角波轨迹（踏步轻，落地匀速）
        # phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices_tensor * 2.0) - 1.0, 0.0, 1.0) * 2.0)

        foot_height = self.rigid_state[:, self.feet_indices, 2] - self._get_foot_heights() - 0.04
        target_height = self.smoothed_commands[:, 9].unsqueeze(1) * phases
        height_error = target_height - foot_height
        rew_foot_clearance = torch.exp(-torch.square(height_error) / 0.005)
        rew_foot_clearance *= (1 - self.desired_contact_states)

        is_moving = self._get_moving_weight()
        base_height_error = torch.clamp(
            torch.abs(self.smoothed_commands[:, 3] - self.base_height) - 0.01,
            min=0.0,
        )
        # Keep swing-height supervision available while the body is still recovering height.
        height_gate = 0.3 + 0.7 * torch.exp(-base_height_error / 0.03)
        return torch.sum(rew_foot_clearance, dim=1) * is_moving * height_gate

    def _reward_trot_symmetry_positive(self):
        # All hip joints use the same positive rotation direction, so diagonal hips
        # should track equal offsets rather than opposite-sign mirror offsets.
        q_rel = self.dof_pos - self.default_dof_pos
        hip_err   = torch.square(q_rel[:, 0] - q_rel[:, 9]) +  torch.square(q_rel[:, 3] - q_rel[:, 6])
        thigh_err = torch.square(q_rel[:, 1] - q_rel[:, 10]) + torch.square(q_rel[:, 4] - q_rel[:, 7])
        calf_err  = torch.square(q_rel[:, 2] - q_rel[:, 11]) + torch.square(q_rel[:, 5] - q_rel[:, 8])
        total_err = hip_err + thigh_err + calf_err
        reward = torch.exp(-total_err / 0.20)
        maneuver_command = torch.maximum(torch.abs(self.smoothed_commands[:, 1]), torch.abs(self.smoothed_commands[:, 2]))
        is_straight = torch.exp(-maneuver_command / 0.1)
        v_cmd_norm = torch.norm(self.smoothed_commands[:, :2], dim=1)
        is_moving = 1.0 - torch.exp(-v_cmd_norm / 0.1)
        return reward * is_straight * is_moving

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_landing_velocity(self):
        foot_vz = torch.abs(self.foot_velocities[:, :, 2])           # [num_envs, 4] 落地垂直速度
        landing = self.first_contacts.float()                        # [num_envs, 4] 刚落地掩码
        rew = torch.sum(foot_vz * landing, dim=1)                    # 只罚落地那帧
        is_moving = self._get_moving_weight()
        return rew * is_moving

    def _reward_dof_pos_symmetry(self):
        """Penalize diagonal joint-position asymmetry with a soft dead-band.
        URDF joint order: FL->FR->RL->RR
        hip: FL=RR, FR=RL (all hip joints share the same positive direction)
        thigh/calf: FL~=RR, FR~=RL
        """
        q_rel = self.dof_pos - self.default_dof_pos
        err_hip = torch.square(q_rel[:, 0] - q_rel[:, 9]) + \
                torch.square(q_rel[:, 3] - q_rel[:, 6])
        err_thigh = torch.square(q_rel[:, 1] - q_rel[:, 10]) + \
                    torch.square(q_rel[:, 4] - q_rel[:, 7])
        err_calf = torch.square(q_rel[:, 2] - q_rel[:, 11]) + \
                torch.square(q_rel[:, 5] - q_rel[:, 8])
        total_err = err_hip + err_thigh + err_calf

        dead_band = 0.01 #0.05
        safe_err = torch.clamp(total_err - dead_band, min=0.0)
        rew = 1.0 - torch.exp(-safe_err / 0.1)

        maneuver_command = torch.maximum(torch.abs(self.smoothed_commands[:, 1]), torch.abs(self.smoothed_commands[:, 2]))
        is_straight = torch.exp(-maneuver_command / 0.1)
        return rew * is_straight

    # def _reward_default_hip_pos(self):
    #     """Penalize hip deviation from default（而非绝对 0，兼容 default_joint_angles 内八偏置）."""
    #     joint_diff = torch.abs(self.dof_pos[:, 0] - self.default_dof_pos[:, 0]) + \
    #                  torch.abs(self.dof_pos[:, 3] - self.default_dof_pos[:, 3]) + \
    #                  torch.abs(self.dof_pos[:, 6] - self.default_dof_pos[:, 6]) + \
    #                  torch.abs(self.dof_pos[:, 9] - self.default_dof_pos[:, 9])

    #     is_standstill = (
    #         torch.abs(self.smoothed_commands[:, 0]) < 0.1
    #     ) & (
    #         torch.abs(self.smoothed_commands[:, 1]) < 0.1
    #     ) & (
    #         torch.abs(self.smoothed_commands[:, 2]) < 0.1
    #     )

    #     actual_pitch = self.base_euler_rpy[:, 1]
    #     pitch_error = torch.abs(-self.smoothed_commands[:, 10] - actual_pitch)
    #     is_pitching = (pitch_error > 0.1).float()
    #     amplifier = 1.0 + 2.0 * (is_standstill.float() * (1.0 - is_pitching))
    #     maneuver_command = torch.maximum(torch.abs(self.smoothed_commands[:, 1]), torch.abs(self.smoothed_commands[:, 2]))
    #     straight_weight = torch.exp(-maneuver_command / 0.1)
    #     return joint_diff * amplifier * straight_weight

    def _reward_default_hip_pos(self):
        """Keep hips neutral during straight motion or in-place pitch adjustment."""
        hip_indices = [0, 3, 6, 9]
        joint_diff = torch.sum(
            torch.abs(self.dof_pos[:, hip_indices] - self.default_dof_pos[:, hip_indices]),
            dim=1,
        )
        # Lateral and yaw motion need the hips for posture adjustment. Keep the
        # constraint for pure forward/backward motion, and for in-place pitch
        # adjustment where the hip joints should remain neutral.
        vx = torch.abs(self.smoothed_commands[:, 0])
        vy = torch.abs(self.smoothed_commands[:, 1])
        wz = torch.abs(self.smoothed_commands[:, 2])
        pure_vx = (vx > 0.1) & (vy < 0.1) & (wz < 0.1)
        in_place_pitch = (vx < 0.1) & (vy < 0.1) & (wz < 0.1) & (
            torch.abs(self.smoothed_commands[:, 10]) > 0.05
        )
        return joint_diff * (pure_vx | in_place_pitch).float()

