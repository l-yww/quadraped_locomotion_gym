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


from .quadruped_wtw_him_arm_fix_height_scan_config import QuadWtwCfg_HIM, QuadWtwCfgPPO_HIM
from wheel_legged_gym.envs.quadruped.quadruped_env import QuadEnv

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import random
import numpy as np
from wheel_legged_gym.envs.quadruped.legged_robot import get_euler_rpy_tensor
from wheel_legged_gym.utils.math import wrap_to_pi, get_scale_shift, quat_apply_yaw


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
        self.foot_swing_peak_height = torch.zeros_like(self.foot_height)
        self.foot_swing_target_height = torch.zeros_like(self.foot_height)
        self.last_foot_in_swing = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        virtual_radius = self.cfg.rewards.virtual_collision_radius
        self.virtual_collision_offsets = torch.tensor(
            [
                [-virtual_radius, -virtual_radius],
                [-virtual_radius, 0.0],
                [-virtual_radius, virtual_radius],
                [0.0, -virtual_radius],
                [0.0, 0.0],
                [0.0, virtual_radius],
                [virtual_radius, -virtual_radius],
                [virtual_radius, 0.0],
                [virtual_radius, virtual_radius],
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        
        self.smoothed_commands = torch.zeros_like(self.commands)

        self._init_height_scan_buffers()

    def _init_height_scan_buffers(self):
        """Allocate one cached height map and its independent 10 Hz sensor clock.

        The physics loop runs at 200 Hz. Height maps are sampled only when
        this clock reaches 100 ms; all other policy observations reuse the
        latest scan, which is inserted into each actor observation frame.
        """
        self._height_scan_enabled = bool(
            getattr(self.cfg.env, "actor_observe_heightmap", False)
        )
        self._actor_proprio_obs_dim = self.cfg.env.num_proprio_obs
        n_points = self.cfg.env.num_height_scan_points
        n_scan = self.cfg.env.num_height_scan_input
        expected_scan_dim = n_points if self._height_scan_enabled else 0
        if n_scan != expected_scan_dim:
            raise ValueError(
                "num_height_scan_input must follow actor_observe_heightmap: "
                f"expected {expected_scan_dim}, got {n_scan}"
            )

        expected_single_obs = self._actor_proprio_obs_dim + n_scan
        if self.cfg.env.num_single_obs != expected_single_obs:
            raise ValueError(
                "num_single_obs must contain proprioception and the configured "
                f"height scan: expected {expected_single_obs}, got "
                f"{self.cfg.env.num_single_obs}"
            )
        expected_history_obs = self.cfg.env.frame_stack * expected_single_obs
        if self.cfg.env.num_observations != expected_history_obs:
            raise ValueError(
                "num_observations must be frame_stack * num_single_obs: "
                f"expected {expected_history_obs}, got {self.cfg.env.num_observations}"
            )
        if (
            self.obs_history.maxlen != self.cfg.env.frame_stack
            or self.obs_history[0].shape[1] != expected_single_obs
        ):
            raise RuntimeError(
                "obs_history was not allocated for the configured actor frame"
            )

        if not self._height_scan_enabled:
            return

        if n_points != self.num_height_points:
            raise ValueError(
                f"num_height_scan_points={n_points} does not match "
                f"the terrain grid ({self.num_height_points})"
            )
        self.num_obs = self.cfg.env.num_observations
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float,
        )
        self.last_actor_heights = torch.zeros(self.num_envs, n_scan, device=self.device)
        self.last_critic_heights = torch.zeros(self.num_envs, n_points, device=self.device)

        self.height_update_interval_steps = max(
            1, int(round(1.0 / (self.cfg.sim.dt * self.cfg.domain_rand.height_update_hz)))
        )
        self._height_sim_step_counter = 0

        self.height_episode_offset = torch.zeros(self.num_envs, device=self.device)
        self.height_yaw_noise = torch.zeros(self.num_envs, device=self.device)
        self.height_pitch_bias = torch.zeros(self.num_envs, device=self.device)
        self.height_roll_bias = torch.zeros(self.num_envs, device=self.device)

        self._n_grid_x = len(self.cfg.terrain.measured_points_x)
        self._n_grid_y = len(self.cfg.terrain.measured_points_y)
        x_points = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        y_points = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        x_max = max(abs(self.cfg.terrain.measured_points_x[0]), abs(self.cfg.terrain.measured_points_x[-1]))
        y_max = max(abs(self.cfg.terrain.measured_points_y[0]), abs(self.cfg.terrain.measured_points_y[-1]))
        grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing="ij")
        self._tilt_x_norm = (grid_x / x_max).reshape(-1)
        self._tilt_y_norm = (grid_y / y_max).reshape(-1)

        self.previous_height_scan = torch.zeros(
            self.num_envs, n_points, device=self.device,
        )
        self.height_scan_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )

    def _resample_height_episode_state(self, env_ids):
        if not self._height_scan_enabled or len(env_ids) == 0:
            return

        cfg = self.cfg.domain_rand
        count = len(env_ids)
        if cfg.randomize_height_offset:
            self.height_episode_offset[env_ids] = torch_rand_float(
                cfg.height_offset_range[0], cfg.height_offset_range[1],
                (count, 1), device=self.device,
            ).squeeze(1)
        if cfg.randomize_height_yaw:
            self.height_yaw_noise[env_ids] = torch_rand_float(
                cfg.height_yaw_noise_range[0], cfg.height_yaw_noise_range[1],
                (count, 1), device=self.device,
            ).squeeze(1)
        if cfg.randomize_height_roll_pitch:
            self.height_pitch_bias[env_ids] = torch_rand_float(
                cfg.height_pitch_bias_range[0], cfg.height_pitch_bias_range[1],
                (count, 1), device=self.device,
            ).squeeze(1)
            self.height_roll_bias[env_ids] = torch_rand_float(
                cfg.height_roll_bias_range[0], cfg.height_roll_bias_range[1],
                (count, 1), device=self.device,
            ).squeeze(1)

    def _update_height_scan_10hz(self):
        """Sample and process the complete height map at one 10 Hz sensor tick."""
        cfg = self.cfg.domain_rand
        self.base_quat[:] = self.root_states[:, 3:7]
        terrain_heights = self._get_heights()
        base_ref = getattr(self.cfg.rewards, "base_height_target", 0.4)
        clean = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - base_ref - terrain_heights,
            -1.0, 1.0,
        )
        self.last_critic_heights.copy_(clean * self.obs_scales.height_measurements)

        # First choose the spatial sampling frame. All remaining perturbations
        # are additive or temporal, so no enabled randomization overwrites another.
        if cfg.randomize_height_yaw:
            sampled_terrain_heights = self._resample_heightmap_with_yaw()
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1)
                - base_ref
                - sampled_terrain_heights,
                -1.0,
                1.0,
            )
        else:
            heights = clean

        if cfg.randomize_height_offset:
            heights += self.height_episode_offset.unsqueeze(1)

        if cfg.add_height_noise:
            if cfg.add_height_gaussian_noise:
                heights += torch.randn_like(heights) * cfg.height_gaussian_noise
            if cfg.add_height_spike_noise:
                spike_mask = torch.rand_like(heights) < 0.05
                spike_amplitude = torch.empty(
                    self.num_envs, 1, device=self.device,
                ).uniform_(*cfg.height_spike_noise_range)
                heights += spike_mask * torch.randn_like(heights) * spike_amplitude

        if cfg.randomize_height_roll_pitch:
            heights += (
                self.height_pitch_bias.unsqueeze(1) * self._tilt_x_norm.unsqueeze(0)
                + self.height_roll_bias.unsqueeze(1) * self._tilt_y_norm.unsqueeze(0)
            )

        if cfg.height_repeat_probability > 0.0:
            repeat_mask = torch.rand(self.num_envs, device=self.device) < cfg.height_repeat_probability
            repeat_mask &= self.height_scan_initialized
            visible = torch.where(
                repeat_mask.unsqueeze(1), self.previous_height_scan, heights
            )
        else:
            visible = heights

        self.previous_height_scan.copy_(visible)
        self.height_scan_initialized[:] = True
        self.last_actor_heights.copy_(visible * self.obs_scales.height_measurements)

    def _resample_heightmap_with_yaw(self):
        yaw = self.height_yaw_noise
        yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        yaw_quat[:, 2] = torch.sin(yaw / 2.0)
        yaw_quat[:, 3] = torch.cos(yaw / 2.0)
        local_points = quat_apply_yaw(
            yaw_quat.repeat(1, self.num_height_points).reshape(-1, 4),
            self.height_points.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_height_points, 3)
        world_points = quat_apply_yaw(
            self.base_quat.repeat(1, self.num_height_points).reshape(-1, 4),
            local_points.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_height_points, 3)
        world_points += self.root_states[:, :3].unsqueeze(1)
        grid_points = (
            world_points[:, :, :2] + self.terrain.cfg.border_size
        ) / self.terrain.cfg.horizontal_scale
        px = grid_points[:, :, 0].reshape(-1)
        py = grid_points[:, :, 1].reshape(-1)
        rows, cols = self.height_samples.shape
        px0 = torch.clip(px.floor().long(), 0, rows - 2)
        py0 = torch.clip(py.floor().long(), 0, cols - 2)
        px1 = (px0 + 1).clamp(max=rows - 1)
        py1 = (py0 + 1).clamp(max=cols - 1)
        terrain_height = torch.min(
            torch.min(self.height_samples[px0, py0], self.height_samples[px0, py1]),
            torch.min(self.height_samples[px1, py0], self.height_samples[px1, py1]),
        )
        return terrain_height.reshape(self.num_envs, self.num_height_points) * self.terrain.cfg.vertical_scale

    def _draw_randomized_heightmap_vis(self):
        """Draw the height map exactly as the actor currently receives it."""
        if (
            not self._height_scan_enabled
            or not getattr(self.cfg.viewer, "draw_randomized_heightmap", False)
        ):
            return

        env_id = int(
            getattr(self.cfg.viewer, "randomized_heightmap_env_id", self.lookat_id)
        )
        env_id = min(max(env_id, 0), self.num_envs - 1)

        # This tensor already includes noise, yaw/tilt bias, and possible
        # repeated samples. Convert it from the policy normalization back to
        # the height convention used by the sensor.
        sensor_heights = (
            self.last_actor_heights[env_id] / self.obs_scales.height_measurements
        )
        base_ref = getattr(self.cfg.rewards, "base_height_target", 0.4)
        apparent_terrain_z = self.root_states[env_id, 2] - base_ref - sensor_heights
        world_points = quat_apply_yaw(
            self.base_quat[env_id].repeat(self.num_height_points, 1),
            self.height_points[env_id],
        ) + self.root_states[env_id, :3].unsqueeze(0)

        perceived_geom = gymutil.WireframeSphereGeometry(
            0.025, 4, 4, None, color=(1.0, 0.0, 0.0)
        )
        for point_idx in range(self.num_height_points):
            point = world_points[point_idx]
            pose = gymapi.Transform(
                gymapi.Vec3(
                    point[0].item(),
                    point[1].item(),
                    apparent_terrain_z[point_idx].item(),
                ),
                r=None,
            )
            gymutil.draw_lines(
                perceived_geom, self.gym, self.viewer, self.envs[env_id], pose
            )

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        if getattr(self.cfg.viewer, "draw_clean_heightmap", False):
            super()._draw_debug_vis()
        self._draw_randomized_heightmap_vis()

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
        if not hasattr(self, "_use_heading_mode"):
            self._use_heading_mode = self.cfg.commands.heading_command
        if self._use_heading_mode and not hasattr(self, "target_heading"):
            self.target_heading = torch.zeros(self.num_envs, device=self.device)
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        if self._use_heading_mode:
            self.target_heading[env_ids] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = 0.0  
        else:
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

        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.1).unsqueeze(1)

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.smoothed_commands[:, 4]
            phases = self.smoothed_commands[:, 5]
            offsets = self.smoothed_commands[:, 6]
            bounds = self.smoothed_commands[:, 7]
            durations = self.smoothed_commands[:, 8]

            zero_cmd_mask = (torch.norm(self.smoothed_commands[:, :2], dim=1) < 0.1) & \
                            (torch.abs(self.smoothed_commands[:, 2]) < 0.1)
            gait_increment = self.dt * frequencies * (~zero_cmd_mask).float()

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

            clock_vals = torch.stack([torch.sin(2 * np.pi * foot_indices[i]) for i in range(4)], dim=1)
            self.clock_inputs = clock_vals * (~zero_cmd_mask).unsqueeze(1).float()
            self.clock_inputs = clock_vals * (~zero_cmd_mask).unsqueeze(1).float()

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
                self.desired_contact_states[:, i] = torch.where(
                    zero_cmd_mask, torch.ones_like(desired_contact), desired_contact
                )

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

        # 【核心修改】：统一一阶低通滤波指令。让所有 12 维指令都具有平滑过渡（0.8 阻尼系数）
        self.smoothed_commands = 0.8 * self.smoothed_commands + 0.2 * self.commands

        self._step_contact_targets()
        if getattr(self, "_use_heading_mode", self.cfg.commands.heading_command):
            if not hasattr(self, "_use_heading_mode"):
                self._use_heading_mode = self.cfg.commands.heading_command
            if not hasattr(self, "target_heading"):
                self.target_heading = torch.zeros(self.num_envs, device=self.device)
            forward = quat_apply(self.base_quat, self.forward_vec)
            cur_heading = torch.atan2(forward[:, 1], forward[:, 0])
            yaw_max = self.command_ranges["ang_vel_yaw"][1]
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.target_heading - cur_heading), -yaw_max, yaw_max
            )

        # ------- 获取高程图 ------- #
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # ------- 获取基座相对地面的高度 ------- #
        if getattr(self.cfg.terrain, "use_support_plane_base_height", True):
            self.base_height = self._get_support_plane_base_height()
        else:
            # 保留原始算法：基座原点到全部高程图采样点的平均垂直距离。
            self.base_height = torch.mean(
                self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
                dim=1,
            )

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

        if getattr(self.cfg.terrain, "use_support_plane_base_height", True):
            # Rigid-body foot states are refreshed only after the next simulation
            # step. Use terrain below the reset base for this one support-plane frame.
            reset_ground_height = self._sample_terrain_height_at_xy(
                self.root_states[env_ids, :2],
            )
            self.base_height[env_ids] = (
                self.root_states[env_ids, 2] - reset_ground_height
            )

        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_rpy = get_euler_rpy_tensor(self.feet_quat)
        self.gait_indices[env_ids] = 0
        self.last_foot_z[env_ids] = 0.
        self.foot_height[env_ids] = 0.
        self.foot_swing_peak_height[env_ids] = 0.
        self.foot_swing_target_height[env_ids] = 0.
        self.last_foot_in_swing[env_ids] = False

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

        if self._height_scan_enabled:
            self.last_actor_heights[env_ids] = 0.0
            self.last_critic_heights[env_ids] = 0.0
            self.previous_height_scan[env_ids] = 0.0
            self.height_scan_initialized[env_ids] = False
            self._resample_height_episode_state(env_ids)

    # ==================================================================================================================== #
    # ================================================ Step ============================================================== #
    # ==================================================================================================================== #
    def step(self, actions):
        self.global_counter += 1

        actions = torch.clamp(actions, min=-self.cfg.normalization.clip_actions, max=self.cfg.normalization.clip_actions)
        self.actions = actions
        if self.cfg.control.action_smoothness:

            # ratio = self.cfg.control.ratio
            # self.actions = ratio * self.actions + (1 - ratio) * self.last_actions
            vxy_norm = torch.norm(self.smoothed_commands[:, :2], dim=1).unsqueeze(1)    # 速度越快，ratio 越趋近 1.0 (越不平滑，响应越快)
            dynamic_ratio = torch.clamp(0.85 + 0.1 * vxy_norm, 0.85, 0.95) 
            self.actions = dynamic_ratio * self.actions + (1 - dynamic_ratio) * self.last_actions

        self.render()
        self.pre_physics_step()

        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(actions).view(self.torques.shape)
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

            # Keep the height sensor on its own 10 Hz clock.  This counter is
            # advanced inside the 200 Hz physics loop, not in policy/obs code.
            if self._height_scan_enabled:
                self._height_sim_step_counter += 1
                if self._height_sim_step_counter >= self.height_update_interval_steps:
                    self._height_sim_step_counter = 0
                    self.gym.refresh_actor_root_state_tensor(self.sim)
                    self._update_height_scan_10hz()

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
            critic_heights = (
                self.last_critic_heights
                if self._height_scan_enabled
                else heights * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 critic_heights), dim=1)

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

        # Actor proprioception is built first.  The height scan is appended to
        # this single frame before the frame enters obs_history.
        proprio_obs = torch.cat((
            self.smoothed_commands * self.commands_scale,
            lagged_q,
            lagged_dq,
            self.actions,
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,
        ), dim=-1)
        
        if self.cfg.env.projected_gravity:
            proprio_obs = torch.cat((
                proprio_obs,
                self.lagged_projected_gravity * self.obs_scales.quat,
            ), dim=-1)
        else:
            proprio_obs = torch.cat((
                proprio_obs,
                self.lagged_base_euler_rpy[:, :2] * self.obs_scales.quat,
            ), dim=-1)
            
        # Add timing signals to actor obs
        if self.cfg.env.observe_timing_parameter:
            proprio_obs = torch.cat((proprio_obs, self.gait_indices.unsqueeze(1)), dim=-1)
        if self.cfg.env.observe_clock_inputs:
            proprio_obs = torch.cat((proprio_obs, self.clock_inputs), dim=-1)

        if proprio_obs.shape[1] != self._actor_proprio_obs_dim:
            raise RuntimeError(
                f"Actor proprioception has {proprio_obs.shape[1]} features, expected "
                f"{self._actor_proprio_obs_dim}"
            )

        if self._height_scan_enabled:
            # The cached 10 Hz scan is part of this 50 Hz observation frame.
            # It is deliberately duplicated in frames between sensor updates.
            obs_buf = torch.cat((proprio_obs, self.last_actor_heights), dim=-1)
        else:
            obs_buf = proprio_obs

        if obs_buf.shape[1] != self.cfg.env.num_single_obs:
            raise RuntimeError(
                f"Actor frame has {obs_buf.shape[1]} features, expected "
                f"{self.cfg.env.num_single_obs}"
            )
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
        if self.obs_buf.shape[1] != self.cfg.env.num_observations:
            raise RuntimeError(
                f"Actor history has {self.obs_buf.shape[1]} features, expected "
                f"{self.cfg.env.num_observations}"
            )
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def get_command(self):
        return self.smoothed_commands * self.commands_scale,

    def compute_privileged_observations(self, env_ids):
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights,  #0.5
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

        # heightmap 特权观测(必须与 compute_observations 一致，否则 termination_privileged_obs 维度对不上)
        if self.cfg.env.priv_observe_heightmap:
            critic_heights = self.last_critic_heights if self._height_scan_enabled else heights
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            critic_heights), dim=1)
                                            
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

    # ------------------------------------------------------------------------------#
    # --------------------------- base / regularization ---------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_base_acc(self):
        base_lin_acc = torch.norm(self.last_root_vel[:, 0:3] - self.root_states[:, 7:10], dim=1) / self.cfg.sim.dt
        base_ang_acc = torch.norm(self.last_root_vel[:, 3:6] - self.root_states[:, 10:13], dim=1) / self.cfg.sim.dt
        rew = base_lin_acc + 0.02 * base_ang_acc
        return rew

    # ------------------------------------------------------------------------------#
    # ------------------------- stand still rewards --------------------------------#
    # ------------------------------------------------------------------------------#
    def _get_stand_still_mask(self):
        """Return true where the commanded planar/yaw speed is near zero."""
        return torch.norm(self.smoothed_commands[:, :3], dim=1) < 0.1

    def _get_walking_mask(self):
        """Return true where gait-tracking rewards should be active."""
        return torch.norm(self.smoothed_commands[:, :3], dim=1) > 0.1

    def _sample_terrain_height_at_xy(self, xy):
        """Sample conservative terrain heights at world-frame XY points."""
        if self.cfg.terrain.mesh_type not in ("heightfield", "trimesh"):
            return torch.zeros(xy.shape[:-1], device=self.device)

        grid = (xy + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale
        px = grid[..., 0].long().reshape(-1)
        py = grid[..., 1].long().reshape(-1)

        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights = torch.min(
            self.height_samples[px, py],
            self.height_samples[px + 1, py],
        )
        heights = torch.min(heights, self.height_samples[px, py + 1])
        return heights.reshape(xy.shape[:-1]) * self.terrain.cfg.vertical_scale

    def _get_support_plane_base_height(self):
        """Return vertical base height relative to the local four-foot plane."""
        if self.cfg.terrain.mesh_type not in ("heightfield", "trimesh"):
            # Keep the flat-plane behavior exact and avoid contact-solver noise.
            num_feet = self.feet_positions.shape[1]
            self.terrain_ground_heights = torch.zeros(
                self.num_envs, num_feet, device=self.device,
            )
            self.foot_ground_heights = self.terrain_ground_heights
            self.support_plane_slopes = torch.zeros(
                self.num_envs, 2, device=self.device,
            )
            self.support_plane_height = torch.zeros(
                self.num_envs, device=self.device,
            )
            return self.root_states[:, 2]

        foot_xy = self.feet_positions[:, :, :2]
        terrain_ground_heights = self._sample_terrain_height_at_xy(foot_xy)

        # A stance foot gives a better measurement than a height-map cell at a
        # stair edge. Swing feet must still use the map, because their link
        # height is intentionally above the ground and would bias the plane.
        foot_surface_height = self.feet_positions[:, :, 2] - getattr(
            self.cfg.asset, "foot_radius", 0.045,
        )
        foot_contact_forces = self.contact_forces[:, self.feet_indices]
        vertical_contact = foot_contact_forces[:, :, 2] > 5.0
        side_contact = (
            torch.norm(foot_contact_forces[:, :, :2], dim=-1)
            > 5.0 * torch.abs(foot_contact_forces[:, :, 2])
        )
        contact_mask = vertical_contact & ~side_contact
        foot_ground_heights = torch.where(
            contact_mask, foot_surface_height, terrain_ground_heights,
        )

        # Fit z = ax + by + c after centering the four ground samples. Sampling
        # terrain under each foot keeps swing-foot clearance out of this estimate.
        relative_xy = foot_xy - self.root_states[:, None, :2]
        mean_xy = torch.mean(relative_xy, dim=1, keepdim=True)
        mean_height = torch.mean(foot_ground_heights, dim=1, keepdim=True)
        centered_xy = relative_xy - mean_xy
        centered_height = foot_ground_heights - mean_height

        dx = centered_xy[:, :, 0]
        dy = centered_xy[:, :, 1]
        covariance_xx = torch.sum(dx * dx, dim=1) + 1e-6
        covariance_xy = torch.sum(dx * dy, dim=1)
        covariance_yy = torch.sum(dy * dy, dim=1) + 1e-6
        height_covariance_x = torch.sum(dx * centered_height, dim=1)
        height_covariance_y = torch.sum(dy * centered_height, dim=1)
        determinant_raw = (
            covariance_xx * covariance_yy
            - covariance_xy * covariance_xy
        )
        # The nominal foot rectangle has a much larger determinant than this;
        # reject nearly collinear/folded configurations before extrapolating.
        plane_valid = determinant_raw > 1e-4
        determinant = determinant_raw.clamp_min(1e-8)
        slope_x = (
            covariance_yy * height_covariance_x
            - covariance_xy * height_covariance_y
        ) / determinant
        slope_y = (
            covariance_xx * height_covariance_y
            - covariance_xy * height_covariance_x
        ) / determinant
        slopes = torch.stack((slope_x, slope_y), dim=1)
        slopes = torch.where(
            plane_valid.unsqueeze(1), slopes, torch.zeros_like(slopes),
        )
        # Four feet can become nearly collinear during a fall. Limit the
        # extrapolated plane so one degenerate frame cannot create an enormous
        # height target or NaN reward.
        max_slope = getattr(self.cfg.terrain, "support_plane_max_slope", 2.0)
        slope_norm = torch.norm(slopes, dim=1, keepdim=True).clamp_min(1e-8)
        slopes = slopes * torch.clamp(max_slope / slope_norm, max=1.0)

        plane_height_at_base = mean_height.squeeze(1) - torch.sum(
            slopes * mean_xy.squeeze(1), dim=1,
        )
        # Keep these tensors available for diagnostics and terrain visualization.
        self.terrain_ground_heights = terrain_ground_heights
        self.foot_ground_heights = foot_ground_heights
        self.support_plane_slopes = slopes
        self.support_plane_height = plane_height_at_base
        # Keep the same vertical-z convention as body_height_cmd. The fitted
        # plane only removes the terrain/tilt bias; it must not change height
        # units to distance along the plane normal.
        return self.root_states[:, 2] - plane_height_at_base

    def _get_adaptive_swing_height_target(self):
        nominal = self.smoothed_commands[:, 9].unsqueeze(1)
        frequency = self.smoothed_commands[:, 4].clamp_min(0.1)
        swing_duration = (1.0 - self.smoothed_commands[:, 8]) / frequency

        takeoff_xy = self.foot_positions[:, :, :2]
        base_vel_xy_world = self.root_states[:, 7:9]
        landing_xy = takeoff_xy + (
            base_vel_xy_world.unsqueeze(1) * swing_duration[:, None, None]
        )

        alpha = torch.linspace(0.0, 1.0, 7, device=self.device).view(1, 1, 7, 1)
        path_xy = takeoff_xy.unsqueeze(2) + (
            landing_xy - takeoff_xy
        ).unsqueeze(2) * alpha

        path_heights = self._sample_terrain_height_at_xy(path_xy)
        takeoff_height = path_heights[:, :, 0]
        step_up = torch.clamp(
            torch.max(path_heights, dim=2).values - takeoff_height,
            min=0.0,
        )

        return torch.clamp(
            nominal + step_up + self.cfg.rewards.adaptive_clearance_margin,
            max=self.cfg.rewards.adaptive_clearance_max,
        )

    def _reward_stand_still(self):
        """Penalize all joint deviations from the default pose while standing."""
        joint_error_sq = torch.sum(
            torch.square(self.dof_pos - self.default_dof_pos), dim=1
        )
        return joint_error_sq * self._get_stand_still_mask()

    def _reward_stand_base_vel_penality(self):
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * self._get_stand_still_mask()

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
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.smoothed_commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_default_joint_pos(self):
        rew = torch.norm(self.leg_pos, dim=1)
        if self.reward_scales["default_joint_pos"] < 0:
            return rew
        else:
            return torch.exp(-20 * rew)

    def _reward_default_hip_pos(self):
        """Strongly keep hip roll and softly regularize hip pitch at default pose."""
        if not hasattr(self, "_hip_roll_pitch_dof_indices"):
            hip_roll_names = (
                "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            )
            hip_pitch_names = (
                "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            )
            try:
                roll_indices = [self.dof_names_to_idx[name] for name in hip_roll_names]
                pitch_indices = [self.dof_names_to_idx[name] for name in hip_pitch_names]
            except KeyError as error:
                raise RuntimeError(
                    f"Missing hip roll/pitch joint in asset DOF names: {error.args[0]}"
                ) from error
            self._hip_roll_pitch_dof_indices = (
                torch.tensor(roll_indices, dtype=torch.long, device=self.device),
                torch.tensor(pitch_indices, dtype=torch.long, device=self.device),
            )

        roll_indices, pitch_indices = self._hip_roll_pitch_dof_indices
        hip_roll_error = (
            self.dof_pos[:, roll_indices]
            - self.default_dof_pos[:, roll_indices]
        )
        hip_pitch_error = (
            self.dof_pos[:, pitch_indices]
            - self.default_dof_pos[:, pitch_indices]
        )
        hip_roll_cost = torch.sum(torch.square(hip_roll_error), dim=1)
        hip_pitch_cost = torch.sum(torch.square(hip_pitch_error), dim=1)
        return hip_roll_cost + 0.5 * hip_pitch_cost

    # ------------------------------------------------------------------------------#
    # ---------------------- common regularization rewards -------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

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
        base_height = self.base_height  
        target_height = self.smoothed_commands[:, 3]
        return torch.abs(base_height - target_height)

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

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_pre_motor_torque_limits(self):
        """Penalize raw PD commands as they approach the pre-efficiency clips."""
        if not getattr(self.cfg.control, "enable_pre_motor_torque_clip", True):
            return torch.zeros(self.num_envs, device=self.device)

        hip_indices = self.cfg.control.torque_vel_hip_indices
        calf_indices = self.cfg.control.torque_vel_calf_indices
        soft_ratio = self.cfg.rewards.pre_motor_torque_soft_ratio
        hip_soft_limit = self.cfg.control.pre_torque_vel_clip_hip * soft_ratio
        calf_soft_limit = self.cfg.control.pre_torque_vel_clip_calf * soft_ratio
        hip_excess = (torch.abs(self.torques_cmd[:, hip_indices]) - hip_soft_limit).clip(min=0.)
        calf_excess = (torch.abs(self.torques_cmd[:, calf_indices]) - calf_soft_limit).clip(min=0.)
        return torch.sum(hip_excess, dim=1) + torch.sum(calf_excess, dim=1)

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


    # def _reward_feet_clearance(self):
    #     """Penalize a swing only when its peak clearance is below the target."""
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
    #     foot_z = self.rigid_state[:, self.feet_indices, 2] - 0.04
    #     self.foot_height += foot_z - self.last_foot_z
    #     self.last_foot_z = foot_z

    #     # Use the gait state rather than physical contact so an early touchdown
    #     # still concludes at the scheduled end of the swing.
    #     foot_in_swing = self.desired_contact_states < 0.5
    #     swing_started = foot_in_swing & ~self.last_foot_in_swing
    #     swing_ended = ~foot_in_swing & self.last_foot_in_swing

    #     target_height = self.smoothed_commands[:, 9].unsqueeze(1)
    #     self.foot_swing_peak_height[swing_started] = 0.0
    #     self.foot_swing_target_height[swing_started] = target_height.expand_as(
    #         self.foot_swing_target_height
    #     )[swing_started]

    #     current_peak = torch.maximum(self.foot_swing_peak_height, self.foot_height)
    #     self.foot_swing_peak_height[foot_in_swing] = current_peak[foot_in_swing]

    #     # Only a shortfall is penalized; exceeding the target peak is free.
    #     height_shortfall = torch.clamp(
    #         self.foot_swing_target_height - self.foot_swing_peak_height, min=0.0
    #     )
    #     swing_cost = 1.0 - torch.exp(-torch.square(height_shortfall) / 0.01)
    #     swing_cost *= swing_ended

    #     # The next swing starts from the actual touchdown height.
    #     self.foot_height[contact] = 0.0
    #     self.last_foot_in_swing = foot_in_swing

    #     return torch.sum(swing_cost, dim=1) * self._get_walking_mask()


    def _reward_feet_clearance(self):
        """Track a terrain-adaptive swing-foot peak-clearance target."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_z = self.rigid_state[:, self.feet_indices, 2] - 0.04
        self.foot_height += foot_z - self.last_foot_z
        self.last_foot_z = foot_z

        foot_in_swing = self.desired_contact_states < 0.5
        swing_started = foot_in_swing & ~self.last_foot_in_swing
        swing_ended = ~foot_in_swing & self.last_foot_in_swing

        adaptive_target = self._get_adaptive_swing_height_target()

        # Lock the target for this swing so it remains stable through the trajectory.
        self.foot_swing_peak_height[swing_started] = 0.0
        self.foot_swing_target_height[swing_started] = adaptive_target[swing_started]

        current_peak = torch.maximum(
            self.foot_swing_peak_height, self.foot_height
        )
        self.foot_swing_peak_height[foot_in_swing] = current_peak[foot_in_swing]

        # Strongly penalize insufficient clearance. Penalize excess height only
        # after a small tolerance, so flat terrain does not learn maximum lifting.
        height_shortfall = torch.clamp(
            self.foot_swing_target_height - self.foot_swing_peak_height,
            min=0.0,
        )
        overheight_tolerance = getattr(
            self.cfg.rewards, "clearance_overheight_tolerance", 0.03
        )
        height_excess = torch.clamp(
            self.foot_swing_peak_height
            - self.foot_swing_target_height
            - overheight_tolerance,
            min=0.0,
        )

        shortfall_cost = 1.0 - torch.exp(-torch.square(height_shortfall) / 0.01)
        excess_cost = 1.0 - torch.exp(-torch.square(height_excess) / 0.01)
        swing_cost = (shortfall_cost + 0.15 * excess_cost) * swing_ended

        # Inflate nearby terrain by a small horizontal radius. This gives a
        # differentiable-in-time penalty when a swing foot enters a step edge,
        # instead of waiting for the end-of-swing peak-height evaluation.
        virtual_surface_xy = (
            self.foot_positions[:, :, :2].unsqueeze(2)
            + self.virtual_collision_offsets.view(1, 1, -1, 2)
        )
        virtual_surface_height = torch.max(
            self._sample_terrain_height_at_xy(virtual_surface_xy), dim=2
        ).values
        clearance_deficit = torch.clamp(
            virtual_surface_height
            + self.cfg.rewards.virtual_collision_clearance
            - foot_z,
            min=0.0,
        )
        collision_cost = 1.0 - torch.exp(
            -torch.square(clearance_deficit)
            / (self.cfg.rewards.virtual_collision_sigma ** 2)
        )
        collision_cost *= foot_in_swing

        self.foot_height[contact] = 0.0
        self.last_foot_in_swing = foot_in_swing

        return torch.sum(
            swing_cost
            + self.cfg.rewards.virtual_collision_weight * collision_cost,
            dim=1,
        ) * self._get_walking_mask()

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_foot_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
