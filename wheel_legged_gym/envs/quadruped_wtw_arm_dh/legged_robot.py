# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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


import os
import numpy as np
from time import *
import time as ti

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque
import random

import torch, torchvision
import sys
import itertools
import cv2

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from .base_task import BaseTask
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from wheel_legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.terrain import  Terrain

# from wheel_legged_gym.sensors.warp.warp_sensor import WarpSensor

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_rpy_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_rpy = torch.stack((r, p, w), dim=1)
    euler_rpy[euler_rpy > np.pi] -= 2 * np.pi
    return euler_rpy

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.sim_params = sim_params
        self.height_samples = None
        # for debug
        self.debug_viz = self.cfg.viewer.debug_viz  # 显示scan dots
        self.lookat_id = 0
        self.init_done = False
        self._parse_cfg(self.cfg)

        self.num_lower_dof = getattr(self.cfg.env, 'num_lower_dof', self.cfg.env.num_actions)
        
        super().__init__(self.cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        if 'P3O' in self.train_cfg.runner_class_name:
            self._prepare_cost_function()
        self.init_done = True

        self.global_counter = 0

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[0], self.cfg.depth.resized[1]),
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    
    def _call_train_eval(self, func, env_ids):

        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None: ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

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

        actions = torch.clamp(actions, min=-self.cfg.normalization.clip_actions, max=self.cfg.normalization.clip_actions)
        self.actions = actions

        if self.cfg.control.action_smoothness:
            ratio = self.cfg.control.ratio
            self.actions = ratio * self.actions + (1 - ratio) * self.last_actions

        self.render()
        
        self.pre_physics_step()
           
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(actions).view(self.torques.shape)
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

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        # self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                                self.envs[i],
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            # if(i == 0): print(torch.mean(depth_image)) # for debug, sometimes isaacgym will return all -inf depth image if not config properly

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                                 dim=0)
        self.gym.end_access_image_tensors(self.sim)



    def compute_dof_vel(self):
        """" 采用位置差速得到 dof_vel """
        self.last_dof_vel_200hz[:] = self.dof_vel[:] # 新版 0818
        diff = (
            torch.remainder(self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi)
            - self.pi
        )
        self.dof_pos_dot = diff / self.sim_params.dt
        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot
        self.last_dof_pos[:] = self.dof_pos[:]      
        self.dof_acc_200hz = (self.dof_vel - self.last_dof_vel_200hz) / self.dt # 新版 0818

    def reset(self):
        """ Reset all robots """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if 'HIM' in self.train_cfg.runner_class_name:
            obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        elif 'P3O' in self.train_cfg.runner_class_name:
            obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        else:
            obs, privileged_obs, _, _, _ = self.step(torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def _compute_feet_states(self):
        """ update feet positions and velocities """
        self.feet_positions = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        # self.feet_velocities = (self.feet_positions - self.last_feet_positions) / self.dt
        self.feet_velocities = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

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
        
    def _draw_base_com_vis(self):
        debug_base_com = self.base_com + self.root_states[:, :3]
        
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = debug_base_com.cpu().numpy()
            x = base_pos[i, 0]
            y = base_pos[i, 1]
            z = base_pos[i, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
    
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
        # update action curriculum for specific dofs
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

        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
            
        if self.cfg.env.action_curriculum:
            self.random_upper_actions[env_ids] = 0.
            self.current_upper_actions[env_ids] = 0.
            self.delta_upper_actions[env_ids] = 0.


    # agibot
    def randomize_lag_props(self,env_ids): 
        """ random add lag """
        if self.cfg.domain_rand.add_action_lag:   
            self.lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0], 
                                                           self.cfg.domain_rand.lag_timesteps_range[1]+1,(len(env_ids),),device=self.device) 
                if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                    self.last_lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]
            else:
                self.lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]

        if self.cfg.domain_rand.add_dof_lag:
            dof_lag_buffer_init = self.default_dof_pos.unsqueeze(1).expand(-1, self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, -1).clone()
            dof_lag_buffer_init = dof_lag_buffer_init.transpose(1, 2)
            self.dof_lag_buffer[env_ids, :, :] = dof_lag_buffer_init[env_ids,:, :]
            self.dof_vel_lag_buffer[env_ids, :, :] = 0.0

            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                self.dof_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep[env_ids] = self.cfg.domain_rand.dof_lag_timesteps_range[1]
            else:
                self.dof_lag_timestep[env_ids] = self.cfg.domain_rand.dof_lag_timesteps_range[1]

        if self.cfg.domain_rand.add_imu_lag:                
            self.imu_lag_buffer[env_ids, :, :] = 0.0   
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep[env_ids] = self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep[env_ids] = self.cfg.domain_rand.imu_lag_timesteps_range[1]

        if 'Map' in self.train_cfg.runner_class_name:
            if self.cfg.domain_rand.add_height_lag:
                self.height_lag_buffer[env_ids, :, :] = 0.0
                if self.cfg.domain_rand.randomize_height_lag_timesteps:
                    self.height_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.height_lag_timesteps_range[0],
                                                            self.cfg.domain_rand.height_lag_timesteps_range[1]+1, (len(env_ids),),device=self.device)
                    if self.cfg.domain_rand.randomize_height_lag_timesteps_perstep:
                        self.last_height_lag_timestep[env_ids] = self.cfg.domain_rand.height_lag_timesteps_range[1]
                else:
                    self.height_lag_timestep[env_ids] = self.cfg.domain_rand.height_lag_timesteps_range[1]

        if hasattr(self, "volume_sample_points_vel"):
            self.volume_sample_points_vel[env_ids] = 0.

    # NOTE: cost
    def compute_cost(self):
        self.cost_buf[:] = 0.

        for i in range(len(self.cost_functions)):
            name = self.cost_names[i]
            cost = self.cost_functions[i]() * self.cost_scales[name]
            self.cost_buf[:] += cost
            self.cost_episode_sums[name] += cost
  
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly

        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode

        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _push_robots(self, env_ids):
        """Random pushes the robots."""
        if len(env_ids) == 0:
            return
        # print(' Pushing robots ')
        max_push_force = (
            self.body_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.cfg.sim.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )
        self.rigid_body_external_forces[env_ids, 0, 0:3] = quat_rotate(
            self.base_quat[env_ids], rigid_body_external_forces[env_ids]
        )
        self.rigid_body_external_forces[env_ids, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

# random_function
# ============================================================
    def _randomize_dof_props(self, env_ids):
        if len(env_ids) == 0:
            return
        # print('randomizing dof props ')
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if self.cfg.domain_rand.randomize_PD_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

        # rand joint friction set in sim
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_friction_range'
                    friction_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_friction_coeffs[env_ids, i] = torch_rand_float(friction_range[0], friction_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:                      
                joint_friction_range = self.cfg.domain_rand.joint_friction_range
                self.joint_friction_coeffs[env_ids] = torch_rand_float(joint_friction_range[0], joint_friction_range[1], (len(env_ids), 1), device=self.device)

        # rand joint damping set in sim
        if self.cfg.domain_rand.randomize_joint_damping:
            if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_damping_range'
                    damping_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_damping_coeffs[env_ids, i] = torch_rand_float(damping_range[0], damping_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_damping_range = self.cfg.domain_rand.joint_damping_range
                self.joint_damping_coeffs[env_ids] = torch_rand_float(joint_damping_range[0], joint_damping_range[1], (len(env_ids), 1), device=self.device)
        
        # rand joint armature inertia set in sim
        if self.cfg.domain_rand.randomize_joint_armature:
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                for i in range(self.num_dofs):
                    range_key = f'joint_{i+1}_armature_range'
                    armature_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_armatures_coeffs[env_ids, i] = torch_rand_float(armature_range[0], armature_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_armature_range = self.cfg.domain_rand.joint_armature_range
                self.joint_armatures_coeffs[env_ids] = torch_rand_float(joint_armature_range[0], joint_armature_range[1], (len(env_ids), 1), device=self.device)

    def _randomize_rigid_body_props(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = self.cfg.domain_rand.added_mass_range
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
        
        if self.cfg.domain_rand.randomize_inertia:
            min_inertia, max_inertia = self.cfg.domain_rand.randomize_inertia_range
            self.inertia_scale[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_inertia - min_inertia) + min_inertia
        
        if self.cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = self.cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3 * self.num_bodies, dtype=torch.float, device=self.device,requires_grad=False) * (max_com_displacement - min_com_displacement) + min_com_displacement

        if self.cfg.domain_rand.randomize_friction:
            min_friction, max_friction = self.cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 2, dtype=torch.float, device=self.device,requires_grad=False) * (max_friction - min_friction) + min_friction

        if self.cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = self.cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props
    

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            print('self.cfg.domain_rand.default_joint_friction',self.cfg.domain_rand.default_joint_friction)
            print('self.cfg.domain_rand.default_joint_damping',self.cfg.domain_rand.default_joint_damping)
            print('self.cfg.domain_rand.default_joint_armature',self.cfg.domain_rand.default_joint_armature)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                props["friction"][i] = self.cfg.domain_rand.default_joint_friction[i]
                props["damping"][i] = self.cfg.domain_rand.default_joint_damping[i]
                props["armature"][i] = self.cfg.domain_rand.default_joint_armature[i]

                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props
    
    def _process_rigid_body_props(self, props, env_id):
        """" 质量随机化 """
        self.init_body_mass[env_id] = props[0].mass
        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass += self.payloads[env_id]  # 加负载

        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale =  np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        self.body_mass[env_id] = props[0].mass

        """" 质心随机化 """
        self.init_base_com[env_id, :] = torch.tensor([props[0].com.x, props[0].com.y, props[0].com.z])
        if self.cfg.domain_rand.randomize_com_displacement:
            current_com = props[0].com
            new_com_x = current_com.x + self.com_displacements[env_id,0].item()
            new_com_y = current_com.y + self.com_displacements[env_id,1].item() * 0.2
            new_com_z = current_com.z + self.com_displacements[env_id,2].item() 
            props[0].com = gymapi.Vec3(new_com_x,new_com_y,new_com_z)
        self.base_com[env_id, :] = torch.tensor([props[0].com.x, props[0].com.y, props[0].com.z])
    
        return props

    def refresh_actor_rigid_body_props(self, env_ids):
        if len(env_ids) == 0:
            return
        """  质量随机化 """
        for env_id in env_ids:
            rigid_body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], 0)
            if self.cfg.domain_rand.randomize_base_mass:
                rigid_body_props[0].mass = self.init_body_mass[env_id] + self.payloads[env_id]
            if self.cfg.domain_rand.randomize_inertia:
                rigid_body_props[0].mass *= self.inertia_scale[env_id]
                rigid_body_props[0].inertia.x.x *= self.inertia_scale[env_id]
                rigid_body_props[0].inertia.y.y *= self.inertia_scale[env_id]
                rigid_body_props[0].inertia.z.z *= self.inertia_scale[env_id]

            """" 质心随机化 """
            if self.cfg.domain_rand.randomize_com_displacement:
                current_com = self.init_base_com[env_id]
                new_com_x = current_com[0] + self.com_displacements[env_id,0].item()
                new_com_y = current_com[1] + self.com_displacements[env_id,1].item() * 0.5
                new_com_z = current_com[2] + self.com_displacements[env_id,2].item() 
                rigid_body_props[0].com = gymapi.Vec3(new_com_x,new_com_y,new_com_z)
            self.base_com[env_id, :] = torch.tensor([rigid_body_props[0].com.x, rigid_body_props[0].com.y, rigid_body_props[0].com.z])

            self.gym.set_actor_rigid_body_properties(self.envs[env_id], 0, rigid_body_props)

    def refresh_actor_rigid_shape_props(self, env_ids):
        if len(env_ids) == 0:
            return
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)
            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _teleport_robots(self, env_ids):
        """ Teleports any robots that are too close to the edge to the other side
        """
        if self.cfg.terrain.teleport_robots:
            thresh = self.cfg.terrain.teleport_thresh

            x_offset = int(self.cfg.terrain.x_offset * self.cfg.terrain.horizontal_scale)

            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += self.cfg.terrain.terrain_length * (self.cfg.terrain.num_rows - 1)

            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > self.cfg.terrain.terrain_length * self.cfg.terrain.num_rows - thresh + x_offset]
            self.root_states[high_x_ids, 0] -= self.cfg.terrain.terrain_length * (self.cfg.terrain.num_rows - 1)

            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += self.cfg.terrain.terrain_width * (self.cfg.terrain.num_cols - 1)

            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > self.cfg.terrain.terrain_width * self.cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= self.cfg.terrain.terrain_width * (self.cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

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

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        # """
        if len(env_ids) == 0:
            return
        # print(' resample commands ')
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0] , self.command_ranges["lin_vel_y"][1] , (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # ----------- resample command based on limx dynamic ------------ #
        # set 1/3 of resample to go straight, set ang_vel = 0
        resample_nums = len(env_ids)
        env_list = list(range(resample_nums))
        half_env_list = random.sample(env_list, resample_nums // 3)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[env_ids[half_env_list]], \
                                 self.forward_vec[env_ids[half_env_list]])
            heading = torch.atan2(forward[:,1], forward[:,0])
        else:
            self.commands[env_ids[half_env_list], 1] = 0.0

        # # set 1/3 of resample to be stand still
        # rest_env_list = list(set(env_list) - set(half_env_list))
        # zero_cmd_env_idx_ = random.sample(rest_env_list, resample_nums // 3)

        # self.commands[env_ids[zero_cmd_env_idx_], 0] = 0.0
        # self.commands[env_ids[zero_cmd_env_idx_], 1] = 0.0
        # if self.cfg.commands.heading_command:
        #     forward = quat_apply(self.base_quat[env_ids[zero_cmd_env_idx_]], \
        #                          self.forward_vec[env_ids[zero_cmd_env_idx_]])
        #     heading = torch.atan2(forward[:,1], forward[:,0])
        #     self.commands[env_ids[zero_cmd_env_idx_], 3] = heading

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)

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
            self.lagged_actions = self.lag_buffer[torch.arange(self.num_envs),:,self.lag_timestep.long()]
        else:
            self.lagged_actions = actions

        if self.cfg.control.control_type == "M":
            # Compute position reference from actions
            pos_ref = self.lagged_actions * self.cfg.control.action_scale

            if not self.cfg.mode.use_net:
                pos_ref = self.ref_dof_pos

            self.joint_pos_target = pos_ref + self.default_dof_pos

            if self.cfg.domain_rand.randomize_PD_factor:
                p_gains = self.Kp_factors * self.p_gains
                d_gains = self.Kd_factors * self.d_gains
            else:
                p_gains = self.p_gains
                d_gains = self.d_gains

            # PD torque for ALL joints
            torques = p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos + self.motor_offsets) - d_gains * self.Kd_factors * self.dof_vel
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

            if self.cfg.domain_rand.randomize_motor_strength:
                torques *= self.motor_strengths

            # Lower body: PD torque control (force tensor)
            # Upper body: direct position target (position target tensor)
            return torch.cat((torques[..., :self.num_lower_dof], self.joint_pos_target[..., self.num_lower_dof:]), dim=-1)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.raw_default_dof_pos
        self.dof_vel[env_ids] = 0.

        if self.cfg.init_state.rand_init_dof:
            self.dof_pos[env_ids] += torch_rand_float(-self.cfg.init_state.rand_init_dof_range,
                                                      self.cfg.init_state.rand_init_dof_range, (len(env_ids), self.num_dof), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0:1] += torch_rand_float(-self.cfg.terrain.x_init_range,
                                                               self.cfg.terrain.x_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[env_ids, 1:2] += torch_rand_float(-self.cfg.terrain.y_init_range,
                                                               self.cfg.terrain.y_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[env_ids, 0] += self.cfg.terrain.x_init_offset
            self.root_states[env_ids, 1] += self.cfg.terrain.y_init_offset
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]        
        
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        
        # base yaws
        init_yaws = torch_rand_float(-self.cfg.terrain.yaw_init_range,
                                     self.cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[env_ids, 3:7] = quat

        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += self.cfg.asset.fix_base_link_height

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_action_curriculum(self, env_ids):
        """Implements a curriculum of increasing upper body perturbation.
        
        If tracking reward exceeds 80% of maximum, increase perturbation ratio.
        """
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.action_curriculum_ratio += 0.05
            self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        # move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        move_down = (
            self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s
            < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.4
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., self.cfg.commands.max_curriculum)

    def _init_custom_buffers__(self):
        self.friction_coeffs = 1 * torch.ones(self.num_envs, 2, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = 1 * torch.ones(self.num_envs, 2, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.inertia_scale = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3 * self.num_bodies , dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False) 
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))      
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                self.joint_friction_coeffs = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)
            else:
                self.joint_friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        if self.cfg.domain_rand.randomize_joint_damping:    
            if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                self.joint_damping_coeffs = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)
            else:
                self.joint_damping_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        if self.cfg.domain_rand.randomize_joint_armature:      
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                self.joint_armatures_coeffs = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)  
            else:
                self.joint_armatures_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)

        self.default_joint_friction = self.cfg.domain_rand.default_joint_friction
        self.default_joint_damping = self.cfg.domain_rand.default_joint_damping
        self.default_joint_armature = self.cfg.domain_rand.default_joint_armature

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.last_dof_pos = torch.zeros_like(self.dof_pos)

        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.power = torch.zeros_like(self.dof_vel)
        self.dof_acc_50hz = torch.zeros_like(self.dof_vel)
        self.dof_acc_200hz = torch.zeros_like(self.dof_vel)
        self.last_dof_vel_50hz = torch.zeros_like(self.dof_vel)
        self.last_dof_vel_200hz = torch.zeros_like(self.dof_vel)

        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.base_pos_init = torch.zeros_like(self.base_pos)
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_rpy = get_euler_rpy_tensor(self.base_quat)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        
        self.feet_positions = self.rigid_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.last_feet_positions = torch.zeros_like(self.feet_positions)
        self.feet_heights = torch.zeros_like(self.feet_positions)
        self.feet_velocities = torch.zeros_like(self.feet_positions)
        self.feet_velocities_f = torch.zeros_like(self.feet_positions)
        self.feet_relative_velocities = torch.zeros_like(self.feet_velocities)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        self.fail_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.envs_steps_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.forward_torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.action_plot = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_acc = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dq = torch.zeros_like(self.dof_vel)
        self.acc_peak_penalty= torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_acc = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.ang_vel, self.obs_scales.height_measurements], device=self.device, requires_grad=False,) 
        
        # feet
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                        device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.last_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                device=self.device, requires_grad=False)
        self.current_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                   device=self.device, requires_grad=False)
        
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
                    if not found:
                        self.p_gains[:, i] = 0.
                        self.d_gains[:, i] = 0.
                        print(f"PD gain of joint {name} were not defined, setting them to zero")
        print('PD controller stiffness', self.p_gains[0, :])
        print('PD controller damping', self.d_gains[0, :])
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        if self.cfg.domain_rand.add_action_lag:   
            self.lag_buffer = torch.zeros(self.num_envs, self.num_dof, self.cfg.domain_rand.lag_timesteps_range[1] + 1, device=self.device)
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_timestep = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0],
                                                  self.cfg.domain_rand.lag_timesteps_range[1]+1,(self.num_envs,),device=self.device) 
                if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                    self.last_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.lag_timesteps_range[1]
            else:
                self.lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.lag_timesteps_range[1]

        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buffer = self.default_dof_pos.unsqueeze(1).expand(-1, self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, -1).clone()
            self.dof_lag_buffer = self.dof_lag_buffer.transpose(1, 2)
            self.dof_vel_lag_buffer = torch.zeros(self.num_envs,self.num_dof,self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,device=self.device)

            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.dof_lag_timesteps_range[1]

            else:
                self.dof_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.dof_lag_timesteps_range[1]

        if self.cfg.domain_rand.add_imu_lag:
            self.imu_lag_buffer = torch.zeros(self.num_envs, 6, self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,device=self.device)  #self.num_actions=6
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.imu_lag_timesteps_range[1]
        
        if 'Map' in self.train_cfg.runner_class_name:
            if self.cfg.domain_rand.add_height_lag:
                self.height_lag_buffer = torch.zeros(self.num_envs, self.cfg.env.num_height_maps, self.cfg.domain_rand.height_lag_timesteps_range[1]+1,device=self.device)
                if self.cfg.domain_rand.randomize_height_lag_timesteps:
                    self.height_lag_timestep = torch.randint(self.cfg.domain_rand.height_lag_timesteps_range[0],
                                                            self.cfg.domain_rand.height_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                    if self.cfg.domain_rand.randomize_height_lag_timesteps_perstep:
                        self.last_height_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.height_lag_timesteps_range[1]
                else:
                    self.height_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.height_lag_timesteps_range[1]

        if hasattr(self.cfg.sim, "body_measure_points"):
            self._init_body_volume_points()
            self._init_volume_sample_points()
            print("Total number of volume estimation points for each robot is:", self.volume_sample_points.shape[1])

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state) # (num_envs*num_bodies, 13)

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)

        # dynamic sigma for tracking rewards based on terrain difficulty
        if hasattr(self.cfg.rewards, 'dynamic_sigma') and self.cfg.rewards.dynamic_sigma:
            self.dynamic_sigma_cfg = self.cfg.rewards.dynamic_sigma
            self.terrain_max_sigmas = torch.tensor(
                self.dynamic_sigma_cfg["max_sigma"], device=self.device, requires_grad=False
            )

        # --- Upper body curriculum buffers ---
        self.upper_num_dof = self.num_dof - self.num_lower_dof  # 6

        # action space limits (in action space, not joint space)
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale

        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.random_upper_actions = torch.zeros((self.num_envs, self.upper_num_dof), device=self.device)
        self.current_upper_actions = torch.zeros((self.num_envs, self.upper_num_dof), device=self.device)
        self.delta_upper_actions = torch.zeros((self.num_envs, self.upper_num_dof), device=self.device)

        # convert upper_interval from seconds to policy steps
        self.cfg.domain_rand.upper_interval = int(np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt))

    def _init_body_volume_points(self):
        """ Generate a series of points grid so that they can be bind to those rigid bodies
        By 'those rigid bodies', we mean the rigid bodies that are specified in the `body_measure_points`
        """
        # read and specify the order of which body to sample from and its relative sample points.
        self.body_measure_name_order = [] # order specified
        self.body_sample_indices = [] # index in environment domain
        # NOTE: assuming all envs have the same number of actors and rigid bodies
        rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        for name, measure_name in itertools.product(rigid_body_names, self.cfg.sim.body_measure_points.keys()):
            if measure_name in name:
                self.body_sample_indices.append(
                    self.gym.find_actor_rigid_body_index(
                        self.envs[0],
                        self.actor_handles[0],
                        name,
                        gymapi.IndexDomain.DOMAIN_ENV,
                ))
                self.body_measure_name_order.append(measure_name) # order specified
        self.body_sample_indices = torch.tensor(self.body_sample_indices, device= self.sim_device).flatten() # n_bodies (each env)

        # compute and store each sample points in body frame.
        self.body_volume_points = dict()
        for measure_name, points_cfg in self.cfg.sim.body_measure_points.items():
            x = torch.tensor(points_cfg["x"], device= self.device, dtype= torch.float32, requires_grad= False)
            y = torch.tensor(points_cfg["y"], device= self.device, dtype= torch.float32, requires_grad= False)
            z = torch.tensor(points_cfg["z"], device= self.device, dtype= torch.float32, requires_grad= False)
            t = torch.tensor(points_cfg["transform"][0:3], device= self.device, dtype= torch.float32, requires_grad= False)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid_points = torch.stack([
                grid_x.flatten(),
                grid_y.flatten(),
                grid_z.flatten(),
            ], dim= -1) # n_points, 3
            if "points" in points_cfg.keys():
                # additional unstructured points
                unstructured_points = torch.tensor(points_cfg["points"], device= self.device, dtype= torch.float32)
                assert len(unstructured_points.shape) == 2 and unstructured_points.shape[1] == 3, "Unstructured points must be a list of 3D points"
                grid_points = torch.cat([
                    grid_points,
                    unstructured_points,
                ], dim= 0)
            q = quat_from_euler_xyz(
                torch.tensor(points_cfg["transform"][3], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][4], device= self.sim_device, dtype= torch.float32),
                torch.tensor(points_cfg["transform"][5], device= self.sim_device, dtype= torch.float32),
            )
            self.body_volume_points[measure_name] = tf_apply(
                q.expand(grid_points.shape[0], 4),
                t.expand(grid_points.shape[0], 3),
                grid_points,
            )

    def _init_volume_sample_points(self):
        """ Build sample points for penetration volume estimation
        NOTE: self.cfg.sim.body_measure_points must be a dict with
            key: body name (or part of the body name) to estimate
            value: dict(
                x, y, z: sample points to form a meshgrid
                transform: [x, y, z, roll, pitch, yaw] for transforming the meshgrid w.r.t body frame
            )
        """
        num_sample_points_per_env = 0
        for body_name in self.body_measure_name_order:
            for measure_name in self.body_volume_points.keys():
                if measure_name in body_name:
                    num_sample_points_per_env += self.body_volume_points[measure_name].shape[0]
        self.volume_sample_points = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )
        self.volume_sample_points_vel = torch.zeros(
            (self.num_envs, num_sample_points_per_env, 3),
            device= self.device,
            dtype= torch.float32,    
        )

    def refresh_volume_sample_points(self):
        """ NOTE: call this method whenever you need to access `volume_sample_points` and `volume_sample_points_vel`,
        Don't worry about repeated calls, it will only compute once in each env.step() call.
        """
        if self.volume_sample_points_refreshed:
            # use `volume_sample_points_refreshed` to avoid repeated computation
            return
        sample_points_start_idx = 0
        for body_idx, body_measure_name in enumerate(self.body_measure_name_order):
            volume_points = self.body_volume_points[body_measure_name] # (n_points, 3)
            num_volume_points = volume_points.shape[0]
            rigid_body_index = self.body_sample_indices[body_idx:body_idx+1] # size: torch.Size([1])
            point_positions_w, point_velocities_w = self._get_target_pos_vel(
                rigid_body_index.expand(num_volume_points,),
                volume_points,
                domain= gymapi.DOMAIN_ENV,
            )
            self.volume_sample_points_vel[
                :,
                sample_points_start_idx: sample_points_start_idx + num_volume_points,
            ] = point_velocities_w
            self.volume_sample_points[
                :,
                sample_points_start_idx: sample_points_start_idx + num_volume_points,
            ] = point_positions_w
            sample_points_start_idx += num_volume_points
        self.volume_sample_points_refreshed = True

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _prepare_cost_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.cost_scales.keys()):
            scale = self.cost_scales[key]
            if scale == 0:
                self.cost_scales.pop(key) 
            else:
                self.cost_scales[key] *= self.dt

        self.cost_functions = []
        self.cost_names = []

        for name, scale in self.cost_scales.items():
            self.cost_names.append(name)
            name = '_cost_' + name
            self.cost_functions.append(getattr(self, name))
        self.cost_episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                  for name in self.cost_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[1]
            camera_props.height = self.cfg.depth.original[0]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_y_angle = np.random.uniform(config.y_angle[0], config.y_angle[1])

            camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
            camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])


            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_x_angle),
                                                           np.radians(camera_y_angle), np.radians(camera_z_angle))
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_joints = self.gym.get_asset_joint_count(robot_asset)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        print('body_names_to_idx: {}'.format(sorted(list(self.body_names_to_idx.items()), key=lambda x: x[1])))

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        print('dof_names_to_idx: {}'.format(sorted(list(self.dof_names_to_idx.items()), key=lambda x: x[1])))

        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        print('self.cfg.asset.',self.cfg.asset)
        self.foot_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.foot_nums = len(self.foot_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_base_com = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device, requires_grad=False)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.init_body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        

        self._init_custom_buffers__()
        self._randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            
            dof_props = self._process_dof_props(dof_props_asset, i)
            dof_props["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][12:] = [40., 40., 40., 40., 40., 40.]
            dof_props["damping"][12:] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.cfg.depth.use_camera:
                self.attach_camera(i, env_handle, actor_handle)

        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))
        
        self.feet_indices = torch.zeros(len(self.foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.foot_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.foot_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # a function to update dof friction, damping, armature
    def _refresh_actor_dof_props(self, env_ids):
        ''' Refresh the dof properties of the actor in the given environments, i.e.
            dof friction, damping, armature
        '''
        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)
            for i in range(self.num_dof):
                if self.cfg.domain_rand.randomize_joint_friction:
                    if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                        dof_props["friction"][i] = self.default_joint_friction[i]*self.joint_friction_coeffs[env_id, i]
                    else:    
                        dof_props["friction"][i] = self.default_joint_friction[i]*self.joint_friction_coeffs[env_id, 0]
                        
                if self.cfg.domain_rand.randomize_joint_damping:
                    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                        dof_props["damping"][i] = self.default_joint_damping[i]*self.joint_damping_coeffs[env_id, i]
                    else:
                        dof_props["damping"][i] = self.default_joint_damping[i]*self.joint_damping_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_joint_armature:
                    if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                        dof_props["armature"][i] = self.default_joint_armature[i]*self.joint_armatures_coeffs[env_id, i]
                    else:
                        dof_props["armature"][i] = self.default_joint_armature[i]*self.joint_armatures_coeffs[env_id, 0]

            self.gym.set_actor_dof_properties(self.envs[env_id], 0, dof_props)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            # map terrain_types (column index) to terrain type index for dynamic_sigma
            if hasattr(self.terrain, 'cols2id') and len(self.terrain.cols2id) > 0:
                self.terrain_ids = torch.tensor(self.terrain.cols2id, device=self.device)[self.terrain_types]
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_x_max = (
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_x_min = -self.cfg.terrain.border_size
            self.terrain_y_max = (
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_y_min = -self.cfg.terrain.border_size
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.interal_dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        if 'P3O' in self.train_cfg.runner_class_name:
            self.cost_scales = class_to_dict(self.cfg.costs.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    def _draw_commands_vis(self):
        """ """
        xy_commands = (self.commands[:, :3] * getattr(self.obs_scales, "commands", 1.)).clone()
        yaw_commands = xy_commands.clone()
        xy_commands[:, 2] = 2.
        yaw_commands[:, :2] = 20.
        color = gymapi.Vec3(*self.cfg.viewer.commands.color)
        xy_commands_global = tf_apply(self.root_states[:, 3:7], self.root_states[:, :3], xy_commands * self.cfg.viewer.commands.size)
        yaw_commands_global = tf_apply(self.root_states[:, 3:7], self.root_states[:, :3], yaw_commands * self.cfg.viewer.commands.size)
        for i in range(self.num_envs):
            gymutil.draw_line(
                gymapi.Vec3(*self.root_states[i, :3].cpu().tolist()),
                gymapi.Vec3(*xy_commands_global[i].cpu().tolist()),
                color,
                self.gym,
                self.viewer,
                self.envs[i],
            )
            gymutil.draw_line(
                gymapi.Vec3(*self.root_states[i, :3].cpu().tolist()),
                gymapi.Vec3(*yaw_commands_global[i].cpu().tolist()),
                color,
                self.gym,
                self.viewer,
                self.envs[i],
            )

    def _draw_volume_sample_points_vis(self):
        self.gym.clear_lines(self.viewer)
        self.refresh_volume_sample_points()
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(0.5, 0., 0.5))
        for env_idx in range(self.num_envs):
            for point_idx in range(self.volume_sample_points.shape[1]):
                sphere_pose = gymapi.Transform(gymapi.Vec3(*self.volume_sample_points[env_idx, point_idx]), r= None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        if hasattr(self, "volume_sample_points") and getattr(self.cfg.viewer, "draw_volume_sample_points", False):
            self._draw_volume_sample_points_vis()

        # if self.cfg.viewer.draw_commands:
        #     self._draw_commands_vis()
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

        if 'Map' in self.train_cfg.runner_class_name:

            lagged_sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                processed_heights = self.lagged_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(processed_heights.shape[0]), self.height_points[i]).cpu().numpy()
                
                for j in range(processed_heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = processed_heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(lagged_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heightXBotL = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heightXBotL)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_target_pos_vel(self, target_link_indices, target_local_pos, domain= gymapi.DOMAIN_SIM):
        """ Get the target pos/vel in world frame, given the body_indices (SIM_DOMAIN) and
        the local position of which this target is rigidly attached to.
        
        NOTE: Before this method, `refresh_rigid_body_state_tensor` must be called before this 
        function and after each `gym.simulate`.

        Args:
            target_link_indices (torch.Tensor): shape (n_envs*n_targets_per_robot) for DOMAIN_SIM,
                or (n_targets_per_robot,) for DOMAIN_ENV
            target_local_pos (torch.Tensor): shape (n_targets_per_robot, 3)
            domain: gymapi.DOMAIN_SIM or gymapi.DOMAIN_ENV, not recommending using gymapi.DOMAIN_ACTOR
                since a env may contain multiple actors with different bodies.

        Returns:
            target_pos_world (torch.Tensor): shape (n_envs, n_targets_per_robot, 3)
            target_vel_world (torch.Tensor): shape (n_envs, n_targets_per_robot, 3)
        """
        # shape: (n_envs, n_targets_per_robot, 13)
        if domain == gymapi.DOMAIN_SIM:
            target_body_states = self.all_rigid_body_states[target_link_indices].view(self.num_envs, -1, 13)
        elif domain == gymapi.DOMAIN_ENV:
            # NOTE: maybe also acceping DOMAIN_ACTOR, but do this at your own risk
            target_body_states = self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, target_link_indices]
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        # shape: (n_envs, n_targets_per_robot, 3)
        target_pos = target_body_states[:, :, 0:3]
        # shape: (n_envs, n_targets_per_robot, 4)
        target_quat = target_body_states[:, :, 3:7]
        # shape: (n_envs * n_targets_per_robot, 3)
        target_pos_world_ = tf_apply(
            target_quat.view(-1, 4),
            target_pos.view(-1, 3),
            target_local_pos.unsqueeze(0).expand(self.num_envs, *target_local_pos.shape).reshape(-1, 3), # using reshape because of contiguous issue
        )
        # shape: (n_envs, n_targets_per_robot, 3)
        target_pos_world = target_pos_world_.view(self.num_envs, -1, 3)
        # shape: (n_envs, n_targets_per_robot, 3)
        # NOTE: assuming the angular velocity here is the same as the time derivative of the axis-angle
        target_vel_world = torch.cross(
            target_body_states[:, :, 10:13],
            target_local_pos.unsqueeze(0).expand(self.num_envs, *target_local_pos.shape),
            dim= -1,
        )
        return target_pos_world, target_vel_world

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()

        self.volume_sample_points_refreshed = False

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        if self.cfg.rewards.dynamic_sigma is None:
            sigma_x = sigma_y = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_lin_vel"]
            vmax = self.dynamic_sigma_cfg["max_lin_vel"]
            sigma_x = self._get_dynamic_sigma(torch.abs(self.commands[:, 0]), vmin, vmax)
            sigma_y = self._get_dynamic_sigma(torch.abs(self.commands[:, 1]), vmin, vmax)
        lin_vel_error_sq = torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2])
        scaled_error = lin_vel_error_sq[:, 0] / sigma_x + lin_vel_error_sq[:, 1] / sigma_y
        # print(f"{self.base_lin_vel[:, :2]=}, {lin_vel_error_sq=}")
        return torch.exp(-scaled_error)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        if self.cfg.rewards.dynamic_sigma is None:
            sigma = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_ang_vel"]
            vmax = self.dynamic_sigma_cfg["max_ang_vel"]
            sigma = self._get_dynamic_sigma(torch.abs(self.commands[:, 2]), vmin, vmax)
        ang_vel_error_sq = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error_sq/sigma)

    def _get_base_height(self):
        if not self.cfg.terrain.measure_heights:
            return self.root_states[:, 2]
        # 根据高度扫描点计算base link到地面估计高度
        masked_heights = self.measured_heights * self.base_height_scan_mask.unsqueeze(0)
        sum_heights = masked_heights.sum(dim=1)
        estimated_ground_z = sum_heights / self.num_base_height_scan_points

        base_z = self.root_states[:, 2]
        base_height = base_z - estimated_ground_z  # (N,)
        return base_height

    def _reward_correct_base_height(self):
        base_height = self._get_base_height()
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew

    def _reward_feet_regulation(self):
        # CTS抬腿正则奖励, 在脚末端速度增大同时, 要求高度尽可能高
        base_height = self._get_base_height()  # 更新刚体空间位置 (开悟比赛无法修改环境, 只能在奖励中完成计算了)
    
    def _reward_tracking_lin_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (
            self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev
        )
        return delta_phi
    
    def _reward_tracking_ang_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (
            self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev
        )
        return delta_phi

    def _get_dynamic_sigma(self, target_vel_abs, v_min, v_max):
        """Compute dynamic sigma based on terrain level and command velocity.
        Sigma interpolates between default and terrain-specific values based on velocity.
        """
        default_sigma = self.cfg.rewards.tracking_sigma
        if not self.cfg.terrain.curriculum or not hasattr(self, 'terrain_ids'):
            return torch.full_like(target_vel_abs, default_sigma)

        target_sigmas = self.terrain_max_sigmas[self.terrain_ids]
        sigma = torch.full_like(target_vel_abs, default_sigma)

        # v_min <= v < v_max: linear interpolation
        mask = (target_vel_abs >= v_min) & (target_vel_abs < v_max)
        if mask.any():
            ratio = (target_vel_abs[mask] - v_min) / (v_max - v_min)
            sigma[mask] = default_sigma + ratio * (target_sigmas[mask] - default_sigma)

        # v >= v_max: use target sigma
        mask = target_vel_abs >= v_max
        if mask.any():
            sigma[mask] = target_sigmas[mask]

        # Scale by terrain level: higher level -> closer to target sigma
        level_scale = torch.clamp(
            torch.exp((self.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0
        )
        sigma = default_sigma + level_scale * (sigma - default_sigma)
        return sigma