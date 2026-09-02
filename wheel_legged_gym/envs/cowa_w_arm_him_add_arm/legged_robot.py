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

import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from .base_task import BaseTask
# from humanoid.utils.terrain import Terrain
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from wheel_legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from wheel_legged_gym.utils.terrain import  Terrain

import sys
import IPython; eee = IPython.embed
# from wheel_legged_gym.utils.web_visualizer.isaac_visualizer_client import create_isaac_visualizer, bind_visualizer_to_gym, set_gpu_pipeline
# from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

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

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
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
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = self.cfg.viewer.debug_viz  # 显示scan dots
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.last_print_time = ti.time()
        
    
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
    
    # <><><><><> zsy 根据键盘输入操作机械臂关节 <><><><><>：
    def render_keyboard_ui(self,  sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            # print(self.ref_buff.shape)
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                # ==== 上一个关节 ==== 
                elif evt.action == "A" and evt.value > 0:  
                    self.dof_pos_count -= 1 
                    if self.dof_pos_count <0: self.dof_pos_count = 0                     #限制关节索引范围
                    elif self.dof_pos_count >5: self.dof_pos_count = 5  
                    print(f"当前关节：{self.dof_pos_count}, 当前关节角度：{self.ref_buff[0,self.dof_pos_count]}")
                # ==== 下一个关节 ==== 
                elif evt.action == "D" and evt.value > 0:  
                    self.dof_pos_count += 1 
                    if self.dof_pos_count <0: self.dof_pos_count = 0 
                    elif self.dof_pos_count >5: self.dof_pos_count = 5  
                    print(f"当前关节：{self.dof_pos_count}, 当前关节角度：{self.ref_buff[0,self.dof_pos_count]}")
                # ==== 增加关节角度 ==== 
                elif evt.action == "W" and evt.value > 0:   
                    self.ref_buff[:,self.dof_pos_count] += self.dof_pos_delta
                    print(f"当前关节：{self.dof_pos_count}, 当前关节角度：{self.ref_buff[0,self.dof_pos_count]}")
                # ==== 减少关节角度 ==== 
                elif evt.action == "S" and evt.value > 0:   
                    self.ref_buff[:,self.dof_pos_count] -= self.dof_pos_delta
                    print(f"当前关节：{self.dof_pos_count}, 当前关节角度：{self.ref_buff[0,self.dof_pos_count]}")
                # 回到初始default状态   
                elif evt.action == "R" and evt.value > 0:  
                    self.ref_buff[:,:6] = self.default_dof_pos[:,:6].clone()
                    self.command_vel_x = 0.0 
                    self.command_height_z = 0.0
                    self.command_vel_y = 0.0
                    print("reset to default")
                # 速度
                elif evt.action == "UP" and evt.value > 0:  
                    self.command_vel_x += 0.1
                    print(f"当前速度  {self.command_vel_x}")
                elif evt.action == "DOWN" and evt.value > 0:  
                    self.command_vel_x -= 0.1
                    print(f"当前速度  {self.command_vel_x}")
                # 高度
                elif evt.action == "+" and evt.value > 0:  
                    self.command_height_z += 0.01
                    print(f"当前高度  {self.cfg.init_state.pos[2] + self.command_height_z}")
                    # print(torch.norm(self.contact_forces[:, self.feet_indices, 2],dim=-1))
                elif evt.action == "-" and evt.value > 0:  
                    self.command_height_z -= 0.01
                    print(f"当前高度  {self.cfg.init_state.pos[2] + self.command_height_z}")
                    # print(torch.norm(self.contact_forces[:, self.feet_indices, 2],dim=-1))


            # print(self.ref_buff.shape)
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)



    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()

    # <><><><<><>< STEP ><><><><><><><><><
    def step(self, actions):

        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # start_time = ti.time()
        actions = torch.clamp(actions, min = -self.cfg.normalization.clip_actions, max = self.cfg.normalization.clip_actions)
        self.actions = actions

        if self.cfg.control.action_smoothness:
            ratio = self.cfg.control.ratio
            self.actions = ratio * self.actions + (1 - ratio) * self.last_actions  # debug plot

        if self.cfg.control.control_test:
            self.render_keyboard_ui() 
        else:
            self.render()
        self.pre_physics_step()
        # self._process_global_com_arm()
        # if self.cfg.domain_rand.randomize_lag_timesteps:    
        #     self.lag_buffer = self.lag_buffer[1:] + [actions.clone()] 
        #     action_delayed = self.lag_buffer[0]
        # else:
        #     action_delayed = actions 
        
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.leg_post_physics_step()
            # self.compute_ref_state()
            # ------------------ 用来模拟torque的延时 ------------------- --
            # 带机械臂轨迹的torque 使用_compute_torques_ARM进行测试   NOTE: 如果机械臂不摆动，请调节arms_play.py的control_test，不要怀疑是代码的问题，谢谢
            if not self.cfg.control.control_test:
                self.ref_buff = self.Curriculum_choice_arm(torch.tensor(self.cfg.control.curriculumn_arm_select)) # 机械臂轨迹 
            
            # NOTE self.ref_buff是传给计算力矩函数的，也就是下一句函数：_compute_torques  zsy注
            torques = self._compute_torques(actions).view(self.torques.shape)
            
            if self.cfg.domain_rand.add_dof_lag:
                q = self.dof_pos
                q_arm = self.dof_pos_arm
                self.dof_lag_buffer[:,:,1:] = self.dof_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_lag_buffer[:,:,0] = q.clone()
                self.dof_lag_buffer_arm[:,:,1:] = self.dof_lag_buffer_arm[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone() # NOTE 机械臂延时 zsy
                self.dof_lag_buffer_arm[:,:,0] = q_arm.clone()   # NOTE 机械臂延时 zsy
                dq = self.dof_vel
                dq_arm = self.dof_vel_arm
                self.dof_vel_lag_buffer[:,:,1:] = self.dof_vel_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_vel_lag_buffer[:,:,0] = dq.clone()
                # zsy
                self.dof_vel_lag_buffer_arm[:,:,1:] = self.dof_vel_lag_buffer_arm[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_vel_lag_buffer_arm[:,:,0] = dq_arm.clone()

            if self.cfg.domain_rand.add_imu_lag:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
                self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                self.imu_lag_buffer[:,:,1:] = self.imu_lag_buffer[:,:,:self.cfg.domain_rand.imu_lag_timesteps_range[1]].clone()
                if self.cfg.control.projected_gravity == True:
                    self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.projected_gravity ), 1).clone()
                else:
                    self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.base_euler_xyz ), 1).clone()
            if self.cfg.domain_rand.randomize_torque_delay:
                # delay_step = np.random.choice([0,1])    # 随机延迟 2或1 帧
                delay_step = 0  # 固定延迟2帧
                self.torques = self.hist_torques[delay_step].to(self.device)
                for i in range(delay_step-1):
                    self.hist_torques[i] = self.hist_torques[delay_step].clone()
                self.hist_torques.popleft()
                self.hist_torques.append(torques)
            else:
                self.torques = torques
            # print(self.torques) 
            # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(torch.cat((self.ref_buff,torch.empty_like(self.torques)), dim=-1))) # zsy add 直接设置机械臂关节角度，不过PD
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            
            # self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.ref_buff), gymtorch.unwrap_tensor(torch.tensor([0,1,2,3,4,5], device=self.device)), int(6)) # zsy add 直接设置机械臂关节角度，不过PD
            # self.gym.set_dof_actuation_force_tensor_indexed(self.sim, gymtorch.unwrap_tensor(torch.cat((self.torques, torch.zeros_like(self.ref_buff,  device=self.device)),dim=-1)), gymtorch.unwrap_tensor(torch.tensor([6,7,8,9,10,11],device=self.device)), int(6))
            
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        
        # 获取termination_privileged_obs(终止环境被刷新前的特权观测)
        termination_ids, termination_priveleged_obs = self.post_physics_step()
        # <><><><> test <><><><>
        self._process_arm_mid_points_projected()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs



    def compute_dof_vel(self):
        diff = (
            torch.remainder(self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi)
            - self.pi
        )
        self.dof_pos_dot = diff / self.sim_params.dt
        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot
        self.last_dof_pos[:] = self.dof_pos[:]


    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _, = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def _compute_feet_states(self):
        # add foot positions
        self.foot_positions = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        # add foot velocities
        self.foot_velocities = (self.foot_positions - self.last_foot_positions) / self.dt


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

        # self.torques_change_acc = (self.last_torques - self.torques) / self.dt
        # self.ang_vel_change_acc = (self.last_base_ang_vel - self.base_ang_vel[:,2]) / self.dt

        # prepare quantities
        self.wheel_pos[:] = self.rigid_state[:,[9,12],0:3] #zsya dd
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        
        # VMC
        self.leg_post_physics_step()

        self._post_physics_step_callback()

        # 計算全局質心位置

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() 
        termination_privileged_obs = self.compute_privileged_observations()[env_ids]
        self.reset_idx(env_ids) 
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # actions是clamp之后的
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        # action_real是网络实际输出的  
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]
        self.last_foot_positions[:] = self.foot_positions[:]
        self.last_base_pos[:] = self.base_pos[:] 
        self.last_wheel_pos[:] = self.wheel_pos[:]# zsy add
        self.last_torques[:] = self.torques[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:,2]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        
        if self.viewer and self.cfg.viewer.draw_base_com:
            self._draw_base_com_vis()
        if 0:
            self._draw_global_com_vis() #
        if 0:
            self._draw_arm_mid_points_debug(self.arm_2_root_pos) #
        if 1:
            self._draw_foot_pos_debug(self.wheel_pos)
        # 返回值
        return env_ids, termination_privileged_obs




    def leg_post_physics_step(self):
        
        self.theta1 = torch.cat(
            (self.pi - self.dof_pos[:, 0].unsqueeze(1) + 0.13433,
             self.pi - self.dof_pos[:, 3].unsqueeze(1) + 0.13433),
            dim=1)
        self.theta2 = torch.cat(
            (
                (self.pi - self.dof_pos[:, 1] - 0.26866).unsqueeze(1),
                (self.pi - self.dof_pos[:, 1] - 0.26866).unsqueeze(1),
            ),
            dim=1,
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)

        self.L0_dot, self.theta0_dot = self.calculate_vmc_vel()

    def forward_kinematics(self, theta1, theta2):
        end_x = (
                self.cfg.asset.offset
                + self.cfg.asset.l1 * torch.cos(theta1)
                + self.cfg.asset.l2 * torch.cos(theta1 + theta2)
        )
        end_y = self.cfg.asset.l1 * torch.sin(theta1) + self.cfg.asset.l2 * torch.sin(
            theta1 + theta2
        )
        L0 = torch.sqrt(end_x ** 2 + end_y ** 2)
        theta0 = - torch.arctan2(end_y, end_x) + self.pi / 2
        # print("theta0",theta0)
        return L0, theta0
    
    def calculate_vmc_vel(self):
        l1 = self.cfg.asset.l1
        l2 = self.cfg.asset.l2
        theta1 = self.theta1
        theta2 = self.theta2
        x = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
        y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)

        dx_dphi1 = -l1 * torch.sin(theta1) - l2 * torch.sin(theta1 + theta2)
        dx_dphi2 = -l2 * torch.sin(theta1 + theta2)
        dy_dphi1 =  l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
        dy_dphi2 =  l2 * torch.cos(theta1 + theta2)
        dr_dphi1 = (dx_dphi1 * x + dy_dphi1 * y) / self.L0
        dr_dphi2 = (dx_dphi2 * x + dy_dphi2 * y) / self.L0
        dtheta_dphi1 = (dy_dphi1 * x - dx_dphi1 * y) / (torch.square(self.L0))
        dtheta_dphi2 = (dy_dphi2 * x - dx_dphi2 * y) / (torch.square(self.L0))
        jacobian = [[dr_dphi1, dr_dphi2],[dtheta_dphi1, dtheta_dphi2]]

        L0_dot = jacobian[0][0] * self.dof_vel[:,[0, 3]] + jacobian[0][1] * self.dof_vel[:,[1, 4]]
        theta0_dot = jacobian[1][0] * self.dof_vel[:,[0, 3]] + jacobian[1][1] * self.dof_vel[:,[1, 4]]
        return L0_dot, theta0_dot
    


    # TODO
    def check_termination(self):
        """ Check if environments need to be reset
        """

        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 8.0, # previous value = 10 
            dim=1,
        )
        fail_buf |= self.projected_gravity[:, 2] > -0.2
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
            # | self.edge_reset_buf
        )

    # TODO
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
        # zsy add
        
        self._resample_commands(env_ids)
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # xxx added_rand
        self._randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        if self.cfg.domain_rand.randomize_rigids_after_start:
            self.refresh_actor_rigid_shape_props(env_ids,self.cfg)
        
        # xxx
        self.randomize_lag_props(env_ids)


        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0

        # fill extras  
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():  
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s  
            # print(f"{key}!!!!!!!!!!",self.extras["episode"]['rew_' + key])
            self.episode_sums[key][env_ids] = 0.    
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # fix reset gravity bug
        # zsy add -->
        self.base_pos_init[env_ids] = self.root_states[env_ids, 0:3]  
        rigid_state_buff = self.rigid_state[:, [9,12], 0:3].clone()
        self.wheel_pos_init[env_ids] = rigid_state_buff[env_ids] + self.env_origins[env_ids].unsqueeze(1)   #zsy add
        # eee()

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)


    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    # agibot
    def randomize_lag_props(self,env_ids): 
        """ random add lag
        """
        if self.cfg.domain_rand.add_lag:   
            self.lag_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0], 
                                                           self.cfg.domain_rand.lag_timesteps_range[1]+1,(len(env_ids),),device=self.device) 
                if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                    self.last_lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]
            else:
                self.lag_timestep[env_ids] = self.cfg.domain_rand.lag_timesteps_range[1]
        if self.cfg.domain_rand.add_dof_lag:
            dof_lag_buffer_init = self.default_dof_pos[:,6:].unsqueeze(1).expand(-1, self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, -1).clone()
            dof_lag_buffer_init = dof_lag_buffer_init.transpose(1, 2)
            self.dof_lag_buffer[env_ids, :, :] = dof_lag_buffer_init[env_ids,:, :]
            # NOTE zsy add 机械臂延时初始化 ----
            dof_lag_buffer_init_arm = self.default_dof_pos[:,0:6].unsqueeze(1).expand(-1, self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, -1).clone()
            dof_lag_buffer_init_arm = dof_lag_buffer_init_arm.transpose(1, 2)
            self.dof_lag_buffer_arm[env_ids, :, :] = dof_lag_buffer_init_arm[env_ids,:, :]

            self.dof_vel_lag_buffer[env_ids, :, :] = 0.0
            self.dof_vel_lag_buffer_arm[env_ids, :, :] = 0.0
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

        # 重置actionbuffer
        # if self.cfg.domain_rand.randomize_lag_timesteps:  
        #     for i in range(len(self.lag_buffer)):
        #         self.lag_buffer[i][env_ids, :] = 0
        
        if self.cfg.domain_rand.randomize_torque_delay:
            for i in range(len(self.hist_torques)):
                self.hist_torques[i][env_ids, :] = 0.
        
        if self.cfg.domain_rand.randomize_obs_delay:
            for i in range(len(self.hist_obs)):
                self.hist_obs[i][env_ids, :] = 0.
            


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = torch.clip(
                rew,
                -self.cfg.rewards.clip_single_reward * self.dt,
                self.cfg.rewards.clip_single_reward * self.dt,
            )       
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
        # if self.cfg.sim.web_vis == True:
        #     start_zmq_server_as_subprocess()
        #     create_isaac_visualizer(port=self.cfg.sim.port, host="localhost", keep_default_viewer=self.cfg.sim.keep_default_viewer, max_env=self.cfg.sim.web_vis_envs)
        #     self.gym = bind_visualizer_to_gym(self.gym, self.sim)
        #     set_gpu_pipeline(self.sim_params.use_gpu_pipeline)
        
        self._create_envs()

    # def _push_robots(self):
    #     """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
    #     """
    #     max_vel = self.cfg.domain_rand.max_push_vel_xy
    #     max_push_angular = self.cfg.domain_rand.max_push_ang_vel
    #     self.rand_push_force[:, :2] = torch_rand_float(
    #         -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
    #     self.root_states[:, 7:9] = self.rand_push_force[:, :2]

    #     self.rand_push_torque = torch_rand_float(
    #         -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

    #     self.root_states[:, 10:13] = self.rand_push_torque

    #     self.gym.set_actor_root_state_tensor(
    #         self.sim, gymtorch.unwrap_tensor(self.root_states))
    def _push_robots(self):
        """Random pushes the robots."""
        env_ids = (
            (
                self.envs_steps_buf
                % int(self.cfg.domain_rand.push_interval_s / self.cfg.sim.dt) # 0.002  
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return
        max_push_force = (
            self.body_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.sim_params.dt
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
    def _randomize_dof_props(self, env_ids ):
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = self.default_motor_offset
            self.motor_offsets[env_ids, :] += torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
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
        # rand joint friciton on torque
        if self.cfg.domain_rand.randomize_coulomb_friction:
            min_joint_coulomb, max_joint_coulomb = self.cfg.domain_rand.joint_coulomb_friction_range
            self.randomized_joint_coulomb_friction[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_joint_coulomb - min_joint_coulomb) + min_joint_coulomb

            min_joint_stick_friction, max_joint_stick_friction = self.cfg.domain_rand.joint_stick_friction_range
            self.randomized_joint_stick_friction[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_joint_stick_friction - min_joint_stick_friction) + min_joint_stick_friction


        # rand joint friction set in sim
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                for i in range(self.num_dofs-6):
                    range_key = f'joint_{i+1}_friction_range'
                    friction_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_friction_coeffs[env_ids, i] = torch_rand_float(friction_range[0], friction_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:                      
                joint_friction_range = self.cfg.domain_rand.joint_friction_range
                self.joint_friction_coeffs[env_ids] = torch_rand_float(joint_friction_range[0], joint_friction_range[1], (len(env_ids), 1), device=self.device)

        # rand joint damping set in sim
        if self.cfg.domain_rand.randomize_joint_damping:
            if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                for i in range(self.num_dofs-6):
                    range_key = f'joint_{i+1}_damping_range'
                    damping_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_damping_coeffs[env_ids, i] = torch_rand_float(damping_range[0], damping_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_damping_range = self.cfg.domain_rand.joint_damping_range
                self.joint_damping_coeffs[env_ids] = torch_rand_float(joint_damping_range[0], joint_damping_range[1], (len(env_ids), 1), device=self.device)
        
        # rand joint armature inertia set in sim
        if self.cfg.domain_rand.randomize_joint_armature:
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                for i in range(self.num_dofs-6):
                    range_key = f'joint_{i+1}_armature_range'
                    armature_range = getattr(self.cfg.domain_rand, range_key)
                    self.joint_armatures_coeffs[env_ids, i] = torch_rand_float(armature_range[0], armature_range[1], (len(env_ids), 1), device=self.device).reshape(-1)
            else:
                joint_armature_range = self.cfg.domain_rand.joint_armature_range
                self.joint_armatures_coeffs[env_ids] = torch_rand_float(joint_armature_range[0], joint_armature_range[1], (len(env_ids), 1), device=self.device)

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
            
            # min_mass_scale, max_mass_scale = cfg.domain_rand.randomize_mass_range
            # self.mass_scale[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
            #                                     requires_grad=False) * (max_mass_scale - min_mass_scale) + min_mass_scale

        
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3 * self.num_bodies, dtype=torch.float, device=self.device,requires_grad=False) * (max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 2, dtype=torch.float, device=self.device,requires_grad=False) * (max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution


    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity:
            if external_force is not None:
                self.gravities[:, :] = external_force.unsqueeze(0)
            elif self.cfg.domain_rand.randomize_gravity:
                min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
                external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                            requires_grad=False) * (max_gravity - min_gravity) + min_gravity

                self.gravities[:, :] = external_force.unsqueeze(0)

            sim_params = self.gym.get_sim_params(self.sim)
            gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
            self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
            sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
            self.gym.set_sim_params(self.sim, sim_params)


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
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

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
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_acc_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                props["friction"][i] = self.cfg.domain_rand.default_joint_friction[i]
                props["damping"][i] = self.cfg.domain_rand.default_joint_damping[i]
                props["armature"][i] = self.cfg.domain_rand.default_joint_armature[i]
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit
        return props


    # zsy add
    def _process_arm_mid_points_projected(self):
        """
        通过关节刚体中心获得机械臂的中心位置，估计
        params: env_ids 被终止的id
        """
        # 15 ~ 20 机械臂的顺序
        self.arm_point_pos = self._rigid_body_pos[:,1:7,:]       # 第一个index是base_link
        mean_points_x = self.arm_point_pos[...,0].mean(dim=1) # 6个机械臂
        mean_points_y = self.arm_point_pos[...,1].mean(dim=1) 
        mean_points_z = self.arm_point_pos[...,2].mean(dim=1) 
        ee_points_xyz = self._rigid_body_pos[:,6,:].clone()        # 机械臂末端点的位置
        self.mean_points = torch.stack((mean_points_x, mean_points_y, mean_points_z), dim=1) # dim = num_envs x 3
        # mean_points_buff = (self.mean_points - self.root_states[:,:3]).clone() #相对值
        local_base_pos           = quat_rotate_inverse(self.base_quat,  self.root_states[:,:3])
        local_mean_points_pos    = quat_rotate_inverse(self.base_quat,  self.mean_points)

        ee_local_base_pos        = quat_rotate_inverse(self.base_quat,  self.root_states[:,:3])
        ee_local_mean_points_pos = quat_rotate_inverse(self.base_quat,  ee_points_xyz)

        self.arm_2_root_pos = local_mean_points_pos - local_base_pos  
        self.ee_arm_2_root_pos = ee_local_mean_points_pos - ee_local_base_pos 

        # aaa = self.arm_2_root_pos.clone()
        # bbb = self.ee_arm_2_root_pos.clone()
        # print(delta_local_pos[0,:]) #tensor([ 0.1747, -0.0112,  0.4969], device='cuda:0')
        # print(bbb) #[ 0.4004,  0.0014,  0.3970],
        # self._draw_arm_mid_points_debug(aaa)
        # self._draw_arm_mid_points_debug(bbb)
        # eee()




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


    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
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


    def _draw_base_com_vis(self):
        # print("<><><><>",self.base_com)
        debug_base_com = self.base_com + self.root_states[0, :3]
        
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        sphere_geom_self = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
        for i in range(self.num_envs):
            base_pos = debug_base_com.cpu().numpy()
            x = base_pos[i, 0]
            y = base_pos[i, 1]
            z = base_pos[i, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
        # zsy add                         
        for i in range(self.num_envs):
            self_pos = self.root_states[:,:3].cpu().numpy()
            x = self_pos[i, 0]
            y = self_pos[i, 1]
            z = self_pos[i, 2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom_self, self.gym, self.viewer, self.envs[i], sphere_pose) 




    # for debug 
    def _draw_arm_mid_points_debug(self, points):
        debug_base_com = points[0,:]
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 0, 0)) # 
        if 0:   
            for i in range(6): # arm_nums=6
                base_pos = debug_base_com.cpu().numpy().copy()
                x = base_pos[i,0]
                y = base_pos[i,1]
                z = base_pos[i,2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
        if 1:   
            for i in range(1): # arm_nums=6
                base_pos = debug_base_com.cpu().numpy().copy()
                x = base_pos[0]
                y = base_pos[1]
                z = base_pos[2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 



    # zsy add for debug only
    def _draw_foot_pos_debug(self, points):
        debug_base_com = points[0,...]
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.16, 4, 4, None, color=(0, 1, 0)) # 
        if 1:
            for i in range(2): # arm_nums=6
                base_pos = debug_base_com.cpu().numpy().copy()
                x = base_pos[i,0]
                y = base_pos[i,1]
                z = base_pos[i,2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
        else:
            for i in range(1): # arm_nums=6
                base_pos = debug_base_com.cpu().numpy().copy()
                x = base_pos[0]
                y = base_pos[1]
                z = base_pos[2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 



    # zsy add
    def _process_global_com_arm(self):
        """
        获得机器人初始时刻的重心位置
        """
        start = ti.time()
        props = self.gym.get_actor_rigid_body_properties(self.env_handle, self.actor_handle)
        self.total_base_com = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        for env_id in range(self.num_envs):
            # create env instance  
            props = self.gym.get_actor_rigid_body_properties(self.env_handle, self.actor_handle)
            new_com_x_sum,  new_com_y_sum,  new_com_z_sum,  new_mass_sum = 0,0,0,0
            # process rigid body properties
            for i in range(len(props)): #13
                current_com = props[i].com
                current_mass = props[i].mass
                new_com_x_sum += current_com.x * current_mass
                new_com_y_sum += current_com.y * current_mass
                new_com_z_sum += current_com.z * current_mass
                new_mass_sum += current_mass
            new_com_x = new_com_x_sum / new_mass_sum
            new_com_y = new_com_y_sum / new_mass_sum
            new_com_z = new_com_z_sum / new_mass_sum
            self.total_base_com[env_id, :] = torch.tensor([new_com_x, new_com_y, new_com_z])
        stop = ti.time()   
        delta = stop-start
        print("time get total base com:",delta)



    def _draw_global_com_vis(self):
        debug_base_com = self.total_base_com[0, :3] + self.root_states[0, :3]
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))
        for i in range(self.num_envs):
            base_pos = debug_base_com.cpu().numpy()
            x = base_pos[0]
            y = base_pos[1]
            z = base_pos[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>




    # TODO
    # 增加机械臂后的刚体顺序 #NOTE：['base_link', 'link15', 'link16', 'link17', 'link18', 'link19', 'link20', 'left_hip_pitch_link', 'left_knee_pitch_link', 'left_wheel_link', 'right_hip_pitch_link', 'right_knee_pitch_link', 'right_wheel_link']
    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        # self.default_body_mass = 28.9
        # props[0].mass = self.default_body_mass
        # props[0].com.x = -0.0103984
        # props[0].com.y = 0
        # props[0].com.z = 0.1402415 
        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass += self.payloads[env_id]  # 加负载

        if self.cfg.domain_rand.randomize_com_displacement:
            if self.cfg.domain_rand.randomize_each_link:
                for i in range(len(props)):
                    current_com = props[i].com
                    if i==0: # baselink的com随机化最大  
                        new_com_x = current_com.x + self.com_displacements[env_id,0].item()
                        new_com_y = current_com.y + self.com_displacements[env_id,1].item()
                        new_com_z = current_com.z + self.com_displacements[env_id,2].item() * 0.5 # NOTE: zsy修改 质心偏移z方向小
                    elif i == 3+6 or i == 6+6 or i in [1,2,3,4,5,6]: # 轮子是对的 #NOTE: zsy修改 机械臂没有质心偏移
                        new_com_x = current_com.x
                        new_com_y = current_com.y
                        new_com_z = current_com.z
                    else: # 其他link的x方向随机化大,其他方向小  
                        new_com_x = current_com.x + self.com_displacements[env_id,3*i+0].item() * self.cfg.domain_rand.link_com_displacement_range_factor
                        new_com_y = current_com.y + self.com_displacements[env_id,3*i+1].item() * self.cfg.domain_rand.link_com_displacement_range_factor
                        new_com_z = current_com.z + self.com_displacements[env_id,3*i+2].item() * self.cfg.domain_rand.link_com_displacement_range_factor 
                    props[i].com = gymapi.Vec3(new_com_x,new_com_y,new_com_z)
            else:   
                current_com = props[0].com
                new_com_x = current_com.x + self.com_displacements[env_id,0].item() # debug
                new_com_y = current_com.y + self.com_displacements[env_id,1].item() 
                new_com_z = current_com.z + self.com_displacements[env_id,2].item() 
                props[0].com = gymapi.Vec3(new_com_x,new_com_y,new_com_z)
                
        self.base_com[env_id, :] = torch.tensor([props[0].com.x,props[0].com.y,props[0].com.z])
        
        # xxx  
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):  
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale =  np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
        self.body_mass[env_id] = props[0].mass
        return props

    # wh: walk_these_way
    def _teleport_robots(self, env_ids, cfg):
        """ Teleports any robots that are too close to the edge to the other side
        """
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 1] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._randomize_dof_props(env_ids)

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

        # 重力随机化，根据step_counter，间接性添加
        if self.cfg.domain_rand.randomize_gravity:
            if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
                self._randomize_gravity()
            if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                    self.cfg.domain_rand.gravity_rand_interval) == 0:
                self._randomize_gravity(torch.tensor([0, 0, 0]))

        # 形体参数随机化
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self.refresh_actor_rigid_shape_props(env_ids,self.cfg)
    # TODO
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        # """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["height"][0] , self.command_ranges["height"][1] , (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)


    # TODO
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # 机械臂轨迹处理：

        if self.cfg.domain_rand.add_lag:
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
        # print("self.lag_timestep=",self.lag_timestep) 
        
        pos_ref = self.lagged_actions_scaled * self.cfg.control.pos_action_scale
        pos_ref[:, 2] *= 0
        pos_ref[:, 5] *= 0
        vel_ref = self.lagged_actions_scaled * self.cfg.control.vel_action_scale
        vel_ref[:, :2] *= 0
        vel_ref[:, 3:5] *= 0
        # pd controller # 加上机械臂参考轨迹
        
        self.joint_pos_target = torch.cat((self.ref_buff , pos_ref), dim=-1) # + self.default_dof_pos
        
        if self.cfg.domain_rand.randomize_PD_factor:
            p_gains = self.Kp_factors * self.p_gains
            d_gains = self.Kd_factors * self.d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if self.cfg.domain_rand.randomize_coulomb_friction: #True
            torques = p_gains * (self.joint_pos_target - self.dof_pos_all + self.motor_offsets) +\
            d_gains * (torch.cat((torch.zeros_like(self.dof_vel_arm), vel_ref), dim=-1) - self.dof_vel_all) -\
            self.randomized_joint_coulomb_friction * self.dof_vel_all -\
            self.randomized_joint_stick_friction * torch.sign(self.dof_vel_all)
        else: 
            torques = p_gains * (self.joint_pos_target - self.dof_pos_all + self.motor_offsets) + d_gains * (torch.cat((torch.zeros_like(self.dof_vel_arm), vel_ref), dim=-1) - self.dof_vel_all)
        
        torques[:,[1+6,4+6]] += 7.0 #補償

        if self.cfg.domain_rand.randomize_motor_strength:
            torques *= self.motor_strengths

        return torch.clip(torques, -self.torque_limits, self.torque_limits)



    def compute_motor_torque(self, F, T):
        l1 = self.cfg.asset.l1
        l2 = self.cfg.asset.l2
        theta1 = self.theta1
        theta2 = self.theta2
        # print("theta1", theta1)
        # print("theta2", theta2)
        # change from original for the joint tf is different！
        x = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
        y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)

        dx_dphi1 = -l1 * torch.sin(theta1) - l2 * torch.sin(theta1 + theta2)
        dx_dphi2 = -l2 * torch.sin(theta1 + theta2)
        dy_dphi1 = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
        dy_dphi2 = l2 * torch.cos(theta1 + theta2)
        dr_dphi1 = (dx_dphi1 * x + dy_dphi1 * y) / self.L0
        dr_dphi2 = (dx_dphi2 * x + dy_dphi2 * y) / self.L0
        dtheta_dphi1 = (dy_dphi1 * x - dx_dphi1 * y) / (torch.square(self.L0))
        dtheta_dphi2 = (dy_dphi2 * x - dx_dphi2 * y) / (torch.square(self.L0))
        jacobian = [[dr_dphi1, dr_dphi2], [dtheta_dphi1, dtheta_dphi2]]
        jacobian_transpose = [[jacobian[0][0], jacobian[1][0]],
                              [jacobian[0][1], jacobian[1][1]]]

        T1 = jacobian_transpose[0][0] * F + jacobian_transpose[0][1] * T
        T2 = jacobian_transpose[1][0] * F + jacobian_transpose[1][1] * T
        return T1, T2

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos_all[env_ids] = self.raw_default_dof_pos
        
        self.dof_vel_all[env_ids] = 0.

        # reset的默认角度会被选择：
        self.Curriculum_choice_arm_default_pos(env_ids, torch.tensor(self.cfg.control.curriculumn_arm_select))
        # if self.cfg.init_state.rand_init_dof:
        #     self.dof_pos_arm[env_ids] += torch_rand_float(-self.cfg.init_state.rand_init_dof_range,
        #                                               self.cfg.init_state.rand_init_dof_range, (len(env_ids), self.num_dof), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    # TODO
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
            # self.root_states[env_ids, :3] += self.env_origins[env_ids]     
        
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
            < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.5  
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        # # wh: 增加升级和降级的env_ids标志
        # mask = self.terrain_levels[env_ids] >= self.max_terrain_levels
        # self.success_ids = env_ids[mask]
        # mask = self.terrain_levels[env_ids] < self.max_terrain_levels
        # self.fail_ids = env_ids[mask]

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

    # def update_command_curriculum(self, env_ids):
    #     """Implements a curriculum of increasing commands

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     def is_successful(env_ids, reward_key, threshold):
    #         """Check if the reward for given environments exceeds the threshold"""
    #         return (
    #             torch.mean(self.episode_sums[reward_key][env_ids]) / self.max_episode_length
    #             > threshold * self.reward_scales[reward_key]
    #         )

    #     # # If the tracking reward is above 80% of the maximum, increase the range of commands
    #     # if self.cfg.terrain.curriculum and len(self.success_ids) != 0:
    #     #     mask = is_successful(self.success_ids, "tracking_lin_vel_x", self.cfg.commands.curriculum_threshold)
    #     #     success_ids = self.success_ids[mask]

    #     #     basic_ids = torch.any(
    #     #         success_ids.unsqueeze(1) == self.basic_terrain_idx.unsqueeze(0), dim=1
    #     #     )
    #     #     basic_ids = success_ids[basic_ids]

    #     #     # Update command ranges for successful environments
    #     #     self.command_ranges["lin_vel_x"][success_ids, 0] -= 0.05
    #     #     self.command_ranges["lin_vel_x"][success_ids, 1] += 0.05
    #     #     self.command_ranges["lin_vel_x"][basic_ids, 0] -= 0.45
    #     #     self.command_ranges["lin_vel_x"][basic_ids, 1] += 0.45

    #     #     # Clip the command ranges for basic and advanced terrain
    #     #     for terrain_idx, max_curriculum in [
    #     #         (self.basic_terrain_idx, self.cfg.commands.basic_max_curriculum),
    #     #         (self.advanced_terrain_idx, self.cfg.commands.advanced_max_curriculum)
    #     #     ]:
    #     #         self.command_ranges["lin_vel_x"][terrain_idx, :] = torch.clip(
    #     #             self.command_ranges["lin_vel_x"][terrain_idx, :],
    #     #             -max_curriculum,
    #     #             max_curriculum,
    #     #         )

    #     if not self.cfg.terrain.curriculum:
    #         if (
    #             is_successful(env_ids, "tracking_lin_vel_x", self.cfg.commands.curriculum_threshold) and
    #             is_successful(env_ids, "tracking_ang_vel", self.cfg.commands.curriculum_threshold * 0.8)
    #         ):
    #             # Expand the command ranges for all environments
    #             self.command_ranges["lin_vel_x"][:, 0] = torch.clip(
    #                 self.command_ranges["lin_vel_x"][:, 0] - 0.1,
    #                 -self.cfg.commands.basic_max_curriculum,
    #                 0.0,
    #             )
    #             self.command_ranges["lin_vel_x"][:, 1] = torch.clip(
    #                 self.command_ranges["lin_vel_x"][:, 1] + 0.1,
    #                 0.0,
    #                 self.cfg.commands.basic_max_curriculum,
    #             )

    # xxx added_rand 
    def _init_custom_buffers__(self):
        self.friction_coeffs = 1 * torch.ones(self.num_envs, 2, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = 1 * torch.ones(self.num_envs, 2, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.mass_scale = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3 * self.num_bodies , dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False) 
        self.default_motor_offset = torch.tensor(self.cfg.domain_rand.default_motor_offset, dtype=torch.float, device=self.device,
                                         requires_grad=False)   
        self.motor_offsets[:] = self.default_motor_offset                           
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        if self.cfg.domain_rand.randomize_coulomb_friction:
            self.randomized_joint_coulomb_friction = torch.zeros(self.num_envs,self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.randomized_joint_stick_friction = torch.zeros(self.num_envs,self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)        
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
            

    # xxx
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) # 刚体属性

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos_all = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_pos = self.dof_pos_all[:,6:] #zsy add arm的意思是让机械臂单独跟随这个
        self.dof_pos_arm = self.dof_pos_all[:,0:6] #zsy add arm的意思是让机械臂单独跟随这个
        self.last_dof_pos = torch.zeros_like(self.dof_pos) # 用于模拟双tbox的观测延时
        self.dof_vel_all = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_vel = self.dof_vel_all[:,6:] #zsy add arm的意思是让机械臂单独跟随这个
        self.dof_vel_arm = self.dof_vel_all[:,0:6] #zsy add arm的意思是让机械臂单独跟随这个
        self.dof_acc = torch.zeros_like(self.dof_vel)
        # zsy add indices for gym
        # self.actor_indices = torch.zeros_like(self.dof_pos_all) # dim = num x 12 
        # self.actor_indices += torch.tensor([i for i in range(12)],device=self.device) #arm wheel
        # 初始化机械臂随机参数
        self.Curriculum_random_init_buffers()

        self.last_dof_vel = torch.zeros_like(self.dof_vel) # 用于模拟双tbox的观测延时
        self.base_pos = self.root_states[:, 0:3]
        self.base_pos_init = torch.zeros_like(self.base_pos)
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # zsy add 刚体的属性
        self._rigid_body_pos        = self.rigid_state[..., :3]
        self._rigid_body_rot        = self.rigid_state[..., 3:7]
        self._rigid_body_vel        = self.rigid_state[..., 7:10]
        self._rigid_body_ang_vel    = self.rigid_state[..., 10:13]

        # zsy add 用于轮子奖励函数 
        self.wheel_pos = self._rigid_body_pos[:, [9,12] ,:3] #- self.base_pos.unsqueeze(1)    # 轮子的刚体位置
        self.wheel_pos_init = torch.zeros_like(self.wheel_pos)  

        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.last_wheel_pos = torch.zeros_like(self.wheel_pos) # zsya dd
        self.foot_positions = self.rigid_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.last_foot_positions = torch.zeros_like(self.foot_positions)
        self.foot_heights = torch.zeros_like(self.foot_positions)
        self.foot_velocities = torch.zeros_like(self.foot_positions)
        self.foot_velocities_f = torch.zeros_like(self.foot_positions)
        self.foot_relative_velocities = torch.zeros_like(self.foot_velocities)

        # VMC
        self.last_T1 = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_T2 = torch.zeros(self.num_envs, 2, device=self.device)
        
        self.L0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.L0_dot = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta0_dot = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta1 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta2 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.torques_change_acc = torch.zeros(self.num_envs, 12, device=self.device)
        self.ang_vel_change_acc = torch.zeros(self.num_envs, 1, device=self.device)

        self.last_torques  = torch.zeros_like(self.torques_change_acc)
        # self.last_torques  = torch.zeros_like(self.torques)
        self.last_base_ang_vel = torch.zeros_like(self.root_states[:, 12])

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
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # zsy修改
        self.forward_torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.action_plot = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_acc = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dq = torch.zeros_like(self.dof_vel)
        self.acc_peak_penalty= torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_acc = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.ang_vel, self.obs_scales.height_measurements], device=self.device, requires_grad=False,) 
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                        device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
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

        # 初始化机械臂随机参数
        self.Curriculum_random_init_buffers()
        # self.Curriculum_random_init()
        
        # 用来模拟torque的延时n步
        if self.cfg.domain_rand.randomize_torque_delay:
            torque_delay_steps = self.cfg.domain_rand.torque_delay_steps
            self.hist_torques = deque(maxlen=torque_delay_steps)
            for _ in range(torque_delay_steps):
                self.hist_torques.append(torch.zeros(self.num_envs, self.num_dof))

        # 用来模拟obs的延时n步
        if self.cfg.domain_rand.randomize_obs_delay:
            obs_delay_steps = self.cfg.domain_rand.obs_delay_steps
            self.hist_obs = deque(maxlen=obs_delay_steps)
            for _ in range(obs_delay_steps):
                self.hist_obs.append(torch.zeros(self.num_envs, self.cfg.env.num_single_obs))

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


        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        # <><><><><><> zsy add <><><><><><><><>
        self.ref_buff = self.default_dof_pos[:,:6].clone()   #机械臂角度的cache
        self.arm_point_pos = torch.zeros((self.num_envs, (self.num_dof - 6), 3), dtype=torch.float32, device=self.device)
        self.arm_2_root_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ee_arm_2_root_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)


        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
        # xxx added_rand
        # self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        # xxx 
        # agibot 
        if self.cfg.domain_rand.add_lag:   
            self.lag_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.lag_timesteps_range[1]+1,device=self.device)
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_timestep = torch.randint(self.cfg.domain_rand.lag_timesteps_range[0],
                                                  self.cfg.domain_rand.lag_timesteps_range[1]+1,(self.num_envs,),device=self.device) 
                if self.cfg.domain_rand.randomize_lag_timesteps_perstep:
                    self.last_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.lag_timesteps_range[1]
            else:
                self.lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.lag_timesteps_range[1]

        if self.cfg.domain_rand.add_dof_lag:
            # self.dof_lag_buffer = self.default_dof_pos.expand(self.num_envs, self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, self.num_dof ).clone()
            self.dof_lag_buffer = self.default_dof_pos[:,6:].unsqueeze(1).expand(-1, self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, -1).clone()
            self.dof_lag_buffer = self.dof_lag_buffer.transpose(1, 2)

            self.dof_lag_buffer_arm = self.default_dof_pos[:,0:6].unsqueeze(1).expand(-1, self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, -1).clone() # NOTE zsy add 机械臂的观测值延时
            self.dof_lag_buffer_arm = self.dof_lag_buffer_arm.transpose(1, 2)  # NOTE zsy add 机械臂的观测值延时
            
            self.dof_vel_lag_buffer     = torch.zeros(self.num_envs,  self.num_actions,     self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,device=self.device)
            self.dof_vel_lag_buffer_arm = torch.zeros(self.num_envs,  6,                    self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,device=self.device) # zsy add 6 is num of arm joints

            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.dof_lag_timesteps_range[1]

            else:
                self.dof_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.dof_lag_timesteps_range[1]

        if self.cfg.domain_rand.add_imu_lag:
            self.imu_lag_buffer = torch.zeros(self.num_envs, self.num_actions, self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,device=self.device)  #self.num_actions=6
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.imu_lag_timesteps_range[1]

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
# TODO
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
        num_joints = self.gym.get_asset_joint_count(robot_asset)
        print("Number of joints in the asset:", num_joints)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset) 
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # import ipdb; ipdb.set_trace()
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # print(self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
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
        self.base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.total_base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) #带机械臂的整体的质心
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        # xxx added_rand
        self._init_custom_buffers__()
        self._randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        # xxx

        for i in range(self.num_envs):
            # create env instance
            self.env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            self.actor_handle = self.gym.create_actor(self.env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
                        
            self.gym.set_actor_dof_properties(self.env_handle, self.actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(self.env_handle, self.actor_handle)
            # self._process_global_com(body_props,i)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(self.env_handle, self.actor_handle, body_props, recomputeInertia=True)
            self.envs.append(self.env_handle)
            self.actor_handles.append(self.actor_handle)

        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))  # 智元 dof random
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], 0)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        # self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(knee_names)):
        #     self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

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
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, i]
                    else:    
                        dof_props["friction"][i] *= self.joint_friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_joint_damping:
                    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, i]
                    else:
                        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, 0]
                        
                if self.cfg.domain_rand.randomize_joint_armature:
                    if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                        dof_props["armature"][i] *= self.joint_armatures_coeffs[env_id, i]
                    else:
                        dof_props["armature"][i] *= self.joint_armatures_coeffs[env_id, 0]
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
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        # xxx added_rand

        self.cfg.domain_rand.rand_interval = np.ceil(self.cfg.domain_rand.rand_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_duration = np.ceil(self.cfg.domain_rand.gravity_rand_interval * self.cfg.domain_rand.gravity_impulse_duration)

        # xxx added



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
    
    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return 0.8 * torch.exp(-10 * lin_vel_error) + 0.2 * torch.exp(-40 * lin_vel_error)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_lin_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (
            self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev
        )
        # return lin_vel_error
        return delta_phi
    
    def _reward_tracking_ang_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (
            self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev
        )
        # return ang_vel_error
        return delta_phi
        