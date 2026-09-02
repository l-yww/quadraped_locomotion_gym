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
# import cv2
import numpy as np
from isaacgym import gymapi
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import time
# import isaacgym
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from wheel_legged_gym.utils import Logger_pd, Logger_foot
from isaacgym.torch_utils import *
import torch.nn.functional as F

import torch
from tqdm import tqdm
from datetime import datetime

def play(args):

    args.run_name = args.task
    print('args.task-------------------------',args.task)
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.mode.use_net = USE_NET

    env_cfg.seed = 10
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    '-----观测噪声----'
    env_cfg.noise.add_noise = False #True
    env_cfg.noise.noise_level = 1
    '-----扰动、外界摩擦和恢复系数----'
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_rigids_after_start = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False 
    '-----质量、质心、转动惯量----'
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_inertia = False 
    env_cfg.domain_rand.randomize_com_displacement = False
    '-----电机能力----'
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_PD_factor = False
    '-----编码器的随机偏置----'
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    '-----固定延迟模拟----'
    env_cfg.domain_rand.fixed_action_delay = False
    env_cfg.domain_rand.fixed_torque_delay = False
    env_cfg.domain_rand.fixed_obs_delay = False
    '-----随机延迟模拟----'
    env_cfg.domain_rand.add_dof_lag = False
    env_cfg.domain_rand.add_imu_lag = False
    env_cfg.domain_rand.add_action_lag = True
    '-----电机的阻尼摩擦特性----'
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False

    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'polices')
        if not os.path.exists(path):
            os.makedirs(path)
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger_pd = Logger_foot(env.dt)  #Logger_pd(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_state_log = 700 # number of steps before plotting states

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.8)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 100.0, (1920, 1080))

    CoM_offset_compensate = False
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(stop_state_log * 10):

        if FIX_COMMAND:
            env.commands[:, 0] = 0.6
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.36
        else:
            env.commands[:, 0] = env.command_vel_x
            env.commands[:, 1] = env.command_vel_y
            env.commands[:, 2] = env_cfg.init_state.pos[2] + env.command_height_z

        if i > stop_state_log - 3 and i < stop_state_log - 1:
            logger_pd.plot_states()  # 绘图
            logger_pd.print_rewards()

        actions = policy(obs.detach()) 

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_pos[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        logger_pd.log_states(
            {
                # "command_height": env.commands[robot_index, 2].item(),
                # "base_height": env.base_height[robot_index].item(),
                # "command_ang_vel": env.commands[robot_index, 1].item(),
                # "base_ang_vel": env.base_ang_vel[robot_index, 2].item(),
                # "command_vel_x": env.commands[robot_index, 0].item(),
                # "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                # "base_roll": env.base_euler_rpy[robot_index, 0].item(),
                # "base_pitch": env.base_euler_rpy[robot_index, 1].item(),
                # "base_yaw": env.base_euler_rpy[robot_index, 2].item(),
                
                # "left_hip_vel": env.dof_vel[robot_index, 0].item(),
                # "left_hip_torque": env.torques[robot_index, 0].item(),
                # "left_knee_vel": env.dof_vel[robot_index, 1].item(),
                # "left_knee_torque": env.torques[robot_index, 1].item(),
                # "left_wheel_vel": env.dof_vel[robot_index, 2].item(),
                # "left_wheel_torque": env.torques[robot_index, 2].item(),
                # "right_hip_vel": env.dof_vel[robot_index, 3].item(),
                # "right_hip_torque": env.torques[robot_index, 3].item(),
                # "right_knee_vel": env.dof_vel[robot_index, 4].item(),
                # "right_knee_torque": env.torques[robot_index, 4].item(),
                # "right_wheel_vel": env.dof_vel[robot_index, 5].item(),
                # "right_wheel_torque": env.torques[robot_index, 5].item(),
                # # 加入关节角度显示
                
                # "left_hip_pos": env.dof_pos[robot_index, 0].item(),
                # "left_knee_pos": env.dof_pos[robot_index, 1].item(),
                # "left_wheel_pos": env.dof_pos[robot_index, 2].item(),
                # "right_hip_pos": env.dof_pos[robot_index, 3].item(),
                # "right_knee_pos": env.dof_pos[robot_index, 4].item(),
                # "right_wheel_pos": env.dof_pos[robot_index, 5].item(),


                # "left_hip_action": env.actions[robot_index, 0].item() * env_cfg.control.pos_action_scale,
                # "left_knee_action": env.actions[robot_index, 1].item() * env_cfg.control.pos_action_scale,
                # "left_wheel_action": env.actions[robot_index, 2].item() * env_cfg.control.vel_action_scale,
                # "right_hip_action": env.actions[robot_index, 3].item() * env_cfg.control.pos_action_scale,
                # "right_knee_action": env.actions[robot_index, 4].item() * env_cfg.control.pos_action_scale,
                # "right_wheel_action": env.actions[robot_index, 5].item() * env_cfg.control.vel_action_scale,

                # "left_hip_ref_pos": env.ref_dof_pos[robot_index, 0].item(),
                # "left_knee_ref_pos": env.ref_dof_pos[robot_index, 1].item(),
                # "left_wheel_ref_vel": env.ref_dof_vel[robot_index, 2].item(),
                # "right_hip_ref_pos": env.ref_dof_pos[robot_index, 3].item(),
                # "right_knee_ref_pos": env.ref_dof_pos[robot_index, 4].item(),
                # "right_wheel_ref_vel": env.ref_dof_vel[robot_index, 5].item(),
                # NOTE delta pos x
                # "delta_foot_pos_x_l": env.feet_obs_l[robot_index,0].item(),
                # "delta_foot_pos_x_r": env.feet_obs_r[robot_index,0].item(),
                # NOTE base height 
                "base_height": env.base_height[robot_index].item(),
                "base_height_cmd": env.commands[robot_index,2].item(),
                # NOTE contact force
                # "contact_forces_buff_l_x": contact_force_left[robot_index,0].item(),
                # "contact_forces_buff_l_y": contact_force_left[robot_index,1].item(),
                # # "contact_forces_buff_l_z": contact_force_left[robot_index,2].item(),
                # "contact_forces_buff_r_x": contact_force_right[robot_index,0].item(),
                # "contact_forces_buff_r_y": contact_force_right[robot_index,1].item(),
                # # "contact_forces_buff_r_z": contact_force_right[robot_index,2].item(),

                # NOTE torque
                "left_hip_roll_torque": env.torques[robot_index, 0].item(),
                "left_hip_pitch_torque": env.torques[robot_index, 1].item(),
                "left_knee_torque": env.torques[robot_index, 2].item(),
                "left_foot_torque": env.torques[robot_index, 3].item(),
                "left_wheel_torque": env.torques[robot_index, 4].item(),
                "right_hip_roll_torque": env.torques[robot_index, 5].item(),
                "right_hip_pitch_torque": env.torques[robot_index, 6].item(),
                "right_knee_torque": env.torques[robot_index, 7].item(),
                "right_foot_torque": env.torques[robot_index, 8].item(),
                "right_wheel_torque": env.torques[robot_index, 9].item(),

                # NOTE real vel
                "left_hip_roll_vel": env.dof_vel[robot_index, 0].item(),
                "left_hip_pitch_vel": env.dof_vel[robot_index, 1].item(),
                "left_knee_vel": env.dof_vel[robot_index, 2].item(),
                "left_foot_vel": env.dof_vel[robot_index, 3].item(),
                "left_wheel_vel": env.dof_vel[robot_index, 4].item(),
                "right_hip_roll_vel": env.dof_vel[robot_index, 5].item(),
                "right_hip_pitch_vel": env.dof_vel[robot_index, 6].item(),
                "right_knee_vel": env.dof_vel[robot_index, 7].item(),
                "right_foot_vel": env.dof_vel[robot_index, 8].item(),
                "right_wheel_vel": env.dof_vel[robot_index, 9].item(),

                # NOTE  real pos
                "left_hip_roll_pos": env.dof_pos[robot_index, 0].item(),
                "left_hip_pitch_pos": env.dof_pos[robot_index, 1].item(),
                "left_knee_pos": env.dof_pos[robot_index, 2].item(),
                "left_foot_pos": env.dof_pos[robot_index, 3].item(),
                "left_wheel_pos": env.dof_pos[robot_index, 4].item(),
                "right_hip_roll_pos": env.dof_pos[robot_index, 5].item(),
                "right_hip_pitch_pos": env.dof_pos[robot_index, 6].item(),
                "right_knee_pos": env.dof_pos[robot_index, 7].item(),
                "right_foot_pos": env.dof_pos[robot_index, 8].item(),
                "right_wheel_pos": env.dof_pos[robot_index, 9].item(),

                # NOTE # 加入关节角度显示
                "base_vel_x": env.base_lin_vel[robot_index,0].item(),
                "base_vel_y": env.base_lin_vel[robot_index,1].item(),
                "base_vel_z": env.base_lin_vel[robot_index,2].item(),
                "cmd_vel_x": env.commands[robot_index,0].item(),
                
                # NOTE action
                "left_hip_roll_action": env.actions[robot_index, 0].item() * env_cfg.control.pos_action_scale,
                "left_hip_pitch_action": env.actions[robot_index, 1].item() * env_cfg.control.pos_action_scale,
                "left_knee_action": env.actions[robot_index, 2].item() * env_cfg.control.pos_action_scale,
                "left_foot_action": env.actions[robot_index, 3].item() * env_cfg.control.pos_action_scale,
                "left_wheel_action": env.actions[robot_index, 4].item() * env_cfg.control.vel_action_scale,
                "right_hip_roll_action": env.actions[robot_index, 5].item() * env_cfg.control.pos_action_scale,
                "right_hip_pitch_action": env.actions[robot_index, 6].item() * env_cfg.control.pos_action_scale,
                "right_knee_action": env.actions[robot_index, 7].item() * env_cfg.control.pos_action_scale,
                "right_foot_action": env.actions[robot_index, 8].item() * env_cfg.control.pos_action_scale,
                "right_wheel_action": env.actions[robot_index, 9].item() * env_cfg.control.vel_action_scale,
  

                # NOTE ref pos
                "left_hip_roll_ref_pos": env.ref_dof_pos[robot_index, 0].item(),
                "left_hip_pitch_ref_pos": env.ref_dof_pos[robot_index, 1].item(),
                "left_knee_ref_pos": env.ref_dof_pos[robot_index, 2].item(),
                "left_foot_ref_pos": env.ref_dof_pos[robot_index, 3].item(),
                "left_wheel_ref_pos": env.ref_dof_vel[robot_index, 4].item(),
                "right_hip_roll_ref_pos": env.ref_dof_pos[robot_index, 5].item(),
                "right_hip_pitch_ref_pos": env.ref_dof_pos[robot_index, 6].item(),
                "right_knee_ref_pos": env.ref_dof_pos[robot_index, 7].item(),
                "right_foot_ref_pos": env.ref_dof_pos[robot_index, 8].item(),
                "right_wheel_ref_pos": env.ref_dof_vel[robot_index, 9].item(),

                # NOTE IMU
                "base_roll": env.base_euler_rpy[robot_index, 0].item(),
                "base_pitch": env.base_euler_rpy[robot_index, 1].item(),
                "base_yaw": env.base_euler_rpy[robot_index, 2].item(),
                "ang_vel_x": env.base_ang_vel[robot_index, 0].item(),
                "ang_vel_y": env.base_ang_vel[robot_index, 1].item(),
                "ang_vel_z": env.base_ang_vel[robot_index, 2].item(),
                "cmd_ang_vel": env.commands[robot_index,1].item(),
                
                # NOTE base pos
                "base_pos_x": env.base_pos[robot_index, 0].item(),
                "base_pos_y": env.base_pos[robot_index, 1].item(),
                "base_pos_z": env.base_pos[robot_index, 2].item(),
            }
            )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger_pd.log_rewards(infos["episode"], num_episodes)
    if RENDER:
        video.release()
    logger_pd.print_rewards()
    
    
if __name__ == '__main__':
    x_vel_cmd , y_vel_cmd , height_cmd = 0., 0., 0.
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    USE_NET = True
    EXPORT_POLICY = True
    RENDER = False
    FIX_COMMAND = False
    MOVE_CAMERA = False
    args = get_args()
    args.task= 'cowa_net_sin_10dof' # "cowa_net_sin"
    args.num_envs = 1
    args.experiment_name = args.task
    args.load_run = "Dec23_13-32-21_v2_net_sin_10dof_[w_RD]"
    args.checkpoint = -1
    args.headless = False
    play(args)