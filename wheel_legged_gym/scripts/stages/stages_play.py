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
import cv2
import numpy as np
from isaacgym import gymapi
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import time
# import isaacgym
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from wheel_legged_gym.utils import Loggerxlh as Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import json
import copy


# ==================== est_policy =======================

class Integrated_EST_policy(torch.nn.Module):
    def __init__(self, actor, estimator, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack

    def forward(self, obs):
        latent = self.estimator.forward(obs)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        output = torch.cat((action_mean, latent), dim=-1)
        return output

    def export(self, path):
        path = os.path.join(path, "my_policy_est.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 


# ==================== play =======================
def play(args):
    args.run_name = args.task
    print('args.task-------------------------',args.task)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.fix_base_link = False    
    # env_cfg.control.play_test = True     
    # env_cfg.terrain.mesh_type = 'plane'   
    # env_cfg.terrain.num_rows = 4  
    # env_cfg.terrain.num_cols = 4  
    # env_cfg.terrain.curriculum = False     
    # env_cfg.terrain.max_init_terrain_level = 5
    # env_cfg.init_state.rand_init_dof = False

    # env_cfg.noise.add_noise = True
    # env_cfg.noise.noise_level = 0.2

    # env_cfg.domain_rand.push_robots = False 
    # env_cfg.domain_rand.push_interval_s = 2

    # env_cfg.domain_rand.randomize_com_displacement = False
    # env_cfg.domain_rand.randomize_inertia = False 

    # env_cfg.domain_rand.randomize_rigids_after_start = False 
    # env_cfg.domain_rand.randomize_motor_strength = False   
    # env_cfg.domain_rand.randomize_motor_offset = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_default_dof_pos = False

    # env_cfg.domain_rand.add_dof_lag = True
    # env_cfg.domain_rand.add_imu_lag = True
    # env_cfg.domain_rand.add_lag = True

    # env_cfg.domain_rand.randomize_base_mass = False

    # env_cfg.domain_rand.action_noise = 0.0 # 0.02
    # env_cfg.domain_rand.action_delay = 0. # 0.1

    # env_cfg.control.action_smoothness = False
    # env_cfg.noise.curriculum = False
    
    # env_cfg.terrain.static_friction = 1
    # env_cfg.terrain.dynamic_friction = 1


    train_cfg.seed = 123
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
        path = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
        if args.task == "cowa_w_arm_est":
            print("export est policy",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_EST_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.estimator,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            integrated_policy.export(path)
        else:  
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_state_log = 60000 # number of steps before plotting states

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 100.0, (1920, 1080))



    CoM_offset_compensate = True
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    command = [0, 0, 0]
    # for i in tqdm(range(stop_state_log)):
    for i in range(stop_state_log):
        if i > stop_state_log - 4 and i < stop_state_log - 2:
            logger.plot_states()  # 绘图
        if args.task == "cowa_rma":
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
        elif args.task == "cowa_w_arm_est" : #or args.task == "cowa_plane"
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
        else :
            actions = policy(obs.detach()) 
            # print(obs.detach().shape)

        if FIX_COMMAND:
            env.commands[:, 0] = 0.001 # 1.0
            env.commands[:, 1] = 0.001
            env.commands[:, 2] = 0.33
        else:
            env.commands[:, 0] = env.command_vel_x
            env.commands[:, 1] = env.command_vel_y  # 0
            env.commands[:, 2] = env_cfg.init_state.pos[2] + env.command_height_z
            

            if CoM_offset_compensate:
                if i > 200 and i < 600:
                    vel_cmd[:] = env.commands[:, 0] #* np.clip((i - 200) / 400.0, 0, 1)
                # elif i>= 800 and i < 1400:
                #     vel_cmd[:] = env.commands[:, 0] * np.clip((1400 - i) / 600.0, 0, 1)
                else:
                    vel_cmd[:] = 0
                vel_err_intergral += (
                    (vel_cmd - env.base_lin_vel[:, 0])
                    * env.dt
                    * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
                )
                vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
                env.commands[:, 0] = vel_cmd + vel_err_intergral


        if args.task=="cowa_him":
            obs, critic_obs, rews, dones, infos, _, _ = env.step(actions.detach(), HIM = True)
        else:
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
        
        # motor_vel_list.append(env.dof_vel.cpu().detach().numpy().tolist()[0])
        # ang_vel_list.append(env.base_ang_vel[robot_index, 2].item())
        # feet_height_list.append((env.rigid_state[robot_index, 5, 2] - 0.14).item())
        # air_time_list.append(env.feet_air_time[robot_index, 0].item())
        # feet_height_smooth_list.append(env.feet_height_smooth_sum[robot_index].item())
        # for i in range(3):
        #     vel_list[i].append(env.base_lin_vel[robot_index, i].item())
        # a = [0,1,2,3,4,5]
        # print("arm dof ======\n")
        # print(env.q_arm[0,0].item())
        # print(env.q_arm[0,1].item())
        # print(env.q_arm[0,2].item())
        # print(env.q_arm[0,3].item())
        # print(env.q_arm[0,4].item())
        # print(env.q_arm[0,5].item()) 

        logger.log_states(
            {
                "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                "left_hip_vel": env.dof_vel[robot_index, 0].item(),
                "left_hip_torque": env.torques[robot_index, 0].item(),
                "left_knee_vel": env.dof_vel[robot_index, 1].item(),
                "left_knee_torque": env.torques[robot_index, 1].item(),
                "left_wheel_vel": env.dof_vel[robot_index, 2].item(),
                "left_wheel_torque": env.torques[robot_index, 2].item(),
                "right_hip_vel": env.dof_vel[robot_index, 3].item(),
                "right_hip_torque": env.torques[robot_index, 3].item(),
                "right_knee_vel": env.dof_vel[robot_index, 4].item(),
                "right_knee_torque": env.torques[robot_index, 4].item(),
                "right_wheel_vel": env.dof_vel[robot_index, 5].item(),
                "right_wheel_torque": env.torques[robot_index, 5].item(),
                # 加入关节角度显示
                
                "left_hip_pos": env.dof_pos[robot_index, 0].item(),
                "left_knee_pos": env.dof_pos[robot_index, 1].item(),
                "right_hip_pos": env.dof_pos[robot_index, 3].item(),
                "right_knee_pos": env.dof_pos[robot_index, 4].item(),
                
                "left_hip_action": env.actions[robot_index, 0].item(),
                "left_knee_action": env.actions[robot_index, 1].item(),
                "left_wheel_action": env.actions[robot_index, 2].item(),
                "right_hip_action": env.actions[robot_index, 3].item(),
                "right_knee_action": env.actions[robot_index, 4].item(),
                "right_wheel_action": env.actions[robot_index, 5].item(),

                "left_hip_ref_pos": env.ref_dof_pos[robot_index, 0].item(),
                "left_knee_ref_pos": env.ref_dof_pos[robot_index, 1].item(),
                "left_wheel_ref_vel": env.ref_dof_vel[robot_index, 2].item(),
                "right_hip_ref_pos": env.ref_dof_pos[robot_index, 3].item(),
                "right_knee_ref_pos": env.ref_dof_pos[robot_index, 4].item(),
                "right_wheel_ref_vel": env.ref_dof_vel[robot_index, 5].item(),
            }
            )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
    if RENDER:
        video.release()


if __name__ == '__main__':
    # NOTE 更改play_test用来启用机械臂角度的源，是随机课程轨迹还是键盘识别 env.cfg.control.play_test = True 启用键盘，反之
    x_vel_cmd , y_vel_cmd , height_cmd = 0., 0., 0.
    EXPORT_POLICY = True
    RENDER = False
    FIX_COMMAND = True
    MOVE_CAMERA = False
    args = get_args()
    args.task="cowa_stages_jump"
    args.num_envs = 10
    args.experiment_name = "cowa_stages_jump"
    args.load_run="Apr01_16-53-42_更改奖励函数-调整jump高度为0.5"      
    # args.checkpoint = 3000


    play(args) 


