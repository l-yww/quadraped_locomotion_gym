# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, task_registry
import torch
import os

def create_folder(task_name):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    gym_dir = os.path.dirname(script_dir)
    base_dir = os.path.dirname(gym_dir)
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir,exist_ok=True)
    name = task_name
    path = os.path.join(logs_dir, name)
    os.makedirs(path, exist_ok=True)

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )
    task_registry.save_cfgs(name=args.task)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

if __name__ == '__main__':

    args = get_args()
    args.task = 'quadruped_wtw_him_arm_fix_height_scan'
    create_folder(args.task)
    args.num_envs = 4096
    args.experiment_name = args.task
    args.run_name = 'wtw_height_scan_[step_height_max_0.20]_[freq_1.5]_[torque_clip_100_200]_[无编码器随机化]_[measured_points_x_0.5_1_y_-0.4_0.4]_[height_offset_range_-0.05_0.05]'
    args.resume = True #False
    args.headless = True
    args.load_run = 'base_model' #'Aug13_12-58-50_wtw_height_scan_[wo_DR]_[random_latest]_[step_height_max_0.20]_[default_hip_pos_-2]_[stand_base_vel_penality_-4]_[measured_points_x_0.5_1]_[freq_1.5]_[torque_clip_100_200]_[short_frame_stack_10]'
    args.checkpoint = 50000
    args.max_iterations = 5000000000
    cuda = 2
    args.sim_device=f"cuda:{cuda}"
    args.rl_device=f"cuda:{cuda}"
    train(args)
