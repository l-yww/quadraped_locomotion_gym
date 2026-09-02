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
from wheel_legged_gym.utils import get_args, task_registry_stage
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
    env, env_cfg = task_registry_stage.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry_stage.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args,
    )
    task_registry_stage.save_cfgs(name=args.task)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

if __name__ == '__main__':
    args = get_args()
 
    args.task = 'cowa_wbc_stage' #'cowa_10dof' #'cowa_dh' #'cowa_net_sin'
    create_folder(args.task)
    args.num_envs = 4096
    args.experiment_name = args.task
    args.run_name = 'v2_wbc_stage_[command_set_maker]_[w_DR]_[same_foot_x_position=-0.1]' #'v2_dh_10dof_[w_DR]_[com_-0.10_0.10]_[stand_still=1]_[urdf_roll_35]_[arm_pingju]'
    args.resume = False #True
    args.headless = True
    args.load_run = 'Mar19_09-18-48_test_[lower_arm_pd]'
    args.checkpoint = -1

    args.resume_arm = True
    args.load_run_arm="Mar31_18-25-23_v2_arm_[w_DR]_[command_set_maker]"
    args.checkpoint_arm = -1
    
    args.max_iterations = 50000

    cuda = 0
    args.sim_device=f"cuda:{cuda}"
    args.rl_device=f"cuda:{cuda}"
    train(args)