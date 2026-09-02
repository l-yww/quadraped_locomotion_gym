# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, task_name)
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
    args.task = 'quadruped_arm_him'
    create_folder(args.task)
    args.num_envs = 4096
    args.experiment_name = args.task
    args.run_name = '优化静止'
    args.resume = False
    # args.resume = True
    args.headless = True
    args.load_run = 'Jul04_11-21-55_加入步态时钟2'
    args.checkpoint = -1
    args.max_iterations = 5000000
    cuda = 4
    args.sim_device = f"cuda:{cuda}"
    args.rl_device = f"cuda:{cuda}"
    train(args)
