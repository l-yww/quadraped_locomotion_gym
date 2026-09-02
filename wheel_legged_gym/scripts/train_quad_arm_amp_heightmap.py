# quadruped_arm_him_amp + actor 前视高程图 训练入口。
# 用法与 train_quad_arm_him_amp.py 一致, 仅 task/experiment_name 换成 heightmap 版。

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
    args.task = 'quadruped_arm_amp_heightmap'
    create_folder(args.task)
    args.num_envs = 4096
    args.experiment_name = args.task
    args.run_name = 'trimesh-heightmap'
    args.resume = False
    args.headless = True
    args.load_run = ''
    args.checkpoint = -1
    args.max_iterations = 5000000000
    cuda = 4
    args.sim_device = f"cuda:{cuda}"
    args.rl_device = f"cuda:{cuda}"
    train(args)
