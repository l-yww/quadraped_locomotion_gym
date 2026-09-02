"""悬空 sin 跟踪训练入口。

用法：
    python3 wheel_legged_gym/scripts/train_joint_track.py --headless

如需覆盖参数（num_envs / max_iterations / resume 等），改下方 __main__ 或用命令行 args。
"""
import numpy as np
import os
from datetime import datetime

import isaacgym  # noqa: F401  必须在 torch 之前 import
from wheel_legged_gym.envs import *  # noqa: F401,F403  触发 task_registry.register
from wheel_legged_gym.utils import get_args, task_registry
import torch


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
    args.task = 'quadruped_joint_track'
    create_folder(args.task)
    args.num_envs = 4096
    args.experiment_name = args.task
    args.run_name = 'joint_track_smooth'   # 加了 action 正则 + 降 entropy 的版本
    args.resume = False
    args.headless = True
    args.load_run = ''
    args.checkpoint = -1
    args.max_iterations = 10001
    train(args)
