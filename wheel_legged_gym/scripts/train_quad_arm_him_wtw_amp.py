"""wtw + AMP 融合训练入口。

用法：
    python wheel_legged_gym/scripts/train_wtw_him_arm_fix_amp.py

融合环境：wtw 姿态调整 + AMP 专家柔和先验（判别器姿态免疫）。
预期：姿态调整功能保留（wtw）+ 踏步变轻（专家先验）。
"""
import os
import isaacgym  # noqa: F401  必须在 torch 之前
from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils import get_args, task_registry


def create_folder(task_name):
    script_path = os.path.abspath(__file__)
    gym_dir = os.path.dirname(os.path.dirname(script_path))
    base_dir = os.path.dirname(gym_dir)
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, task_name), exist_ok=True)


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
    args.task = 'quadruped_wtw_him_arm_fix_amp'
    create_folder(args.task)
    args.num_envs = 4096
    args.experiment_name = args.task
    args.run_name = 'wtw_amp_v1'
    args.resume = False
    args.headless = True
    args.load_run = ''
    args.checkpoint = -1
    args.max_iterations = 5000000
    cuda = 2
    args.sim_device=f"cuda:{cuda}"
    args.rl_device=f"cuda:{cuda}"
    train(args)

