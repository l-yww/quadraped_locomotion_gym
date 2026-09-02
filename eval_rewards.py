# -*- coding: utf-8 -*-
"""
针对 quadruped_arm_him_amp 训练环境,逐个 checkpoint 评估每个 reward 分项。

原理:
  - 复用 task_registry 建 env + AMPRunner_HIM (不进训练循环)
  - 对 run 目录下每个 model_*.pt: runner.load(path) → 跑纯推理 rollout (复刻 learn 的 rollout 段,不调 alg.update())
  - rollout 中收集每个结束 episode 的 infos['episode']['rew_*'],求均值
      * rew_xxx: env 原生 task reward 分项 (tracking/orientation/foot_slip/...)
      * rew_amp: 判别器给的 style reward (runner 在 rollout 里写入)
  - 终端逐模型逐项打印; 最后按 task 总和 (所有 rew_xxx 之和, 不含 rew_amp) 选最高模型

用法:
    python eval_rewards.py --load_run 0713                    # 评估 logs/quadruped_arm_him_amp/0713 下所有 ckpt
    python eval_rewards.py --load_run 0713 --num_steps 800    # 每个 ckpt 跑 800 步
    python eval_rewards.py --load_run 0713 --ckpt 91000,81500 # 只评估指定几个 ckpt (逗号分隔)
    python eval_rewards.py --list_runs                         # 列出所有 run 目录

  --task / --load_run / --num_envs / --headless 直接走 isaacgym 的参数解析,
  --num_steps / --ckpt / --list_runs 是本脚本自定义参数。

注意: 需要 isaacgym 环境。import isaacgym 必须在 torch 之前。
"""
import os
import sys
import glob
import argparse
import re
from collections import defaultdict

# import isaacgym 必须在 torch 前
import isaacgym  # noqa: F401
import torch
import numpy as np

# 关键: 必须先 import envs 包 (触发 envs/__init__.py 注册所有环境 + algo 初始化链),
# 再 import task_registry. 顺序反了会触发循环 import (task_registry 内部 eval(runner_class_name)
# 依赖 envs 已注册)。play 脚本就是这个顺序。
from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils import get_args, task_registry
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR


def list_runs(experiment_name="quadruped_arm_him_amp"):
    log_root = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs', experiment_name)
    if not os.path.isdir(log_root):
        print(f"[ERR] log root 不存在: {log_root}")
        return
    runs = sorted([d for d in os.listdir(log_root)
                   if os.path.isdir(os.path.join(log_root, d))])
    print(f"log root: {log_root}")
    print(f"runs ({len(runs)}):")
    for r in runs:
        n = len(glob.glob(os.path.join(log_root, r, "model_*.pt")))
        print(f"  {r}  ({n} checkpoints)")


def collect_checkpoints(run_dir, ckpt_filter=None):
    """收集 run_dir 下所有 model_*.pt, 返回 [(iter_int, path), ...] 按 iter 升序."""
    files = glob.glob(os.path.join(run_dir, "model_*.pt"))
    out = []
    for f in files:
        m = re.search(r"model_(\d+)\.pt$", f)
        if not m:
            continue
        it = int(m.group(1))
        if ckpt_filter and it not in ckpt_filter:
            continue
        out.append((it, f))
    out.sort(key=lambda x: x[0])
    return out


def build_env_and_runner(args):
    """建 env + runner (复刻 play 脚本的 cfg 覆盖, 但用评估规模)."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 评估规模 (用户选 "快(小)")
    env_cfg.env.num_envs = 1024
    env_cfg.sim.max_gpu_contact_pairs = 2 ** 14
    env_cfg.env.env_spacing = 1.0

    # 用训练地形 (用户选)
    # terrain_proportions 保持 config 原样, 不覆盖

    env_cfg.env.episode_length_s = 200
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s

    # 评估走 mesh
    env_cfg.terrain.mesh_type = 'plane' #plane trimesh
    print("当前评估地形：" + env_cfg.terrain.mesh_type)

    # 噪声/扰动保持 config (贴近训练分布)

    # 不建 log_dir (评估不需要 tensorboard)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = False  # 我们手动 load, 不让 make_alg_runner 自动 load
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    return env, ppo_runner, train_cfg, env_cfg


def eval_one_checkpoint(runner, env, num_steps):
    """对已 load 的模型跑纯推理 rollout, 收集 episode 分项 reward.
    复刻 AMPRunner_HIM.learn 的 rollout 段 (line 334-398), 但去掉 alg.update() 和 estimator 训练.

    返回: dict {rew_name: mean_value}, 含所有 rew_* (task 分项 + rew_amp)
    """
    alg = runner.alg
    device = runner.device
    disc = alg.discriminator

    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    amp_obs = env.get_amp_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs = obs.to(device)
    critic_obs = critic_obs.to(device)
    amp_obs = amp_obs.to(device)

    # 评估: 策略走推理 (inference), 判别器 eval
    alg.actor_critic.eval()
    disc.eval()

    ep_reward_sums = defaultdict(list)  # rew_name -> [per-episode mean values]

    with torch.inference_mode():
        for i in range(num_steps):
            actions = alg.act(obs, critic_obs, amp_obs)
            # 注意: AMP env.step 返回 8 元组
            obs, privileged_obs, rewards, dones, infos, reset_env_ids, \
                termination_privileged_obs, terminal_amp_states = env.step(actions)
            next_amp_obs = env.get_amp_observations()

            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs = obs.to(device)
            critic_obs = critic_obs.to(device)
            next_amp_obs = next_amp_obs.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            # terminal states: 用 reset 前的 amp 状态
            if reset_env_ids.numel() > 0:
                next_amp_obs_with_term = torch.clone(next_amp_obs)
                next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
            else:
                next_amp_obs_with_term = next_amp_obs

            # 判别器算 style reward (predict_amp_reward 内部会 eval/train 切换,
            # 但我们外层再强制 eval, 保证不更新 BN/running stats)
            rewards, amp_reward = disc.predict_amp_reward(
                amp_obs, next_amp_obs_with_term, rewards,
                normalizer=alg.amp_normalizer,
                style_reward_normalizer=alg.style_reward_normalizer,
            )
            amp_obs = next_amp_obs

            # 复刻 runner: 把 rew_amp 写进 episode info
            if 'episode' in infos:
                infos['episode']['rew_amp'] = amp_reward / env.dt
                ep = infos['episode']
                for k, v in ep.items():
                    if k.startswith('rew_'):
                        # v 是该 batch 中所有结束 env 的均值 (标量 tensor)
                        try:
                            ep_reward_sums[k].append(float(v.mean().item()))
                        except Exception:
                            pass

            # process_env_step 主要是把 transition 存进 amp_storage, 评估不需要存
            # 但 reset 处理 (env 内部已做), 这里只保证 obs/amp_obs 状态推进

    # 汇总每个 rew_* 的均值 (跨所有 episode)
    result = {}
    for k, vals in ep_reward_sums.items():
        result[k] = float(np.mean(vals)) if vals else 0.0
    return result


def main():
    # 自己解析 --task/--load_run (get_args 给 --task 设了默认值 anymal_c_flat 会干扰, 所以自己拿)
    # --num_steps/--ckpt/--list_runs 是本脚本自定义; 其余 (--headless/--rl_device 等) 交给 get_args
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="quadruped_arm_him_amp",
                        help="环境注册名 (experiment_name)")
    parser.add_argument("--load_run", type=str, default=None,
                        help="run 目录名 (logs/<task>/<load_run>), 不给则列出所有 run")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="每个 ckpt 跑多少 rollout 步")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="只评估指定 iter 的 ckpt, 逗号分隔, 如 --ckpt 91000,81500")
    parser.add_argument("--list_runs", action="store_true",
                        help="只列出所有 run 目录, 不评估")
    args_ext, remaining = parser.parse_known_args()

    if args_ext.list_runs:
        list_runs(args_ext.task)
        return

    if not args_ext.load_run:
        print("未指定 --load_run, 列出所有 run:")
        list_runs(args_ext.task)
        return

    # 清掉 get_args 不认识的参数 (我们自己解析的), 只留 isaacgym 认识的
    sys.argv = [sys.argv[0]] + remaining

    isaac_args = get_args()
    isaac_args.headless = True
    isaac_args.task = args_ext.task  # 强制 task, 不用 get_args 默认的 anymal_c_flat
    isaac_args.num_envs = 1024

    run_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs',
                           args_ext.task, args_ext.load_run)
    if not os.path.isdir(run_dir):
        print(f"[ERR] run 目录不存在: {run_dir}")
        return

    ckpt_filter = None
    if args_ext.ckpt:
        ckpt_filter = {int(x.strip()) for x in args_ext.ckpt.split(",") if x.strip()}
    ckpts = collect_checkpoints(run_dir, ckpt_filter)
    if not ckpts:
        print(f"[ERR] {run_dir} 下没有 model_*.pt")
        return

    num_envs = 1024
    print(f"共 {len(ckpts)} 个 checkpoint, 每个 rollout {args_ext.num_steps} 步, num_envs={num_envs}")
    print(f"{'='*80}")

    # 建 env + runner 一次, 复用
    env, runner, train_cfg, env_cfg = build_env_and_runner(isaac_args)
    print(f"env: {env.num_envs} envs, terrain_proportions={env_cfg.terrain.terrain_proportions}")
    print(f"{'='*80}")

    all_results = []  # [(iter, result_dict), ...]
    for idx, (it, path) in enumerate(ckpts):
        print(f"\n[{idx+1}/{len(ckpts)}] model_{it}.pt  ({os.path.basename(path)})")
        try:
            runner.load(path, load_optimizer=False)
        except Exception as e:
            print(f"  [LOAD FAIL] {e}")
            continue

        result = eval_one_checkpoint(runner, env, args_ext.num_steps)
        if not result:
            print("  (无 episode 结束, 步数太少, 增大 --num_steps)")
            continue

        # 打印每项 reward
        task_total = 0.0
        style_val = 0.0
        print(f"  --- reward 分项 (均值) ---")
        for name in sorted(result.keys()):
            v = result[name]
            if name == 'rew_amp':
                style_val = v
                print(f"    {name:30s}: {v:+10.4f}   [style]")
            else:
                task_total += v
                print(f"    {name:30s}: {v:+10.4f}")
        print(f"  {'-'*40}")
        print(f"  {'task 总和 (不含 style)':30s}: {task_total:+10.4f}")
        print(f"  {'style (rew_amp)':30s}: {style_val:+10.4f}")
        print(f"  {'total (task+style)':30s}: {task_total+style_val:+10.4f}")

        all_results.append((it, result, task_total))

    # 选最高
    print(f"\n{'='*80}")
    print("=== 汇总 (按 task 总和排序) ===")
    if not all_results:
        print("无有效结果")
        return
    all_results.sort(key=lambda x: x[2], reverse=True)
    print(f"{'iter':>8}  {'task_total':>12}  {'style':>10}  {'total':>12}")
    for it, result, task_total in all_results:
        style_val = result.get('rew_amp', 0.0)
        print(f"{it:>8}  {task_total:>+12.4f}  {style_val:>+10.4f}  {task_total+style_val:>+12.4f}")

    best_it, best_result, best_task = all_results[0]
    print(f"\n>>> 最高 (按 task 总和): model_{best_it}.pt  task_total={best_task:+.4f}")
    print(f">>> 路径: {os.path.join(run_dir, f'model_{best_it}.pt')}")


if __name__ == "__main__":
    main()
