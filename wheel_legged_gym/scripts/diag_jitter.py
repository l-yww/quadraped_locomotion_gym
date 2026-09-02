"""诊断腿抖根源：加载策略，对比 dof_pos / dof_vel / torque 的高频振荡。
- dof_pos 平滑但 dof_vel/torque 高频振荡 → PD 放大抖（sigma 太紧逼策略微调）
- dof_pos 也抖 → 策略 action 抖"""
import isaacgym  # noqa
import torch
from wheel_legged_gym.envs import task_registry
from wheel_legged_gym.utils import get_args


def high_freq_energy(series):
    """序列的高频能量占比：先 detrend（去趋势），看残差的标准差。
    跟踪正弦是低频的，抖动是高频的，残差大=抖。"""
    import numpy as np
    s = np.asarray(series, dtype=float)
    # 简单 detrend：减去 5 点滑动平均
    if len(s) > 10:
        k = 5
        smooth = np.convolve(s, np.ones(k)/k, mode='same')
        resid = s - smooth
        return float(resid.std())
    return float(s.std())


def main():
    args = get_args()
    args.task = 'quadruped_joint_track'
    args.headless = True
    args.resume = True
    args.load_run = -1
    args.checkpoint = -1
    args.num_envs = 8

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    policy = ppo_runner.get_inference_policy(device=env.device)

    std = ppo_runner.alg.actor_critic.std.detach().mean().item()
    print(f"\npolicy std: {std:.4f}  sigma: {env_cfg.rewards.tracking_sigma}")

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    N = 200
    pos_hist, vel_hist, torq_hist = [], [], []
    track_hist = []
    for s in range(N):
        with torch.no_grad():
            action = policy(obs)
        out = env.step(action)
        obs = out[0]
        env.compute_ref_state()
        track_hist.append(env._reward_tracking_joint_pos().mean().item())
        pos_hist.append(env.dof_pos[0].cpu().numpy().copy())
        vel_hist.append(env.dof_vel[0].cpu().numpy().copy())
        torq_hist.append(env.torques[0].cpu().numpy().copy())

    import numpy as np
    pos_hist = np.array(pos_hist)      # (N,12)
    vel_hist = np.array(vel_hist)
    torq_hist = np.array(torq_hist)

    print(f"\n========== RESULT ==========")
    print(f"deterministic tracking reward: {np.mean(track_hist):.4f}  (oracle 0.93)")

    names = ['FL_hip','FL_th','FL_cf','FR_hip','FR_th','FR_cf','RL_hip','RL_th','RL_cf','RR_hip','RR_th','RR_cf']
    print(f"\n=== 逐关节「高频残差」（detrend 后 std，越小越平滑）===")
    print(f"{'joint':8s} {'pos':>9s} {'vel':>9s} {'torque':>9s}")
    for j, n in enumerate(names):
        pe = high_freq_energy(pos_hist[:, j])
        ve = high_freq_energy(vel_hist[:, j])
        te = high_freq_energy(torq_hist[:, j])
        flag = "  <-- vel/torque 高频大" if (ve > 1.0 or te > 5.0) else ""
        print(f"{n:8s} {pe:9.4f} {ve:9.3f} {te:9.2f}{flag}")

    print(f"\n=== 判读 ===")
    pos_jitter = np.mean([high_freq_energy(pos_hist[:, j]) for j in range(12)])
    vel_jitter = np.mean([high_freq_energy(vel_hist[:, j]) for j in range(12)])
    print(f"  pos 平均高频残差: {pos_jitter:.4f} rad")
    print(f"  vel 平均高频残差: {vel_jitter:.3f} rad/s")
    if vel_jitter > 1.0 and pos_jitter < 0.03:
        print("  → pos 平滑但 vel 高频振荡：PD 放大抖！sigma 太紧(0.02)逼策略高频微调。")
        print("    修法：sigma 调大到 0.05~0.08（放松跟踪带，减少微调），或降 stiffness。")
    elif pos_jitter > 0.03:
        print("  → pos 本身在抖：策略 action 抖，加 action_rate 正则。")


if __name__ == '__main__':
    main()
