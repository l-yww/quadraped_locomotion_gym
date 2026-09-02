"""悬空 sin 跟踪可视化：加载训练好的策略，画 12 关节 dof_pos vs ref_dof_pos 对比图。

用法：
    python3 wheel_legged_gym/scripts/play_joint_track.py
    # 默认加载最近一次训练的最新 checkpoint（load_run=-1, checkpoint=-1）

输出：
- terminal 打印逐关节 RMSE 和平均 tracking reward
- 保存 PNG：track_plot.png（12 子图，ref 虚线 + actual 实线）
"""
import isaacgym  # noqa
import os
import torch
import matplotlib
matplotlib.use('Agg')            # 无显示也能存图
import matplotlib.pyplot as plt

from wheel_legged_gym.envs import task_registry
from wheel_legged_gym.utils import get_args

JOINT_NAMES = ['FL_hip', 'FL_thigh', 'FL_calf',
               'FR_hip', 'FR_thigh', 'FR_calf',
               'RL_hip', 'RL_thigh', 'RL_calf',
               'RR_hip', 'RR_thigh', 'RR_calf']


def _save_plot(ref_hist, act_hist, track_rs, n_steps):
    """存 track_plot.png 并打印逐关节 RMSE。Agg 后端，不弹窗。"""
    import numpy as np
    track_mean = sum(track_rs) / max(len(track_rs), 1)
    print(f"--- 窗口平均 tracking reward: {track_mean:.4f}  (oracle 上限 0.93) ---")
    rmses = []
    for j, name in enumerate(JOINT_NAMES):
        err = np.array(act_hist[j]) - np.array(ref_hist[j])
        rmse = float((err ** 2).mean() ** 0.5) if len(err) > 0 else 0.0
        rmses.append(rmse)
    print("逐关节 RMSE [deg]: " + "  ".join(f"{n.split('_')[0]}{n.split('_')[1][:2]}={rmses[i]*57.3:.1f}" for i, n in enumerate(JOINT_NAMES)))

    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle(f'Joint tracking (window avg reward={track_mean:.3f}, oracle ceiling=0.93)\n'
                 f'actual (blue solid) vs ref (red dashed)', fontsize=12)
    xs = list(range(len(ref_hist[0])))
    for j, name in enumerate(JOINT_NAMES):
        axes[j].plot(xs, ref_hist[j], 'r--', lw=1, label='ref')
        axes[j].plot(xs, act_hist[j], 'b-', lw=1.2, label='actual')
        axes[j].set_title(f'{name} (RMSE={rmses[j]*57.3:.1f}°)', fontsize=9)
        axes[j].grid(True, alpha=0.3)
        if j % 3 == 0:
            axes[j].set_ylabel('rad')
    axes[-1].set_xlabel('step')
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), 'track_plot.png')
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"图已保存: {out_path}")


def _save_torque_plot(torq_hist, env):
    """存 torque_plot.png：12 关节力矩曲线 + 力矩限位线，看是否超限/抖动。"""
    import numpy as np
    import os
    # torq_hist 存储结构：[关节0的序列, 关节1的序列, ...] → np.array 后是 (12, T)
    torq_arr = np.array(torq_hist)   # (12, T)
    if torq_arr.ndim == 2 and torq_arr.shape[0] == 12:
        pass   # 已是 (12, T)
    else:
        torq_arr = torq_arr.T        # 兜底：(T,12) → (12,T)
    T = torq_arr.shape[1]
    torque_limits = env.torque_limits.cpu().numpy()   # (12,)
    print(f"--- 力矩统计 ---")
    print(f"  最大力矩: {np.abs(torq_arr).max():.1f} Nm")
    print(f"  平均力矩: {np.abs(torq_arr).mean():.1f} Nm")
    # 逐关节比较超限：(12,T) vs (12,1)
    over = np.abs(torq_arr) > torque_limits[:, None]
    print(f"  超限占比: {over.mean()*100:.2f}%")
    print(f"  力矩限位: {torque_limits}")

    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle('Joint torque (Nm)  red dashed = torque limit', fontsize=12)
    xs = list(range(T))
    for j, name in enumerate(JOINT_NAMES):
        axes[j].plot(xs, torq_arr[j], 'g-', lw=1, label='torque')   # (12,T) → 取第 j 行
        axes[j].axhline(torque_limits[j], color='r', ls='--', lw=0.8)
        axes[j].axhline(-torque_limits[j], color='r', ls='--', lw=0.8)
        axes[j].set_title(f'{name} (max={np.abs(torq_arr[:, j]).max():.0f}Nm)', fontsize=9)
        axes[j].grid(True, alpha=0.3)
        if j % 3 == 0:
            axes[j].set_ylabel('Nm')
    axes[-1].set_xlabel('step')
    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), 'torque_plot.png')
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"图已保存: {out_path}")


def main():
    args = get_args()
    args.task = 'quadruped_joint_track'
    args.headless = True           # 临时headless，看机器人腿实时跟踪
    # 调试卡顿：可临时改成 True 不开 viewer，只存图。若 headless 能跑通则卡在 viewer。
    # args.headless = True
    args.resume = True             # play 必须加载 checkpoint
    if args.load_run is None:
        args.load_run = -1
    if args.checkpoint is None:
        args.checkpoint = -1
    args.num_envs = 1

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # ★ play/eval 时关掉所有 domain_rand 随机化，测标称参数下的真实表现（与 mujoco deploy 对齐）。
    # 保留 default_joint_*（辨识值），只关随机化开关。否则 play 带随机扰动，和 mujoco 固定参数不公平。
    dr = env_cfg.domain_rand
    dr.use_random = True
    dr.push_robots = False
    dr.randomize_friction = False
    dr.randomize_restitution = False
    dr.randomize_base_mass = False
    dr.randomize_inertia = False
    dr.randomize_com_displacement = False
    dr.randomize_motor_strength = True
    # 按 hip/thigh/calf 三种电机分别设 motor_strength（[min,max]相同=固定不随机）
    # 关节索引：hip=[0,3,6,9], thigh=[1,4,7,10], calf=[2,5,8,11]
    # 1.0=标称，<1 电机偏弱，>1 偏强。按真机各电机 gap 分别调。
    # dr.motor_strength_range = [1.0, 1.0]            # 兜底默认（没配下面三组时用这个）
    dr.motor_strength_hip_range = [0.7, 0.7]          # hip 电机
    dr.motor_strength_thigh_range = [0.55, 0.55]      # thigh 电机
    dr.motor_strength_calf_range = [0.45, 0.45]       # calf 电机
    dr.randomize_PD_factor = False
    dr.randomize_motor_offset = False
    dr.add_action_lag = False
    dr.add_dof_lag = False
    dr.add_imu_lag = False
    # ★ 关节物理参数随机化（闭环外，能真正影响力矩，不像 motor_strength 被闭环补偿）
    # friction/damping/armature 改的是仿真器物理参数，PD 要输出更大力矩克服 → 净力矩升高。
    # play 用统一范围（each_joint=False），设固定值 [x,x] = 不随机。
    # 想让力矩升高匹配真机：friction 调大（>1），damping 调大（>1）。
    dr.randomize_joint_friction = False
    dr.randomize_joint_friction_each_joint = False   # 统一范围，不用 per-joint
    dr.joint_friction_range = [1.0, 1.0]             # [min,max]，基准 default_joint_friction × 此值。1.0=辨识值，>1 加摩擦
    dr.randomize_joint_damping = False
    dr.randomize_joint_damping_each_joint = False
    dr.joint_damping_range = [1.0, 1.0]              # 1.0=辨识值，>1 加阻尼
    dr.randomize_joint_armature = False               # armature 先不动（影响惯量，非力矩）
    # 关掉随机初始相位：play/eval 时 init_phase=0，与 mujoco deploy(init_phase=0)对齐，
    # 否则两边相位错开（视觉上像"慢/反相"），其实是相位起点不同。
    env_cfg.ref.random_init_phase = False
    print("[1/4] cfg loaded (domain_rand 全关 + init_phase=0，标称参数)")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[2/4] env created, viewer =", getattr(env, 'viewer', None))
    # env 创建后强制 init_phase=0（保险，防止基类 reset 已随机过）
    if hasattr(env, 'init_phase'):
        env.init_phase[:] = 0.0
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    print("[3/4] runner + policy loaded")
    policy = ppo_runner.get_inference_policy(device=env.device)
    if env.viewer is not None:
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    print(f"[4/4] ready. viewer={'on' if env.viewer is not None else 'off(headless)'}")

    # ========== 导出 onnx（用于部署） ==========
    # 普通 PPO actor：输入 (1, num_observations=190) → 输出 (1, 12) action mean
    # 用 deepcopy 在 CPU 上导出，不动原 actor（原 actor 留在 GPU 给推理用），
    # 否则 actor.to('cpu') 是 in-place，会让 policy(obs) 设备不匹配报错。
    import os as _os
    import copy as _copy
    try:
        actor_cpu = _copy.deepcopy(ppo_runner.alg.actor_critic.actor).to("cpu").eval()
        num_obs = env_cfg.env.num_observations    # 190 = frame_stack(5) × num_single_obs(38)
        dummy = torch.zeros(1, num_obs)
        # 导出到 logs/quadruped_joint_track/exported/。
        # 必须叫 "exported"：get_load_path 内部 runs.remove("exported") 会跳过它，
        # 否则 load_run=-1 会把导出目录当最新 run，里面没 model_*.pt → 越界。
        export_dir = _os.path.join("logs", "quadruped_joint_track", "exported")
        _os.makedirs(export_dir, exist_ok=True)
        onnx_path = _os.path.join(export_dir, "policy.onnx")
        torch.onnx.export(
            actor_cpu, dummy, onnx_path,
            input_names=["obs"], output_names=["action"],
            opset_version=14,
        )
        # 同时导出 jit（部分部署链路用）
        jit_path = _os.path.join(export_dir, "policy.pt")
        torch.jit.script(actor_cpu).save(jit_path)
        print(f"导出 onnx: {onnx_path}  (输入{num_obs}维 → 输出12维)")
        print(f"导出 jit:  {jit_path}")
        del actor_cpu   # 副本用完释放，原 actor 全程在 GPU 未被动过
    except Exception as e:
        print(f"[warn] onnx 导出失败: {e}")

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    # ★ 诊断：打印 motor_strengths 实际值，确认 [2.2,2.2] 是否生效
    if hasattr(env, 'motor_strengths'):
        print(f"[diag] motor_strengths[0] = {env.motor_strengths[0].cpu().numpy()}")
        print(f"[diag] randomize_motor_strength = {env.cfg.domain_rand.randomize_motor_strength}")
        print(f"[diag] motor_strength_range = {env.cfg.domain_rand.motor_strength_range}")

    # 记录全程数据（位置/转速/力矩/base），和 mujoco deploy 同格式，便于对比
    log_t, log_q, log_ref, log_dq, log_tau = [], [], [], [], []
    log_base_z, log_base_ang = [], []
    dt = env.dt

    N_MAX_STEPS = 500   # 跑 500 步(≈10s)自动退出保存；headless 无 viewer 时靠这个退出
    print(f"Running {N_MAX_STEPS} 步(≈{N_MAX_STEPS*dt:.1f}s)... 按 ESC 提前退出。退出后保存 4 张图")
    step = 0
    while step < N_MAX_STEPS:
        with torch.no_grad():
            action = policy(obs)
        out = env.step(action)
        obs = out[0]
        env.compute_ref_state()
        # 记录（env.torques 是经过训练完整力矩链限幅的真实下发力矩）
        log_t.append(step * dt)
        log_q.append(env.dof_pos[0].cpu().numpy().copy())
        log_ref.append(env.ref_dof_pos[0].cpu().numpy().copy())
        log_dq.append(env.dof_vel[0].cpu().numpy().copy())
        # 记录 PD 命令力矩（乘 motor_strength 之前），反映策略+PD 要求的力矩，不被电机效率/闭环补偿掩盖
        log_tau.append(env.torques_cmd[0].cpu().numpy().copy())
        log_base_z.append(env.root_states[0, 2].item())
        log_base_ang.append(env.base_ang_vel[0].cpu().numpy().copy())

        # isaacgym viewer 刷新（看腿实时跟踪）
        if env.viewer is not None:
            env.gym.query_viewer_has_closed(env.viewer)
            env.render()
            if env.gym.query_viewer_has_closed(env.viewer):
                print("Viewer closed, exiting.")
                break

        if step % 100 == 0:
            print(f"step {step:5d}  tracking={env._reward_tracking_joint_pos()[0].item():.3f}")
        step += 1

    # ============ 画 4 张图（与 mujoco deploy 同结构，存 sim2sim/logs/ 便于对比）============
    import numpy as np
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_t = np.array(log_t)
    log_q = np.array(log_q)
    log_ref = np.array(log_ref)
    log_dq = np.array(log_dq)
    log_tau = np.array(log_tau)
    log_base_z = np.array(log_base_z)
    log_base_ang = np.array(log_base_ang)

    err = log_q - log_ref
    rmse = np.sqrt((err ** 2).mean(axis=0))
    print(f"\n=== isaacgym play RESULT ===")
    print(f"平均跟踪误差 RMSE: {rmse.mean():.4f} rad ({np.degrees(rmse.mean()):.2f}°)")
    for n, r in zip(JOINT_NAMES, rmse):
        print(f"  {n:10s}: {r:.4f} rad ({np.degrees(r):.2f}°)")
    # ★ 力矩统计：PD 命令力矩（乘 motor_strength 之前）
    print(f"[diag] motor_strengths[0]={env.motor_strengths[0].mean().item():.2f}")
    print(f"[diag] PD命令力矩(乘ms前) 绝对值最大: {np.abs(log_tau).max():.2f} Nm  均值: {np.abs(log_tau).mean():.2f} Nm")
    print(f"[diag] 逐关节峰值: {np.round(np.abs(log_tau).max(axis=0),1)}")

    log_path = os.path.join("sim2sim", "logs")
    os.makedirs(log_path, exist_ok=True)
    names = JOINT_NAMES

    # Plot 1: 位置 vs ref
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle(f"isaacgym play: Positions vs Ref (avg RMSE={rmse.mean():.4f} rad)")
    for j in range(12):
        axes[j].plot(log_t, log_q[:, j], "b-", lw=1.0, label="actual")
        axes[j].plot(log_t, log_ref[:, j], "r--", lw=1.0, label="ref")
        axes[j].set_title(f"{names[j]} (RMSE={rmse[j]:.4f} rad)", fontsize=9)
        axes[j].set_ylabel("rad"); axes[j].grid(True, alpha=0.3)
        if j == 0: axes[j].legend(fontsize=7)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig(os.path.join(log_path, "isaac_qpos_vs_ref.png"), dpi=150); plt.close()

    # Plot 2: 转速
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle("isaacgym play: Joint Velocities")
    for j in range(12):
        axes[j].plot(log_t, log_dq[:, j], "g-", lw=1.0)
        axes[j].set_title(names[j], fontsize=9); axes[j].set_ylabel("rad/s"); axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig(os.path.join(log_path, "isaac_qvel.png"), dpi=150); plt.close()

    # Plot 3: 力矩（env.torques，经过训练完整力矩链限幅）
    # isaacgym 的 dof_vel 是差分算的（脏），kd*D项 会放大噪声 → 力矩毛刺多。
    # 画原始(淡) + 移动平均(实) 两条线，平均后能看出 sin 趋势。
    def smooth(y, w=11):  # 11点 ≈ 0.22s 窗口
        if len(y) < w:
            return y
        return np.convolve(y, np.ones(w) / w, mode="same")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle("isaacgym play: PD 命令力矩 (乘 motor_strength 之前; 紫=原始, 蓝=移动平均)")
    for j in range(12):
        axes[j].plot(log_t, log_tau[:, j], color="purple", lw=0.5, alpha=0.4)   # 原始(淡)
        axes[j].plot(log_t, smooth(log_tau[:, j]), "b-", lw=1.3)                # 平滑(实)
        axes[j].set_title(f"{names[j]} (max={np.abs(log_tau[:, j]).max():.0f}Nm)", fontsize=9)
        axes[j].set_ylabel("Nm"); axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig(os.path.join(log_path, "isaac_torque_cmd.png"), dpi=150); plt.close()

    # Plot 4: base 稳定性
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("isaacgym play: Base Stability (悬空应稳定)")
    ax1.plot(log_t, log_base_z, "b-", lw=1.5, label="base z"); ax1.set_ylabel("m"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.plot(log_t, log_base_ang[:, 0], lw=1.0, label="ang_vel_x")
    ax2.plot(log_t, log_base_ang[:, 1], lw=1.0, label="ang_vel_y")
    ax2.plot(log_t, log_base_ang[:, 2], lw=1.0, label="ang_vel_z")
    ax2.set_ylabel("rad/s"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig(os.path.join(log_path, "isaac_base_stability.png"), dpi=150); plt.close()

    print(f"\n4 张图已保存到 {log_path}/:")
    print(f"  isaac_qpos_vs_ref.png / isaac_qvel.png / isaac_torque_cmd.png(命令力矩) / isaac_base_stability.png")


if __name__ == '__main__':
    main()

