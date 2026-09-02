"""对比 isaacgym vs mujoco 在同一策略下的关节跟踪序列。
isaacgym 侧 headless 跑（关 domain_rand 标称参数），mujoco 侧用 deploy 同条件，
两边各跑 6s，叠加 thigh 关节曲线，定位 sim2sim 差异（幅值/相位）。
"""
import isaacgym  # noqa
import os, numpy as np, torch

# ============ isaacgym 侧 ============
from wheel_legged_gym.envs import task_registry
from wheel_legged_gym.utils import get_args


def run_isaac():
    args = get_args()
    args.task = 'quadruped_joint_track'
    args.headless = True
    args.resume = True
    args.load_run = -1
    args.checkpoint = -1
    args.num_envs = 1
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    # 关 domain_rand，标称参数（与 mujoco deploy 对齐）
    dr = env_cfg.domain_rand
    for attr in ['use_random', 'push_robots', 'randomize_friction', 'randomize_restitution',
                 'randomize_base_mass', 'randomize_inertia', 'randomize_com_displacement',
                 'randomize_motor_strength', 'randomize_PD_factor', 'randomize_motor_offset',
                 'add_action_lag', 'add_dof_lag', 'add_imu_lag',
                 'randomize_joint_friction', 'randomize_joint_damping', 'randomize_joint_armature']:
        setattr(dr, attr, False)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    policy = runner.get_inference_policy(device=env.device)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    # 强制 isaacgym 初始相位=0，与 mujoco deploy(init_phase=0)对齐，
    # 否则 random_init_phase 会让两边相位错开（视觉上像"慢/反相"）
    if hasattr(env, 'init_phase'):
        env.init_phase[:] = 0.0
    N = 300   # 6s / 0.02 = 300 策略步
    t_log, q_log, ref_log = [], [], []
    dt = env.dt
    act_log = []
    dq_log = []
    for s in range(N):
        with torch.no_grad():
            action = policy(obs)
        act_log.append(action[0].cpu().numpy().copy())
        dq_log.append(env.dof_vel[0].cpu().numpy().copy())
        out = env.step(action)
        obs = out[0]
        env.compute_ref_state()
        t_log.append(s * dt)
        q_log.append(env.dof_pos[0].cpu().numpy().copy())
        ref_log.append(env.ref_dof_pos[0].cpu().numpy().copy())
    return np.array(t_log), np.array(q_log), np.array(ref_log), np.array(act_log), np.array(dq_log)


def main():
    print("=== 跑 isaacgym (headless, 标称参数) ===")
    t_i, q_i, ref_i, a_i, dq_i = run_isaac()
    # 存 isaac 数据
    np.savez("/tmp/isaac_track.npz", t=t_i, q=q_i, ref=ref_i, act=a_i, dq=dq_i)
    print(f"isaacgym thigh(idx1) 幅值={(q_i[:,1].max()-q_i[:,1].min())/2:.4f}  末值={q_i[-1,1]:.4f}")
    # action 抖动 = 逐帧差分均值
    i_adiff = np.mean(np.abs(np.diff(a_i, axis=0)))
    i_jitter = np.std(np.diff(a_i, axis=0))
    print(f"isaacgym action 逐帧|Δ|均值={i_adiff:.5f}  std={i_jitter:.5f}")
    # dof_vel 抖动（高频残差：减 5 点滑动平均后的 std）
    def hf(s):
        s = np.asarray(s, float)
        if len(s) > 10:
            sm = np.convolve(s, np.ones(5)/5, mode='same')
            return float((s-sm).std())
        return float(s.std())
    i_dqjitter = np.mean([hf(dq_i[:, j]) for j in range(12)])
    print(f"isaacgym dof_vel 高频残差均值={i_dqjitter:.4f}")

    # ============ mujoco 侧（直接调 deploy 逻辑）============
    import mujoco
    import onnxruntime as ort
    KP, KD, DT, DEC, ASC, FS, NS = 160.0, 5.0, 0.005, 4, 0.20, 5, 38
    A = np.array([0.2,0.3,0.3]*4, np.float32); CY = 2.0
    sess = ort.InferenceSession('logs/quadruped_joint_track/exported/policy.onnx')
    m = mujoco.MjModel.from_xml_path('sim2sim/cowa2_description_mujoco/xml/cowa2_d1_arm_2.xml')
    m.opt.timestep = DT
    d = mujoco.MjData(m); d.qpos[7:] = 0; mujoco.mj_forward(m, d)
    hist = np.zeros(FS*NS, np.float32); act = np.zeros(12, np.float32); la = np.zeros(12, np.float32)
    def build(qj, dqj, phase):
        o = np.zeros(NS, np.float32); o[:12]=qj*1.0; o[12:24]=dqj*0.05; o[24:36]=la
        o[36]=np.sin(2*np.pi*phase); o[37]=np.cos(2*np.pi*phase); return np.clip(o,-100,100)
    N = 300 * DEC  # mujoco 用 sim 步，300 策略步 = 1200 sim 步
    t_m, q_m, ref_m, act_m, dq_m = [], [], [], [], []
    sim_t = 0.0
    for i in range(N):
        if i % DEC == 0:
            phase = sim_t / CY
            o = build(d.qpos[7:19].astype(np.float32), d.qvel[6:18].astype(np.float32), phase)
            hist[:(FS-1)*NS] = hist[NS:]; hist[(FS-1)*NS:] = o
            act = sess.run(['action'], {'obs': hist[np.newaxis].astype(np.float32)})[0].squeeze().astype(np.float32)
            act = np.clip(act, -20, 20)
        tgt = act * ASC
        qj = d.qpos[7:19]; dqj = d.qvel[6:18]
        d.ctrl[:] = (tgt-qj)*KP + (0-dqj)*KD
        d.qpos[0:3]=[0,0,0.5]; d.qpos[3:7]=[0,0,0,1]; d.qvel[0:6]=0
        if i % DEC == 0:   # 只记录策略步（和 isaac 对齐）
            t_m.append(sim_t); q_m.append(qj.copy())
            ref_m.append(A * np.sin(2*np.pi*sim_t/CY))
            act_m.append(act.copy())
            dq_m.append(dqj.copy())
        mujoco.mj_step(m, d); sim_t += DT; la = act.copy()
    t_m, q_m, ref_m, act_m, dq_m = np.array(t_m), np.array(q_m), np.array(ref_m), np.array(act_m), np.array(dq_m)
    print(f"mujoco   thigh(idx1) 幅值={(q_m[:,1].max()-q_m[:,1].min())/2:.4f}  末值={q_m[-1,1]:.4f}")
    m_adiff = np.mean(np.abs(np.diff(act_m, axis=0)))
    m_jitter = np.std(np.diff(act_m, axis=0))
    print(f"mujoco   action 逐帧|Δ|均值={m_adiff:.5f}  std={m_jitter:.5f}")
    m_dqjitter = np.mean([hf(dq_m[:, j]) for j in range(12)])
    print(f"mujoco   dof_vel 高频残差均值={m_dqjitter:.4f}")
    print(f"→ action 抖动比 isaac/mujoco = {i_adiff/max(m_adiff,1e-9):.2f}  (>1 说明 isaac action 更跳)")
    print(f"→ dof_vel 抖动比 isaac/mujoco = {i_dqjitter/max(m_dqjitter,1e-9):.2f}  (>1 说明 isaac vel 更脏)")

    # ============ 叠加对比图（12 关节，isaac vs mujoco vs ref）============
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle("isaacgym vs mujoco joint tracking (same policy, nominal params)")
    names = ['FL_hip','FL_thigh','FL_calf','FR_hip','FR_thigh','FR_calf',
             'RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf']
    for j in range(12):
        axes[j].plot(t_i, ref_i[:, j], 'k--', lw=1, label='ref', alpha=0.7)
        axes[j].plot(t_i, q_i[:, j], 'b-', lw=1.3, label='isaac')
        axes[j].plot(t_m, q_m[:, j], 'r-', lw=1.0, label='mujoco', alpha=0.8)
        axes[j].set_title(names[j], fontsize=9)
        axes[j].grid(True, alpha=0.3)
        if j == 0: axes[j].legend(fontsize=7)
    axes[-1].set_xlabel('time [s]')
    plt.tight_layout()
    out = 'sim2sim/isaac_vs_mujoco.png'
    plt.savefig(out, dpi=100)
    print(f"\n对比图已保存: {out}")
    print("蓝=isaacgym  红=mujoco  黑虚=ref。看蓝红是否重合/相位差。")


if __name__ == '__main__':
    main()
