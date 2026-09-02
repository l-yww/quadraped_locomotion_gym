"""quadruped_joint_track 的 mujoco 部署：悬空 + 闭环策略跟踪 sin。

训练环境配置对齐（quadruped_joint_track_config.py）：
  fix_base_link=True（悬空），obs=dof_pos(12)+dof_vel(12)+last_action(12)+sin(1)+cos(1)=38，frame_stack=5 → 190
  action_scale=0.20，PD: kp=160, kd=5，D 项目标速度=0
  ref = default + offset + A*sin(2π·t/cycle_time + φ)，cycle_time=2，phase_offset 全0，amplitude=[0.2,0.3,0.3]×4
  sim_dt=0.005，decimation=4 → 策略 50Hz

用法：
  python sim2sim/deploy_joint_track.py --onnx logs/quadruped_joint_track/exported/policy.onnx
  # 默认加载最新 exported policy；悬空固定 base，画 dof_pos vs ref 对比图
"""
import os
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 训练对齐参数（与 quadruped_joint_track_config.py 一致）----
SIM_DT = 0.005
DECIMATION = 4              # 策略 50Hz
ACTION_SCALE = 0.20
KP = 160.0
KD = 5.0
NUM_ACTIONS = 12
NUM_SINGLE_OBS = 38
FRAME_STACK = 5
CLIP_ACTION = 20.0
CLIP_OBS = 100.0

# ---- 力矩-速度曲线限幅（与训练 _compute_torques 完全一致）----
# 训练里 PD 力矩要经过电机力矩-速度包络限幅 + 硬限幅，deploy 之前漏了，导致力矩图虚高。
# hip/thigh: indices [0,1,3,4,6,7,9,10]；calf: indices [2,5,8,11]
HIP_INDICES = [0, 1, 3, 4, 6, 7, 9, 10]
HIP_MAX_VEL = 20.0       # rad/s
HIP_VEL_1 = 7.28         # rad/s
HIP_MAX_TORQUE = 200.0   # Nm
CALF_INDICES = [2, 5, 8, 11]
CALF_MAX_VEL = 12.0
CALF_VEL_1 = 6.6
CALF_MAX_TORQUE = 330.0
# 硬限幅（URDF effort，与训练 torque_limits 一致）
TORQUE_LIMITS = np.array([200, 200, 330, 200, 200, 330, 200, 200, 330, 200, 200, 330], dtype=np.float32)

# obs 分段索引
Q_SLICE = slice(0, 12)        # dof_pos
DQ_SLICE = slice(12, 24)      # dof_vel
ACT_SLICE = slice(24, 36)     # last_action
SIN_SLICE = 36               # sin
COS_SLICE = 37               # cos

# obs scales
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

# sin 轨迹参数
CYCLE_TIME = 2.0
AMPLITUDE = np.array([0.2, 0.3, 0.3,
                      0.2, 0.3, 0.3,
                      0.2, 0.3, 0.3,
                      0.2, 0.3, 0.3], dtype=np.float32)
PHASE_OFFSET = np.zeros(12, dtype=np.float32)
OFFSET = np.zeros(12, dtype=np.float32)
DEFAULT_ANGLES = np.zeros(12, dtype=np.float32)   # default_joint_angles=0

# mujoco xml（freejoint base，关节顺序 FL/FR/RL/RR 按腿分组，与训练一致）
XML_PATH = "sim2sim/cowa2_description_mujoco/xml/cowa2_d1_arm_2.xml"
# XML_PATH = "sim2sim/cowa2_description_mujoco/xml/cowa2.xml"
# XML_PATH = "sim2sim/cowa2_description_mujoco/xml/scene.xml"

# 关节名（mujoco joint 顺序，与 dof_pos 索引一致）
JOINT_NAMES = ["FL_hip", "FL_thigh", "FL_calf",
               "FR_hip", "FR_thigh", "FR_calf",
               "RL_hip", "RL_thigh", "RL_calf",
               "RR_hip", "RR_thigh", "RR_calf"]


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """与训练 _compute_torques 一致：tau = kp*(target-q) + kd*(target_dq-dq)，D项目标=0"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def apply_torque_vel_limits(tau, dq):
    """力矩-速度曲线限幅 + 硬限幅，与训练 _compute_torques 完全一致。
    电机特性：转速越快，能输出的力矩越小（平行四边形包络）。
    之前 deploy 漏了这步，导致记录的力矩虚高、和训练不一致。"""
    tau = tau.copy()
    # ---- HIP/THIGH ----
    K = HIP_MAX_TORQUE / max(HIP_MAX_VEL - HIP_VEL_1, 1e-6)
    vel = dq[HIP_INDICES]
    upper = np.clip(-K * (vel - HIP_MAX_VEL), -HIP_MAX_TORQUE, HIP_MAX_TORQUE)
    lower = np.clip(-K * (vel + HIP_MAX_VEL), -HIP_MAX_TORQUE, HIP_MAX_TORQUE)
    v_int = HIP_MAX_VEL + HIP_MAX_TORQUE / K
    over = np.abs(vel) > v_int
    upper = np.where(over, 0.0, upper)
    lower = np.where(over, 0.0, lower)
    tau[HIP_INDICES] = np.clip(tau[HIP_INDICES], lower, upper)
    # ---- CALF ----
    K = CALF_MAX_TORQUE / max(CALF_MAX_VEL - CALF_VEL_1, 1e-6)
    vel = dq[CALF_INDICES]
    upper = np.clip(-K * (vel - CALF_MAX_VEL), -CALF_MAX_TORQUE, CALF_MAX_TORQUE)
    lower = np.clip(-K * (vel + CALF_MAX_VEL), -CALF_MAX_TORQUE, CALF_MAX_TORQUE)
    v_int = CALF_MAX_VEL + CALF_MAX_TORQUE / K
    over = np.abs(vel) > v_int
    upper = np.where(over, 0.0, upper)
    lower = np.where(over, 0.0, lower)
    tau[CALF_INDICES] = np.clip(tau[CALF_INDICES], lower, upper)
    # ---- 硬限幅 ----
    return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)


def compute_ref(t):
    """ref_dof_pos = default + offset + A*sin(2π·t/cycle_time + φ)，与 compute_ref_state 一致"""
    phase = t / CYCLE_TIME
    arg = 2 * np.pi * phase + PHASE_OFFSET
    return DEFAULT_ANGLES + OFFSET + AMPLITUDE * np.sin(arg)


def build_obs(qj, dqj, last_action, phase):
    """拼 38 维单帧 obs，与 compute_observations 一致。dof_pos 减 default(0) 不变。"""
    obs = np.zeros(NUM_SINGLE_OBS, dtype=np.float32)
    obs[Q_SLICE] = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
    obs[DQ_SLICE] = dqj * DOF_VEL_SCALE
    obs[ACT_SLICE] = last_action
    obs[SIN_SLICE] = np.sin(2 * np.pi * phase)
    obs[COS_SLICE] = np.cos(2 * np.pi * phase)
    return np.clip(obs, -CLIP_OBS, CLIP_OBS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="logs/quadruped_joint_track/exported/policy.onnx",
                        help="policy onnx path")
    parser.add_argument("--no-viewer", action="store_true", help="run headless, only plot")
    parser.add_argument("--duration", type=float, default=10.0, help="simulation seconds")
    args = parser.parse_args()

    # 加载策略
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx)
        onnx_in = sess.get_inputs()[0].name
        print(f"loaded onnx: {args.onnx}  input={onnx_in}")
    except Exception as e:
        print(f"[ERR] onnx 加载失败: {e}\n先用 play_quad_joint_track.py 导出 onnx。")
        return

    # 加载 mujoco 模型
    xml = os.path.abspath(XML_PATH)
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    # 初始化关节到 default
    d.qpos[7:] = DEFAULT_ANGLES.copy()
    mujoco.mj_forward(m, d)

    # obs history（frame_stack=5 滚动窗口，与训练 obs_buf 一致：新帧在末尾）
    obs_hist = np.zeros(FRAME_STACK * NUM_SINGLE_OBS, dtype=np.float32)

    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # 日志
    log_t, log_q, log_ref, log_tau, log_dq, log_base_z, log_base_ang = [], [], [], [], [], [], []

    sim_time = 0.0
    counter = 0
    N_STEPS = int(args.duration / SIM_DT)

    print(f"Running {args.duration}s (悬空 base 固定，闭环策略跟踪 sin)...")

    def run_step():
        nonlocal sim_time, counter, action, last_action, obs_hist
        if counter % DECIMATION == 0:
            # 策略步：拼 obs → 滚动 history → onnx 推理
            phase = sim_time / CYCLE_TIME
            qj = d.qpos[7:7 + NUM_ACTIONS].astype(np.float32)
            dqj = d.qvel[6:6 + NUM_ACTIONS].astype(np.float32)
            obs_now = build_obs(qj, dqj, last_action, phase)
            # 滑动窗口：旧帧前移，新帧放末尾
            obs_hist[: (FRAME_STACK - 1) * NUM_SINGLE_OBS] = obs_hist[NUM_SINGLE_OBS:]
            obs_hist[(FRAME_STACK - 1) * NUM_SINGLE_OBS:] = obs_now

            obs_in = obs_hist.astype(np.float32)[np.newaxis, :]
            action = sess.run(["action"], {onnx_in: obs_in})[0].squeeze().astype(np.float32)
            action = np.clip(action, -CLIP_ACTION, CLIP_ACTION)

        # PD：target = default + action * action_scale，D 目标速度=0
        target_q = DEFAULT_ANGLES + action * ACTION_SCALE
        qj = d.qpos[7:7 + NUM_ACTIONS]
        dqj = d.qvel[6:6 + NUM_ACTIONS]
        tau = pd_control(target_q, qj, KP, np.zeros_like(dqj), dqj, KD)
        # ★ 力矩-速度曲线限幅 + 硬限幅（与训练 _compute_torques 一致），之前漏了
        tau = apply_torque_vel_limits(tau, dqj)
        d.ctrl[:] = tau

        # ★ 悬空：固定 base（锁死 pos/quat/vel），与训练 fix_base_link 等效
        d.qpos[0:3] = [0.0, 0.0, 0.5]   # base pos
        d.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]   # base quat（水平）
        d.qvel[0:6] = 0.0               # base 线/角速度清零

        # 记录：只在策略步(50Hz)记录，避免 200Hz sim 步的 PD 高频毛刺把图画密
        # （力矩每 sim 步随 q/dq 微变，是 PD 正常行为，全画会成密集毛刺）
        if counter % DECIMATION == 0:
            ref = compute_ref(sim_time)
            log_t.append(sim_time)
            log_q.append(qj.copy())
            log_ref.append(ref.copy())
            log_tau.append(tau.copy())
            log_dq.append(dqj.copy())
            log_base_z.append(d.qpos[2].copy())
            log_base_ang.append(np.array([d.qvel[3], d.qvel[4], d.qvel[5]]).copy())

        mujoco.mj_step(m, d)
        sim_time += SIM_DT
        counter += 1
        last_action = action.copy()

    if args.no_viewer:
        for _ in range(N_STEPS):
            run_step()
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            for _ in range(N_STEPS):
                run_step()
                viewer.sync()
                if not viewer.is_running():
                    break

    # ============ 画图（位置/转速/力矩/base，与其他 deploy 一致）============
    log_t = np.array(log_t)
    log_q = np.array(log_q)
    log_ref = np.array(log_ref)
    log_tau = np.array(log_tau)
    log_dq = np.array(log_dq)
    log_base_z = np.array(log_base_z)
    log_base_ang = np.array(log_base_ang)

    err = log_q - log_ref
    rmse = np.sqrt((err ** 2).mean(axis=0))
    print(f"\n=== RESULT ===")
    print(f"平均跟踪误差 RMSE: {rmse.mean():.4f} rad ({np.degrees(rmse.mean()):.2f}°)")
    for n, r in zip(JOINT_NAMES, rmse):
        print(f"  {n:10s}: {r:.4f} rad ({np.degrees(r):.2f}°)")

    log_path = os.path.join("sim2sim", "logs")
    os.makedirs(log_path, exist_ok=True)

    # Plot 1: 关节位置 vs ref
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle(f"Joint Track: Positions vs Ref (avg RMSE={rmse.mean():.4f} rad)")
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j]
        ax.plot(log_t, log_q[:, j], "b-", lw=1.0, label="actual")
        ax.plot(log_t, log_ref[:, j], "r--", lw=1.0, label="ref")
        ax.set_title(f"{name} (RMSE={rmse[j]:.4f} rad)", fontsize=9)
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "joint_track_qpos_vs_ref.png"), dpi=150)
    plt.close()

    # Plot 2: 关节转速
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Joint Track: Joint Velocities")
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j]
        ax.plot(log_t, log_dq[:, j], "g-", lw=1.0)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "joint_track_qvel.png"), dpi=150)
    plt.close()

    # Plot 3: 关节力矩
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Joint Track: Joint Torques")
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j]
        ax.plot(log_t, log_tau[:, j], "m-", lw=1.0)
        ax.set_title(f"{name} (max={np.abs(log_tau[:, j]).max():.0f}Nm)", fontsize=9)
        ax.set_ylabel("Nm")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "joint_track_torque.png"), dpi=150)
    plt.close()

    # Plot 4: base 高度 + 角速度（悬空时应稳定；高频尖峰=抖动）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Joint Track: Base Stability (悬空应稳定)")
    ax1.plot(log_t, log_base_z, "b-", lw=1.5, label="base z")
    ax1.set_ylabel("m")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(log_t, log_base_ang[:, 0], lw=1.0, label="ang_vel_x")
    ax2.plot(log_t, log_base_ang[:, 1], lw=1.0, label="ang_vel_y")
    ax2.plot(log_t, log_base_ang[:, 2], lw=1.0, label="ang_vel_z")
    ax2.set_ylabel("rad/s")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, "joint_track_base_stability.png"), dpi=150)
    plt.close()

    print(f"\n4 张图已保存到 {log_path}/:")
    print(f"  joint_track_qpos_vs_ref.png  (位置 vs 目标)")
    print(f"  joint_track_qvel.png         (转速)")
    print(f"  joint_track_torque.png       (力矩)")
    print(f"  joint_track_base_stability.png (base 高度/角速度，抖动指标)")


if __name__ == "__main__":
    main()
