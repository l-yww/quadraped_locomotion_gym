import sys
from pathlib import Path
PATH_PARENT = Path(__file__).parent
PATH_ROOT = PATH_PARENT.parent
sys.path.append(str(PATH_PARENT))
sys.path.append(str(PATH_ROOT))
from utils import MujocoRenderUtils

import os
import time
import mujoco.viewer
import mujoco
import numpy as np
LEGGED_GYM_ROOT_DIR = str(PATH_ROOT)
import torch
import onnxruntime as ort
import yaml
import imageio
from argparse import ArgumentParser
import pygame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def quat_to_yaw(q):
    """四元数 [w,x,y,z] → 航向角 yaw (rad, [-pi, pi])。"""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
    return float(np.arctan2(siny_cosp, cosy_cosp))

def wrap_to_pi(angle):
    """把角度归一化到 [-pi, pi]。"""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# 全局指令变量 (纯 3 维)
CMD_VX, CMD_VY, CMD_YAW = 0.0, 0.0, 0.0
CMD_STEP = 0.1

# ===== 航向锁(PID 角度补偿) =====
# 目标航向角(弧度)，由 Q/E 累积设定；航向锁开启时，PID 把航向误差转成 wz 命令喂给策略。
TARGET_HEADING = 0.0
# 用列表存标志位，避免在按键回调里用 global 声明(嵌套作用域 global 语法限制)
HEADING_LOCK = [True]         # [True]=航向锁(给目标角度)，[False]=直接 yaw(给角速度)
HEADING_STEP = 0.1           # Q/E 每次调整目标航向的步长 [rad]
# PID 增益(从 cmd 的 wz 层面补偿，输出 clip 到 yaw 范围)
KP_HEADING = 1.5             # 比例:误差 0.1rad → wz 0.15
KI_HEADING = 0.5             # 积分:消除稳态偏差
KD_HEADING = 0.2             # 微分:抑制超调
heading_integral = [0.0]    # PID 积分项(列表包装避免 global 声明)
last_heading_error = [0.0]  # PID 上次误差
YAW_LIMIT = 0.5              # wz 命令上限(对齐 cmd_range ang_vel_yaw)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    args = parser.parse_args()
    save_video = args.save_video
    config_file = "cowa2.yaml" 

    pygame.init()
    screen = pygame.display.set_mode((200, 100))
    pygame.display.set_caption("Keyboard Control")
    print("Keyboard: W/S=vx, A/D=vy, Q/E=目标航向(±0.1rad), X=切换航向锁, Z=reset")

    with open(f"sim2sim/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        
        # 只取前 3 个 cmd_scale
        cmd_scale = np.array(config["cmd_scale"][:3], dtype=np.float32) 

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd_init = np.array(config["cmd_init"][:3], dtype=np.float32)
        if "cmd_range" in config:
            cmd_limits = np.array(config["cmd_range"][:3], dtype=np.float32)
        else:
            max_cmd = np.array(config["max_cmd"][:3], dtype=np.float32)
            cmd_limits = np.stack((-max_cmd, max_cmd), axis=1)

        # HIMLoco cowa 训练环境直接使用 URDF/asset DOF 顺序，和当前 MuJoCo XML 顺序一致：
        # FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, ...
        # cowa2.yaml 里的 model_joint_names 是 wheel_legged_gym helper 顺序，不能用于这个 HIMLoco policy。
        idx_model2mj = np.arange(num_actions, dtype=np.int64)
        idx_mj2model = np.arange(num_actions, dtype=np.int64)
        default_angles_model = default_angles.copy()
        default_angles_mj = default_angles.copy()

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    # 自动检测 ONNX
    use_onnx = policy_path.endswith('.onnx')
    if not use_onnx:
        onnx_candidate = policy_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_candidate):
            policy_path = onnx_candidate
            use_onnx = True
            print(f"Auto-detected ONNX: {policy_path}")

    model_name = os.path.basename(policy_path).split('.')[0]

    # 上下文变量定义
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles_mj.copy()
    
    # 核心：观测历史缓冲区 (Frame Stacking)
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_frame_stack = config.get("frame_stack", 6) # 默认堆叠 5 帧
    obs_history = np.zeros(obs_frame_stack * num_obs, dtype=np.float32)
    
    counter = 0

    # 数据记录列表
    log_time, log_qpos, log_action, log_target_dof_pos = [], [], [], []
    log_base_z, log_cmd, log_ang_vel = [], [], []

    # 加载 MuJoCo 模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 初始化姿态，高度微调防止弹跳
    d.qpos[2] = 0.42
    d.qpos[7:] = default_angles_mj.copy()
    mujoco.mj_forward(m, d)

    # 稳定化阶段 (Settling)：让机器狗先依靠 PD 站稳 1 秒，再接入策略
    print("Stabilizing robot (base z=%.2f)..." % d.qpos[2])
    settle_steps = int(1.0 / simulation_dt) 
    for _ in range(settle_steps):
        tau = pd_control(default_angles_mj, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

    renderer = mujoco.Renderer(m, height=360, width=640)

    # 加载策略
    if use_onnx:
        sess = ort.InferenceSession(policy_path)
        print(f"Loaded ONNX model: {policy_path}")
    else:
        policy = torch.jit.load(policy_path)
        print(f"Loaded PT model: {policy_path}")

    video_fps = 50
    if save_video:
        cmd_str = f"cmd_{cmd_init[0]}_{cmd_init[1]}_{cmd_init[2]}"
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = max(1, int(sim_fps / video_fps))
        writer = imageio.get_writer(video_path, fps=video_fps)
    
    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        start_wall_time = time.time()
        start_sim_time = d.time
        render_decimation = config.get("render_decimation", control_decimation)
        while viewer.is_running() and time.time() - start_wall_time < simulation_duration:
            if counter % control_decimation == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        viewer.close()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        key = event.key
                        # 纯净的 3D 速度指令控制
                        if key == pygame.K_w:   CMD_VX = min(CMD_VX + CMD_STEP, cmd_limits[0][1])
                        elif key == pygame.K_s: CMD_VX = max(CMD_VX - CMD_STEP, cmd_limits[0][0])
                        elif key == pygame.K_a: CMD_VY = min(CMD_VY + CMD_STEP, cmd_limits[1][1])
                        elif key == pygame.K_d: CMD_VY = max(CMD_VY - CMD_STEP, cmd_limits[1][0])
                        elif key == pygame.K_q:
                            # 航向锁:调整目标航向；直接模式:调整 yaw 角速度
                            if HEADING_LOCK[0]:
                                TARGET_HEADING = wrap_to_pi(TARGET_HEADING + HEADING_STEP)
                            else:
                                CMD_YAW = max(CMD_YAW - CMD_STEP, cmd_limits[2][0])
                        elif key == pygame.K_e:
                            if HEADING_LOCK[0]:
                                TARGET_HEADING = wrap_to_pi(TARGET_HEADING - HEADING_STEP)
                            else:
                                CMD_YAW = min(CMD_YAW + CMD_STEP, cmd_limits[2][1])
                        elif key == pygame.K_x:
                            # 切换航向锁开/关
                            HEADING_LOCK[0] = not HEADING_LOCK[0]
                            heading_integral[0] = 0.0
                            last_heading_error[0] = 0.0
                            print(f"\n航向锁: {'ON' if HEADING_LOCK[0] else 'OFF'}")
                        elif key == pygame.K_z:
                            CMD_VX, CMD_VY, CMD_YAW = 0.0, 0.0, 0.0
                            TARGET_HEADING = quat_to_yaw(d.qpos[3:7])  # reset 到当前航向
                            heading_integral[0] = 0.0
                            last_heading_error[0] = 0.0

                # ===== 航向锁 PID:把目标航向角误差转成 wz 角速度命令 =====
                if HEADING_LOCK[0]:
                    cur_yaw = quat_to_yaw(d.qpos[3:7])
                    err = wrap_to_pi(TARGET_HEADING - cur_yaw)      # 航向误差 [-pi, pi]
                    # 积分(抗饱和:误差大时不累积)
                    if abs(err) < 0.5:
                        heading_integral[0] += err * simulation_dt * control_decimation
                    else:
                        heading_integral[0] *= 0.5   # 大误差时积分衰减，防 windup
                    heading_integral[0] = np.clip(heading_integral[0], -0.5, 0.5)
                    # 微分
                    derr = (err - last_heading_error[0]) / (simulation_dt * control_decimation)
                    last_heading_error[0] = err
                    # PID 输出 → wz 命令
                    wz_cmd = KP_HEADING * err + KI_HEADING * heading_integral[0] + KD_HEADING * derr
                    wz_cmd = float(np.clip(wz_cmd, -YAW_LIMIT, YAW_LIMIT))
                    cmd_yaw_final = wz_cmd
                else:
                    cmd_yaw_final = CMD_YAW

                cmd = np.array([CMD_VX, CMD_VY, cmd_yaw_final], dtype=np.float32)
                
                # 终端状态打印
                local_vel = quat_rotate_inverse(d.qpos[3:7], d.qvel[:3])
                # MuJoCo free-joint qvel[3:6] is used here as base-frame angular velocity,
                # matching the training observation's base_ang_vel semantics.
                local_ang_vel = d.qvel[3:6]
                show_str = (
                    f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:+.2f} | "
                    f"Real: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:+.2f}\n"
                    f"{'[LOCK]' if HEADING_LOCK[0] else '[FREE]'} tgt_yaw={TARGET_HEADING:+.2f} | "
                    f"FL: h={action[0]:+5.2f} t={action[1]:+5.2f} c={action[2]:+5.2f} | "
                    f"FR: h={action[3]:+5.2f} t={action[4]:+5.2f} c={action[5]:+5.2f}"
                )
                print(show_str, end='\r')

            # PD 控制与力矩限幅 (安全机制)
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            torque_clip = np.array([190, 190, 290, 190, 190, 290, 190, 190, 290, 190, 190, 290], dtype=np.float32)
            tau = np.clip(tau, -torque_clip, torque_clip)
            d.ctrl[:] = tau
            
            mujoco.mj_step(m, d)
            mujoco_render_utils.update(cmd, d)

            if save_video and counter % frame_skip == 0:
                renderer.update_scene(d, camera=viewer.cam)
                mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                writer.append_data(renderer.render())

            counter += 1
            if counter % control_decimation == 0:
                # 1. 提取物理状态。HIMLoco cowa policy 使用 URDF/MuJoCo 顺序，无需重排。
                q_mj = d.qpos[7:]
                dq_mj = d.qvel[6:]
                q_model = q_mj[idx_mj2model]
                dq_model = dq_mj[idx_mj2model]
                qj = (q_model - default_angles_model) * dof_pos_scale
                dqj = dq_model * dof_vel_scale
                quat = d.qpos[3:7]
                gravity_orientation = get_gravity_orientation(quat)
                # HIMLoco cowa 训练观测为 base-frame base_ang_vel * 0.25；MuJoCo qvel[3:6]
                # 在当前 deploy 中按机体系角速度使用，避免重复 quat_rotate_inverse 和重复缩放。
                ang_vel = d.qvel[3:6] * ang_vel_scale

                # 2. 按 HIMLoco cowa 训练环境组装单帧 Observation:
                # cmd(3), base_ang_vel(3), projected_gravity(3), dof_pos(12), dof_vel(12), action(12)
                obs[:3] = cmd * cmd_scale                 # [0, 1, 2]       -> commands (3)
                obs[3:6] = ang_vel                        # [3, 4, 5]       -> base_ang_vel (3)
                obs[6:9] = gravity_orientation            # [6, 7, 8]       -> projected_gravity (3)
                obs[9:21] = qj                            # [9 到 20]       -> dof_pos (12)
                obs[21:33] = dqj                          # [21 到 32]      -> dof_vel (12)
                obs[33:45] = action                       # [33 到 44]      -> actions (12)
                obs = np.clip(obs, -100.0, 100.0)
                
                # 3. 维护历史观测队列 (Frame Stacking)：HIMLoco 训练为当前帧在最前，历史帧后移。
                obs_history[num_obs:] = obs_history[:obs_frame_stack * num_obs - num_obs]
                obs_history[:num_obs] = obs

                if counter == control_decimation: # 第一次采样，填满缓冲区
                    for k in range(obs_frame_stack):
                        obs_history[k * num_obs:(k + 1) * num_obs] = obs
                
                # 4. 策略推理
                if use_onnx:
                    obs_input = obs_history.astype(np.float32)[np.newaxis, :]
                    action = sess.run(['actions'], {'obs': obs_input})[0].squeeze()
                else:
                    obs_tensor = torch.from_numpy(obs_history).unsqueeze(0)
                    result = policy(obs_tensor)
                    if isinstance(result, tuple):
                        action = result[0].detach().numpy().squeeze()
                    else:
                        action = result.detach().cpu().numpy().squeeze()
                
                # 动作限幅与目标关节计算。
                action = np.clip(action, -100.0, 100.0)
                target_dof_pos_model = action * action_scale + default_angles_model
                target_dof_pos = target_dof_pos_model[idx_model2mj]

                # 记录绘图数据
                log_time.append(counter * simulation_dt)
                log_qpos.append(d.qpos[7:].copy())
                log_action.append(action.copy())
                log_base_z.append(d.qpos[2].copy())
                log_cmd.append(cmd.copy())
                log_ang_vel.append(ang_vel.copy())

            if counter % render_decimation == 0:
                mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
                viewer.sync()

            target_wall_time = start_wall_time + (d.time - start_sim_time)
            time_to_wait = target_wall_time - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    if save_video:
        writer.close()
        print(f"\nVideo saved successfully to {video_path}")

    # ===== 绘制分析曲线 =====
    log_path = str(PATH_PARENT / "logs")
    os.makedirs(log_path, exist_ok=True)
    joint_names_config = config.get("mujoco_joint_names", [f"joint_{i}" for i in range(num_actions)])
    
    log_time = np.array(log_time)
    log_qpos = np.array(log_qpos)
    log_action = np.array(log_action)
    log_base_z = np.array(log_base_z)
    log_cmd = np.array(log_cmd)
    
    if len(log_time) > 0:
        # Plot Base Height & Cmds
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle('Body Height & Commands')
        ax1.plot(log_time, log_base_z, label='base z', linewidth=1.5, color='b')
        ax1.axhline(y=0.4, color='r', linestyle='--', linewidth=1.0, label='target')
        ax1.set_ylabel('m')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(log_time, log_cmd[:, 0], label='cmd_vx', linewidth=1.5)
        ax2.plot(log_time, log_cmd[:, 1], label='cmd_vy', linewidth=1.5)
        ax2.plot(log_time, log_cmd[:, 2], label='cmd_yaw', linewidth=1.5)
        ax2.set_ylabel('cmd')
        ax2.legend()
        ax2.grid(True)
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'height_and_cmd.png'), dpi=150)
        plt.close()
        print(f"Curves saved to {log_path}/")