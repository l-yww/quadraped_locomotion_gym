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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


CMD_VX, CMD_VY, CMD_YAW = 0.0, 0.0, 0.0
CMD_STEP = 0.1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    args = parser.parse_args()
    save_video = args.save_video
    config_file = "cowa2_him_wtw.yaml"

    pygame.init()
    screen = pygame.display.set_mode((200, 100))
    pygame.display.set_caption("Keyboard Control - HIM")
    print("Keyboard control: W/S=vx(+/-0.2), A/D=vy(+/-0.2), Q/E=yaw(+/-0.2) — persistent, Z to reset")
    print("  T/G: gait freq +/-  |  9/0: gait dur +/-  |  1/2: foot swing +/-  |  X/C: body pitch +/-  |  R/F: body height +/-  |  Z: reset")

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
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]  # 59 dims per frame
        obs_frame_stack = config.get("frame_stack", 5)  # frame_stack, 默认5向后兼容

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        if "cmd_range" in config:
            cmd_limits = np.array(config["cmd_range"], dtype=np.float32)
        else:
            max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            cmd_limits = np.stack((-max_cmd, max_cmd), axis=1)

        gait_frequency = config.get("gait_frequency", 2.0)
        gait_duration = config.get("gait_duration", 0.5)
        footswing_height = config.get("footswing_height", 0.20)
        body_height_cmd = config.get("body_height_cmd", 0.4)
        body_pitch = config.get("body_pitch", 0.0)
        control_dt = simulation_dt * control_decimation

        gait_frequency_range = config.get("gait_frequency_range", [2.0, 4.0])
        gait_duration_range = config.get("gait_duration_range", [0.35, 0.65])
        footswing_height_range = config.get("footswing_height_range", [0.03, 0.35])
        body_pitch_range = config.get("body_pitch_range", [-0.52, 0.52])
        body_height_range = config.get("body_height_range", [0.3, 0.5])

        active_gait = {
            "name": "trot",
            "frequency": gait_frequency,
            "phase": 0.5,
            "offset": 0.0,
            "bound": 0.0,
            "duration": gait_duration,
            "footswing_height": footswing_height,
            "body_height_cmd": body_height_cmd,
            "body_pitch": body_pitch,
            "body_roll": 0.0,
        }

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    # Auto-detect ONNX: if .onnx exists next to .pt, prefer ONNX
    use_onnx = policy_path.endswith('.onnx')
    if not use_onnx:
        onnx_candidate = policy_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_candidate):
            policy_path = onnx_candidate
            use_onnx = True
            print(f"Auto-detected ONNX: {policy_path}")

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history = np.zeros(obs_frame_stack * num_obs, dtype=np.float32)  # frame_stack * 59 = 1770
    gait_index = 0.0
    clock_inputs = np.zeros(4, dtype=np.float32)

    counter = 0

    # Data recording lists
    log_time = []
    log_qpos = []
    log_action = []
    log_target_dof_pos = []
    log_base_z = []
    log_cmd = []
    log_ang_vel = []
    log_torque_time = []
    log_torque = []
    log_motor_vel_time = []
    log_motor_vel = []

    # 初始化 12 维的指令和平滑缓冲
    smoothed_cmd = np.zeros(12, dtype=np.float32)
    target_cmd = np.zeros(12, dtype=np.float32)

    # torque-vel 约束参数 (从 yaml 读, 见 cowa2_wtw_him.yaml 的 torque_vel_limits)
    # 与训练侧 LeggedRobot._compute_torques 的平行四边形包络逐行一致, sim2real 必须对齐
    tv = config["torque_vel_limits"]
    torque_vel_hip_thigh = {
        "indices": np.array(tv["hip_thigh"]["indices"], dtype=np.int64),
        "max_vel": float(tv["hip_thigh"]["max_vel"]),
        "vel_1": float(tv["hip_thigh"]["vel_1"]),
        "max_torque": float(tv["hip_thigh"]["max_torque"]),
        "directional_brake": bool(tv["hip_thigh"].get("directional_brake", False)),
    }
    torque_vel_calf = {
        "indices": np.array(tv["calf"]["indices"], dtype=np.int64),
        "max_vel": float(tv["calf"]["max_vel"]),
        "vel_1": float(tv["calf"]["vel_1"]),
        "max_torque": float(tv["calf"]["max_torque"]),
        "directional_brake": bool(tv["calf"].get("directional_brake", False)),
    }
    top_clip = np.array(tv["top_clip"], dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Initialize joint positions
    d.qpos[2] = 0.40  # lower base z from 0.4 to 0.39
    d.qpos[7:] = default_angles.copy()
    target_dof_pos = default_angles.copy()
    mujoco.mj_forward(m, d)

    # Stabilize: let robot settle for a few steps before enabling policy control
    print("Stabilizing robot (base z=%.2f)..." % d.qpos[2])
    settle_steps = 200
    for _ in range(settle_steps):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        tau = np.clip(tau, -top_clip, top_clip)  # 静止下只需顶层兜底
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

    # renderer = mujoco.Renderer(m, height=360, width=640)

    # load policy
    if use_onnx:
        sess = ort.InferenceSession(policy_path)
        onnx_input_name = sess.get_inputs()[0].name
        print(f"Loaded ONNX model: {policy_path} (input: '{onnx_input_name}')")
    else:
        policy = torch.jit.load(policy_path)
        print(f"Loaded HIM PT model: {policy_path}")

    video_fps = 50
    if save_video:
        renderer = mujoco.Renderer(m, height=360, width=640)
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        print(f"Video recording will be saved to: {video_path}")
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
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
        
        print("\n")  # 强行为终端刷新腾出位置

        while viewer.is_running() and time.time() - start_wall_time < simulation_duration:
            # vel = d.qvel[:3]
            # ang_vel = d.qvel[3:6]
            # local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            # local_ang_vel = quat_rotate_inverse(d.qpos[3:7], ang_vel)
            vel = d.qvel[:3]
            local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            local_ang_vel = d.qvel[3:6] # 直接读取局部角速度

            show_str = f"[HIM] Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}"

            if counter % control_decimation == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        viewer.close()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        key = event.key
                        if key == pygame.K_w:
                            CMD_VX = min(CMD_VX + CMD_STEP, cmd_limits[0][1])
                        elif key == pygame.K_s:
                            CMD_VX = max(CMD_VX - CMD_STEP, cmd_limits[0][0])
                        elif key == pygame.K_a:
                            CMD_VY = min(CMD_VY + CMD_STEP, cmd_limits[1][1])
                        elif key == pygame.K_d:
                            CMD_VY = max(CMD_VY - CMD_STEP, cmd_limits[1][0])
                        elif key == pygame.K_q:
                            CMD_YAW = min(CMD_YAW + CMD_STEP, cmd_limits[2][1])
                        elif key == pygame.K_e:
                            CMD_YAW = max(CMD_YAW - CMD_STEP, cmd_limits[2][0])
                        elif key == pygame.K_t:
                            active_gait["frequency"] = min(active_gait["frequency"] + 0.1, gait_frequency_range[1])
                        elif key == pygame.K_g:
                            active_gait["frequency"] = max(active_gait["frequency"] - 0.1, gait_frequency_range[0])
                        elif key == pygame.K_9:
                            active_gait["duration"] = min(active_gait["duration"] + 0.05, gait_duration_range[1])
                        elif key == pygame.K_0:
                            active_gait["duration"] = max(active_gait["duration"] - 0.05, gait_duration_range[0])
                        elif key == pygame.K_1:
                            active_gait["footswing_height"] = min(active_gait["footswing_height"] + 0.05, footswing_height_range[1])
                        elif key == pygame.K_2:
                            active_gait["footswing_height"] = max(active_gait["footswing_height"] - 0.05, footswing_height_range[0])
                        elif key == pygame.K_x:
                            active_gait["body_pitch"] = min(active_gait["body_pitch"] + 0.1, body_pitch_range[1])
                        elif key == pygame.K_c:
                            active_gait["body_pitch"] = max(active_gait["body_pitch"] - 0.1, body_pitch_range[0])
                        elif key == pygame.K_r:
                            active_gait["body_height_cmd"] = min(active_gait["body_height_cmd"] + 0.05, body_height_range[1])
                        elif key == pygame.K_f:
                            active_gait["body_height_cmd"] = max(active_gait["body_height_cmd"] - 0.05, body_height_range[0])
                        elif key == pygame.K_z:
                            CMD_VX = 0.0
                            CMD_VY = 0.0
                            CMD_YAW = 0.0
                            active_gait["frequency"] = gait_frequency
                            active_gait["duration"] = gait_duration
                            active_gait["footswing_height"] = footswing_height
                            active_gait["body_pitch"] = body_pitch
                            active_gait["body_height_cmd"] = body_height_cmd

                # 键盘日志打印依然使用原始目标值，便于观察
                show_str += (
                    f" | Cmd: Vx={CMD_VX:.2f}, Vy={CMD_VY:.2f}, Wz={CMD_YAW:.2f}, "
                    f"Gait={active_gait['name']}, freq={active_gait['frequency']:.1f}, "
                    f"dur={active_gait['duration']:.2f}, swing={active_gait['footswing_height']:.2f}, "
                    f"pitch={active_gait['body_pitch']:.2f}, height={active_gait['body_height_cmd']:.2f}"
                )
                act_str = (
                    f"Act FL: h={action[0]:+5.2f} t={action[1]:+5.2f} c={action[2]:+5.2f} | "
                    f"FR: h={action[3]:+5.2f} t={action[4]:+5.2f} c={action[5]:+5.2f} | "
                    f"RL: h={action[6]:+5.2f} t={action[7]:+5.2f} c={action[8]:+5.2f} | "
                    f"RR: h={action[9]:+5.2f} t={action[10]:+5.2f} c={action[11]:+5.2f}"
                )
                sys.stdout.write(f"\033[A\r\033[2K{show_str}\n\r\033[2K{act_str}")
                sys.stdout.flush()

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)

            # ===== torque-vel 约束 (逐行对齐训练父类 LeggedRobot._compute_torques) =====
            # 平行四边形包络: upper/lower 为两条带符号斜线 -K*(vel ∓ max_vel),
            # 在两条斜线交点之外 (|vel| > v_int) 把 limit 归零。不区分 directional_brake (与训练侧实际行为一致)。
            qvel = d.qvel[6:]  # 12 关节速度
            for group in (torque_vel_hip_thigh, torque_vel_calf):
                idx = group["indices"]
                vel = qvel[idx]
                max_vel = group["max_vel"]
                vel_1 = group["vel_1"]
                max_torque = group["max_torque"]
                # 斜率 K = max_torque / (max_vel - vel_1), 与训练侧一致
                vel_range = max(max_vel - vel_1, 1e-6)
                K = max_torque / vel_range
                # 平行四边形的动态上下限, 硬截断在 ±max_torque
                upper_limit = np.clip(-K * (vel - max_vel), -max_torque, max_torque)
                lower_limit = np.clip(-K * (vel + max_vel), -max_torque, max_torque)
                # 两条斜线交点速度 v_int = max_vel + max_torque/K, 超过则两侧 limit 归零
                v_int = max_vel + max_torque / K
                over_intersection = np.abs(vel) > v_int
                upper_limit = np.where(over_intersection, 0.0, upper_limit)
                lower_limit = np.where(over_intersection, 0.0, lower_limit)
                tau[idx] = np.clip(tau[idx], lower_limit, upper_limit)

            # 顶层静态 clip (兜底, 与训练侧 torque_limits 对齐: 取分组 max_torque)
            tau = np.clip(tau, -top_clip, top_clip)
            d.ctrl[:] = tau
            # 记录最终下发的力矩和关节速度 (用于数据分析)
            qvel = d.qvel[6:]  # 12 关节速度
            log_torque_time.append(counter * simulation_dt)
            log_torque.append(tau.copy())
            log_motor_vel_time.append(counter * simulation_dt)
            log_motor_vel.append(qvel.copy())
            mujoco.mj_step(m, d)
            # 渲染时的 cmd 仅用于箭头方向等可视化，可以用一个 3 维临时数组
            mujoco_render_utils.update(np.array([CMD_VX, CMD_VY, CMD_YAW]), d)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"\nError rendering frame: {e}\n") 

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                gravity_orientation = get_gravity_orientation(quat)
                # ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])
                ang_vel = d.qvel[3:6] # 直接读取局部角速度

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale

                # 1. 构建 12 维完整目标指令
                target_cmd = np.array([
                    CMD_VX, CMD_VY, CMD_YAW,
                    active_gait["body_height_cmd"], active_gait["frequency"],
                    active_gait["phase"], active_gait["offset"], active_gait["bound"], active_gait["duration"],
                    active_gait["footswing_height"], active_gait["body_pitch"], active_gait["body_roll"]
                ], dtype=np.float32)

                # 2. 统一平滑处理
                smoothed_cmd = 0.8 * smoothed_cmd + 0.2 * target_cmd
                # smoothed_cmd = target_cmd


                # 3. 最终使用平滑后的向量构建观测
                full_cmd = smoothed_cmd

                # 4. 更新步态时钟 (强制基于平滑后的参数计算)
                zero_cmd = np.all(np.abs(smoothed_cmd[:3]) < 0.1)
                if zero_cmd:
                    clock_inputs[:] = 0.0
                else:
                    # 提取平滑后的动态步态参数
                    s_freq = smoothed_cmd[4]
                    s_phase = smoothed_cmd[5]
                    s_offset = smoothed_cmd[6]
                    s_bound = smoothed_cmd[7]
                    s_duration = smoothed_cmd[8]

                    gait_index = (gait_index + control_dt * s_freq) % 1.0
                    
                    foot_phases = [
                        gait_index + s_phase + s_offset + s_bound,
                        gait_index + s_bound,
                        gait_index + s_offset,
                        gait_index + s_phase
                    ]
                    for i in range(4):
                        fp = foot_phases[i] % 1.0
                        if fp < s_duration:
                            fp = fp * (0.5 / s_duration)
                        else:
                            fp = 0.5 + (fp - s_duration) * (0.5 / (1.0 - s_duration))
                        clock_inputs[i] = np.sin(2 * np.pi * fp)

                # 构建 59 维观测
                obs[:12] = full_cmd * cmd_scale
                obs[12:24] = qj
                obs[24:36] = dqj
                obs[36:48] = action
                obs[48:51] = ang_vel * ang_vel_scale
                obs[51:54] = gravity_orientation
                obs[54] = gait_index
                obs[55:59] = clock_inputs

                # 与训练一致：观测裁剪到 ±100 (clip_observations)。
                # yaw 转向时 ang_vel/dof_vel 会变大，不裁剪会超出训练分布(>100)把策略推到 OOD → action 爆炸。
                obs = np.clip(obs, -100.0, 100.0)

                obs_history[:obs_frame_stack * num_obs - num_obs] = obs_history[num_obs:]
                obs_history[obs_frame_stack * num_obs - num_obs:] = obs
                if counter == 0:
                    for k in range(obs_frame_stack):
                        obs_history[k * num_obs:(k + 1) * num_obs] = obs

                last_action = action
                if use_onnx:
                    obs_input = obs_history.astype(np.float32)[np.newaxis, :]
                    action = sess.run(['actions'], {onnx_input_name: obs_input})[0].squeeze()
                else:
                    obs_tensor = torch.from_numpy(obs_history).unsqueeze(0)
                    result = policy(obs_tensor)
                    if isinstance(result, tuple):
                        action = result[0].detach().numpy().squeeze()
                    else:
                        action = result.detach().cpu().numpy().squeeze()

                action = np.clip(action, -20.0, 20.0)
                # 与训练一致：action 低通平滑 (action_smoothness=True, ratio=0.9)。
                # 转向需要更激进的非对称 action，缺这层平滑易激振失稳。
                action = 0.9 * action + 0.1 * last_action
                target_dof_pos = action * action_scale + default_angles

                # Record data
                log_time.append(counter * simulation_dt)
                log_qpos.append(d.qpos[7:].copy())
                log_action.append(action.copy())
                log_target_dof_pos.append(target_dof_pos.copy())
                log_base_z.append(d.qpos[2].copy())
                log_cmd.append(np.array([CMD_VX, CMD_VY, CMD_YAW]).copy()) # 记录原始目标命令
                log_ang_vel.append(ang_vel.copy())

            if counter % control_decimation == 0:
                mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
                viewer.sync()

            target_wall_time = start_wall_time + (d.time - start_sim_time)
            time_to_wait = target_wall_time - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    if save_video:
        print(f"\n\nVideo saved successfully to {video_path}")
        writer.close()

    # Plot curves
    log_path = str(PATH_PARENT / "logs")
    os.makedirs(log_path, exist_ok=True)

    joint_names_config = config.get("mujoco_joint_names", [f"joint_{i}" for i in range(num_actions)])

    log_time = np.array(log_time)
    log_qpos = np.array(log_qpos)
    log_action = np.array(log_action)
    log_target_dof_pos = np.array(log_target_dof_pos)
    log_base_z = np.array(log_base_z)
    log_cmd = np.array(log_cmd)
    log_ang_vel = np.array(log_ang_vel)
    log_torque_time = np.array(log_torque_time)
    log_torque = np.array(log_torque)
    log_motor_vel_time = np.array(log_motor_vel_time)
    log_motor_vel = np.array(log_motor_vel)

    # 打印第 5s 时刻下发的关节力矩 (取最接近 5.0s 的采样点)
    if len(log_torque_time) > 0:
        idx_5s = int(np.argmin(np.abs(log_torque_time - 5.0)))
        t_5s = log_torque_time[idx_5s]
        tau_5s = log_torque[idx_5s]
        print(f"\n[Torque @ ~5s] t={t_5s:.4f}s (sample #{idx_5s})")
        print("  joint   : " + "  ".join(f"{n:>10s}" for n in joint_names_config))
        print("  torque  : " + "  ".join(f"{v:+10.3f}" for v in tau_5s))
        print(f"  abs max : {np.max(np.abs(tau_5s)):.3f} Nm")

    n = len(log_time)
    if n > 0:
        # Plot 1: Joint positions vs time
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('[HIM] Joint Positions (Qpos) vs Time')
        for i in range(num_actions):
            ax = axes[i // 3][i % 3]
            ax.plot(log_time, log_qpos[:, i], label=f'{joint_names_config[i]} qpos', linewidth=1.5)
            ax.axhline(y=default_angles[i], color='r', linestyle='--', linewidth=1.0, label='default')
            ax.set_ylabel('rad')
            ax.legend(fontsize=7)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'qpos_vs_time.png'), dpi=150)
        plt.close()

        # Plot 2: Actions vs time
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('[HIM] Policy Actions vs Time')
        for i in range(num_actions):
            ax = axes[i // 3][i % 3]
            ax.plot(log_time, log_action[:, i], label=f'{joint_names_config[i]} action', linewidth=1.5)
            ax.set_ylabel('')
            ax.legend(fontsize=7)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'actions_vs_time.png'), dpi=150)
        plt.close()

        # Plot 3: Body height and command velocity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle('[HIM] Body Height & Commands')
        ax1.plot(log_time, log_base_z, label='base z', linewidth=1.5, color='b')
        ax1.axhline(y=active_gait["body_height_cmd"], color='r', linestyle='--', linewidth=1.0, label='target')
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

        # Plot 4: Angular velocity
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(log_time, log_ang_vel[:, 0], label='ang_vel_x', linewidth=1.5)
        ax.plot(log_time, log_ang_vel[:, 1], label='ang_vel_y', linewidth=1.5)
        ax.plot(log_time, log_ang_vel[:, 2], label='ang_vel_z', linewidth=1.5)
        ax.set_ylabel('rad/s')
        ax.legend()
        ax.grid(True)
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'ang_vel_vs_time.png'), dpi=150)
        plt.close()

        if len(log_torque_time) > 0:
            # Plot 5: 最终下发力矩 (经过 torque-vel 约束 + top_clip 后) vs 时间
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle('[HIM] Applied Joint Torques vs Time')
            for i in range(num_actions):
                ax = axes[i // 3][i % 3]
                ax.plot(log_torque_time, log_torque[:, i], label=f'{joint_names_config[i]} torque', linewidth=1.0)
                ax.axhline(y=top_clip[i], color='r', linestyle='--', linewidth=0.8, label='+clip')
                ax.axhline(y=-top_clip[i], color='r', linestyle='--', linewidth=0.8, label='-clip')
                ax.set_ylabel('Nm')
                ax.legend(fontsize=7)
                ax.grid(True)
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'torques_vs_time.png'), dpi=150)
            plt.close()

            torque_abs = np.abs(log_torque)
            torque_max = np.max(torque_abs, axis=0)
            torque_rms = np.sqrt(np.mean(np.square(log_torque), axis=0))

            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            x = np.arange(num_actions)
            ax.bar(x - 0.18, torque_max, width=0.36, label='max |torque|')
            ax.bar(x + 0.18, torque_rms, width=0.36, label='rms torque')
            ax.set_xticks(x)
            ax.set_xticklabels(joint_names_config, rotation=35, ha='right')
            ax.set_ylabel('Nm')
            ax.set_title('[HIM] Torque Summary')
            ax.grid(True, axis='y')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'torque_summary.png'), dpi=150)
            plt.close()

            print("Torque max |Nm|:", np.array2string(torque_max, precision=2, separator=', '))
            print("Torque RMS Nm:", np.array2string(torque_rms, precision=2, separator=', '))

        if len(log_motor_vel_time) > 0:
            # Plot 6: 关节电机速度 vs 时间
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle('[HIM] Motor Velocities vs Time')
            for i in range(num_actions):
                ax = axes[i // 3][i % 3]
                ax.plot(log_motor_vel_time, log_motor_vel[:, i], label=f'{joint_names_config[i]} vel', linewidth=1.0)
                ax.set_ylabel('rad/s')
                ax.legend(fontsize=7)
                ax.grid(True)
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'motor_vel_vs_time.png'), dpi=150)
            plt.close()

            motor_vel_abs = np.abs(log_motor_vel)
            motor_vel_max = np.max(motor_vel_abs, axis=0)
            motor_vel_rms = np.sqrt(np.mean(np.square(log_motor_vel), axis=0))

            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            x = np.arange(num_actions)
            ax.bar(x - 0.18, motor_vel_max, width=0.36, label='max |motor vel|')
            ax.bar(x + 0.18, motor_vel_rms, width=0.36, label='rms motor vel')
            ax.set_xticks(x)
            ax.set_xticklabels(joint_names_config, rotation=35, ha='right')
            ax.set_ylabel('rad/s')
            ax.set_title('[HIM] Motor Velocity Summary')
            ax.grid(True, axis='y')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'motor_vel_summary.png'), dpi=150)
            plt.close()

            print("Motor vel max |rad/s|:", np.array2string(motor_vel_max, precision=2, separator=', '))
            print("Motor vel RMS rad/s:", np.array2string(motor_vel_rms, precision=2, separator=', '))

        print(f"\nCurves saved to {log_path}/")