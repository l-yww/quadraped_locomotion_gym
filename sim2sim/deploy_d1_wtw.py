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
import os
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

CMD_VX, CMD_VY, CMD_YAW = 1.0, 0.0, 0.0
CMD_STEP = 0.1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    args = parser.parse_args()
    save_video = args.save_video
    config_file = "cowa2.yaml"

    pygame.init()
    screen = pygame.display.set_mode((200, 100))
    pygame.display.set_caption("Keyboard Control")
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
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        if "cmd_range" in config:
            cmd_limits = np.array(config["cmd_range"], dtype=np.float32)
        else:
            max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            cmd_limits = np.stack((-max_cmd, max_cmd), axis=1)

        # gait parameters
        gait_frequency = config.get("gait_frequency", 2.0)
        gait_duration = config.get("gait_duration", 0.5)
        footswing_height = config.get("footswing_height", 0.20)
        body_height_cmd = config.get("body_height_cmd", 0.4)
        body_pitch = config.get("body_pitch", 0.0)
        control_dt = simulation_dt * control_decimation

        # Trot gait parameters (fixed for trot-only training)
        # phase=0.5 (diagonal legs in phase), offset=0.0, bound=0.0, body_roll=0.0
        # Ranges from training config
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
    # observation history buffer for frame stacking
    obs_frame_stack = 5  # should match training config's frame_stack
    obs_history = np.zeros(obs_frame_stack * num_obs, dtype=np.float32)
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

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Initialize joint positions: keep all at 0 for max leg reach (training default_dof_pos)
    # But lower base z by ~1cm so feet touch the ground
    # (base_link collision: z=0.1±0.1, hip: z=0.105, leg length: ~0.453m)
    # foot ~ 0.4+0.105-0.453 = 0.052m above ground with all joints at 0
    # foot sphere radius 0.045m → ~7mm gap → lower base by 1cm
    d.qpos[2] = 0.39  # lower base z from 0.4 to 0.39
    d.qpos[7:] = default_angles.copy()  # all joints at 0 (max reach)
    target_dof_pos = default_angles.copy()  # PD initially holds joints at 0
    mujoco.mj_forward(m, d)

    # Stabilize: let robot settle for a few steps before enabling policy control
    print("Stabilizing robot (base z=%.2f)..." % d.qpos[2])
    settle_steps = 200  # 200 * 0.005s = 1s
    for _ in range(settle_steps):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

    renderer = mujoco.Renderer(m, height=360, width=640)

    # load policy (auto-detect PT vs ONNX)
    if use_onnx:
        sess = ort.InferenceSession(policy_path)
        print(f"Loaded ONNX model: {policy_path}")
    else:
        policy = torch.jit.load(policy_path)
        print(f"Loaded PT model: {policy_path}")

    video_fps = 50
    if save_video:
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        print(f"Video recording will be saved to: {video_path}")
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")
    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    with mujoco.viewer.launch_passive(m, d) as viewer:

        # set viewer.camera to follow robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            vel = d.qvel[:3]
            ang_vel = d.qvel[3:6]
            local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            local_ang_vel = quat_rotate_inverse(d.qpos[3:7], ang_vel)
            show_str = f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}, "
            step_start = time.time()

            if counter % control_decimation == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        viewer.close()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        key = event.key
                        # velocity commands (persistent, step by CMD_STEP)
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
                        # gait frequency adjustment
                        elif key == pygame.K_t:
                            active_gait["frequency"] = min(active_gait["frequency"] + 0.1, gait_frequency_range[1])
                        elif key == pygame.K_g:
                            active_gait["frequency"] = max(active_gait["frequency"] - 0.1, gait_frequency_range[0])
                        # gait duration adjustment
                        elif key == pygame.K_9:
                            active_gait["duration"] = min(active_gait["duration"] + 0.05, gait_duration_range[1])
                        elif key == pygame.K_0:
                            active_gait["duration"] = max(active_gait["duration"] - 0.05, gait_duration_range[0])
                        # foot swing height adjustment
                        elif key == pygame.K_1:
                            active_gait["footswing_height"] = min(active_gait["footswing_height"] + 0.05, footswing_height_range[1])
                        elif key == pygame.K_2:
                            active_gait["footswing_height"] = max(active_gait["footswing_height"] - 0.05, footswing_height_range[0])
                        # body pitch adjustment
                        elif key == pygame.K_x:
                            active_gait["body_pitch"] = min(active_gait["body_pitch"] + 0.1, body_pitch_range[1])
                        elif key == pygame.K_c:
                            active_gait["body_pitch"] = max(active_gait["body_pitch"] - 0.1, body_pitch_range[0])
                        # body height adjustment
                        elif key == pygame.K_r:
                            active_gait["body_height_cmd"] = min(active_gait["body_height_cmd"] + 0.1, body_height_range[1])
                        elif key == pygame.K_f:
                            active_gait["body_height_cmd"] = max(active_gait["body_height_cmd"] - 0.1, body_height_range[0])
                        # reset all commands
                        elif key == pygame.K_z:
                            CMD_VX = 0.0
                            CMD_VY = 0.0
                            CMD_YAW = 0.0
                            active_gait["frequency"] = gait_frequency
                            active_gait["duration"] = gait_duration
                            active_gait["footswing_height"] = footswing_height
                            active_gait["body_pitch"] = body_pitch
                            active_gait["body_height_cmd"] = body_height_cmd

                cmd = np.array([CMD_VX, CMD_VY, CMD_YAW], dtype=np.float32)
                show_str += (
                    # f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}, "
                    # f"Gait={active_gait['name']}, freq={active_gait['frequency']:.1f}, "
                    # f"dur={active_gait['duration']:.2f}, swing={active_gait['footswing_height']:.2f}, "
                    # f"pitch={active_gait['body_pitch']:.2f}, height={active_gait['body_height_cmd']:.2f}\n"
                    f"Act FL: h={action[0]:+5.2f} t={action[1]:+5.2f} c={action[2]:+5.2f} | "
                    f"FR: h={action[3]:+5.2f} t={action[4]:+5.2f} c={action[5]:+5.2f} | "
                    f"RL: h={action[6]:+5.2f} t={action[7]:+5.2f} c={action[8]:+5.2f} | "
                    f"RR: h={action[9]:+5.2f} t={action[10]:+5.2f} c={action[11]:+5.2f}"
                )
                print(show_str, end='\n')

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            # Clip torques to match training torque_limits (URDF effort values)
            # hips: 190, thighs: 190, calves: 290 (mujoco order: FL_hip,FL_th,FL_cf,FR_hip,FR_th,FR_cf,RL_hip,RL_th,RL_cf,RR_hip,RR_th,RR_cf)
            torque_clip = np.array([190, 190, 290, 190, 190, 290, 190, 190, 290, 190, 190, 290], dtype=np.float32)
            tau = np.clip(tau, -torque_clip, torque_clip)
            # --- DEBUG: skip torque ---
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            mujoco_render_utils.update(cmd, d)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                gravity_orientation = get_gravity_orientation(quat)
                ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale

                # 12-dim command: [vx, vy, yaw, body_height, freq, phase, offset, bound, duration, footswing, pitch, roll]
                full_cmd = np.array([
                    cmd[0], cmd[1], cmd[2],
                    active_gait["body_height_cmd"], active_gait["frequency"],
                    active_gait["phase"], active_gait["offset"], active_gait["bound"], active_gait["duration"],
                    active_gait["footswing_height"], active_gait["body_pitch"], active_gait["body_roll"]
                ], dtype=np.float32)

                zero_cmd = np.all(np.abs(cmd[:3]) < 0.1)

                if zero_cmd:
                    clock_inputs[:] = 0.0
                else:
                    # step gait_index and clock_inputs
                    gait_index = (gait_index + control_dt * active_gait["frequency"]) % 1.0
                    foot_phases = [
                        gait_index + active_gait["phase"] + active_gait["offset"] + active_gait["bound"],
                        gait_index + active_gait["bound"],
                        gait_index + active_gait["offset"],
                        gait_index + active_gait["phase"]
                    ]
                    for i in range(4):
                        fp = foot_phases[i] % 1.0
                        if fp < active_gait["duration"]:
                            fp = fp * (0.5 / active_gait["duration"])
                        else:
                            fp = 0.5 + (fp - active_gait["duration"]) * (0.5 / (1.0 - active_gait["duration"]))
                        clock_inputs[i] = np.sin(2 * np.pi * fp)

                

                # Observation order must match training (quadruped_wtw_slope):
                # [commands(12), dof_pos(12), dof_vel(12), actions(12), ang_vel(3), gravity(3), gait_index(1), clock(4)]
                obs[:12] = full_cmd * cmd_scale
                obs[12:24] = qj #* 0.0
                obs[24:36] = dqj #* 0.0
                obs[36:48] = action
                obs[48:51] = ang_vel * ang_vel_scale #* 0.0
                obs[51:54] = gravity_orientation #* 0.0
                obs[54] = gait_index
                obs[55:59] = clock_inputs
                # maintain observation history buffer (sliding window)
                obs_history[:obs_frame_stack * num_obs - num_obs] = obs_history[num_obs:]
                obs_history[obs_frame_stack * num_obs - num_obs:] = obs
                # initialize history buffer on first step to avoid all-zero frames
                if counter == 0:
                    for k in range(obs_frame_stack):
                        obs_history[k * num_obs:(k + 1) * num_obs] = obs
                obs_tensor = torch.from_numpy(obs_history).unsqueeze(0)
                # policy inference
                # last_action = action
                # if zero_cmd:
                #     target_dof_pos = default_angles.copy()
                #     # action = np.zeros(num_actions)
                # else:
                #     result = policy(obs_tensor)
                #     if isinstance(result, tuple):
                #         action, (weights, latent) = result
                #         action = action.detach().numpy().squeeze()[idx_model2mj]
                #         weights = weights.detach().numpy().squeeze()
                #         latent = latent.detach().numpy().squeeze()
                #     else:
                #         action = result.detach().cpu().numpy().squeeze()[idx_model2mj]
                #     # transform action to target_dof_pos
                #     target_dof_pos = action * action_scale + default_angles
                last_action = action
                if use_onnx:
                    obs_input = obs_history.astype(np.float32)[np.newaxis, :]
                    action = sess.run(['actions'], {'obs': obs_input})[0].squeeze()
                else:
                    result = policy(obs_tensor)
                    if isinstance(result, tuple):
                        action, (weights, latent) = result
                        action = action.detach().numpy().squeeze()
                        weights = weights.detach().numpy().squeeze()
                        latent = latent.detach().numpy().squeeze()
                    else:
                        action = result.detach().cpu().numpy().squeeze()
                # clip actions to match training (clip_actions = 20)
                action = np.clip(action, -20.0, 20.0)
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                # Record data
                log_time.append(counter * simulation_dt)
                log_qpos.append(d.qpos[7:].copy())
                log_action.append(action.copy())
                log_target_dof_pos.append(target_dof_pos.copy())
                log_base_z.append(d.qpos[2].copy())
                log_cmd.append(cmd.copy())
                log_ang_vel.append(ang_vel.copy())


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            # time_until_next_step = m.opt.timestep - (time.time() - step_start) - 0.1
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

    # writer.close()
    if save_video:
        print(f"Video saved successfully to {video_path}")
        writer.close()

    # Plot curves
    log_path = str(PATH_PARENT / "logs")
    os.makedirs(log_path, exist_ok=True)

    joint_names_config = config.get("mujoco_joint_names", [f"joint_{i}" for i in range(num_actions)])

    # Convert lists to arrays
    log_time = np.array(log_time)
    log_qpos = np.array(log_qpos)
    log_action = np.array(log_action)
    log_target_dof_pos = np.array(log_target_dof_pos)
    log_base_z = np.array(log_base_z)
    log_cmd = np.array(log_cmd)
    log_ang_vel = np.array(log_ang_vel)

    n = len(log_time)
    print(f"[debug] 退出循环，开始画图。n={n}, log_action shape={np.array(log_action).shape}")
    if n > 0:
        # ===== Plot 0: scale 前的 12 个 action，仅取前 5s =====
        T_WINDOW = 5.0   # 取前 5 秒数据
        mask_5s = log_time <= (log_time[0] + T_WINDOW)
        t_5s = log_time[mask_5s]
        a_5s = log_action[mask_5s]   # scale 前的策略原始输出(已 clip±20)
        print(f"[debug] 前5s数据点数: {len(t_5s)}, 时间范围 {t_5s[0] if len(t_5s)>0 else 'N/A'}~{t_5s[-1] if len(t_5s)>0 else 'N/A'}")
        if len(t_5s) > 0:
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle(f'Raw Policy Actions (pre action_scale), first {T_WINDOW}s')
            for i in range(num_actions):
                ax = axes[i // 3][i % 3]
                ax.plot(t_5s, a_5s[:, i], label=f'{joint_names_config[i]}', linewidth=1.5)
                ax.axhline(y=0.0, color='r', linestyle='--', linewidth=0.8)
                ax.set_ylabel('raw action')
                ax.set_xlabel('time [s]')
                ax.legend(fontsize=7)
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'actions_raw_5s.png'), dpi=150)
            plt.close()
            print(f"[debug] 已保存: {os.path.join(log_path, 'actions_raw_5s.png')}")
        else:
            print("[debug] 前5s无数据，跳过画图")

        # Plot 1: Joint positions vs time
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Joint Positions (Qpos) vs Time')
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
        fig.suptitle('Policy Actions vs Time')
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
        fig.suptitle('Body Height & Commands')
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

        print(f"\nCurves saved to {log_path}/")
