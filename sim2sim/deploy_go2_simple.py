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
import yaml
import imageio
from argparse import ArgumentParser
import pygame


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
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
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_keyboard_command(keys, cmd_limits):
    cmd_x = 0.0
    cmd_y = 0.0
    cmd_yaw = 0.0
    if keys[pygame.K_w]:    cmd_x += cmd_limits[0, 1]
    if keys[pygame.K_s]:    cmd_x += cmd_limits[0, 0]
    if keys[pygame.K_a]:    cmd_y += cmd_limits[1, 1]
    if keys[pygame.K_d]:    cmd_y += cmd_limits[1, 0]
    if keys[pygame.K_q]:    cmd_yaw += cmd_limits[2, 1]
    if keys[pygame.K_e]:    cmd_yaw += cmd_limits[2, 0]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    save_video = args.save_video
    config_file = "go2_simple.yaml"

    pygame.init()
    screen = pygame.display.set_mode((200, 100))
    pygame.display.set_caption("Keyboard Control")
    print("Keyboard control: W/S=vx, A/D=vy, Q/E=yaw")

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

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cycle_time = config["cycle_time"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        cmd_limits = np.array(config["cmd_range"], dtype=np.float32)

        control_dt = simulation_dt * control_decimation

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    # context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    episode_length = 0

    clip_obs = 100.0
    clip_actions = 100.0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # warmup: PD control to default pose before policy takes over
    for _ in range(1000):
        tau = pd_control(default_angles, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

    renderer = mujoco.Renderer(m, height=360, width=640)
    policy = torch.jit.load(policy_path)

    video_fps = 50
    if save_video:
        model_name = os.path.basename(policy_path).split('.')[0]
        cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = max(int(sim_fps / video_fps), 1)
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Video: {video_path}")

    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            if counter % control_decimation == 0:
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                cmd = get_keyboard_command(keys, cmd_limits)

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
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
                episode_length += 1
                quat = d.qpos[3:7]
                ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])
                qj = (d.qpos[7:] - default_angles) * dof_pos_scale
                dqj = d.qvel[6:] * dof_vel_scale

                phase = (episode_length * control_dt) % cycle_time / cycle_time
                sin_pos = np.sin(2 * np.pi * phase)
                cos_pos = np.cos(2 * np.pi * phase)

                obs[:3] = ang_vel * ang_vel_scale
                obs[3] = sin_pos
                obs[4] = cos_pos
                obs[5:8] = cmd * cmd_scale
                obs[8:8 + num_actions] = qj[idx_mj2model]
                obs[8 + num_actions:8 + 2 * num_actions] = dqj[idx_mj2model]
                obs[8 + 2 * num_actions:8 + 3 * num_actions] = action[idx_mj2model]
                obs[8 + 3 * num_actions:8 + 4 * num_actions] = last_action[idx_mj2model]
                obs = np.clip(obs, -clip_obs, clip_obs)

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                last_action = action.copy()
                result = policy(obs_tensor)
                if isinstance(result, tuple):
                    action = result[0].detach().numpy().squeeze()[idx_model2mj]
                else:
                    action = result.detach().cpu().numpy().squeeze()[idx_model2mj]
                action = np.clip(action, -clip_actions, clip_actions)
                target_dof_pos = action * action_scale + default_angles

                vel = d.qvel[:3]
                local_vel = quat_rotate_inverse(quat, vel)
                local_ang_vel = ang_vel
                print(f"Speed: Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}, "
                      f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}", end='\r')

            mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
            viewer.sync()

    if save_video:
        print(f"\nVideo saved to {video_path}")
        writer.close()
