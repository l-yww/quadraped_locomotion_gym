import sys
from pathlib import Path
PATH_PARENT = Path(__file__).parent
sys.path.append(str(PATH_PARENT))
from utils import MujocoRenderUtils

import os
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import imageio
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    args = parser.parse_args()
    save_video = args.save_video
    config_file = "cowa2.yaml"

    with open(str(PATH_PARENT / "configs" / config_file), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"]
        # Resolve relative paths relative to project root (PATH_PARENT.parent)
        if not os.path.isabs(xml_path):
            xml_path = str((PATH_PARENT.parent / xml_path).resolve())

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        num_actions = config["num_actions"]
        control_dt = simulation_dt * control_decimation

    # Sin trajectory parameters matching cowa2_test training config
    # cycle_time = 2.0s -> frequency = 0.5 Hz
    sin_frequency = 0.5
    sin_amplitude = 0.3  # rad, matching all joint amplitudes in training

    # Half-wave sinusoidal: only positive half-cycle produces motion
    def half_wave(t):
        s = np.sin(2 * np.pi * sin_frequency * t)
        return np.maximum(s, 0.0)

    # Trot gait: diagonal pairs for hip/thigh, all calf joints same sign
    # FL/RR: no phase offset, use hw(t)
    # FR/RL: 0.5 cycle phase offset = hw(t + 0.5/freq) = hw at opposite phase
    # Calf joints all use same sign (axis is 0 1 0 for all 4)
    trot_config = {
        # FL leg (indices 0,1,2) - phase 0.0
        0:   (0.0,  1.0),   # FL_hip
        1:   (0.0,  1.0),   # FL_thigh
        2:   (0.0, -1.0),   # FL_calf
        # FR leg (indices 3,4,5) - phase 0.5 (opposite)
        3:   (0.5,  1.0),   # FR_hip
        4:   (0.5,  1.0),   # FR_thigh
        5:   (0.5, -1.0),   # FR_calf
        # RL leg (indices 6,7,8) - phase 0.5 (opposite)
        6:   (0.5,  1.0),   # RL_hip
        7:   (0.5,  1.0),   # RL_thigh
        8:   (0.5, -1.0),   # RL_calf
        # RR leg (indices 9,10,11) - phase 0.0
        9:   (0.0,  1.0),   # RR_hip
        10:  (0.0,  1.0),   # RR_thigh
        11:  (0.0, -1.0),   # RR_calf
    }

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Initialize joint positions to default angles
    d.qpos[7:] = default_angles.copy()
    mujoco.mj_forward(m, d)

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    counter = 0
    log_time = []
    log_qpos = []
    log_qvel = []
    log_target = []
    log_torque = []
    log_base_z = []
    log_base_ang_vel = []

    print(f"Running trot trajectory test (half-wave sin): freq={sin_frequency}Hz, amplitude={sin_amplitude}rad")
    print(f"All 12 joints: trot gait with diagonal pairs")
    print(f"Simulation dt={simulation_dt*1000:.1f}ms, control dt={control_dt*1000:.1f}ms")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            sim_time = time.time() - start

            if counter % control_decimation == 0:
                # Generate trot trajectory target matching cowa2_test training
                target_pos = default_angles.copy()
                t = sim_time
                for joint_idx, (phase_offset, sign) in trot_config.items():
                    hw = half_wave(t + phase_offset / sin_frequency)
                    target_pos[joint_idx] = default_angles[joint_idx] + sign * hw * sin_amplitude

                # PD control
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                tau = pd_control(target_pos, qj, kps, np.zeros_like(kds), dqj, kds)
                d.ctrl[:] = tau

                # Log data
                log_time.append(t)
                log_qpos.append(qj.copy())
                log_qvel.append(dqj.copy())
                log_target.append(target_pos.copy())
                log_torque.append(tau.copy())
                log_base_z.append(d.qpos[2].copy())
                quat = d.qpos[3:7]
                log_base_ang_vel.append(np.array([d.qvel[3], d.qvel[4], d.qvel[5]]).copy())

            mujoco.mj_step(m, d)
            viewer.sync()
            counter += 1

    print(f"\nSimulation finished, logging {len(log_time)} steps")

    # Save plots
    log_path = str(PATH_PARENT / "logs")
    os.makedirs(log_path, exist_ok=True)

    joint_names = config.get("mujoco_joint_names", [f"joint_{i}" for i in range(num_actions)])
    log_time = np.array(log_time)
    log_qpos = np.array(log_qpos)
    log_qvel = np.array(log_qvel)
    log_target = np.array(log_target)
    log_torque = np.array(log_torque)
    log_base_z = np.array(log_base_z)
    log_base_ang_vel = np.array(log_base_ang_vel)

    n = len(log_time)
    if n > 0:
        # Plot 1: Joint positions vs target
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Sin Trajectory Test: Joint Positions vs Target')
        for i in range(num_actions):
            ax = axes[i // 3][i % 3]
            ax.plot(log_time, log_qpos[:, i], label=f'{joint_names[i]} actual', linewidth=1.0)
            ax.plot(log_time, log_target[:, i], label=f'{joint_names[i]} target', linewidth=1.0, linestyle='--')
            ax.set_ylabel('rad')
            ax.legend(fontsize=6)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'sin_qpos_vs_target.png'), dpi=150)
        plt.close()

        # Plot 2: Joint velocities
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Sin Trajectory Test: Joint Velocities')
        for i in range(num_actions):
            ax = axes[i // 3][i % 3]
            ax.plot(log_time, log_qvel[:, i], label=f'{joint_names[i]}', linewidth=1.0)
            ax.set_ylabel('rad/s')
            ax.legend(fontsize=6)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'sin_qvel.png'), dpi=150)
        plt.close()

        # Plot 3: Torques
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Sin Trajectory Test: Joint Torques')
        for i in range(num_actions):
            ax = axes[i // 3][i % 3]
            ax.plot(log_time, log_torque[:, i], label=f'{joint_names[i]}', linewidth=1.0)
            ax.set_ylabel('Nm')
            ax.legend(fontsize=6)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'sin_torque.png'), dpi=150)
        plt.close()

        # Plot 4: Base height and angular velocity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle('Sin Trajectory Test: Base Stability')
        ax1.plot(log_time, log_base_z, label='base z', linewidth=1.5, color='b')
        ax1.set_ylabel('m')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(log_time, log_base_ang_vel[:, 0], label='ang_vel_x', linewidth=1.0)
        ax2.plot(log_time, log_base_ang_vel[:, 1], label='ang_vel_y', linewidth=1.0)
        ax2.plot(log_time, log_base_ang_vel[:, 2], label='ang_vel_z', linewidth=1.0)
        ax2.set_ylabel('rad/s')
        ax2.legend()
        ax2.grid(True)
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'sin_base_stability.png'), dpi=150)
        plt.close()

        print(f"Plots saved to {log_path}/")
        print("Check sin_base_stability.png for jitter indicators (high angular velocity spikes = jitter)")
