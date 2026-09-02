# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ... (保持原有版权声明)

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger_8dof:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 7))  # 画hip_roll
        fig, axs1 = plt.subplots(nb_rows, nb_cols, figsize=(10, 7)) # 画hip_pitch
        fig, axs2 = plt.subplots(nb_rows, nb_cols, figsize=(10, 7)) # 画knee
        fig, axs3 = plt.subplots(nb_rows, nb_cols, figsize=(10, 7)) # 画wheel
        _, axs_extra = plt.subplots(3, 2, figsize=(10, 7))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log

        # ######################
        a = axs_extra[0, 0]
        if log["base_height"] != []:  a.plot(time, log["base_height"], label="real")
        if log["base_height_cmd"] != []:  a.plot(time, log["base_height_cmd"], label="cmd")
        a.set(xlabel="time [s]", ylabel="base height [m]", title="base height tracking")
        a.legend()

        a = axs_extra[0, 1]
        if log["base_vel_x"] != []:  a.plot(time, log["base_vel_x"], label="real-x")
        if log["base_vel_y"] != []:  a.plot(time, log["base_vel_y"], label="real-y")
        if log["base_vel_z"] != []:  a.plot(time, log["base_vel_z"], label="real-z")
        if log["cmd_vel_x"] != []:  a.plot(time, log["cmd_vel_x"], label="cmd-x")
        a.set(xlabel="time [s]", ylabel="base_vel", title="Base Vel")
        a.legend()

        a = axs_extra[1, 0]
        if log["base_roll"] != []:  a.plot(time, log["base_roll"], label="real roll")
        if log["base_pitch"] != []:  a.plot(time, log["base_pitch"], label="real pitch")
        if log["base_yaw"] != []:  a.plot(time, log["base_yaw"], label="real yaw")
        if log["ang_vel_x"] != []:  a.plot(time, log["ang_vel_x"], label="ang_vel_x")
        if log["ang_vel_y"] != []:  a.plot(time, log["ang_vel_y"], label="ang_vel_y")
        if log["ang_vel_z"] != []:  a.plot(time, log["ang_vel_z"], label="ang_vel_z")
        if log["cmd_ang_vel"] != []:  a.plot(time, log["cmd_ang_vel"], label="cmd_ang_vel")
        a.set(xlabel="time [s]", ylabel="base euler angles & angular velocity", title="Base Angles & Angular Velocity")
        a.legend()

        a = axs_extra[1, 1]
        if log["base_pos_x"] != []:  a.plot(time, log["base_pos_x"], label="x")
        if log["base_pos_y"] != []:  a.plot(time, log["base_pos_y"], label="y")
        if log["base_pos_z"] != []:  a.plot(time, log["base_pos_z"], label="z")
        a.set(xlabel="time [s]", ylabel="base pos [m]", title="Base Pos")        
        a.legend()

        a = axs_extra[2, 0]
        if log["contact_forces_l_x"] != []:  a.plot(time, log["contact_forces_l_x"], label="x")
        if log["contact_forces_l_y"] != []:  a.plot(time, log["contact_forces_l_y"], label="y")
        if log["contact_forces_l_z"] != []:  a.plot(time, log["contact_forces_l_z"], label="z")
        a.set(xlabel="time [s]", ylabel="contact_forces_l", title="contact_forces_l")    
        a.legend()

        a = axs_extra[2, 1]
        if log["contact_forces_r_x"] != []:  a.plot(time, log["contact_forces_r_x"], label="x")
        if log["contact_forces_r_y"] != []:  a.plot(time, log["contact_forces_r_y"], label="y")
        if log["contact_forces_r_z"] != []:  a.plot(time, log["contact_forces_r_z"], label="z")
        a.set(xlabel="time [s]", ylabel="contact_forces_r", title="contact_forces_r")        
        a.legend()

        # hip roll
        # 力矩 时间
        a = axs[0, 0]
        if log["left_hip_roll_torque"] != []:
            a.plot(time, log["left_hip_roll_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Left Hip Roll Torque [Nm]", title="Left Hip Roll Torque")
        a.legend()
        # 速度 时间
        a = axs[1, 0]
        if log["left_hip_roll_vel"] != []:
            a.plot(time, log["left_hip_roll_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="Left Hip Roll Vel[rad/s]", title="Left Hip Roll Velocity")
        a.legend()
        # 力矩 速度
        a = axs[0, 2]
        if log["left_hip_roll_vel"] != [] and log["left_hip_roll_torque"] != []:
            a.plot(log["left_hip_roll_vel"], log["left_hip_roll_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs[0, 3]
        if log["left_hip_roll_power"] != []:
            a.plot(time, log["left_hip_roll_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Left Hip Roll Power[rad/s]", title="Left Hip Roll Power")
        a.legend()

        a = axs[0, 1]
        if log["right_hip_roll_torque"] != []:
            a.plot(time, log["right_hip_roll_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="right Hip Torque [Nm]", title="right Hip Torque")
        a.legend()
        # 速度 时间
        a = axs[1, 1]
        if log["right_hip_roll_vel"] != []:
            a.plot(time, log["right_hip_roll_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="right Hip Vel[rad/s]", title="right Hip Velocity")
        a.legend()
        # 力矩 速度
        a = axs[1, 2]
        if log["right_hip_roll_vel"] != [] and log["right_hip_roll_torque"] != []:
            a.plot(log["right_hip_roll_vel"], log["right_hip_roll_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs[1, 3]
        if log["right_hip_roll_power"] != []:
            a.plot(time, log["right_hip_roll_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Right Hip Roll Power[rad/s]", title="Right Hip Roll Power")
        a.legend()

        plt.tight_layout()
        # 角度 时间
        a = axs[2, 0]
        if log["left_hip_roll_pos"] != []:
            a.plot(time, log["left_hip_roll_pos"], label="real pos")
        if log["left_hip_roll_action"] != []:
            a.plot(time, log["left_hip_roll_action"], label="left hip action")
        if log["left_hip_roll_ref_pos"] != []:
            a.plot(time, log["left_hip_roll_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="Left Hip Pos[rad]", title="Left Hip Pos")
        a.legend()
        a = axs[2, 1]
        if log["right_hip_roll_pos"] != []:
            a.plot(time, log["right_hip_roll_pos"], label="real pos")
        if log["right_hip_roll_action"] != []:
            a.plot(time, log["right_hip_roll_action"], label="right hip action")
        if log["right_hip_roll_ref_pos"] != []:
            a.plot(time, log["right_hip_roll_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="Right Hip Pos[rad]", title="Right Hip Pos")
        a.legend()

        # hip pitch
        # 力矩 时间
        a = axs1[0, 0]
        if log["left_hip_pitch_torque"] != []:
            a.plot(time, log["left_hip_pitch_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Left Hip Torque [Nm]", title="Left Hip Torque")
        a.legend()
        # 速度 时间
        a = axs1[1, 0]
        if log["left_hip_pitch_vel"] != []:
            a.plot(time, log["left_hip_pitch_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="Left Hip Vel[rad/s]", title="Left Hip Velocity")
        a.legend()
        # 力矩 速度
        a = axs1[0, 2]
        if log["left_hip_pitch_vel"] != [] and log["left_hip_pitch_torque"] != []:
            a.plot(log["left_hip_pitch_vel"], log["left_hip_pitch_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs1[0, 3]
        if log["left_hip_pitch_power"] != []:
            a.plot(time, log["left_hip_pitch_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Left Hip Pitch Power[rad/s]", title="Left Hip Pitch Power")
        a.legend()

        a = axs1[0, 1]
        if log["right_hip_pitch_torque"] != []:
            a.plot(time, log["right_hip_pitch_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="right Hip Torque [Nm]", title="right Hip Torque")
        a.legend()
        # 速度 时间
        a = axs1[1, 1]
        if log["right_hip_pitch_vel"] != []:
            a.plot(time, log["right_hip_pitch_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="right Hip Vel[rad/s]", title="right Hip Velocity")
        a.legend()
        # 力矩 速度
        a = axs1[1, 2]
        if log["right_hip_pitch_vel"] != [] and log["right_hip_pitch_torque"] != []:
            a.plot(log["right_hip_pitch_vel"], log["right_hip_pitch_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs1[1, 3]
        if log["right_hip_pitch_power"] != []:
            a.plot(time, log["right_hip_pitch_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Right Hip Pitch Power[rad/s]", title="Right Hip Pitch Power")
        a.legend()

        plt.tight_layout()
        # 角度 时间
        a = axs1[2, 0]
        if log["left_hip_pitch_pos"] != []:
            a.plot(time, log["left_hip_pitch_pos"], label="real pos")
        if log["left_hip_pitch_action"] != []:
            a.plot(time, log["left_hip_pitch_action"], label="left hip action")
        if log["left_hip_pitch_ref_pos"] != []:
            a.plot(time, log["left_hip_pitch_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="Left Hip Pos[rad]", title="Left Hip Pos")
        a.legend()
        a = axs1[2, 1]
        if log["right_hip_pitch_pos"] != []:
            a.plot(time, log["right_hip_pitch_pos"], label="real pos")
        if log["right_hip_pitch_action"] != []:
            a.plot(time, log["right_hip_pitch_action"], label="right hip action")
        if log["right_hip_pitch_ref_pos"] != []:
            a.plot(time, log["right_hip_pitch_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="Right Hip Pos[rad]", title="Right Hip Pos")
        a.legend()

        # knee
        # 力矩 时间
        a = axs2[0, 0]
        if log["left_knee_torque"] != []:
            a.plot(time, log["left_knee_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Left knee Torque [Nm]", title="Left knee Torque")
        a.legend()
        # 速度 时间
        a = axs2[1, 0]
        if log["left_knee_vel"] != []:
            a.plot(time, log["left_knee_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="Left knee Vel[rad/s]", title="Left knee Velocity")
        a.legend()
        # 力矩 速度
        a = axs2[0, 2]
        if log["left_knee_vel"] != [] and log["left_knee_torque"] != []:
            a.plot(log["left_knee_vel"], log["left_knee_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs2[0, 3]
        if log["left_knee_power"] != []:
            a.plot(time, log["left_knee_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Left Knee Power[rad/s]", title="Left Knee Power")
        a.legend()

        a = axs2[0, 1]
        if log["right_knee_torque"] != []:
            a.plot(time, log["right_knee_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="right knee Torque [Nm]", title="right knee Torque")
        a.legend()
        # 速度 时间
        a = axs2[1, 1]
        if log["right_knee_vel"] != []:
            a.plot(time, log["right_knee_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="right knee Vel[rad/s]", title="right knee Velocity")
        a.legend()
        # 力矩 速度
        a = axs2[1, 2]
        if log["right_knee_vel"] != [] and log["right_knee_torque"] != []:
            a.plot(log["right_knee_vel"], log["right_knee_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs2[1, 3]
        if log["right_knee_power"] != []:
            a.plot(time, log["right_knee_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Right Knee Power[rad/s]", title="Right Knee Power")
        a.legend()

        # 角度 时间
        a = axs2[2, 0]
        if log["left_knee_pos"] != []:
            a.plot(time, log["left_knee_pos"], label="left knee pos")
        if log["left_knee_action"] != []:
            a.plot(time, log["left_knee_action"], label="left knee action")
        if log["left_knee_ref_pos"] != []:
            a.plot(time, log["left_knee_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="left knee pos[rad]", title="Left knee Pos")
        a.legend()
        a = axs2[2, 1]
        if log["right_knee_pos"] != []:
            a.plot(time, log["right_knee_pos"], label="right knee pos")
        if log["right_knee_action"] != []:
            a.plot(time, log["right_knee_action"], label="right knee action")
        if log["right_knee_ref_pos"] != []:
            a.plot(time, log["right_knee_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="right knee pos[rad]", title="Right knee Pos")
        a.legend()

        # wheel
        # 力矩 时间
        a = axs3[0, 0]
        if log["left_wheel_torque"] != []:
            a.plot(time, log["left_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Left wheel Torque [Nm]", title="Left wheel Torque")
        a.legend()
        # 速度 时间
        a = axs3[1, 0]
        if log["left_wheel_vel"] != []:
            a.plot(time, log["left_wheel_vel"], label="real vel")
        if log["left_wheel_ref_pos"] != []:
            a.plot(time, log["left_wheel_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="Left wheel Vel[rad/s]", title="Left wheel Velocity")
        a.legend()
        # 力矩 速度
        a = axs3[0, 2]
        if log["left_wheel_vel"] != [] and log["left_wheel_torque"] != []:
            a.plot(log["left_wheel_vel"], log["left_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs3[0, 3]
        if log["left_wheel_power"] != []:
            a.plot(time, log["left_wheel_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Left Wheel Power[rad/s]", title="Left Wheel Power")
        a.legend()

        a = axs3[0, 1]
        if log["right_wheel_torque"] != []:
            a.plot(time, log["right_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="right wheel Torque [Nm]", title="right wheel Torque")
        a.legend()
        # 速度 时间
        a = axs3[1, 1]
        if log["right_wheel_vel"] != []:
            a.plot(time, log["right_wheel_vel"], label="real vel")
        if log["right_wheel_ref_pos"] != []:
            a.plot(time, log["right_wheel_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="right wheel Vel[rad/s]", title="right wheel Velocity")
        a.legend()
        # 力矩 速度
        a = axs3[1, 2]
        if log["right_wheel_vel"] != [] and log["right_wheel_torque"] != []:
            a.plot(log["right_wheel_vel"], log["right_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs3[1, 3]
        if log["right_wheel_power"] != []:
            a.plot(time, log["right_wheel_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="Right Wheel Power[rad/s]", title="Right Wheel Power")
        a.legend()

        # 角度 时间
        a = axs3[2, 0]
        if log["left_wheel_pos"] != []:
            a.plot(time, log["left_wheel_pos"], label="left wheel pos")
        if log["left_wheel_action"] != []:
            a.plot(time, log["left_wheel_action"], label="left wheel action")
        a.set(xlabel="time [s]", ylabel="left wheel pos[rad]", title="Left wheel Pos")
        a.legend()
        a = axs3[2, 1]
        if log["right_wheel_pos"] != []:
            a.plot(time, log["right_wheel_pos"], label="right wheel pos")
        if log["right_wheel_action"] != []:
            a.plot(time, log["right_wheel_action"], label="right wheel action")
        a.set(xlabel="time [s]", ylabel="right wheel pos[rad]", title="Right wheel Pos")
        a.legend()

        plt.tight_layout()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
