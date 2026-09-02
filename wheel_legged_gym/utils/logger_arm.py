# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ... (保持原有版权声明)

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger_arm:
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
        fig, axs5 = plt.subplots(nb_rows, nb_cols-2, figsize=(10, 7)) # 画joint1
        fig, axs6 = plt.subplots(nb_rows, nb_cols-2, figsize=(10, 7)) # 画joint2
        fig, axs7 = plt.subplots(nb_rows, nb_cols-2, figsize=(10, 7)) # 画joint3
        fig, axs8 = plt.subplots(nb_rows, nb_cols-2, figsize=(10, 7)) # 画joint4
        fig, axs9 = plt.subplots(nb_rows, nb_cols-2, figsize=(10, 7)) # 画joint5
        fig, axs10 = plt.subplots(nb_rows, nb_cols-2, figsize=(10, 7)) # 画joint6
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log

        # joint1
        # 力矩 时间
        a = axs5[0, 0]
        if log["joint1_torque"] != []:
            a.plot(time, log["joint1_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="joint1 Torque [Nm]", title="joint1 Torque")
        a.legend()
        # 速度 时间
        a = axs5[1, 0]
        if log["joint1_vel"] != []:
            a.plot(time, log["joint1_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="joint1 Vel[rad/s]", title="joint1 Velocity")
        a.legend()
        # 位置 时间
        a = axs5[2, 0]
        if log["joint1_pos"] != []:
            a.plot(time, log["joint1_pos"], label="real pos")
        if log["joint1_action"] != []:
            a.plot(time, log["joint1_action"], label="joint1 action")
        if log["joint1_ref_pos"] != []:
            a.plot(time, log["joint1_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="joint1 pos[rad]", title="joint1 Pos")
        a.legend()
        # 力矩 速度
        a = axs5[0, 1]
        if log["joint1_vel"] != [] and log["joint1_torque"] != []:
            a.plot(log["joint1_vel"], log["joint1_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        a = axs5[1, 1]
        if log["joint1_power"] != []:
            a.plot(time, log["joint1_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="joint1 Power[rad/s]", title="joint1 Power")
        a.legend()

        # joint2
        # 力矩 时间
        a = axs6[0, 0]
        if log["joint2_torque"] != []:
            a.plot(time, log["joint2_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="joint2 Torque [Nm]", title="joint2 Torque")
        a.legend()
        # 速度 时间
        a = axs6[1, 0]
        if log["joint2_vel"] != []:
            a.plot(time, log["joint2_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="joint2 Vel[rad/s]", title="joint2 Velocity")
        a.legend()
        # 位置 时间
        a = axs6[2, 0]
        if log["joint2_pos"] != []:
            a.plot(time, log["joint2_pos"], label="real pos")
        if log["joint2_action"] != []:
            a.plot(time, log["joint2_action"], label="joint2 action")
        if log["joint2_ref_pos"] != []:
            a.plot(time, log["joint2_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="joint2 pos[rad]", title="joint2 Pos")
        a.legend()
        # 力矩 速度
        a = axs6[0, 1]
        if log["joint2_vel"] != [] and log["joint2_torque"] != []:
            a.plot(log["joint2_vel"], log["joint2_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        a = axs6[1, 1]
        if log["joint2_power"] != []:
            a.plot(time, log["joint2_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="joint2 Power[rad/s]", title="joint2 Power")
        a.legend()

        # joint3
        # 力矩 时间
        a = axs7[0, 0]
        if log["joint3_torque"] != []:
            a.plot(time, log["joint3_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="joint3 Torque [Nm]", title="joint3 Torque")
        a.legend()
        # 速度 时间
        a = axs7[1, 0]
        if log["joint3_vel"] != []:
            a.plot(time, log["joint3_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="joint3 Vel[rad/s]", title="joint3 Velocity")
        a.legend()
        # 位置 时间
        a = axs7[2, 0]
        if log["joint3_pos"] != []:
            a.plot(time, log["joint3_pos"], label="real pos")
        if log["joint3_action"] != []:
            a.plot(time, log["joint3_action"], label="joint3 action")
        if log["joint3_ref_pos"] != []:
            a.plot(time, log["joint3_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="joint3 pos[rad]", title="joint3 Pos")
        a.legend()
        # 力矩 速度
        a = axs7[0, 1]
        if log["joint3_vel"] != [] and log["joint3_torque"] != []:
            a.plot(log["joint3_vel"], log["joint3_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        a = axs7[1, 1]
        if log["joint3_power"] != []:
            a.plot(time, log["joint3_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="joint3 Power[rad/s]", title="joint3 Power")
        a.legend()

        # joint4
        # 力矩 时间
        a = axs8[0, 0]
        if log["joint4_torque"] != []:
            a.plot(time, log["joint4_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="joint4 Torque [Nm]", title="joint4 Torque")
        a.legend()
        # 速度 时间
        a = axs8[1, 0]
        if log["joint4_vel"] != []:
            a.plot(time, log["joint4_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="joint4 Vel[rad/s]", title="joint4 Velocity")
        a.legend()
        # 位置 时间
        a = axs8[2, 0]
        if log["joint4_pos"] != []:
            a.plot(time, log["joint4_pos"], label="real pos")
        if log["joint4_action"] != []:
            a.plot(time, log["joint4_action"], label="joint4 action")
        if log["joint4_ref_pos"] != []:
            a.plot(time, log["joint4_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="joint4 pos[rad]", title="joint4 Pos")
        a.legend()
        # 力矩 速度
        a = axs8[0, 1]
        if log["joint4_vel"] != [] and log["joint4_torque"] != []:
            a.plot(log["joint4_vel"], log["joint4_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        a = axs8[1, 1]
        if log["joint4_power"] != []:
            a.plot(time, log["joint4_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="joint4 Power[rad/s]", title="joint4 Power")
        a.legend()

        # joint5
        # 力矩 时间
        a = axs9[0, 0]
        if log["joint5_torque"] != []:
            a.plot(time, log["joint5_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="joint5 Torque [Nm]", title="joint5 Torque")
        a.legend()
        # 速度 时间
        a = axs9[1, 0]
        if log["joint5_vel"] != []:
            a.plot(time, log["joint5_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="joint5 Vel[rad/s]", title="joint5 Velocity")
        a.legend()
        # 位置 时间
        a = axs9[2, 0]
        if log["joint5_pos"] != []:
            a.plot(time, log["joint5_pos"], label="real pos")
        if log["joint5_action"] != []:
            a.plot(time, log["joint5_action"], label="joint5 action")
        if log["joint5_ref_pos"] != []:
            a.plot(time, log["joint5_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="joint5 pos[rad]", title="joint5 Pos")
        a.legend()
        # 力矩 速度
        a = axs9[0, 1]
        if log["joint5_vel"] != [] and log["joint5_torque"] != []:
            a.plot(log["joint5_vel"], log["joint5_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        a = axs9[1, 1]
        if log["joint5_power"] != []:
            a.plot(time, log["joint5_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="joint5 Power[rad/s]", title="joint5 Power")
        a.legend()

        # joint6
        # 力矩 时间
        a = axs10[0, 0]
        if log["joint6_torque"] != []:
            a.plot(time, log["joint6_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="joint6 Torque [Nm]", title="joint6 Torque")
        a.legend()
        # 速度 时间
        a = axs10[1, 0]
        if log["joint6_vel"] != []:
            a.plot(time, log["joint6_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="joint6 Vel[rad/s]", title="joint6 Velocity")
        a.legend()
        # 位置 时间
        a = axs10[2, 0]
        if log["joint6_pos"] != []:
            a.plot(time, log["joint6_pos"], label="real pos")
        if log["joint6_action"] != []:
            a.plot(time, log["joint6_action"], label="joint6 action")
        if log["joint6_ref_pos"] != []:
            a.plot(time, log["joint6_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="joint6 pos[rad]", title="joint6 Pos")
        a.legend()
        # 力矩 速度
        a = axs10[0, 1]
        if log["joint6_vel"] != [] and log["joint6_torque"] != []:
            a.plot(log["joint6_vel"], log["joint6_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        a = axs10[1, 1]
        if log["joint6_power"] != []:
            a.plot(time, log["joint6_power"], label="real power")
        a.set(xlabel="time [s]", ylabel="joint6 Power[rad/s]", title="joint6 Power")
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
