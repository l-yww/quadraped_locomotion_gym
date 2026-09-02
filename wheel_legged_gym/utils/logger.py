# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger:
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
        # self._plot()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 7))  # 画hip, 第一行左腿，第二行右腿，每一列是转速-s、力矩-s、转速-力矩
        fig, axs1 = plt.subplots(nb_rows, nb_cols, figsize=(10, 7)) # 画knee
        fig, axs2 = plt.subplots(nb_rows, nb_cols, figsize=(10, 7)) # 画wheel
        _, axs_extra = plt.subplots(2, 2)
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
        if log["est_lin_vel_x"] != []:  a.plot(time, log["est_lin_vel_x"], label="est-x")
        if log["est_lin_vel_y"] != []:  a.plot(time, log["est_lin_vel_y"], label="est-y")
        if log["est_lin_vel_z"] != []:  a.plot(time, log["est_lin_vel_z"], label="est-z")
        a.set(xlabel="time [s]", ylabel="base_vel", title="Base Vel")
        a.legend()

        a = axs_extra[1, 0]
        if log["noised_base_roll"] != []:  a.plot(time, log["noised_base_roll"], label="obs roll")
        if log["noised_base_pitch"] != []:  a.plot(time, log["noised_base_pitch"], label="obs yaw")
        if log["noised_base_yaw"] != []:  a.plot(time, log["noised_base_yaw"], label="obs pitch")
        if log["base_roll"] != []:  a.plot(time, log["base_roll"], label="real roll")
        if log["base_pitch"] != []:  a.plot(time, log["base_pitch"], label="real yaw")
        if log["base_yaw"] != []:  a.plot(time, log["base_yaw"], label="real pitch")
        a.set(xlabel="time [s]", ylabel="base euler angles", title="Base Angles")
        a.legend()

        a = axs_extra[1, 1]
        if log["base_pos_x"] != []:  a.plot(time, log["base_pos_x"], label="x")
        if log["base_pos_y"] != []:  a.plot(time, log["base_pos_y"], label="y")
        if log["base_pos_z"] != []:  a.plot(time, log["base_pos_z"], label="z")
        a.set(xlabel="time [s]", ylabel="base pos [m]", title="Base Pos")        
        a.legend()

        # 图零
        # 力矩 时间
        a = axs[0, 0]
        if log["left_hip_torque"] != []:
            a.plot(time, log["left_hip_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Left Hip Torque [Nm]", title="Left Hip Torque")
        a.legend()
        # 速度 时间
        a = axs[1, 0]
        if log["left_hip_vel"] != []:
            a.plot(time, log["left_hip_vel"], label="real vel")
        if log["noised_left_hip_vel"] != []:
            a.plot(time, log["noised_left_hip_vel"], label="obs vel")
        a.set(xlabel="time [s]", ylabel="Left Hip Vel[rad/s]", title="Left Hip Velocity")
        a.legend()
        # 力矩 速度
        a = axs[0, 2]
        if log["left_hip_vel"] != [] and log["left_hip_torque"] != []:
            a.plot(log["left_hip_vel"], log["left_hip_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs[0, 1]
        if log["right_hip_torque"] != []:
            a.plot(time, log["right_hip_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="right Hip Torque [Nm]", title="right Hip Torque")
        a.legend()
        # 速度 时间
        a = axs[1, 1]
        if log["right_hip_vel"] != []:
            a.plot(time, log["right_hip_vel"], label="real vel")
        if log["noised_right_hip_vel"] != []:
            a.plot(time, log["noised_right_hip_vel"], label="obs vel")
        a.set(xlabel="time [s]", ylabel="right Hip Vel[rad/s]", title="right Hip Velocity")
        a.legend()
        # 力矩 速度
        a = axs[1, 2]
        if log["right_hip_vel"] != [] and log["right_hip_torque"] != []:
            a.plot(log["right_hip_vel"], log["right_hip_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        plt.tight_layout()
        # 角度 时间
        a = axs[2, 0]
        if log["noised_left_hip_pos"] != []:
            a.plot(time, log["noised_left_hip_pos"], label="obs pos")
        if log["left_hip_pos"] != []:
            a.plot(time, log["left_hip_pos"], label="real pos")
            a.plot(time, log["left_hip_action"], label="left hip action")
        if log["left_hip_ref_pos"] != []:
            a.plot(time, log["left_hip_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="Left Hip Pos[rad]", title="Left Hip Pos")
        a.legend()
        a = axs[2, 1]
        if log["noised_right_hip_pos"] != []:
            a.plot(time, log["noised_right_hip_pos"], label="obs pos")
        if log["right_hip_pos"] != []:
            a.plot(time, log["right_hip_pos"], label="real pos")
            a.plot(time, log["right_hip_action"], label="right hip action")
        if log["right_hip_ref_pos"] != []:
            a.plot(time, log["right_hip_ref_pos"], label="ref_pos")
        a.set(xlabel="time [s]", ylabel="Right Hip Pos[rad]", title="Right Hip Pos")
        a.legend()

        # 图一 knee
        # 力矩 时间
        a = axs1[0, 0]
        if log["left_knee_torque"] != []:
            a.plot(time, log["left_knee_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Left knee Torque [Nm]", title="Left knee Torque")
        a.legend()
        # 速度 时间
        a = axs1[1, 0]
        if log["noised_left_knee_vel"] != []:
            a.plot(time, log["noised_left_knee_vel"], label="obs vel")
        if log["left_knee_vel"] != []:
            a.plot(time, log["left_knee_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="Left knee Vel[rad/s]", title="Left knee Velocity")
        a.legend()
        # 力矩 速度
        a = axs1[0, 2]
        if log["left_knee_vel"] != [] and log["left_knee_torque"] != []:
            a.plot(log["left_knee_vel"], log["left_knee_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs1[0, 1]
        if log["right_knee_torque"] != []:
            a.plot(time, log["right_knee_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="right knee Torque [Nm]", title="right knee Torque")
        a.legend()
        # 速度 时间
        a = axs1[1, 1]
        if log["noised_right_knee_vel"] != []:
            a.plot(time, log["noised_right_knee_vel"], label="obs vel")
        if log["right_knee_vel"] != []:
            a.plot(time, log["right_knee_vel"], label="real vel")
        a.set(xlabel="time [s]", ylabel="right knee Vel[rad/s]", title="right knee Velocity")
        a.legend()
        # 力矩 速度
        a = axs1[1, 2]
        if log["right_knee_vel"] != [] and log["right_knee_torque"] != []:
            a.plot(log["right_knee_vel"], log["right_knee_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        # 角度 时间
        a = axs1[2, 0]
        if log["noised_left_knee_pos"] != []:
            a.plot(time, log["noised_left_knee_pos"], label="noised left knee pos")
        if log["left_knee_pos"] != []:
            a.plot(time, log["left_knee_pos"], label="left knee pos")
            a.plot(time, log["left_knee_action"], label="left knee action")
        a.set(xlabel="time [s]", ylabel="left knee pos[rad]", title="Left knee Pos")
        a.legend()
        a = axs1[2, 1]
        if log["noised_right_knee_pos"] != []:
            a.plot(time, log["noised_right_knee_pos"], label="noised right knee pos")
        if log["right_knee_pos"] != []:
            a.plot(time, log["right_knee_pos"], label="right knee pos")
            a.plot(time, log["right_knee_action"], label="right knee action")
        a.set(xlabel="time [s]", ylabel="right knee pos[rad]", title="Right knee Pos")
        a.legend()

        # 图二 wheel
        # 力矩 时间
        a = axs2[0, 0]
        if log["left_wheel_torque"] != []:
            a.plot(time, log["left_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Left wheel Torque [Nm]", title="Left wheel Torque")
        a.legend()
        # 速度 时间
        a = axs2[1, 0]
        if log["noised_left_wheel_vel"] != []:
            a.plot(time, log["noised_left_wheel_vel"], label="obs vel")
        if log["left_wheel_vel"] != []:
            a.plot(time, log["left_wheel_vel"], label="real vel")
            a.plot(time, log["left_wheel_action"], label="wheel action")
        if log["left_wheel_ref_vel"] != []:
            a.plot(time, log["left_wheel_ref_vel"], label="ref vel")
        a.set(xlabel="time [s]", ylabel="Left wheel Vel[rad/s]", title="Left wheel Velocity")
        a.legend()
        # 力矩 速度
        a = axs2[0, 2]
        if log["left_wheel_vel"] != [] and log["left_wheel_torque"] != []:
            a.plot(log["left_wheel_vel"], log["left_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()

        a = axs2[0, 1]
        if log["right_wheel_torque"] != []:
            a.plot(time, log["right_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="right wheel Torque [Nm]", title="right wheel Torque")
        a.legend()
        # 速度 时间
        a = axs2[1, 1]
        if log["noised_right_wheel_vel"] != []:
            a.plot(time, log["noised_right_wheel_vel"], label="obs vel")
        if log["right_wheel_vel"] != []:
            a.plot(time, log["right_wheel_vel"], label="real vel")
            a.plot(time, log["right_wheel_action"], label="wheel action")
        if log["right_wheel_ref_vel"] != []:
            a.plot(time, log["right_wheel_ref_vel"], label="ref vel")
        a.set(xlabel="time [s]", ylabel="right wheel Vel[rad/s]", title="right wheel Velocity")
        a.legend()
        # 力矩 速度
        a = axs2[1, 2]
        if log["right_wheel_vel"] != [] and log["right_wheel_torque"] != []:
            a.plot(log["right_wheel_vel"], log["right_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
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
