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


class Logger_quad_wtw:
    def __init__(self, dt, save_prefix="", torque_velocity_envelope=None):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.save_prefix = save_prefix
        self.torque_velocity_envelope = torque_velocity_envelope

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
        self._plot()  # run in main process, not subprocess

    def _plot(self):
        nb_rows = 4
        nb_cols = 3
        leg_names = ["FL", "RL", "FR", "RR"]
        joint_types = ["hip", "thigh", "calf"]

        # ============ Figures 0-3: per-leg joint analysis ============
        leg_figs = []
        leg_axs = []
        for i, leg in enumerate(leg_names):
            fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 9))
            fig.suptitle(f"{leg} Leg", fontsize=14)
            leg_figs.append(fig)
            leg_axs.append(axs)

        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log

        for leg_idx, leg in enumerate(leg_names):
            axs = leg_axs[leg_idx]
            for j_idx, joint in enumerate(joint_types):
                row = j_idx
                torque_key = f"{leg}_{joint}_torque"
                vel_key = f"{leg}_{joint}_vel"
                torque_vel_key = f"{leg}_{joint}_torque_vel"
                unclipped_torque_key = f"{leg}_{joint}_unclipped_torque"
                pos_key = f"{leg}_{joint}_pos"
                action_key = f"{leg}_{joint}_action"

                # Col 0: torque vs time
                a = axs[row, 0]
                if log[torque_key]:
                    a.plot(time, log[torque_key], label="torque")
                a.set(xlabel="time [s]", ylabel="Torque [Nm]", title=f"{torque_key}")
                a.legend()

                # Col 1: velocity vs time
                a = axs[row, 1]
                if log[vel_key]:
                    a.plot(time, log[vel_key], label="vel")
                a.set(xlabel="time [s]", ylabel="Vel [rad/s]", title=f"{vel_key}")
                a.legend()

                # Col 2: torque-velocity curve
                a = axs[row, 2]
                if log[vel_key] and log[torque_key]:
                    velocities = np.asarray(log[vel_key])
                    torque_velocities = (
                        np.asarray(log[torque_vel_key])
                        if log.get(torque_vel_key)
                        else velocities
                    )
                    torques = np.asarray(log[torque_key])

                    if self.torque_velocity_envelope is not None:
                        joint_type = "calf" if joint == "calf" else "hip"
                        params = self.torque_velocity_envelope[joint_type]
                        static_limits = np.asarray(
                            self.torque_velocity_envelope["static_limits"]
                        ).reshape(-1)
                        dof_index = self.torque_velocity_envelope["dof_indices"][leg][joint]
                        static_limit = float(static_limits[dof_index])

                        max_vel = params["max_vel"]
                        vel_1 = params["vel_1"]
                        max_torque = params["max_torque"]
                        slope = max_torque / max(max_vel - vel_1, 1e-6)
                        intersection_velocity = max_vel + max_torque / slope

                        velocity_extent = max(
                            np.max(np.abs(torque_velocities)),
                            1.05 * intersection_velocity,
                        )
                        envelope_x = np.linspace(-velocity_extent, velocity_extent, 600)

                        upper_limit = np.clip(
                            -slope * (envelope_x - max_vel),
                            -max_torque,
                            max_torque,
                        )
                        lower_limit = np.clip(
                            -slope * (envelope_x + max_vel),
                            -max_torque,
                            max_torque,
                        )
                        outside_envelope = np.abs(envelope_x) > intersection_velocity
                        upper_limit[outside_envelope] = 0.0
                        lower_limit[outside_envelope] = 0.0

                        upper_limit = np.minimum(upper_limit, static_limit)
                        lower_limit = np.maximum(lower_limit, -static_limit)

                        a.fill_between(
                            envelope_x,
                            lower_limit,
                            upper_limit,
                            color="tab:blue",
                            alpha=0.12,
                            label="allowed torque",
                        )
                        a.plot(
                            envelope_x,
                            upper_limit,
                            color="tab:red",
                            linewidth=1.5,
                            label="upper limit",
                        )
                        a.plot(
                            envelope_x,
                            lower_limit,
                            color="tab:blue",
                            linewidth=1.5,
                            label="lower limit",
                        )

                    if log.get(unclipped_torque_key):
                        a.plot(
                            torque_velocities,
                            np.asarray(log[unclipped_torque_key]),
                            ".",
                            color="tab:orange",
                            markersize=2,
                            label="pre-limit torque",
                        )
                    a.plot(
                        torque_velocities,
                        torques,
                        ".",
                        color="black",
                        markersize=2,
                        label="applied torque",
                    )
                a.set(xlabel="Joint vel [rad/s]", ylabel="Torque [Nm]", title="Torque/velocity")
                a.legend()

            # Row 3: position vs time for all 3 joints
            a = axs[3, 0]
            for joint in joint_types:
                pos_key = f"{leg}_{joint}_pos"
                action_key = f"{leg}_{joint}_action"
                if log[pos_key]:
                    a.plot(time, log[pos_key], label=f"{joint} pos")
                if log[action_key]:
                    a.plot(time, log[action_key], '--', label=f"{joint} action")
            a.set(xlabel="time [s]", ylabel="Pos [rad]", title=f"{leg} Joint Positions")
            a.legend()

            # Contact force
            contact_key = f"{leg}_contact_force"
            if log[contact_key]:
                a = axs[3, 1]
                a.plot(time, log[contact_key], label="contact force")
                a.set(xlabel="time [s]", ylabel="Force [N]", title=f"{leg} Contact Force")
                a.legend()

            # Joint acceleration
            acc_keys_exist = any(log.get(f"{leg}_{joint}_acc") for joint in joint_types)
            if acc_keys_exist:
                a = axs[3, 2]
                for joint in joint_types:
                    acc_key = f"{leg}_{joint}_acc"
                    if log[acc_key]:
                        a.plot(time, log[acc_key], label=f"{joint} acc")
                a.set(xlabel="time [s]", ylabel="Acc [rad/s²]", title=f"{leg} Joint Accelerations")
                a.legend()

        for fig in leg_figs:
            fig.tight_layout()

        # ============ Figure 4: tracking performance ============
        fig4, axs4 = plt.subplots(2, 2, figsize=(10, 7))
        fig4.suptitle("Tracking Performance", fontsize=14)

        # X velocity
        a = axs4[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="real")
        if log["cmd_vel_x"]:
            a.plot(time, log["cmd_vel_x"], '--', label="commanded")
        a.set(xlabel="time [s]", ylabel="vel [m/s]", title="Base Velocity X")
        a.legend()

        # Yaw angular velocity
        a = axs4[0, 1]
        if log["ang_vel_z"]:
            a.plot(time, log["ang_vel_z"], label="real")
        if log["cmd_ang_vel_z"]:
            a.plot(time, log["cmd_ang_vel_z"], '--', label="commanded")
        a.set(xlabel="time [s]", ylabel="ang vel [rad/s]", title="Base Angular Velocity Z")
        a.legend()

        # Base height
        a = axs4[1, 0]
        if log["base_height"]:
            a.plot(time, log["base_height"], label="real")
        if log["base_height_cmd"]:
            a.plot(time, log["base_height_cmd"], '--', label="commanded")
        a.set(xlabel="time [s]", ylabel="height [m]", title="Base Height")
        a.legend()

        # IMU angles
        a = axs4[1, 1]
        if log["base_pitch"]:
            a.plot(time, log["base_pitch"], label="pitch")
        if log["base_roll"]:
            a.plot(time, log["base_roll"], label="roll")
        if log["base_yaw"]:
            a.plot(time, log["base_yaw"], label="yaw")
        a.set(xlabel="time [s]", ylabel="ang [rad]", title="IMU Angles")
        a.legend()

        fig4.tight_layout()

        # ============ Figure 5: policy actions (12 DOF) ============
        fig5, axs5 = plt.subplots(3, 4, figsize=(14, 8))
        fig5.suptitle("Policy Actions", fontsize=14)
        leg_names = ["FL", "RL", "FR", "RR"]
        joint_types = ["hip", "thigh", "calf"]
        plot_end = min(20.0, time[-1])
        plot_start = 0.0
        mask = (time >= plot_start) & (time <= plot_end)
        t_window = time[mask]
        for j_row, joint in enumerate(joint_types):
            for l_col, leg in enumerate(leg_names):
                a = axs5[j_row, l_col]
                action_key = f"{leg}_{joint}_action"
                if log[action_key]:
                    vals = np.array(log[action_key])[mask]
                    a.plot(t_window, vals, label="action")
                a.set(xlabel="time [s]", ylabel="action", title=action_key)
                a.set_xlim(plot_start, plot_end)
                a.legend()

        fig5.tight_layout()

        print("\nTorque limits over plotted window:")
        static_limits = None
        if self.torque_velocity_envelope is not None:
            static_limits = self.torque_velocity_envelope.get("static_limits", None)
            if static_limits is not None:
                static_limits = np.asarray(static_limits).reshape(-1)
            hip_env = self.torque_velocity_envelope.get("hip", {})
            calf_env = self.torque_velocity_envelope.get("calf", {})
            print(
                "  velocity envelope: "
                f"hip(max_vel={hip_env.get('max_vel')}, vel_1={hip_env.get('vel_1')}, "
                f"max_torque={hip_env.get('max_torque')}), "
                f"calf(max_vel={calf_env.get('max_vel')}, vel_1={calf_env.get('vel_1')}, "
                f"max_torque={calf_env.get('max_torque')})"
            )
        for leg_idx, leg in enumerate(leg_names):
            for j_idx, joint in enumerate(joint_types):
                torque_key = f"{leg}_{joint}_torque"
                torque_values = np.asarray(log[torque_key])[mask] if log[torque_key] else np.array([])
                max_abs_torque = float(np.max(np.abs(torque_values))) if torque_values.size else float("nan")
                joint_idx = leg_idx * 3 + j_idx
                if static_limits is not None and joint_idx < static_limits.size:
                    limit = float(static_limits[joint_idx])
                    print(
                        f"  {torque_key}: limit=[{-limit:.3f}, {limit:.3f}] Nm, "
                        f"max|tau|={max_abs_torque:.3f} Nm"
                    )
                else:
                    print(f"  {torque_key}: max|tau|={max_abs_torque:.3f} Nm")

        # ============ Figure 6: trajectory (x-y) + base height (z) ============
        fig6, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        fig6.suptitle("Trajectory & Height", fontsize=14)
        # ---- x-y trajectory ----
        if log["base_pos_x"] and log["base_pos_y"]:
            a1.plot(log["base_pos_x"], log["base_pos_y"], linewidth=1.5, label="trajectory")
            a1.scatter(log["base_pos_x"][0], log["base_pos_y"][0], c='g', s=50, marker='o', label='start')
            a1.scatter(log["base_pos_x"][-1], log["base_pos_y"][-1], c='r', s=50, marker='x', label='end')
        a1.set(xlabel="x [m]", ylabel="y [m]", title="Top-down view")
        a1.axis("equal")
        a1.legend()
        a1.grid(True)
        # ---- z vs time ----
        if log["base_pos_z"]:
            a2.plot(time, log["base_pos_z"], linewidth=1.5, label="z")
        if log["base_height_cmd"]:
            a2.plot(time, log["base_height_cmd"], '--', linewidth=1.0, label="cmd")
        a2.set(xlabel="time [s]", ylabel="z [m]", title="Base Height")
        a2.legend()
        a2.grid(True)
        fig6.tight_layout()


        # # ============ Figure 6: base position ============
        # fig6, axs6 = plt.subplots(3, 1, figsize=(10, 7))
        # fig6.suptitle("Base Position", fontsize=14)
        # for i, axis in enumerate(["base_pos_x", "base_pos_y", "base_pos_z"]):
        #     if log[axis]:
        #         axs6[i].plot(time, log[axis], label=axis)
        #     axs6[i].set(xlabel="time [s]", ylabel="[m]", title=axis)
        #     axs6[i].legend()
        #     axs6[i].grid(True)
        # fig6.tight_layout()

        # ============ Figure 6: trajectory (x-y plane) ============
        # fig6 = plt.figure(figsize=(7, 7))
        # fig6.suptitle("Trajectory (top-down view)", fontsize=14)
        # a = fig6.add_subplot(111)
        # if log["base_pos_x"] and log["base_pos_y"]:
        #     a.plot(log["base_pos_x"], log["base_pos_y"], linewidth=1.5, label="trajectory")
        #     a.scatter(log["base_pos_x"][0], log["base_pos_y"][0], c='g', s=50, marker='o', label='start')
        #     a.scatter(log["base_pos_x"][-1], log["base_pos_y"][-1], c='r', s=50, marker='x', label='end')
        # a.set(xlabel="x [m]", ylabel="y [m]")
        # a.axis("equal")
        # a.legend()
        # a.grid(True)
        # fig6.tight_layout()


        # Save all figures
        import os
        save_dir = os.path.join(os.path.dirname(__file__), '../../logs/plots', self.save_prefix)
        os.makedirs(save_dir, exist_ok=True)
        for i, fig in enumerate(leg_figs):
            fig.savefig(os.path.join(save_dir, f'leg_{leg_names[i]}.png'), dpi=150)
        fig4.savefig(os.path.join(save_dir, 'tracking.png'), dpi=150)
        fig5.savefig(os.path.join(save_dir, 'actions.png'), dpi=150)
        fig6.savefig(os.path.join(save_dir, 'base_pos.png'), dpi=150)
        print(f'Plots saved to {save_dir}')

        # plt.show()  # 本地有显示就取消注释

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
