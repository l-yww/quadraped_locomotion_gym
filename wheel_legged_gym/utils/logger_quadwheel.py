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

import os
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Logger_quadwheel:
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

    def plot_states(self, save_dir=None):
        self.plot_process = Process(target=self._plot, args=(save_dir,))
        self.plot_process.start()

    def _plot(self, save_dir=None):
        leg_names = ["FL", "FR", "HL", "HR"]
        joint_types = ["hipx", "hipy", "knee", "wheel"]
        nb_rows = 4
        nb_cols = 4

        log = self.state_log
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        # === Figures 0-3: per-leg joint analysis (4 figures, one per leg) ===
        leg_figs = []
        leg_axs = []
        for i, leg in enumerate(leg_names):
            fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(14, 10))
            fig.suptitle(f"{leg} Leg", fontsize=14)
            leg_figs.append(fig)
            leg_axs.append(axs)

        for leg_idx, leg in enumerate(leg_names):
            axs = leg_axs[leg_idx]
            for j_idx, joint in enumerate(joint_types):
                torque_key = f"{leg}_{joint}_torque"
                vel_key = f"{leg}_{joint}_vel"
                pos_key = f"{leg}_{joint}_pos"
                action_key = f"{leg}_{joint}_action"
                ref_pos_key = f"{leg}_{joint}_ref_pos"
                ref_vel_key = f"{leg}_{joint}_ref_vel"

                # Col 0: torque vs time
                a = axs[j_idx, 0]
                if log[torque_key]:
                    a.plot(time, log[torque_key], label="torque")
                a.set(xlabel="time [s]", ylabel="Torque [Nm]", title=f"{leg} {joint} Torque")
                a.legend()

                # Col 1: velocity vs time
                a = axs[j_idx, 1]
                if log[vel_key]:
                    a.plot(time, log[vel_key], label="vel")
                if log[ref_vel_key]:
                    a.plot(time, log[ref_vel_key], "--", label="ref_vel")
                a.set(xlabel="time [s]", ylabel="Vel [rad/s]", title=f"{leg} {joint} Velocity")
                a.legend()

                # Col 2: torque-velocity scatter
                a = axs[j_idx, 2]
                if log[vel_key] and log[torque_key]:
                    a.scatter(log[vel_key], log[torque_key], s=5, alpha=0.8, label="t-v")
                a.set(xlabel="Joint vel [rad/s]", ylabel="Torque [Nm]", title=f"{leg} {joint} T/V")
                a.legend()

                # Col 3: position vs time
                a = axs[j_idx, 3]
                if log[pos_key]:
                    a.plot(time, log[pos_key], label="pos")
                if log[action_key]:
                    a.plot(time, log[action_key], "--", label="action")
                if log[ref_pos_key]:
                    a.plot(time, log[ref_pos_key], ":", label="ref_pos")
                a.set(xlabel="time [s]", ylabel="Pos [rad]", title=f"{leg} {joint} Position")
                a.legend()

        for fig in leg_figs:
            fig.tight_layout()

        # === Figure 4: base tracking performance ===
        fig_base, axs_base = plt.subplots(2, 3, figsize=(14, 8))
        fig_base.suptitle("Base Tracking", fontsize=14)

        a = axs_base[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="real_x")
        if log["cmd_vel_x"]:
            a.plot(time, log["cmd_vel_x"], "--", label="cmd_x")
        a.set(xlabel="time [s]", ylabel="vel [m/s]", title="Base Velocity X")
        a.legend()

        a = axs_base[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label="real_y")
        if log["cmd_vel_y"]:
            a.plot(time, log["cmd_vel_y"], "--", label="cmd_y")
        a.set(xlabel="time [s]", ylabel="vel [m/s]", title="Base Velocity Y")
        a.legend()

        a = axs_base[0, 2]
        if log["base_ang_vel_z"] or log["ang_vel_z"]:
            key = "base_ang_vel_z" if log["base_ang_vel_z"] else "ang_vel_z"
            a.plot(time, log[key], label="real_z")
        if log["cmd_ang_vel"]:
            a.plot(time, log["cmd_ang_vel"], "--", label="cmd")
        a.set(xlabel="time [s]", ylabel="ang_vel [rad/s]", title="Base Angular Velocity Z")
        a.legend()

        a = axs_base[1, 0]
        if log["base_height"]:
            a.plot(time, log["base_height"], label="real")
        if log["base_height_cmd"]:
            a.plot(time, log["base_height_cmd"], "--", label="cmd")
        a.set(xlabel="time [s]", ylabel="height [m]", title="Base Height")
        a.legend()

        a = axs_base[1, 1]
        if log["base_roll"]:
            a.plot(time, log["base_roll"], label="roll")
        if log["base_pitch"]:
            a.plot(time, log["base_pitch"], label="pitch")
        if log["base_yaw"]:
            a.plot(time, log["base_yaw"], label="yaw")
        a.set(xlabel="time [s]", ylabel="angle [rad]", title="Base Euler Angles")
        a.legend()

        a = axs_base[1, 2]
        if log["ang_vel_x"]:
            a.plot(time, log["ang_vel_x"], label="ang_vel_x")
        if log["ang_vel_y"]:
            a.plot(time, log["ang_vel_y"], label="ang_vel_y")
        a.set(xlabel="time [s]", ylabel="ang_vel [rad/s]", title="Base Angular Velocity XY")
        a.legend()

        fig_base.tight_layout()

        # === Figure 5: contact forces & extra ===
        fig_extra, axs_extra = plt.subplots(2, 2, figsize=(10, 7))
        fig_extra.suptitle("Contact & Extra", fontsize=14)

        a = axs_extra[0, 0]
        if log["contact_forces_l_x"]:
            a.plot(time, log["contact_forces_l_x"], label="x")
        if log["contact_forces_l_y"]:
            a.plot(time, log["contact_forces_l_y"], label="y")
        if log["contact_forces_l_z"]:
            a.plot(time, log["contact_forces_l_z"], label="z")
        a.set(xlabel="time [s]", ylabel="force [N]", title="Contact Forces L")
        a.legend()

        a = axs_extra[0, 1]
        if log["contact_forces_r_x"]:
            a.plot(time, log["contact_forces_r_x"], label="x")
        if log["contact_forces_r_y"]:
            a.plot(time, log["contact_forces_r_y"], label="y")
        if log["contact_forces_r_z"]:
            a.plot(time, log["contact_forces_r_z"], label="z")
        a.set(xlabel="time [s]", ylabel="force [N]", title="Contact Forces R")
        a.legend()

        a = axs_extra[1, 0]
        if log["base_pos_x"]:
            a.plot(time, log["base_pos_x"], label="x")
        if log["base_pos_y"]:
            a.plot(time, log["base_pos_y"], label="y")
        if log["base_pos_z"]:
            a.plot(time, log["base_pos_z"], label="z")
        a.set(xlabel="time [s]", ylabel="pos [m]", title="Base Position")
        a.legend()

        a = axs_extra[1, 1]
        if log["feet_distance_lateral"]:
            a.plot(time, log["feet_distance_lateral"], label="real")
        a.axhline(y=0.2, color="r", linestyle="--", alpha=0.5, label="min(0.2)")
        a.axhline(y=0.40, color="g", linestyle="--", alpha=0.5, label="max(0.40)")
        a.set(xlabel="time [s]", ylabel="distance [m]", title="Feet Distance Lateral")
        a.legend()

        fig_extra.tight_layout()

        # === Figures 6-9: per-leg joint power (4 figures, one per leg, 2x2 subplots) ===
        power_figs = []
        power_axs = []
        for i, leg in enumerate(leg_names):
            fig, axs = plt.subplots(2, 2, figsize=(10, 7))
            fig.suptitle(f"{leg} Leg Joint Power", fontsize=14)
            power_figs.append(fig)
            power_axs.append(axs)

        for leg_idx, leg in enumerate(leg_names):
            for j_idx, joint in enumerate(joint_types):
                row = j_idx // 2
                col = j_idx % 2
                a = power_axs[leg_idx][row, col]
                t_key = f"{leg}_{joint}_torque"
                v_key = f"{leg}_{joint}_vel"
                if log[t_key] and log[v_key]:
                    p = np.array(log[t_key]) * np.array(log[v_key])
                    a.plot(time, p, label=joint)
                    a.plot(time, np.abs(p), label=f"|{joint}|")
                a.set(xlabel="time [s]", ylabel="power [W]", title=f"{leg} {joint} Power")
                a.legend()

        for fig in power_figs:
            fig.tight_layout()

        # === Figure 10: per-leg total power (4 subplots) ===
        fig_total, axs_total = plt.subplots(2, 2, figsize=(10, 7))
        fig_total.suptitle("Per-Leg Total Power", fontsize=14)

        for leg_idx, leg in enumerate(leg_names):
            row = leg_idx // 2
            col = leg_idx % 2
            a = axs_total[row, col]
            power_key = f"{leg}_power"
            if log[power_key]:
                a.plot(time, log[power_key], label="power")
                a.plot(time, np.abs(log[power_key]), label="|power|")
            a.set(xlabel="time [s]", ylabel="power [W]", title=f"{leg} Total Power")
            a.legend()

        fig_total.tight_layout()

        # === Save figures ===
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "../../logs/plots/quadwheel")

        all_figs = leg_figs + [fig_base, fig_extra] + power_figs + [fig_total]
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for i, leg in enumerate(leg_names):
                leg_figs[i].savefig(os.path.join(save_dir, f"{leg.lower()}_leg.png"), dpi=150)
            fig_base.savefig(os.path.join(save_dir, "base_tracking.png"), dpi=150)
            fig_extra.savefig(os.path.join(save_dir, "contact_extra.png"), dpi=150)
            for i, leg in enumerate(leg_names):
                power_figs[i].savefig(os.path.join(save_dir, f"{leg.lower()}_power.png"), dpi=150)
            fig_total.savefig(os.path.join(save_dir, "total_power.png"), dpi=150)
            print(f"[INFO] Saved state plots to {save_dir}")
        for fig in all_figs:
            plt.close(fig)

    def save_csv(self, save_dir):
        import csv
        os.makedirs(save_dir, exist_ok=True)

        leg_names = ["FL", "FR", "HL", "HR"]
        joint_types = ["hipx", "hipy", "knee", "wheel"]

        header = ["time"]
        for leg in leg_names:
            for joint in joint_types:
                header.append(f"{leg}_{joint}_torque")
                header.append(f"{leg}_{joint}_velocity")

        n = len(self.state_log.get("time", []))
        if n == 0:
            for k, v in self.state_log.items():
                n = len(v)
                break
        if n == 0:
            print("[WARNING] No data to save in CSV.")
            return

        time_col = [i * self.dt for i in range(n)]
        rows = []
        for i in range(n):
            row = [time_col[i]]
            for leg in leg_names:
                for joint in joint_types:
                    torque_key = f"{leg}_{joint}_torque"
                    vel_key = f"{leg}_{joint}_vel"
                    row.append(self.state_log[torque_key][i] if torque_key in self.state_log and i < len(self.state_log[torque_key]) else "")
                    row.append(self.state_log[vel_key][i] if vel_key in self.state_log and i < len(self.state_log[vel_key]) else "")
            rows.append(row)

        csv_path = os.path.join(save_dir, "torque_velocity.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"[INFO] Saved torque-velocity CSV to {csv_path}")

    def save_torque_vel_scatter_plots(self, save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)
        log = self.state_log

        leg_names = ["FL", "FR", "HL", "HR"]
        joint_types = ["hipx", "hipy", "knee", "wheel"]

        for leg in leg_names:
            for joint in joint_types:
                torque_key = f"{leg}_{joint}_torque"
                vel_key = f"{leg}_{joint}_vel"
                if vel_key not in log or torque_key not in log or len(log[vel_key]) == 0:
                    continue
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(log[vel_key], log[torque_key], s=5, alpha=0.8)
                ax.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title=f"{leg} {joint} Torque-Velocity")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{leg.lower()}_{joint}_tv_scatter.png"), dpi=150)
                plt.close(fig)
        print(f"[INFO] Saved torque-velocity scatter plots to {save_dir}")

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
