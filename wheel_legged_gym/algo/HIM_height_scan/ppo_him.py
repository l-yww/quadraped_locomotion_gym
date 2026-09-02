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

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic_him import ActorCritic_HIM
from .rollout_storage_him import RolloutStorage_HIM

class PPO_HIM:
    actor_critic: ActorCritic_HIM
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 num_adaptation_module_substeps =1,
                 symmetry_scale=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage_HIM.Transition()
        self.symmetry_scale = symmetry_scale

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # encoder update
        self.num_adaptation_module_substeps = num_adaptation_module_substeps

    def init_storage(self, num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage_HIM(num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs_history, critic_obs):
        # Compute the actions and values
        # Rollout data is fixed during PPO updates, so do not retain a graph
        # here.  Gradients for height_encoder are created later in update().
        with torch.no_grad():
            self.transition.actions = self.actor_critic.act(obs_history)
            self.transition.values = self.actor_critic.evaluate(critic_obs)
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions)
            self.transition.action_mean = self.actor_critic.action_mean
            self.transition.action_sigma = self.actor_critic.action_std
        # need to record obs and critic_obs before env.step()
        self.transition.observations_history = obs_history
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # ==================== Flip functions (from ppo_sym.py) ====================

    @staticmethod
    def _mirror_wtw_gait_phase(source, flipped, gait_index_idx):
        """Keep WTW gait commands and clock state physically consistent.

        The environment defines [FL, FR, RL, RR] phases as
        [g+p+o+b, g+b, g+o, g+p].  A left-right reflection swaps
        FL<->FR and RL<->RR, so it must also change the phase reference.
        """
        phase = source[..., 5]
        offset = source[..., 6]
        flipped[..., 5] = torch.remainder(-phase, 1.0)
        flipped[..., 6] = torch.remainder(-offset, 1.0)
        flipped[..., gait_index_idx] = torch.remainder(
            source[..., gait_index_idx] + phase + offset, 1.0
        )

    def _append_flipped_actor_heightmap(self, flipped_proprio, obs):
        """Append the left-right mirrored heightmap history to actor proprioception."""
        if not self.actor_critic.observe_heightmap:
            return flipped_proprio

        rows, cols = self.actor_critic.height_map_shape
        history_len = self.actor_critic.height_history_len
        heightmap = obs[:, self.actor_critic.num_obs_history:]
        expected_dim = history_len * rows * cols
        if heightmap.shape[1] != expected_dim:
            raise ValueError(
                "Actor heightmap input has an unexpected size: "
                f"expected {expected_dim}, got {heightmap.shape[1]}"
            )
        heightmap = heightmap.reshape(-1, history_len, rows, cols)
        flipped_heightmap = torch.flip(heightmap, dims=(-1,)).reshape(-1, expected_dim)
        return torch.cat((flipped_proprio, flipped_heightmap), dim=-1)

    def _flip_heightmap_critic_obs(self, critic_obs):
        """Mirror a structured WTW critic observation."""
        layout = self.actor_critic.critic_obs_layout
        if layout is None:
            return None

        frame_stack = layout["frame_stack"]
        frame_size = layout["frame_size"]
        if critic_obs.shape[1] != frame_stack * frame_size:
            raise ValueError(
                "Critic observation has an unexpected size: "
                f"expected {frame_stack * frame_size}, got {critic_obs.shape[1]}"
            )

        source = critic_obs.reshape(-1, frame_stack, frame_size)
        flipped = source.clone()

        # Shared WTW proprioception: commands, joint position/velocity/action,
        # angular velocity, and projected gravity occupy the first 54 values.
        flipped[:, :, 1] = -source[:, :, 1]
        flipped[:, :, 2] = -source[:, :, 2]
        flipped[:, :, 11] = -source[:, :, 11]
        if "gait_index" in layout:
            gait_index_start, gait_index_size = layout["gait_index"]
            if gait_index_size != 1:
                raise ValueError("WTW symmetry expects a scalar gait index")
            self._mirror_wtw_gait_phase(source, flipped, gait_index_start)
        swap_idx = torch.tensor(
            [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
            device=critic_obs.device,
            dtype=torch.long,
        )
        hip_flip = torch.tensor(
            [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
             -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
            device=critic_obs.device,
            dtype=critic_obs.dtype,
        )
        for offset in (12, 24, 36):
            flipped[:, :, offset:offset + 12] = source[:, :, offset:offset + 12][:, :, swap_idx] * hip_flip
        flipped[:, :, 48] = -source[:, :, 48]
        flipped[:, :, 50] = -source[:, :, 50]
        flipped[:, :, 52] = -source[:, :, 52]

        if "motor_strength" in layout:
            start, size = layout["motor_strength"]
            if size != 12:
                raise ValueError("WTW symmetry expects 12 motor-strength values")
            flipped[:, :, start:start + size] = source[:, :, start:start + size][:, :, swap_idx]
        if "motor_offset" in layout:
            start, size = layout["motor_offset"]
            if size != 12:
                raise ValueError("WTW symmetry expects 12 motor-offset values")
            flipped[:, :, start:start + size] = source[:, :, start:start + size][:, :, swap_idx] * hip_flip
        if "com_displacement" in layout:
            start, size = layout["com_displacement"]
            if size != 3:
                raise ValueError("WTW symmetry expects a 3D COM displacement")
            flipped[:, :, start + 1] = -source[:, :, start + 1]

        if "heightmap" in layout:
            height_start, height_size = layout["heightmap"]
            rows, cols = self.actor_critic.height_map_shape
            if height_size != rows * cols:
                raise ValueError(
                    "Critic heightmap size does not match height_map_shape: "
                    f"{height_size} != {rows * cols}"
                )
            heightmap = source[:, :, height_start:height_start + height_size]
            flipped[:, :, height_start:height_start + height_size] = torch.flip(
                heightmap.reshape(-1, frame_stack, rows, cols), dims=(-1,)
            ).reshape(-1, frame_stack, height_size)

        if "clock_inputs" in layout:
            start, size = layout["clock_inputs"]
            if size != 4:
                raise ValueError("WTW symmetry expects four clock inputs")
            flipped[:, :, start:start + size] = source[:, :, [start + 1, start, start + 3, start + 2]]

        base_lin_vel_start, base_lin_vel_size = layout["base_lin_vel"]
        if base_lin_vel_size != 3:
            raise ValueError("WTW symmetry expects a 3D base linear velocity")
        flipped[:, :, base_lin_vel_start + 1] = -source[:, :, base_lin_vel_start + 1]
        return flipped.reshape(-1, frame_stack * frame_size)

    def flip_actor_obs(self, obs):
        """Flip actor observation for left-right symmetry.
        Observation layout (one step, 59 dims):
        [0:12]   commands
        [12:24]  dof_pos (URDF order: FL_hip,FL_th,FL_cf,FR_hip,FR_th,FR_cf,RL_hip,RL_th,RL_cf,RR_hip,RR_th,RR_cf)
        [24:36]  dof_vel
        [36:48]  actions
        [48:51]  ang_vel
        [51:54]  projected_gravity
        [54]     gait_index
        [55:59]  clock_inputs [FL,FR,RL,RR]
        """
        num_obs = self.actor_critic.num_proprio_obs
        embedded_height = self.actor_critic.actor_height_in_obs
        base_num_obs = num_obs
        if embedded_height:
            rows, cols = self.actor_critic.height_map_shape
            height_size = rows * cols
            base_num_obs -= height_size
            if base_num_obs != 59:
                raise ValueError(
                    "WTW actor frames with an embedded heightmap must contain "
                    f"59 proprioception values, got {base_num_obs}"
                )
        proprio_dim = self.actor_critic.num_obs_history
        if proprio_dim % num_obs != 0:
            raise ValueError("num_obs_history must be divisible by num_proprio_obs")
        history = proprio_dim // num_obs
        if obs.shape[1] < proprio_dim:
            raise ValueError("Actor observation is shorter than its proprioception history")
        obs_batch = torch.clone(obs[:, :proprio_dim])
        obs_batch = obs_batch.view(-1, history, num_obs)
        flipped = torch.zeros_like(obs_batch)

        if base_num_obs == 45:
            # quadruped 布局: [0:3]cmd + [3:15]dof_pos + [15:27]dof_vel + [27:39]action + [39:42]ang_vel + [42:45]gravity
            # 关节顺序 (URDF/isaacgym dof_names, 按腿分组):
            #   [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
            #    RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
            # 左右镜像: FL<->FR, RL<->RR; hip 关节符号取反 (左右 hip 旋转方向相反),
            #           thigh/calf 不取反 (前后摆腿左右一致)
            flipped[:, :, 0] =  obs_batch[:, :, 0]    # vx
            flipped[:, :, 1] = -obs_batch[:, :, 1]    # vy
            flipped[:, :, 2] = -obs_batch[:, :, 2]    # yaw
            for off in (3, 15, 27):  # dof_pos / dof_vel / action 起始偏移
                # FL(0,1,2) <-> FR(3,4,5)
                flipped[:, :, off + 0] = -obs_batch[:, :, off + 3]   # FL_hip <- FR_hip (取反)
                flipped[:, :, off + 1] =  obs_batch[:, :, off + 4]   # FL_thigh <- FR_thigh
                flipped[:, :, off + 2] =  obs_batch[:, :, off + 5]   # FL_calf <- FR_calf
                flipped[:, :, off + 3] = -obs_batch[:, :, off + 0]   # FR_hip <- FL_hip (取反)
                flipped[:, :, off + 4] =  obs_batch[:, :, off + 1]   # FR_thigh <- FL_thigh
                flipped[:, :, off + 5] =  obs_batch[:, :, off + 2]   # FR_calf <- FL_calf
                # RL(6,7,8) <-> RR(9,10,11)
                flipped[:, :, off + 6] = -obs_batch[:, :, off + 9]   # RL_hip <- RR_hip (取反)
                flipped[:, :, off + 7] =  obs_batch[:, :, off + 10]  # RL_thigh <- RR_thigh
                flipped[:, :, off + 8] =  obs_batch[:, :, off + 11]  # RL_calf <- RR_calf
                flipped[:, :, off + 9] = -obs_batch[:, :, off + 6]   # RR_hip <- RL_hip (取反)
                flipped[:, :, off + 10] = obs_batch[:, :, off + 7]   # RR_thigh <- RL_thigh
                flipped[:, :, off + 11] = obs_batch[:, :, off + 8]   # RR_calf <- RL_calf
            flipped[:, :, 39] = -obs_batch[:, :, 39]   # ang_vel_x
            flipped[:, :, 40] =  obs_batch[:, :, 40]   # ang_vel_y
            flipped[:, :, 41] = -obs_batch[:, :, 41]   # ang_vel_z
            flipped[:, :, 42] = -obs_batch[:, :, 42]   # gravity_x
            flipped[:, :, 43] =  obs_batch[:, :, 43]   # gravity_y
            flipped[:, :, 44] =  obs_batch[:, :, 44]   # gravity_z
            return self._append_flipped_actor_heightmap(
                flipped.view(-1, num_obs * history), obs
            )

        # Commands (0-11) in the WTW 59-D proprioception prefix.
        flipped[:, :, 0] =  obs_batch[:, :, 0]
        flipped[:, :, 1] = -obs_batch[:, :, 1]
        flipped[:, :, 2] = -obs_batch[:, :, 2]
        flipped[:, :, 3] =  obs_batch[:, :, 3]
        flipped[:, :, 4] =  obs_batch[:, :, 4]
        flipped[:, :, 7] =  obs_batch[:, :, 7]
        flipped[:, :, 8] =  obs_batch[:, :, 8]
        flipped[:, :, 9] =  obs_batch[:, :, 9]
        flipped[:, :, 10] =  obs_batch[:, :, 10]
        flipped[:, :, 11] = -obs_batch[:, :, 11]
        self._mirror_wtw_gait_phase(obs_batch, flipped, gait_index_idx=54)

        swap_pos = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2, 6:9, 7:10, 8:11, 9:6, 10:7, 11:8}
        hip_indices = {0, 3, 6, 9}
        for src, dst in swap_pos.items():
            sign = -1.0 if src in hip_indices else 1.0
            flipped[:, :, 12 + src] = sign * obs_batch[:, :, 12 + dst]
            flipped[:, :, 24 + src] = sign * obs_batch[:, :, 24 + dst]
            flipped[:, :, 36 + src] = sign * obs_batch[:, :, 36 + dst]

        flipped[:, :, 48] = -obs_batch[:, :, 48]
        flipped[:, :, 49] =  obs_batch[:, :, 49]
        flipped[:, :, 50] = -obs_batch[:, :, 50]
        flipped[:, :, 51] =  obs_batch[:, :, 51]
        flipped[:, :, 52] = -obs_batch[:, :, 52]
        flipped[:, :, 53] =  obs_batch[:, :, 53]
        flipped[:, :, 55] =  obs_batch[:, :, 56]
        flipped[:, :, 56] =  obs_batch[:, :, 55]
        flipped[:, :, 57] =  obs_batch[:, :, 58]
        flipped[:, :, 58] =  obs_batch[:, :, 57]

        if embedded_height:
            heightmap = obs_batch[:, :, base_num_obs:]
            flipped[:, :, base_num_obs:] = torch.flip(
                heightmap.reshape(-1, history, rows, cols), dims=(-1,)
            ).reshape(-1, history, height_size)
            return flipped.reshape(-1, num_obs * history)

        return self._append_flipped_actor_heightmap(
            flipped.view(-1, num_obs * history), obs
        )

    def flip_critic_obs(self, critic_obs):
        flipped_heightmap_critic = self._flip_heightmap_critic_obs(critic_obs)
        if flipped_heightmap_critic is not None:
            return flipped_heightmap_critic

        history = 1                                          # c_frame_stack
        num_actor_obs = self.actor_critic.num_proprio_obs    # 59 (num_single_obs)
        obs_batch = torch.clone(critic_obs[:, :num_actor_obs * history])
        obs_batch = obs_batch.view(-1, history, num_actor_obs)
        flipped_actor = torch.zeros_like(obs_batch)

        if num_actor_obs == 45:
            # quadruped actor 部分(前 45)镜像，priv(remaining)不动
            # 关节顺序 URDF (按腿分组): FL三 / FR三 / RL三 / RR三; hip 取反, thigh/calf 不取反
            flipped_actor[:, :, 0] =  obs_batch[:, :, 0]
            flipped_actor[:, :, 1] = -obs_batch[:, :, 1]
            flipped_actor[:, :, 2] = -obs_batch[:, :, 2]
            for off in (3, 15, 27):
                # FL(0,1,2) <-> FR(3,4,5)
                flipped_actor[:, :, off + 0] = -obs_batch[:, :, off + 3]
                flipped_actor[:, :, off + 1] =  obs_batch[:, :, off + 4]
                flipped_actor[:, :, off + 2] =  obs_batch[:, :, off + 5]
                flipped_actor[:, :, off + 3] = -obs_batch[:, :, off + 0]
                flipped_actor[:, :, off + 4] =  obs_batch[:, :, off + 1]
                flipped_actor[:, :, off + 5] =  obs_batch[:, :, off + 2]
                # RL(6,7,8) <-> RR(9,10,11)
                flipped_actor[:, :, off + 6] = -obs_batch[:, :, off + 9]
                flipped_actor[:, :, off + 7] =  obs_batch[:, :, off + 10]
                flipped_actor[:, :, off + 8] =  obs_batch[:, :, off + 11]
                flipped_actor[:, :, off + 9] = -obs_batch[:, :, off + 6]
                flipped_actor[:, :, off + 10] = obs_batch[:, :, off + 7]
                flipped_actor[:, :, off + 11] = obs_batch[:, :, off + 8]
            flipped_actor[:, :, 39] = -obs_batch[:, :, 39]
            flipped_actor[:, :, 40] =  obs_batch[:, :, 40]
            flipped_actor[:, :, 41] = -obs_batch[:, :, 41]
            flipped_actor[:, :, 42] = -obs_batch[:, :, 42]
            flipped_actor[:, :, 43] =  obs_batch[:, :, 43]
            flipped_actor[:, :, 44] =  obs_batch[:, :, 44]
            flipped_proprio = flipped_actor.view(-1, num_actor_obs * history)
            remaining = critic_obs[:, num_actor_obs * history:]
            return torch.cat([flipped_proprio, remaining], dim=-1)

        flipped_actor[:, :, 0] =  obs_batch[:, :, 0]
        flipped_actor[:, :, 1] = -obs_batch[:, :, 1]
        flipped_actor[:, :, 2] = -obs_batch[:, :, 2]
        flipped_actor[:, :, 3] =  obs_batch[:, :, 3]
        flipped_actor[:, :, 4] =  obs_batch[:, :, 4]
        flipped_actor[:, :, 7] =  obs_batch[:, :, 7]
        flipped_actor[:, :, 8] =  obs_batch[:, :, 8]
        flipped_actor[:, :, 9] =  obs_batch[:, :, 9]
        flipped_actor[:, :, 10] =  obs_batch[:, :, 10]
        flipped_actor[:, :, 11] = -obs_batch[:, :, 11]
        self._mirror_wtw_gait_phase(obs_batch, flipped_actor, gait_index_idx=54)

        swap_pos = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2, 6:9, 7:10, 8:11, 9:6, 10:7, 11:8}
        hip_indices = {0, 3, 6, 9}
        for src, dst in swap_pos.items():
            sign = -1.0 if src in hip_indices else 1.0
            flipped_actor[:, :, 12 + src] = sign * obs_batch[:, :, 12 + dst]
            flipped_actor[:, :, 24 + src] = sign * obs_batch[:, :, 24 + dst]
            flipped_actor[:, :, 36 + src] = sign * obs_batch[:, :, 36 + dst]

        flipped_actor[:, :, 48] = -obs_batch[:, :, 48]
        flipped_actor[:, :, 49] =  obs_batch[:, :, 49]
        flipped_actor[:, :, 50] = -obs_batch[:, :, 50]
        flipped_actor[:, :, 51] =  obs_batch[:, :, 51]
        flipped_actor[:, :, 52] = -obs_batch[:, :, 52]
        flipped_actor[:, :, 53] =  obs_batch[:, :, 53]
        flipped_actor[:, :, 55] =  obs_batch[:, :, 56]
        flipped_actor[:, :, 56] =  obs_batch[:, :, 55]
        flipped_actor[:, :, 57] =  obs_batch[:, :, 58]
        flipped_actor[:, :, 58] =  obs_batch[:, :, 57]

        flipped_proprio = flipped_actor.view(-1, num_actor_obs * history)
        remaining = critic_obs[:, num_actor_obs * history:]
        return torch.cat([flipped_proprio, remaining], dim=-1)

    def flip_actions(self, actions):
        if self.actor_critic.num_proprio_obs == 45:
            # quadruped 关节顺序 URDF (按腿分组): [FL_hip,FL_thigh,FL_calf, FR_hip,FR_thigh,FR_calf,
            #                                         RL_hip,RL_thigh,RL_calf, RR_hip,RR_thigh,RR_calf]
            # 镜像: FL<->FR, RL<->RR; hip(0,3,6,9) 取反, thigh/calf 不取反
            # swap_idx[i] = 镜像后位置 i 应取原 action 的哪个 index
            swap_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], device=actions.device, dtype=torch.long)
            hip_flip = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], device=actions.device)
        else:
            # wtw 腿分组: FL,FR,RL,RR 各 hip/th/cf
            hip_flip = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], device=actions.device)
            swap_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], device=actions.device, dtype=torch.long)
        flipped = actions[:, swap_idx] * hip_flip
        return flipped.detach()

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_history_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                
                self.actor_critic.act(obs_history_batch)  # 更新action的分布distribution，以便于后续的动作概率actions_log_prob计算
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                for _ in range(self.num_adaptation_module_substeps):
                    # Estimator Update
                    proprio_history_batch = obs_history_batch[:, :self.actor_critic.num_obs_history]
                    estimation_loss, swap_loss = self.actor_critic.estimator.update(
                        proprio_history_batch, next_critic_obs_batch, lr=self.learning_rate
                    )
                    with torch.no_grad():
                        mean_estimation_loss += estimation_loss
                        mean_swap_loss += swap_loss

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Symmetry loss (仅在 symmetry_scale > 0 时计算；flip_actor_obs/flip_critic_obs
                # 硬编码了 59 维 obs 布局，非 59 维(如 quadruped 的 45 维)会越界，故 symmetry_scale=0 时跳过)
                if self.symmetry_scale > 0:
                    flipped_obs_batch = self.flip_actor_obs(obs_history_batch)
                    flipped_critic_obs_batch = self.flip_critic_obs(critic_obs_batch)
                    actor_sym_loss = self.symmetry_scale * torch.mean(
                        torch.sum(torch.square(
                            self.actor_critic.act_inference(flipped_obs_batch) - self.flip_actions(self.actor_critic.act_inference(obs_history_batch))
                        ), dim=-1)
                    )
                    critic_sym_loss = self.symmetry_scale * torch.mean(
                        torch.square(
                            self.actor_critic.evaluate(flipped_critic_obs_batch) - self.actor_critic.evaluate(critic_obs_batch).detach()
                        )
                    )
                    loss = loss + actor_sym_loss + critic_sym_loss
                else:
                    actor_sym_loss = torch.tensor(0., device=self.device)
                    critic_sym_loss = torch.tensor(0., device=self.device)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_actor_sym_loss += actor_sym_loss.item()
                mean_critic_sym_loss += critic_sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates * self.num_adaptation_module_substeps
        mean_swap_loss /= num_updates * self.num_adaptation_module_substeps
        mean_actor_sym_loss /= num_updates
        mean_critic_sym_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss, mean_actor_sym_loss, mean_critic_sym_loss
