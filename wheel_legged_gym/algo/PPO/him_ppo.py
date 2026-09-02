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

from .him_actor_critic import HIMActorCritic
from .him_rollout_storage import HIMRolloutStorage

class HIMPPO:
    actor_critic: HIMActorCritic
    def __init__(self,
                 actor_critic,
                 use_flip = True,
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
                 device='cpu',
                 symmetry_scale=1e-3,
                 ):

        self.device = device
        self.use_flip = use_flip

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition()
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

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        if self.use_flip:
            obs_sym = self.flip_actor_obs(obs)
            critic_obs_sym = self.flip_critic_obs(critic_obs)
            self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
            self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
            self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_sym.actions).detach()
            self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
            self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
            # need to record obs and critic_obs before env.step()
            self.transition_sym.observations = obs_sym
            self.transition_sym.critic_observations = critic_obs_sym
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.use_flip:
            next_critic_obs_sym = self.flip_critic_obs(next_critic_obs)
            self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
            self.transition_sym.rewards = rewards.clone()
            self.transition_sym.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            if self.use_flip:
                self.transition_sym.rewards += self.gamma * torch.squeeze(self.transition_sym.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        # Record the transition
        self.storage.add_transitions(self.transition)
        if self.use_flip:
            self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        if self.use_flip:
            self.transition_sym.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                self.actor_critic.act(obs_batch)
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

                #Estimator Update
                if self.use_flip:
                    flipped_obs_batch = self.flip_actor_obs(obs_batch)
                    flipped_next_critic_obs_batch = self.flip_critic_obs(next_critic_obs_batch)
                    estimator_update_obs_batch =  torch.cat((obs_batch, flipped_obs_batch), dim=0)
                    estimator_update_next_critic_obs_batch = torch.cat((next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0)
                else:
                    estimator_update_obs_batch = obs_batch
                    estimator_update_next_critic_obs_batch = next_critic_obs_batch
                estimation_loss, swap_loss = self.actor_critic.update_estimator(estimator_update_obs_batch, estimator_update_next_critic_obs_batch, lr=self.learning_rate)
                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    
                if self.use_flip:
                    flipped_critic_obs_batch = self.flip_critic_obs(critic_obs_batch)
                    actor_sym_loss = self.symmetry_scale * torch.mean(torch.sum(torch.square(self.actor_critic.act_inference(flipped_obs_batch) - self.flip_actions(self.actor_critic.act_inference(obs_batch))), dim=-1))
                    critic_sym_loss = self.symmetry_scale * torch.mean(torch.square(self.actor_critic.evaluate(flipped_critic_obs_batch) - self.actor_critic.evaluate(critic_obs_batch).detach()))
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + actor_sym_loss + critic_sym_loss
                else:
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss
                if self.use_flip:
                    mean_actor_sym_loss += actor_sym_loss.item()
                    mean_critic_sym_loss += critic_sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        if self.use_flip:
            mean_actor_sym_loss /= num_updates
            mean_critic_sym_loss /= num_updates
        self.storage.clear()

        if self.use_flip:
            return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss
        else:
            return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss, 0, 0
    
    def flip_actor_obs(self, obs):
        """Flip actor observation for left-right symmetry.
        """
        num_obs = self.actor_critic.num_one_step_obs
        history = self.actor_critic.actor_history_length

        if num_obs == 80:
            return self._flip_adam_actor_obs(obs, num_obs, history)
        elif num_obs == 78:
            return self._flip_cowa_actor_obs(obs, num_obs, history)
        else:
            return self._flip_g1_actor_obs(obs, num_obs, history)

    def _flip_adam_actor_obs(self, obs, num_obs, history):
        """Flip Adam SP actor observation.

        Observation layout (one step):
        0-3:   commands (lin_vel_x, lin_vel_y, ang_vel_yaw, height)
        4-6:   ang_vel (roll, pitch, yaw)
        7-9:   projected_gravity (x, y, z)
        10-38: joint_pos (29 DOFs)
        39-67: joint_vel (29 DOFs)
        68-79: actions_lower (12 DOFs)

        DOF order:
        0-5:   Left leg  (hipPitch, hipRoll, hipYaw, kneePitch, anklePitch, ankleRoll)
        6-11:  Right leg (hipPitch, hipRoll, hipYaw, kneePitch, anklePitch, ankleRoll)
        12-14: Waist     (waistRoll, waistPitch, waistYaw)
        15-21: Left arm  (shoulderPitch, shoulderRoll, shoulderYaw, elbow, wristYaw, wristPitch, wristRoll)
        22-28: Right arm (shoulderPitch, shoulderRoll, shoulderYaw, elbow, wristYaw, wristPitch, wristRoll)
        """
        proprioceptive_obs = torch.clone(obs[:, :num_obs * history])
        proprioceptive_obs = proprioceptive_obs.view(-1, history, num_obs)

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)

        # Commands (indices 0-3)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0]  # lin_vel_x stays
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]  # lin_vel_y flips
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # ang_vel_yaw flips
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3]  # height stays

        # Ang vel (indices 4-6)
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4]  # roll flips
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5]  # pitch stays
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6]  # yaw flips

        # Projected gravity (indices 7-9)
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7]  # gravity_x stays
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8]  # gravity_y flips
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9]  # gravity_z stays

        DOF = 29
        joint_pos_start = 10
        joint_vel_start = 10 + DOF  # 39
        actions_start = 10 + 2 * DOF  # 68

        # Define the flip pattern for each joint type
        # flip_flags[i] = True means flip sign when mirroring
        flip_flags = [
            False,  # 0:  hipPitch_L
            True,   # 1:  hipRoll_L
            True,   # 2:  hipYaw_L
            False,  # 3:  kneePitch_L
            False,  # 4:  anklePitch_L
            True,   # 5:  ankleRoll_L
            False,  # 6:  hipPitch_R
            True,   # 7:  hipRoll_R
            True,   # 8:  hipYaw_R
            False,  # 9:  kneePitch_R
            False,  # 10: anklePitch_R
            True,   # 11: ankleRoll_R
            True,   # 12: waistRoll (flip)
            False,  # 13: waistPitch (stay)
            True,   # 14: waistYaw (flip)
            False,  # 15: shoulderPitch_L
            True,   # 16: shoulderRoll_L
            True,   # 17: shoulderYaw_L
            False,  # 18: elbow_L
            True,   # 19: wristYaw_L
            False,  # 20: wristPitch_L
            True,  # 21: wristRoll_L
            False,  # 22: shoulderPitch_R
            True,   # 23: shoulderRoll_R
            True,   # 24: shoulderYaw_R
            False,  # 25: elbow_R
            True,   # 26: wristYaw_R
            False,  # 27: wristPitch_R
            True,  # 28: wristRoll_R
        ]

        # Left-right swap mapping
        swap_map = {
            0: 6,  1: 7,  2: 8,  3: 9,  4: 10, 5: 11,   # Left leg <-> Right leg
            6: 0,  7: 1,  8: 2,  9: 3,  10: 4, 11: 5,
            12: 12, 13: 13, 14: 14,  # Waist maps to itself
            15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27, 21: 28,             # Left arm <-> Right arm
            22: 15, 23: 16, 24: 17, 25: 18, 26: 19, 27: 20, 28: 21,
        }

        # Apply flip to joint_pos, joint_vel, and actions
        for src_idx in range(DOF):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            # joint_pos
            flipped_proprioceptive_obs[:, :, joint_pos_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_pos_start + dst_dof]
            # joint_vel
            flipped_proprioceptive_obs[:, :, joint_vel_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_vel_start + dst_dof]

        # actions_lower (12 dims: first 12 DOFs = legs)
        num_lower = 12
        for src_idx in range(num_lower):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            flipped_proprioceptive_obs[:, :, actions_start + src_idx] = \
                sign * proprioceptive_obs[:, :, actions_start + dst_dof]

        return flipped_proprioceptive_obs.view(-1, num_obs * history)

    def _flip_cowa_actor_obs(self, obs, num_obs, history):
        """Flip Cowa human actor observation.

        Observation layout (one step):
        0-3:   commands (lin_vel_x, lin_vel_y, ang_vel_yaw, height)
        4-6:   ang_vel (roll, pitch, yaw)
        7-9:   projected_gravity (x, y, z)
        10-37: joint_pos (28 DOFs)
        38-65: joint_vel (28 DOFs)
        66-77: actions_lower (12 DOFs)

        DOF order:
        0-5:   Left leg  (hipPitch, hipRoll, hipYaw, kneePitch, anklePitch, ankleRoll)
        6-11:  Right leg (hipPitch, hipRoll, hipYaw, kneePitch, anklePitch, ankleRoll)
        12-13: Waist     (waistYaw, waistPitch)
        14-20: Left arm  (shoulderPitch, shoulderRoll, shoulderYaw, elbow, wristYaw, wristPitch, wristRoll)
        21-27: Right arm (shoulderPitch, shoulderRoll, shoulderYaw, elbow, wristYaw, wristPitch, wristRoll)
        """
        proprioceptive_obs = torch.clone(obs[:, :num_obs * history])
        proprioceptive_obs = proprioceptive_obs.view(-1, history, num_obs)

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)

        # Commands (indices 0-3)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0]  # lin_vel_x stays
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]  # lin_vel_y flips
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]  # ang_vel_yaw flips
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3]  # height stays

        # Ang vel (indices 4-6)
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4]  # roll flips
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5]  # pitch stays
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6]  # yaw flips

        # Projected gravity (indices 7-9)
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7]  # gravity_x stays
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8]  # gravity_y flips
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9]  # gravity_z stays

        DOF = 28
        joint_pos_start = 10
        joint_vel_start = 10 + DOF  # 38
        actions_start = 10 + 2 * DOF  # 66

        # Define the flip pattern for each joint type
        # flip_flags[i] = True means flip sign when mirroring
        flip_flags = [
            False,  # 0:  hipPitch_L
            False,   # 1:  hipRoll_L
            False,   # 2:  hipYaw_L
            False,  # 3:  kneePitch_L
            False,  # 4:  anklePitch_L
            False,   # 5:  ankleRoll_L
            False,  # 6:  hipPitch_R
            False,   # 7:  hipRoll_R
            False,   # 8:  hipYaw_R
            False,  # 9:  kneePitch_R
            False,  # 10: anklePitch_R
            False,   # 11: ankleRoll_R
            True,   # 12: waistYaw (flip)
            False,  # 13: waistPitch (stay)
            True,  # 14: shoulderPitch_L
            False,   # 15: shoulderRoll_L
            False,   # 16: shoulderYaw_L
            True,  # 17: elbow_L
            False,   # 18: wristYaw_L
            True,  # 19: wristPitch_L
            False,  # 20: wristRoll_L
            True,  # 21: shoulderPitch_R
            False,   # 22: shoulderRoll_R
            False,   # 23: shoulderYaw_R
            True,  # 24: elbow_R
            False,   # 25: wristYaw_R
            True,  # 26: wristPitch_R
            False,  # 27: wristRoll_R
        ]

        # Left-right swap mapping
        swap_map = {
            0: 6,  1: 7,  2: 8,  3: 9,  4: 10, 5: 11,   # Left leg <-> Right leg
            6: 0,  7: 1,  8: 2,  9: 3,  10: 4, 11: 5,
            12: 12, 13: 13,  # Waist maps to itself
            14: 21, 15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27,             # Left arm <-> Right arm
            21: 14, 22: 15, 23: 16, 24: 17, 25: 18, 26: 19, 27: 20,
        }

        # Apply flip to joint_pos, joint_vel, and actions
        for src_idx in range(DOF):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            # joint_pos
            flipped_proprioceptive_obs[:, :, joint_pos_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_pos_start + dst_dof]
            # joint_vel
            flipped_proprioceptive_obs[:, :, joint_vel_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_vel_start + dst_dof]

        # actions_lower (12 dims: first 12 DOFs = legs)
        num_lower = 12
        for src_idx in range(num_lower):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            flipped_proprioceptive_obs[:, :, actions_start + src_idx] = \
                sign * proprioceptive_obs[:, :, actions_start + dst_dof]

        return flipped_proprioceptive_obs.view(-1, num_obs * history)
    
    def _flip_g1_actor_obs(self, obs, num_obs, history):
        """Original G1 flip function (76 dims)."""
        proprioceptive_obs = torch.clone(obs[:, :num_obs * history])
        proprioceptive_obs = proprioceptive_obs.view(-1, history, num_obs)

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z

        # joint pos
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 16] # lower
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 19]
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 21]
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 10]
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped_proprioceptive_obs[:, :, 19] =  proprioceptive_obs[:, :, 13]
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 15]

        flipped_proprioceptive_obs[:, :, 22] =  -proprioceptive_obs[:, :, 22] # waist

        flipped_proprioceptive_obs[:, :, 23] =  proprioceptive_obs[:, :, 30] # left shoulder
        flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 31]
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 32]
        flipped_proprioceptive_obs[:, :, 26] =  proprioceptive_obs[:, :, 33] # elbow
        flipped_proprioceptive_obs[:, :, 27] = -proprioceptive_obs[:, :, 34] # wrist
        flipped_proprioceptive_obs[:, :, 28] =  proprioceptive_obs[:, :, 35]
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 36]


        flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 23] # right shoulder
        flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 24]
        flipped_proprioceptive_obs[:, :, 32] = -proprioceptive_obs[:, :, 25]
        flipped_proprioceptive_obs[:, :, 33] =  proprioceptive_obs[:, :, 26] # elbow
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 27] # wrist
        flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 28]
        flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 29]

        # joint vel
        flipped_proprioceptive_obs[:, :, 10+27] =  proprioceptive_obs[:, :, 16+27] # lower
        flipped_proprioceptive_obs[:, :, 11+27] = -proprioceptive_obs[:, :, 17+27]
        flipped_proprioceptive_obs[:, :, 12+27] = -proprioceptive_obs[:, :, 18+27]
        flipped_proprioceptive_obs[:, :, 13+27] =  proprioceptive_obs[:, :, 19+27]
        flipped_proprioceptive_obs[:, :, 14+27] =  proprioceptive_obs[:, :, 20+27]
        flipped_proprioceptive_obs[:, :, 15+27] = -proprioceptive_obs[:, :, 21+27]
        flipped_proprioceptive_obs[:, :, 16+27] =  proprioceptive_obs[:, :, 10+27]
        flipped_proprioceptive_obs[:, :, 17+27] = -proprioceptive_obs[:, :, 11+27]
        flipped_proprioceptive_obs[:, :, 18+27] = -proprioceptive_obs[:, :, 12+27]
        flipped_proprioceptive_obs[:, :, 19+27] =  proprioceptive_obs[:, :, 13+27]
        flipped_proprioceptive_obs[:, :, 20+27] =  proprioceptive_obs[:, :, 14+27]
        flipped_proprioceptive_obs[:, :, 21+27] = -proprioceptive_obs[:, :, 15+27]

        flipped_proprioceptive_obs[:, :, 22+27] =  -proprioceptive_obs[:, :, 22+27] # waist

        flipped_proprioceptive_obs[:, :, 23+27] =  proprioceptive_obs[:, :, 30+27] # left shoulder
        flipped_proprioceptive_obs[:, :, 24+27] = -proprioceptive_obs[:, :, 31+27]
        flipped_proprioceptive_obs[:, :, 25+27] = -proprioceptive_obs[:, :, 32+27]
        flipped_proprioceptive_obs[:, :, 26+27] =  proprioceptive_obs[:, :, 33+27] # elbow
        flipped_proprioceptive_obs[:, :, 27+27] = -proprioceptive_obs[:, :, 34+27] # wrist
        flipped_proprioceptive_obs[:, :, 28+27] =  proprioceptive_obs[:, :, 35+27]
        flipped_proprioceptive_obs[:, :, 29+27] = -proprioceptive_obs[:, :, 36+27]


        flipped_proprioceptive_obs[:, :, 30+27] =  proprioceptive_obs[:, :, 23+27] # right shoulder
        flipped_proprioceptive_obs[:, :, 31+27] = -proprioceptive_obs[:, :, 24+27]
        flipped_proprioceptive_obs[:, :, 32+27] = -proprioceptive_obs[:, :, 25+27]
        flipped_proprioceptive_obs[:, :, 33+27] =  proprioceptive_obs[:, :, 26+27] # elbow
        flipped_proprioceptive_obs[:, :, 34+27] = -proprioceptive_obs[:, :, 27+27] # wrist
        flipped_proprioceptive_obs[:, :, 35+27] =  proprioceptive_obs[:, :, 28+27]
        flipped_proprioceptive_obs[:, :, 36+27] = -proprioceptive_obs[:, :, 29+27]

        # joint target
        flipped_proprioceptive_obs[:, :, 10+54] =  proprioceptive_obs[:, :, 16+54] # lower
        flipped_proprioceptive_obs[:, :, 11+54] = -proprioceptive_obs[:, :, 17+54]
        flipped_proprioceptive_obs[:, :, 12+54] = -proprioceptive_obs[:, :, 18+54]
        flipped_proprioceptive_obs[:, :, 13+54] =  proprioceptive_obs[:, :, 19+54]
        flipped_proprioceptive_obs[:, :, 14+54] =  proprioceptive_obs[:, :, 20+54]
        flipped_proprioceptive_obs[:, :, 15+54] = -proprioceptive_obs[:, :, 21+54]
        flipped_proprioceptive_obs[:, :, 16+54] =  proprioceptive_obs[:, :, 10+54]
        flipped_proprioceptive_obs[:, :, 17+54] = -proprioceptive_obs[:, :, 11+54]
        flipped_proprioceptive_obs[:, :, 18+54] = -proprioceptive_obs[:, :, 12+54]
        flipped_proprioceptive_obs[:, :, 19+54] =  proprioceptive_obs[:, :, 13+54]
        flipped_proprioceptive_obs[:, :, 20+54] =  proprioceptive_obs[:, :, 14+54]
        flipped_proprioceptive_obs[:, :, 21+54] = -proprioceptive_obs[:, :, 15+54]

        return flipped_proprioceptive_obs.view(-1, num_obs * history)                                                                                                                                                                                                                                             
    
    def flip_critic_obs(self, critic_obs):
        """Flip critic observation for left-right symmetry.
        """
        num_obs = self.actor_critic.num_one_step_critic_obs
        history = self.actor_critic.critic_history_length

        if num_obs == 83:
            return self._flip_adam_critic_obs(critic_obs, num_obs, history)
        elif num_obs == 81:
            return self._flip_cowa_critic_obs(critic_obs, num_obs, history)
        else:
            return self._flip_g1_critic_obs(critic_obs, num_obs, history)

    def _flip_adam_critic_obs(self, critic_obs, num_obs, history):
        """Flip Adam SP critic observation (74 dims).

        Observation layout (one step):
        0-3:   commands (lin_vel_x, lin_vel_y, ang_vel_yaw, height)
        4-6:   ang_vel (roll, pitch, yaw)
        7-9:   projected_gravity (x, y, z)
        10-38: joint_pos (29 DOFs)
        39-67: joint_vel (29 DOFs)
        68-79: actions_lower (12 DOFs)
        80-82: base_lin_vel (x, y, z)
        """
        proprioceptive_obs = torch.clone(critic_obs[:, :num_obs * history])
        proprioceptive_obs = proprioceptive_obs.view(-1, history, num_obs)

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)

        DOF = 29
        joint_pos_start = 10
        joint_vel_start = 10 + DOF  # 39
        actions_start = 10 + 2 * DOF  # 68
        lin_vel_start = 10 + 2 * DOF + 12  # 80

        # Commands (indices 0-3)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0]
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3]

        # Ang vel (indices 4-6)
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4]
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5]
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6]

        # Projected gravity (indices 7-9)
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7]
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8]
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9]

        # Flip flags and swap mapping (same as actor)
        flip_flags = [
            False,  # 0:  hipPitch_L
            True,   # 1:  hipRoll_L
            True,   # 2:  hipYaw_L
            False,  # 3:  kneePitch_L
            False,  # 4:  anklePitch_L
            True,   # 5:  ankleRoll_L
            False,  # 6:  hipPitch_R
            True,   # 7:  hipRoll_R
            True,   # 8:  hipYaw_R
            False,  # 9:  kneePitch_R
            False,  # 10: anklePitch_R
            True,   # 11: ankleRoll_R
            True,   # 12: waistRoll (flip)
            False,  # 13: waistPitch (stay)
            True,   # 14: waistYaw (flip)
            False,  # 15: shoulderPitch_L
            True,   # 16: shoulderRoll_L
            True,   # 17: shoulderYaw_L
            False,  # 18: elbow_L
            True,   # 19: wristYaw_L
            False,  # 20: wristPitch_L
            True,  # 21: wristRoll_L
            False,  # 22: shoulderPitch_R
            True,   # 23: shoulderRoll_R
            True,   # 24: shoulderYaw_R
            False,  # 25: elbow_R
            True,   # 26: wristYaw_R
            False,  # 27: wristPitch_R
            True,  # 28: wristRoll_R
        ]

        # Left-right swap mapping
        swap_map = {
            0: 6,  1: 7,  2: 8,  3: 9,  4: 10, 5: 11,   # Left leg <-> Right leg
            6: 0,  7: 1,  8: 2,  9: 3,  10: 4, 11: 5,
            12: 12, 13: 13, 14: 14,  # Waist maps to itself
            15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27, 21: 28,             # Left arm <-> Right arm
            22: 15, 23: 16, 24: 17, 25: 18, 26: 19, 27: 20, 28: 21,
        }

        for src_idx in range(DOF):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            flipped_proprioceptive_obs[:, :, joint_pos_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_pos_start + dst_dof]
            flipped_proprioceptive_obs[:, :, joint_vel_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_vel_start + dst_dof]

        num_lower = 12
        for src_idx in range(num_lower):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            flipped_proprioceptive_obs[:, :, actions_start + src_idx] = \
                sign * proprioceptive_obs[:, :, actions_start + dst_dof]

        # Base linear velocity (x flips, y stays, z stays)
        flipped_proprioceptive_obs[:, :, lin_vel_start + 0] =  proprioceptive_obs[:, :, lin_vel_start + 0]
        flipped_proprioceptive_obs[:, :, lin_vel_start + 1] = -proprioceptive_obs[:, :, lin_vel_start + 1]
        flipped_proprioceptive_obs[:, :, lin_vel_start + 2] =  proprioceptive_obs[:, :, lin_vel_start + 2]

        return flipped_proprioceptive_obs.view(-1, num_obs * history)

    def _flip_cowa_critic_obs(self, critic_obs, num_obs, history):
        """Flip Cowa human critic observation.

        Observation layout (one step):
        0-3:   commands (lin_vel_x, lin_vel_y, ang_vel_yaw, height)
        4-6:   ang_vel (roll, pitch, yaw)
        7-9:   projected_gravity (x, y, z)
        10-37: joint_pos (28 DOFs)
        38-65: joint_vel (28 DOFs)
        66-77: actions_lower (12 DOFs)
        78-80: base_lin_vel (x, y, z)
        """
        proprioceptive_obs = torch.clone(critic_obs[:, :num_obs * history])
        proprioceptive_obs = proprioceptive_obs.view(-1, history, num_obs)

        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)

        DOF = 28
        joint_pos_start = 10
        joint_vel_start = 10 + DOF  # 38
        actions_start = 10 + 2 * DOF  # 66
        lin_vel_start = 10 + 2 * DOF + 12  # 78

        # Commands (indices 0-3)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0]
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1]
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2]
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3]

        # Ang vel (indices 4-6)
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4]
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5]
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6]

        # Projected gravity (indices 7-9)
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7]
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8]
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9]

        # Flip flags and swap mapping (same as actor)
        flip_flags = [
            False,  # 0:  hipPitch_L
            False,   # 1:  hipRoll_L
            False,   # 2:  hipYaw_L
            False,  # 3:  kneePitch_L
            False,  # 4:  anklePitch_L
            False,   # 5:  ankleRoll_L
            False,  # 6:  hipPitch_R
            False,   # 7:  hipRoll_R
            False,   # 8:  hipYaw_R
            False,  # 9:  kneePitch_R
            False,  # 10: anklePitch_R
            False,   # 11: ankleRoll_R
            True,   # 12: waistYaw (flip)
            False,  # 13: waistPitch (stay)
            True,  # 14: shoulderPitch_L
            False,   # 15: shoulderRoll_L
            False,   # 16: shoulderYaw_L
            True,  # 17: elbow_L
            False,   # 18: wristYaw_L
            True,  # 19: wristPitch_L
            False,  # 20: wristRoll_L
            True,  # 21: shoulderPitch_R
            False,   # 22: shoulderRoll_R
            False,   # 23: shoulderYaw_R
            True,  # 24: elbow_R
            False,   # 25: wristYaw_R
            True,  # 26: wristPitch_R
            False,  # 27: wristRoll_R
        ]

        # Left-right swap mapping
        swap_map = {
            0: 6,  1: 7,  2: 8,  3: 9,  4: 10, 5: 11,   # Left leg <-> Right leg
            6: 0,  7: 1,  8: 2,  9: 3,  10: 4, 11: 5,
            12: 12, 13: 13,  # Waist maps to itself
            14: 21, 15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27,             # Left arm <-> Right arm
            21: 14, 22: 15, 23: 16, 24: 17, 25: 18, 26: 19, 27: 20,
        }

        for src_idx in range(DOF):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            flipped_proprioceptive_obs[:, :, joint_pos_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_pos_start + dst_dof]
            flipped_proprioceptive_obs[:, :, joint_vel_start + src_idx] = \
                sign * proprioceptive_obs[:, :, joint_vel_start + dst_dof]

        num_lower = 12
        for src_idx in range(num_lower):
            dst_dof = swap_map[src_idx]
            sign = -1.0 if flip_flags[src_idx] else 1.0
            flipped_proprioceptive_obs[:, :, actions_start + src_idx] = \
                sign * proprioceptive_obs[:, :, actions_start + dst_dof]

        # Base linear velocity (x flips, y stays, z stays)
        flipped_proprioceptive_obs[:, :, lin_vel_start + 0] =  proprioceptive_obs[:, :, lin_vel_start + 0]
        flipped_proprioceptive_obs[:, :, lin_vel_start + 1] = -proprioceptive_obs[:, :, lin_vel_start + 1]
        flipped_proprioceptive_obs[:, :, lin_vel_start + 2] =  proprioceptive_obs[:, :, lin_vel_start + 2]

        return flipped_proprioceptive_obs.view(-1, num_obs * history)
    
    def _flip_g1_critic_obs(self, critic_obs, num_obs, history):
        """Original G1 flip function (79 dims)."""
        proprioceptive_obs = torch.clone(critic_obs[:, :num_obs * history])
        proprioceptive_obs = proprioceptive_obs.view(-1, history, num_obs)
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)

        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z

        # joint pos
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 16]
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 19]
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 21]
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 10]
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped_proprioceptive_obs[:, :, 19] =  proprioceptive_obs[:, :, 13]
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 15]

        flipped_proprioceptive_obs[:, :, 22] =  -proprioceptive_obs[:, :, 22] # waist

        flipped_proprioceptive_obs[:, :, 23] =  proprioceptive_obs[:, :, 30]
        flipped_proprioceptive_obs[:, :, 24] = -proprioceptive_obs[:, :, 31]
        flipped_proprioceptive_obs[:, :, 25] = -proprioceptive_obs[:, :, 32]
        flipped_proprioceptive_obs[:, :, 26] =  proprioceptive_obs[:, :, 33]
        flipped_proprioceptive_obs[:, :, 27] = -proprioceptive_obs[:, :, 34]
        flipped_proprioceptive_obs[:, :, 28] =  proprioceptive_obs[:, :, 35]
        flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 36]


        flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 23]
        flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 24]
        flipped_proprioceptive_obs[:, :, 32] = -proprioceptive_obs[:, :, 25]
        flipped_proprioceptive_obs[:, :, 33] =  proprioceptive_obs[:, :, 26]
        flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 27]
        flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 28]
        flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 29]

        # joint vel
        flipped_proprioceptive_obs[:, :, 10+27] =  proprioceptive_obs[:, :, 16+27]
        flipped_proprioceptive_obs[:, :, 11+27] = -proprioceptive_obs[:, :, 17+27]
        flipped_proprioceptive_obs[:, :, 12+27] = -proprioceptive_obs[:, :, 18+27]
        flipped_proprioceptive_obs[:, :, 13+27] =  proprioceptive_obs[:, :, 19+27]
        flipped_proprioceptive_obs[:, :, 14+27] =  proprioceptive_obs[:, :, 20+27]
        flipped_proprioceptive_obs[:, :, 15+27] = -proprioceptive_obs[:, :, 21+27]
        flipped_proprioceptive_obs[:, :, 16+27] =  proprioceptive_obs[:, :, 10+27]
        flipped_proprioceptive_obs[:, :, 17+27] = -proprioceptive_obs[:, :, 11+27]
        flipped_proprioceptive_obs[:, :, 18+27] = -proprioceptive_obs[:, :, 12+27]
        flipped_proprioceptive_obs[:, :, 19+27] =  proprioceptive_obs[:, :, 13+27]
        flipped_proprioceptive_obs[:, :, 20+27] =  proprioceptive_obs[:, :, 14+27]
        flipped_proprioceptive_obs[:, :, 21+27] = -proprioceptive_obs[:, :, 15+27]

        flipped_proprioceptive_obs[:, :, 22+27] =  -proprioceptive_obs[:, :, 22+27]

        flipped_proprioceptive_obs[:, :, 23+27] =  proprioceptive_obs[:, :, 30+27]
        flipped_proprioceptive_obs[:, :, 24+27] = -proprioceptive_obs[:, :, 31+27]
        flipped_proprioceptive_obs[:, :, 25+27] = -proprioceptive_obs[:, :, 32+27]
        flipped_proprioceptive_obs[:, :, 26+27] =  proprioceptive_obs[:, :, 33+27]
        flipped_proprioceptive_obs[:, :, 27+27] = -proprioceptive_obs[:, :, 34+27]
        flipped_proprioceptive_obs[:, :, 28+27] =  proprioceptive_obs[:, :, 35+27]
        flipped_proprioceptive_obs[:, :, 29+27] = -proprioceptive_obs[:, :, 36+27]


        flipped_proprioceptive_obs[:, :, 30+27] =  proprioceptive_obs[:, :, 23+27]
        flipped_proprioceptive_obs[:, :, 31+27] = -proprioceptive_obs[:, :, 24+27]
        flipped_proprioceptive_obs[:, :, 32+27] = -proprioceptive_obs[:, :, 25+27]
        flipped_proprioceptive_obs[:, :, 33+27] =  proprioceptive_obs[:, :, 26+27]
        flipped_proprioceptive_obs[:, :, 34+27] = -proprioceptive_obs[:, :, 27+27]
        flipped_proprioceptive_obs[:, :, 35+27] =  proprioceptive_obs[:, :, 28+27]
        flipped_proprioceptive_obs[:, :, 36+27] = -proprioceptive_obs[:, :, 29+27]

        # joint target
        flipped_proprioceptive_obs[:, :, 10+54] =  proprioceptive_obs[:, :, 16+54]
        flipped_proprioceptive_obs[:, :, 11+54] = -proprioceptive_obs[:, :, 17+54]
        flipped_proprioceptive_obs[:, :, 12+54] = -proprioceptive_obs[:, :, 18+54]
        flipped_proprioceptive_obs[:, :, 13+54] =  proprioceptive_obs[:, :, 19+54]
        flipped_proprioceptive_obs[:, :, 14+54] =  proprioceptive_obs[:, :, 20+54]
        flipped_proprioceptive_obs[:, :, 15+54] = -proprioceptive_obs[:, :, 21+54]
        flipped_proprioceptive_obs[:, :, 16+54] =  proprioceptive_obs[:, :, 10+54]
        flipped_proprioceptive_obs[:, :, 17+54] = -proprioceptive_obs[:, :, 11+54]
        flipped_proprioceptive_obs[:, :, 18+54] = -proprioceptive_obs[:, :, 12+54]
        flipped_proprioceptive_obs[:, :, 19+54] =  proprioceptive_obs[:, :, 13+54]
        flipped_proprioceptive_obs[:, :, 20+54] =  proprioceptive_obs[:, :, 14+54]
        flipped_proprioceptive_obs[:, :, 21+54] = -proprioceptive_obs[:, :, 15+54]

        flipped_proprioceptive_obs[:, :, 22+54] =  proprioceptive_obs[:, :, 22+54] # base lin vel x
        flipped_proprioceptive_obs[:, :, 23+54] = -proprioceptive_obs[:, :, 23+54] # base lin vel y
        flipped_proprioceptive_obs[:, :, 24+54] =  proprioceptive_obs[:, :, 24+54] # base lin vel z

        return flipped_proprioceptive_obs.view(-1, num_obs * history).detach()
    
    def flip_actions(self, actions):
        """Flip actions for left-right symmetry.
        """
        num_obs = self.actor_critic.num_one_step_obs
        history = self.actor_critic.actor_history_length
        if num_obs == 78:
            return self._flip_cowa_actions(actions)
        else:
            return self._flip_g1_actions(actions)

    def _flip_g1_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)
        flipped_actions[:,  0] =  actions[:, 6]        # 0 "left_hip_pitch_joint",
        flipped_actions[:,  1] = -actions[:, 7]        # 1 "left_hip_roll_joint",
        flipped_actions[:,  2] = -actions[:, 8]        # 2 "left_hip_yaw_joint",
        flipped_actions[:,  3] =  actions[:, 9]        # 3 "left_knee_joint",
        flipped_actions[:,  4] =  actions[:, 10]       # 4 "left_ankle_pitch_joint",
        flipped_actions[:,  5] = -actions[:, 11]       # 5 "left_ankle_roll_joint",
        flipped_actions[:,  6] =  actions[:, 0]        # 6 "right_hip_pitch_joint",
        flipped_actions[:,  7] = -actions[:, 1]        # 7 "right_hip_roll_joint",
        flipped_actions[:,  8] = -actions[:, 2]        # 8 "right_hip_yaw_joint",
        flipped_actions[:,  9] =  actions[:, 3]        # 9 "right_knee_joint",
        flipped_actions[:, 10] =  actions[:, 4]        # 10 "right_ankle_pitch_joint",
        flipped_actions[:, 11] = -actions[:, 5]        # 11 "right_ankle_roll_joint",
        return flipped_actions.detach()
    

    def _flip_cowa_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)
        flipped_actions[:,  0] = actions[:, 6]        # 0 "left_hip_pitch_joint",
        flipped_actions[:,  1] = actions[:, 7]        # 1 "left_hip_roll_joint",
        flipped_actions[:,  2] = actions[:, 8]        # 2 "left_hip_yaw_joint",
        flipped_actions[:,  3] = actions[:, 9]        # 3 "left_knee_joint",
        flipped_actions[:,  4] = actions[:, 10]       # 4 "left_ankle_pitch_joint",
        flipped_actions[:,  5] = actions[:, 11]       # 5 "left_ankle_roll_joint",
        flipped_actions[:,  6] = actions[:, 0]        # 6 "right_hip_pitch_joint",
        flipped_actions[:,  7] = actions[:, 1]        # 7 "right_hip_roll_joint",
        flipped_actions[:,  8] = actions[:, 2]        # 8 "right_hip_yaw_joint",
        flipped_actions[:,  9] = actions[:, 3]        # 9 "right_knee_joint",
        flipped_actions[:, 10] = actions[:, 4]        # 10 "right_ankle_pitch_joint",
        flipped_actions[:, 11] = actions[:, 5]        # 11 "right_ankle_roll_joint",
        return flipped_actions.detach()