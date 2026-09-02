# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

class PPO_SYM:
    """PPO with left-right symmetry constraint via observation mirroring.

    Adapted from HIM PPO: adds flipped observation augmentation and
    actor/critic symmetry loss to enforce symmetric gait.
    """
    actor_critic: ActorCritic
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
                 symmetry_scale=0.01,
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
        self.transition = RolloutStorage.Transition()
        self.transition_sym = RolloutStorage.Transition()
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
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

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
            self.transition_sym.observations = obs_sym
            self.transition_sym.critic_observations = critic_obs_sym
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.use_flip:
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
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
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

                # Symmetry loss (from HIM PPO)
                if self.use_flip:
                    flipped_obs_batch = self.flip_actor_obs(obs_batch)
                    flipped_critic_obs_batch = self.flip_critic_obs(critic_obs_batch)
                    actor_sym_loss = self.symmetry_scale * torch.mean(
                        torch.sum(torch.square(
                            self.actor_critic.act_inference(flipped_obs_batch) - self.flip_actions(self.actor_critic.act_inference(obs_batch))
                        ), dim=-1)
                    )
                    critic_sym_loss = self.symmetry_scale * torch.mean(
                        torch.square(
                            self.actor_critic.evaluate(flipped_critic_obs_batch) - self.actor_critic.evaluate(critic_obs_batch).detach()
                        )
                    )
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
                if self.use_flip:
                    mean_actor_sym_loss += actor_sym_loss.item()
                    mean_critic_sym_loss += critic_sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if self.use_flip:
            mean_actor_sym_loss /= num_updates
            mean_critic_sym_loss /= num_updates
        self.storage.clear()

        if self.use_flip:
            return mean_value_loss, mean_surrogate_loss, mean_actor_sym_loss, mean_critic_sym_loss
        else:
            return mean_value_loss, mean_surrogate_loss

    # ==================== Flip functions for quadruped_wtw_slope ====================

    def flip_actor_obs(self, obs):
        """Flip actor observation for left-right symmetry.

        Observation layout (one step, 59 dims):
        [0:12]   commands (vx,vy,yaw,body_h,freq,phase,offset,bound,dur,swing,pitch,roll)
        [12:24]  dof_pos (URDF order: FL_hip,FL_th,FL_cf,FR_hip,FR_th,FR_cf,RL_hip,RL_th,RL_cf,RR_hip,RR_th,RR_cf)
        [24:36]  dof_vel (same order)
        [36:48]  actions (same order)
        [48:51]  ang_vel (x,y,z)
        [51:54]  projected_gravity (x,y,z)
        [54]     gait_index
        [55:59]  clock_inputs [FL, FR, RL, RR]
        """
        history = 5  # frame_stack
        num_obs = 59
        obs_batch = torch.clone(obs[:, :num_obs * history])
        obs_batch = obs_batch.view(-1, history, num_obs)
        flipped = torch.zeros_like(obs_batch)

        # Commands (0-11)
        flipped[:, :, 0] =  obs_batch[:, :, 0]   # vx stays
        flipped[:, :, 1] = -obs_batch[:, :, 1]   # vy flips
        flipped[:, :, 2] = -obs_batch[:, :, 2]   # yaw flips
        flipped[:, :, 3] =  obs_batch[:, :, 3]   # body_h stays
        flipped[:, :, 4] =  obs_batch[:, :, 4]   # freq stays
        flipped[:, :, 5] =  obs_batch[:, :, 5]   # phase stays
        flipped[:, :, 6] =  obs_batch[:, :, 6]   # offset stays (timing shift, symmetric)
        flipped[:, :, 7] =  obs_batch[:, :, 7]   # bound stays
        flipped[:, :, 8] =  obs_batch[:, :, 8]   # dur stays
        flipped[:, :, 9] =  obs_batch[:, :, 9]   # swing stays
        flipped[:, :, 10] =  obs_batch[:, :, 10] # pitch stays
        flipped[:, :, 11] = -obs_batch[:, :, 11] # roll flips

        # DOF pos (12-23): swap left↔right, flip hip sign
        # URDF order: FL_hip,FL_th,FL_cf, FR_hip,FR_th,FR_cf, RL_hip,RL_th,RL_cf, RR_hip,RR_th,RR_cf
        # Swap: FL(0,1,2)↔FR(3,4,5), RL(6,7,8)↔RR(9,10,11)
        # Sign flip: hips (0↔3, 6↔9), thighs/calves keep sign
        swap_pos = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2, 6:9, 7:10, 8:11, 9:6, 10:7, 11:8}
        hip_indices = {0, 3, 6, 9}
        for src, dst in swap_pos.items():
            sign = -1.0 if src in hip_indices else 1.0
            flipped[:, :, 12 + src] = sign * obs_batch[:, :, 12 + dst]

        # DOF vel (24-35): same pattern
        for src, dst in swap_pos.items():
            sign = -1.0 if src in hip_indices else 1.0
            flipped[:, :, 24 + src] = sign * obs_batch[:, :, 24 + dst]

        # Actions (36-47): same pattern
        for src, dst in swap_pos.items():
            sign = -1.0 if src in hip_indices else 1.0
            flipped[:, :, 36 + src] = sign * obs_batch[:, :, 36 + dst]

        # Ang vel (48-50): roll flips, pitch stays, yaw flips
        flipped[:, :, 48] = -obs_batch[:, :, 48]
        flipped[:, :, 49] =  obs_batch[:, :, 49]
        flipped[:, :, 50] = -obs_batch[:, :, 50]

        # Projected gravity (51-53): x stays, y flips, z stays
        flipped[:, :, 51] =  obs_batch[:, :, 51]
        flipped[:, :, 52] = -obs_batch[:, :, 52]
        flipped[:, :, 53] =  obs_batch[:, :, 53]

        # Gait index (54): stays
        flipped[:, :, 54] = obs_batch[:, :, 54]

        # Clock inputs (55-58): [FL, FR, RL, RR] → swap FL↔FR, RL↔RR
        flipped[:, :, 55] = obs_batch[:, :, 56]  # FL←FR
        flipped[:, :, 56] = obs_batch[:, :, 55]  # FR←FL
        flipped[:, :, 57] = obs_batch[:, :, 58]  # RL←RR
        flipped[:, :, 58] = obs_batch[:, :, 57]  # RR←RL

        return flipped.view(-1, num_obs * history)

    def flip_critic_obs(self, critic_obs):
        """Flip critic observation (privileged obs). Same layout as actor + extra priv info.
        We only flip the proprioceptive part; privileged scalars (payload, inertia, etc.) stay unchanged.
        """
        history = 1  # c_frame_stack
        # The privileged obs starts with the same 59 dims as actor obs per frame
        num_actor_obs = 59
        num_critic_obs = critic_obs.shape[1] // history
        obs_batch = torch.clone(critic_obs[:, :num_actor_obs * history])
        obs_batch = obs_batch.view(-1, history, num_actor_obs)
        flipped_actor = torch.zeros_like(obs_batch)

        # Same flip logic as actor obs for the first 59 dims
        # Commands
        flipped_actor[:, :, 0] =  obs_batch[:, :, 0]
        flipped_actor[:, :, 1] = -obs_batch[:, :, 1]
        flipped_actor[:, :, 2] = -obs_batch[:, :, 2]
        flipped_actor[:, :, 3] =  obs_batch[:, :, 3]
        flipped_actor[:, :, 4] =  obs_batch[:, :, 4]
        flipped_actor[:, :, 5] =  obs_batch[:, :, 5]
        flipped_actor[:, :, 6] =  obs_batch[:, :, 6]
        flipped_actor[:, :, 7] =  obs_batch[:, :, 7]
        flipped_actor[:, :, 8] =  obs_batch[:, :, 8]
        flipped_actor[:, :, 9] =  obs_batch[:, :, 9]
        flipped_actor[:, :, 10] =  obs_batch[:, :, 10]
        flipped_actor[:, :, 11] = -obs_batch[:, :, 11]

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
        flipped_actor[:, :, 54] =  obs_batch[:, :, 54]
        flipped_actor[:, :, 55] =  obs_batch[:, :, 56]
        flipped_actor[:, :, 56] =  obs_batch[:, :, 55]
        flipped_actor[:, :, 57] =  obs_batch[:, :, 58]
        flipped_actor[:, :, 58] =  obs_batch[:, :, 57]

        flipped_proprio = flipped_actor.view(-1, num_actor_obs * history)

        # Keep remaining privileged info (base_lin_vel, height, payload, inertia, etc.) unchanged
        remaining = critic_obs[:, num_actor_obs * history:]
        return torch.cat([flipped_proprio, remaining], dim=-1)

    def flip_actions(self, actions):
        """Flip actions for left-right symmetry.
        URDF order: FL_hip,FL_th,FL_cf, FR_hip,FR_th,FR_cf, RL_hip,RL_th,RL_cf, RR_hip,RR_th,RR_cf
        Swap: FL(0,1,2)↔FR(3,4,5), RL(6,7,8)↔RR(9,10,11)
        Hips flip sign, thighs/calves keep sign.
        """
        flipped = torch.zeros_like(actions)
        hip_flip = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], device=actions.device)
        swap_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], device=actions.device, dtype=torch.long)
        flipped = actions[:, swap_idx] * hip_flip
        return flipped.detach()
