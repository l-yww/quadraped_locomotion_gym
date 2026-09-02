# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

class PPO_SYM_ARM:
    """PPO with symmetry constraint for quadruped_wtw_arm task (71-dim obs)."""
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

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.transition_sym = RolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale

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
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
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
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            if self.use_flip:
                self.transition_sym.rewards += self.gamma * torch.squeeze(self.transition_sym.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        if self.use_flip:
            self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        if self.use_flip:
            self.transition_sym.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
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

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

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

    # ==================== Flip functions for quadruped_wtw_arm ====================

    def flip_actor_obs(self, obs):
        """Flip actor observation for left-right symmetry.

        Observation layout (one step, num_single_obs dims - default 71):
        [0:12]   commands (vx,vy,yaw,body_h,freq,phase,offset,bound,dur,swing,pitch,roll)
        [12:30]  dof_pos × 1.0 (18 DOF, URDF order: FL_hip,FL_th,FL_cf,FR_hip,FR_th,FR_cf,RL_hip,RL_th,RL_cf,RR_hip,RR_th,RR_cf,arm17-22)
        [30:48]  dof_vel × 0.05 (same order)
        [48:60]  actions (12 DOF, only legs: FL_hip,FL_th,FL_cf,FR_hip,FR_th,FR_cf,RL_hip,RL_th,RL_cf,RR_hip,RR_th,RR_cf)
        [60:63]  ang_vel × 0.25 (x,y,z)
        [63:66]  projected_gravity × 1 (x,y,z)
        [66]     gait_index
        [67:71]  clock_inputs [FL, FR, RL, RR]

        Arm joints (indices 12-17 in dof_pos/dof_vel) are NOT flipped (no left-right symmetry).
        """
        history = 5
        num_obs = int(obs.shape[1] / history)
        obs_batch = torch.clone(obs[:, :num_obs * history])
        obs_batch = obs_batch.view(-1, history, num_obs)
        flipped = torch.zeros_like(obs_batch)

        # Commands (0-11)
        flipped[:, :, 0] =  obs_batch[:, :, 0]
        flipped[:, :, 1] = -obs_batch[:, :, 1]
        flipped[:, :, 2] = -obs_batch[:, :, 2]
        flipped[:, :, 3] =  obs_batch[:, :, 3]
        flipped[:, :, 4] =  obs_batch[:, :, 4]
        flipped[:, :, 5] =  obs_batch[:, :, 5]
        flipped[:, :, 6] =  obs_batch[:, :, 6]
        flipped[:, :, 7] =  obs_batch[:, :, 7]
        flipped[:, :, 8] =  obs_batch[:, :, 8]
        flipped[:, :, 9] =  obs_batch[:, :, 9]
        flipped[:, :, 10] =  obs_batch[:, :, 10]
        flipped[:, :, 11] = -obs_batch[:, :, 11]

        DOF = 12 #18 #自由度
        dof_pos_start = 12
        dof_vel_start = 12 + DOF  # 30
        action_start = 12 + 2 * DOF  # 48

        # Leg DOF swap: FL(0,1,2)↔FR(3,4,5), RL(6,7,8)↔RR(9,10,11)
        # Hip flip sign, thigh/calf keep sign
        leg_swap = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2, 6:9, 7:10, 8:11, 9:6, 10:7, 11:8}
        hip_indices = {0, 3, 6, 9}

        for src, dst in leg_swap.items():
            sign = -1.0 if src in hip_indices else 1.0
            flipped[:, :, dof_pos_start + src] = sign * obs_batch[:, :, dof_pos_start + dst]
            flipped[:, :, dof_vel_start + src] = sign * obs_batch[:, :, dof_vel_start + dst]
            flipped[:, :, action_start + src] = sign * obs_batch[:, :, action_start + dst]

        # Arm joints (12-17): unchanged (no left-right mirror)
        for a in range(12, DOF):
            flipped[:, :, dof_pos_start + a] = obs_batch[:, :, dof_pos_start + a]
            flipped[:, :, dof_vel_start + a] = obs_batch[:, :, dof_vel_start + a]

        # Ang vel (60-62): roll flips, pitch stays, yaw flips
        ang_vel_start = action_start + 12  # 60
        flipped[:, :, ang_vel_start + 0] = -obs_batch[:, :, ang_vel_start + 0]
        flipped[:, :, ang_vel_start + 1] =  obs_batch[:, :, ang_vel_start + 1]
        flipped[:, :, ang_vel_start + 2] = -obs_batch[:, :, ang_vel_start + 2]

        # Projected gravity (63-65): x stays, y flips, z stays
        gravity_start = ang_vel_start + 3  # 63
        flipped[:, :, gravity_start + 0] =  obs_batch[:, :, gravity_start + 0]
        flipped[:, :, gravity_start + 1] = -obs_batch[:, :, gravity_start + 1]
        flipped[:, :, gravity_start + 2] =  obs_batch[:, :, gravity_start + 2]

        # Gait index (66): stays
        gait_start = gravity_start + 3  # 66
        flipped[:, :, gait_start] = obs_batch[:, :, gait_start]

        # Clock inputs (67-70): [FL, FR, RL, RR] → swap FL↔FR, RL↔RR
        clock_start = gait_start + 1  # 67
        flipped[:, :, clock_start + 0] = obs_batch[:, :, clock_start + 1]  # FL←FR
        flipped[:, :, clock_start + 1] = obs_batch[:, :, clock_start + 0]  # FR←FL
        flipped[:, :, clock_start + 2] = obs_batch[:, :, clock_start + 3]  # RL←RR
        flipped[:, :, clock_start + 3] = obs_batch[:, :, clock_start + 2]  # RR←RL

        return flipped.view(-1, num_obs * history)

    def flip_critic_obs(self, critic_obs):
        """Flip critic observation. Flips the proprioceptive part; privileged info unchanged."""
        history = 1
        num_actor_obs = int(self.actor_critic.num_one_step_actor_obs) if hasattr(self.actor_critic, 'num_one_step_actor_obs') else critic_obs.shape[1] // history  # default: all is actor
        # For arm task, the privileged obs includes actor obs + extra priv info
        # We use the same actor obs size as computed above
        num_actor_obs_total = critic_obs.shape[1]  # Take all privileged obs
        # Actually the privileged obs has 1 frame stack, same number of dims per frame
        # We'll flip the same structure
        obs_batch = torch.clone(critic_obs[:, :num_actor_obs_total * history])
        obs_batch = obs_batch.view(-1, history, num_actor_obs_total)
        flipped_actor = torch.zeros_like(obs_batch)

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

        DOF = 12 #18
        dof_pos_start = 12
        dof_vel_start = 12 + DOF
        action_start = 12 + 2 * DOF
        ang_vel_start = action_start + 12
        gravity_start = ang_vel_start + 3
        gait_start = gravity_start + 3
        clock_start = gait_start + 1

        leg_swap = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2, 6:9, 7:10, 8:11, 9:6, 10:7, 11:8}
        hip_indices = {0, 3, 6, 9}
        for src, dst in leg_swap.items():
            sign = -1.0 if src in hip_indices else 1.0
            flipped_actor[:, :, dof_pos_start + src] = sign * obs_batch[:, :, dof_pos_start + dst]
            flipped_actor[:, :, dof_vel_start + src] = sign * obs_batch[:, :, dof_vel_start + dst]
            flipped_actor[:, :, action_start + src] = sign * obs_batch[:, :, action_start + dst]

        for a in range(12, DOF):
            flipped_actor[:, :, dof_pos_start + a] = obs_batch[:, :, dof_pos_start + a]
            flipped_actor[:, :, dof_vel_start + a] = obs_batch[:, :, dof_vel_start + a]

        flipped_actor[:, :, ang_vel_start + 0] = -obs_batch[:, :, ang_vel_start + 0]
        flipped_actor[:, :, ang_vel_start + 1] =  obs_batch[:, :, ang_vel_start + 1]
        flipped_actor[:, :, ang_vel_start + 2] = -obs_batch[:, :, ang_vel_start + 2]
        flipped_actor[:, :, gravity_start + 0] =  obs_batch[:, :, gravity_start + 0]
        flipped_actor[:, :, gravity_start + 1] = -obs_batch[:, :, gravity_start + 1]
        flipped_actor[:, :, gravity_start + 2] =  obs_batch[:, :, gravity_start + 2]
        flipped_actor[:, :, gait_start] = obs_batch[:, :, gait_start]
        flipped_actor[:, :, clock_start + 0] = obs_batch[:, :, clock_start + 1]
        flipped_actor[:, :, clock_start + 1] = obs_batch[:, :, clock_start + 0]
        flipped_actor[:, :, clock_start + 2] = obs_batch[:, :, clock_start + 3]
        flipped_actor[:, :, clock_start + 3] = obs_batch[:, :, clock_start + 2]

        flipped_proprio = flipped_actor[:, :, :clock_start + 4].reshape(-1, (clock_start + 4) * history)
        remaining = obs_batch[:, :, clock_start + 4:].reshape(-1, (num_actor_obs_total - clock_start - 4) * history)
        return torch.cat([flipped_proprio, remaining], dim=-1)

    def flip_actions(self, actions):
        """Flip actions for left-right symmetry (12 DOF legs only, URDF order)."""
        hip_flip = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], device=actions.device)
        swap_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], device=actions.device, dtype=torch.long)
        return (actions[:, swap_idx] * hip_flip).detach()
