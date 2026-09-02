# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
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
                 mlp_learning_rate = 5.e-4,
                 num_adaptation_module_substeps =1 ,
                 device='cpu',
                 ):

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # dagger
        self.dagger_optimizer = optim.Adam(self.actor_critic.parameters(), lr=5e-4)
        # xxx
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.estimator_optimizer = optim.Adam(self.actor_critic.estimator.parameters(), lr=mlp_learning_rate)
        self.transition = RolloutStorage.Transition()

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

    def init_storage(self, num_envs, num_transitions_per_env, height_obs_shape, actor_obs_shape, obs_history_shape, critic_obs_shape, action_shape, dagger_on):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, height_obs_shape, actor_obs_shape, obs_history_shape, critic_obs_shape, action_shape, self.device, dagger_on)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, height_obs, obs_input, obs_history, critic_obs, gt_actions, dagger_on=False):
        # Compute the actions and values
        if dagger_on:
            self.transition.actions = self.actor_critic.act(obs_input, obs_history, height_obs).detach()
            self.transition.gt_actions = gt_actions
        else:
            self.transition.actions = self.actor_critic.act(obs_input, obs_history, height_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs_input
        self.transition.observations_history = obs_history  #adaptation
        self.transition.critic_observations = critic_obs
        self.transition.height_obs = height_obs # height scan
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
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

    def update(self, dagger_only=False, dagger_on=False):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_est_loss = 0
        mean_dagger_loss = 0 # ts
        v_avg_diff_x = 0
        v_avg_diff_y = 0
        v_avg_diff_z = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for height_obs_batch, obs_batch, obs_history_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, gt_actions_batch in generator:
                
                self.actor_critic.act(obs_batch, obs_history_batch, height_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
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

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                if not dagger_only:
                    loss.backward()
                else:
                    print("Dagger only, ignoring RL losses")
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                # est
                if dagger_on:
                    for _ in range(self.num_adaptation_module_substeps):
                        self.estimator_optimizer.zero_grad()
                        estimator_batch = self.actor_critic.estimator(obs_history_batch)
                        estimator_loss_dict = self.actor_critic.estimator.loss_fn(estimator_batch, critic_obs_batch, self.actor_critic.num_est_prob)
                        estimator_loss = torch.mean(estimator_loss_dict["loss"])
                        v_avg_diff_x = estimator_loss_dict["v_avg_diff_x"]
                        v_avg_diff_y = estimator_loss_dict["v_avg_diff_y"]
                        v_avg_diff_z = estimator_loss_dict["v_avg_diff_z"]
                        base_height_diff = estimator_loss_dict["base_height_diff"]

                        estimator_loss.backward()
                        self.estimator_optimizer.step()
                        with torch.no_grad():
                            est_loss = torch.mean(estimator_loss_dict["loss"])
                            mean_est_loss += est_loss.item()

                # dagger loss
                if dagger_on:
                    mean_dagger_loss = self._optimize_dagger_loss(obs_batch, obs_history_batch, height_obs_batch, gt_actions_batch)


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if dagger_on:
            mean_est_loss /= num_updates * self.num_adaptation_module_substeps
            mean_dagger_loss /= num_updates
        self.storage.clear()

        if dagger_on:
            return mean_value_loss, mean_surrogate_loss, v_avg_diff_x, v_avg_diff_y, v_avg_diff_z, base_height_diff, mean_est_loss, mean_dagger_loss
        else:
            return mean_value_loss, mean_surrogate_loss, None, None, None,  None, None, None


    def _optimize_dagger_loss(self, obs_batch, obs_history_batch, height_obs_batch, gt_actions_batch):
        # print("Teacher student training") 
        pred_action = self.actor_critic.act_inference(obs_batch, obs_history_batch, height_obs_batch)
        gt_action = gt_actions_batch
        dagger_loss = torch.norm(pred_action - gt_action, dim=-1).mean()  ## RMSE
        dagger_loss = dagger_loss * 1.0 
        self.dagger_optimizer.zero_grad()
        dagger_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.dagger_optimizer.step()
        return dagger_loss





