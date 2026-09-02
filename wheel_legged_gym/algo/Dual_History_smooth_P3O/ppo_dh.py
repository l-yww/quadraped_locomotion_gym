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

from .actor_critic_dh import ActorCritic_DH_Smooth_P3O
from .rollout_storage_dh import RolloutStorage_Estimator_Smooth_P3O

class PPO_DH_Smooth_P3O:
    actor_critic: ActorCritic_DH_Smooth_P3O
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 cost_value_loss_coef=1.0,  
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 mlp_learning_rate = 5.e-4,
                 num_adaptation_module_substeps =1 ,
                 device='cpu',
                 value_smoothness_coef=0.1,
                 smoothness_upper_bound=1.0,
                 smoothness_lower_bound=0.0,
                 advantage_weight=0.5,  
                 cost_advantage_weight=0.5,  
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
        # xxx
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.estimator_optimizer = optim.Adam(self.actor_critic.estimator.parameters(), lr=mlp_learning_rate)
        self.transition = RolloutStorage_Estimator_Smooth_P3O.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        # Dual History
        self.num_short_obs = self.actor_critic.num_short_obs
        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound

        self.advantage_weight = advantage_weight
        self.cost_advantage_weight = cost_advantage_weight

    def init_storage(self, num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage_Estimator_Smooth_P3O(num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs_history, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach() # cost values
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations_history = obs_history  #adaptation
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, costs, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.costs += self.gamma * torch.squeeze(self.transition.cost_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, obs):
        last_cost_values = self.actor_critic.evaluate_cost(obs).detach()
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam)


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_cost_value_loss = 0
        mean_cost_surrogate_loss = 0
        mean_est_loss = 0
        mean_smooth_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_history_batch, critic_obs_batch, next_obs_batch, next_critic_batch, cont_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch in generator:
                
                self.actor_critic.act(obs_history_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                cost_value_batch = self.actor_critic.evaluate_cost(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
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

                # 归一化优势函数
                advantages_normalized = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                cost_advantages_normalized = (cost_advantages_batch - cost_advantages_batch.mean()) / (cost_advantages_batch.std() + 1e-8)
                
                combined_advantages = self.advantage_weight * advantages_normalized + self.cost_advantage_weight * cost_advantages_normalized

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(combined_advantages) * ratio
                surrogate_clipped = -torch.squeeze(combined_advantages) * torch.clamp(ratio, 1.0 - self.clip_param,
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

                # Cost value function loss
                if self.use_clipped_value_loss:
                    cost_value_clipped = target_cost_values_batch + (cost_value_batch - target_cost_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                    cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                    cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
                else:
                    cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

                combine_value_loss = self.cost_value_loss_coef * cost_value_loss + self.value_loss_coef * value_loss
                entropy_loss = - self.entropy_coef * entropy_batch.mean()     

                loss = surrogate_loss + combine_value_loss + entropy_loss 
                # loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Smoothness loss
                epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
                policy_smooth_coef = self.smoothness_upper_bound * epsilon
                value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

                mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                
                mix_obs_batch = obs_history_batch.clone()
                
                short_obs = obs_history_batch[:, -self.num_short_obs:]
                short_next_obs = next_obs_batch[:, -self.num_short_obs:]

                mixed_short_obs = short_obs + mix_weights * (short_next_obs - short_obs)

                mix_obs_batch[:, -self.num_short_obs:] = mixed_short_obs

                mix_critic_batch = critic_obs_batch + mix_weights * (next_critic_batch - critic_obs_batch)

                policy_smooth_loss = torch.square(torch.norm(mu_batch - self.actor_critic.act_inference(mix_obs_batch), dim=-1)).mean()
                value_smooth_loss = torch.square(torch.norm(value_batch - self.actor_critic.evaluate(mix_critic_batch), dim=-1)).mean()
                smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_smooth_loss

                with torch.inference_mode():
                    action_smoothness = torch.norm(mu_batch - self.actor_critic.act_inference(next_obs_batch), dim=-1).mean()
                    
                loss += smooth_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_cost_surrogate_loss += cost_surrogate_loss.item()
                mean_smooth_loss += smooth_loss.item()

                # Estimator
                estimator_input = obs_history_batch[:, -self.num_short_obs:]    # Estimator 的输入为 short history
                for _ in range(self.num_adaptation_module_substeps):
                    self.estimator_optimizer.zero_grad()
                    estimator_batch = self.actor_critic.estimator(estimator_input)
                    estimator_loss = self.actor_critic.est_loss_fn(estimator_batch, critic_obs_batch, self.actor_critic.num_est_prob)
                    estimator_loss.backward()
                    self.estimator_optimizer.step()
                    with torch.no_grad():
                        mean_est_loss += estimator_loss.item()
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_cost_surrogate_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_est_loss /= num_updates * self.num_adaptation_module_substeps
        mean_smooth_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_cost_value_loss, mean_surrogate_loss, mean_cost_surrogate_loss, mean_est_loss, mean_smooth_loss
