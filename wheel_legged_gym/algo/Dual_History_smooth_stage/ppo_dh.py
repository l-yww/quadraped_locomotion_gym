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

from .actor_critic_dh import ActorCritic_DH_Smooth_Stage
from .rollout_storage_dh import RolloutStorage_Estimator_Smooth_Stage

class PPO_DH_Smooth_Stage:
    actor_critic: ActorCritic_DH_Smooth_Stage
    def __init__(self,
                 actor_critic_arm,
                 actor_critic_leg,
                 actor_frame,
                 num_prio_obs,
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
                 value_smoothness_coef=0.1,
                 smoothness_upper_bound=1.0,
                 smoothness_lower_bound=0.0,
                 ):

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic_arm : ActorCritic = actor_critic_arm
        self.actor_critic_leg : ActorCritic = actor_critic_leg
        self.actor_frame = actor_frame
        self.num_prio_obs = num_prio_obs

        self.actor_critic_arm.to(self.device)
        self.actor_critic_leg.to(self.device)
        self.storage = None # initialized later

        self.optimizer_leg = optim.Adam(self.actor_critic_leg.parameters(), lr=learning_rate)

        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.estimator_optimizer_leg = optim.Adam(self.actor_critic_leg.estimator.parameters(), lr=mlp_learning_rate)

        self.transition_leg = RolloutStorage_Estimator_Smooth_Stage.Transition()
        self.transition_arm = RolloutStorage_Estimator_Smooth_Stage.Transition()

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
        # Dual History
        self.num_short_obs_leg = self.actor_critic_leg.num_short_obs
        self.num_short_obs_arm = self.actor_critic_arm.num_short_obs
        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound = smoothness_upper_bound
        self.smoothness_lower_bound = smoothness_lower_bound

    def init_leg_storage(self, num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape):
        #storage covers leg  
        self.storage_leg = RolloutStorage_Estimator_Smooth_Stage(num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape, self.device)

    def init_arm_storage(self, num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape):
        #storage covers arm 
        self.storage_arm = RolloutStorage_Estimator_Smooth_Stage(num_envs, num_transitions_per_env, obs_history_shape, critic_obs_shape, action_shape, self.device)


    def test_mode(self):
        self.actor_critic_arm.test()
        self.actor_critic_leg.test()
    
    def train_mode(self):
        self.actor_critic_arm.eval()
        self.actor_critic_leg.train()

    def act(self, obs_history, critic_obs):

        self.transition_leg.obs_long_history = obs_history.clone().detach()
        num_env = obs_history.shape[0]
        obs_history = obs_history.clone().reshape(num_env, -1, self.num_prio_obs)
        estimated_prob_leg = critic_obs[:,-7:-3]
        estimated_prob_arm = critic_obs[:,-3:]

        # Compute the arm actions and values
        arm_actor_obs = torch.cat([   
                                   obs_history[:,:,3:15],   # arm cmd
                                   obs_history[:,:,15:21],  # arm pos
                                   obs_history[:,:,29:35],  # arm vel
                                   obs_history[:,:,45:51]],dim=2).reshape(num_env,-1).detach() # arm action
        arm_critic_obs = torch.cat([critic_obs[:,3:15],                            # arm cmd
                                    critic_obs[:,15:21],                           # arm pos
                                    critic_obs[:,29:35],                          # arm vel
                                    critic_obs[:,45:51],                          # arm action
                                    critic_obs[:,-3:],                          # 末端执行器相对于机器人基座（或躯干）的位移向量
                                    ],dim=1).reshape(num_env,-1).detach()
        # 使用评估模式获取上肢动作，不计算梯度
        with torch.no_grad():
            self.transition_arm.actions = self.actor_critic_arm.act(arm_actor_obs).detach()
            self.transition_arm.values = self.actor_critic_arm.evaluate(arm_critic_obs).detach()
            self.transition_arm.actions_log_prob = self.actor_critic_arm.get_actions_log_prob(self.transition_arm.actions).detach()
            self.transition_arm.action_mean = self.actor_critic_arm.action_mean.detach()
            self.transition_arm.action_sigma = self.actor_critic_arm.action_std.detach()
        
        # need to record obs and critic_obs before env.step()
        self.transition_arm.observations_history = arm_actor_obs
        self.transition_arm.critic_observations = arm_critic_obs

        # Compute the leg actions and values
        leg_actor_obs = torch.cat([obs_history[:,:,:3],                             # leg cmd
                                    obs_history[:,:,21:29],                           # leg pos
                                    obs_history[:,:,35:45],                          # leg vel
                                    obs_history[:,:,51:61],                          # leg action
                                    obs_history[:,:,61:66]],dim=2).reshape(num_env,-1).detach() 
        # print("leg_actor_obs", leg_actor_obs.shape)

        leg_critic_obs = torch.cat([critic_obs[:,:3],                             # leg cmd
                                    critic_obs[:,21:29],                           # leg pos
                                    critic_obs[:,35:45],                          # leg vel
                                    critic_obs[:,51:61],                          # leg action
                                    critic_obs[:,61:67],                          # base ang vel & project gravity
                                    critic_obs[:,67:71]],dim=1).reshape(num_env,-1).detach()  # base height & base lin vel

        self.transition_leg.actions = self.actor_critic_leg.act(leg_actor_obs).detach()
        self.transition_leg.values = self.actor_critic_leg.evaluate(leg_critic_obs).detach()
        self.transition_leg.actions_log_prob = self.actor_critic_leg.get_actions_log_prob(self.transition_leg.actions).detach()
        self.transition_leg.action_mean = self.actor_critic_leg.action_mean.detach()
        self.transition_leg.action_sigma = self.actor_critic_leg.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition_leg.observations_history = leg_actor_obs
        self.transition_leg.critic_observations = leg_critic_obs.detach()

        # print('self.transition_arm.actions', self.transition_arm.actions)

        return torch.cat([self.transition_arm.actions, self.transition_leg.actions],dim=1)
        
    
    def process_env_step(self, rewards_leg, dones, infos):

        self.transition_leg.rewards = rewards_leg.clone()
        self.transition_leg.dones = dones.clone()
        if 'time_outs' in infos:
            self.transition_leg.rewards += self.gamma * torch.squeeze(self.transition_leg.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.storage_leg.add_transitions(self.transition_leg)
        self.transition_leg.clear()
        self.actor_critic_leg.reset(dones.clone())
    
    def compute_returns(self, last_critic_obs):

        num_env = last_critic_obs.shape[0]
        
        leg_critic_obs = torch.cat([last_critic_obs[:,:3],                             # leg cmd
                                    last_critic_obs[:,21:29],                           # leg pos
                                    last_critic_obs[:,35:45],                          # leg vel
                                    last_critic_obs[:,51:61],                          # leg action
                                    last_critic_obs[:,61:67],                          # base ang vel & project gravity
                                    last_critic_obs[:,67:71]],dim=1).reshape(num_env,-1).detach()  # base height & base lin vel

        last_values_leg= self.actor_critic_leg.evaluate(leg_critic_obs).detach()
        self.storage_leg.compute_returns(last_values_leg, self.gamma, self.lam)


    def update(self):
        mean_value_loss_leg = 0
        mean_surrogate_loss_leg = 0
        mean_est_loss_leg = 0
        mean_smooth_loss_leg = 0

        generator_leg = self.storage_leg.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_history_batch, critic_obs_batch, next_obs_batch, next_critic_batch, cont_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator_leg:
   
                leg_act_obs = []
                self.actor_critic_leg.act(obs_history_batch)  # 更新action的分布distribution，以便于后续的动作概率actions_log_prob计算
                actions_log_prob_batch = self.actor_critic_leg.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic_leg.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic_leg.action_mean
                sigma_batch = self.actor_critic_leg.action_std
                entropy_batch = self.actor_critic_leg.entropy

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
                        
                        for param_group in self.optimizer_leg.param_groups:
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

                # Smoothness loss
                epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
                policy_smooth_coef = self.smoothness_upper_bound * epsilon
                value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef

                mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                
                mix_obs_batch = obs_history_batch.clone()
                
                short_obs = obs_history_batch[:, -self.num_short_obs_leg:]
                short_next_obs = next_obs_batch[:, -self.num_short_obs_leg:]

                mixed_short_obs = short_obs + mix_weights * (short_next_obs - short_obs)

                mix_obs_batch[:, -self.num_short_obs_leg:] = mixed_short_obs

                mix_critic_batch = critic_obs_batch + mix_weights * (next_critic_batch - critic_obs_batch)

                policy_smooth_loss = torch.square(torch.norm(mu_batch - self.actor_critic_leg.act_inference(mix_obs_batch), dim=-1)).mean()
                value_smooth_loss = torch.square(torch.norm(value_batch - self.actor_critic_leg.evaluate(mix_critic_batch), dim=-1)).mean()
                smooth_loss = policy_smooth_coef * policy_smooth_loss + value_smooth_coef * value_smooth_loss

                with torch.inference_mode():
                    action_smoothness = torch.norm(mu_batch - self.actor_critic_leg.act_inference(next_obs_batch), dim=-1).mean()
                    
                loss += smooth_loss

                # Gradient step
                self.optimizer_leg.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic_leg.parameters(), self.max_grad_norm)
                self.optimizer_leg.step()

                mean_value_loss_leg += value_loss.item()
                mean_surrogate_loss_leg += surrogate_loss.item()
                mean_smooth_loss_leg += smooth_loss.item()

                # Estimator
                estimator_input = obs_history_batch[:, -self.num_short_obs_leg:]    # Estimator 的输入为 short history
                for _ in range(self.num_adaptation_module_substeps):
                    self.estimator_optimizer_leg.zero_grad()
                    estimator_batch = self.actor_critic_leg.estimator(estimator_input)
                    estimator_loss = self.actor_critic_leg.est_loss_fn(estimator_batch, critic_obs_batch, self.actor_critic_leg.num_est_prob)
                    estimator_loss.backward()
                    self.estimator_optimizer_leg.step()
                    with torch.no_grad():
                        mean_est_loss_leg += estimator_loss.item()
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss_leg /= num_updates
        mean_surrogate_loss_leg /= num_updates
        mean_est_loss_leg /= num_updates * self.num_adaptation_module_substeps
        mean_smooth_loss_leg /= num_updates
        self.storage_arm.clear()
        self.storage_leg.clear()


        return mean_value_loss_leg, mean_surrogate_loss_leg, mean_est_loss_leg, mean_smooth_loss_leg
