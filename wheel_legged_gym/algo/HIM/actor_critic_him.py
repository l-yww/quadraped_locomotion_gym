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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from ..utils import get_activation
from .estimator_him import Estimator_HIM

class ActorCritic_HIM(nn.Module):
    is_recurrent = False
    def __init__(self,  num_short_obs,
                        num_single_obs,
                        num_est_prob,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        estimator_hidden_dims=[128, 64],
                        enc_hidden_dims=[128, 64, 16],
                        tar_hidden_dims=[128, 64],
                        history_len = 30,   # CNN 时间轴长度(=历史帧数 frame_stack)；CNN 真正的 in_channels 是 num_single_obs(在 Estimator_HIM 内部取)
                        kernel_size=[6, 4],
                        filter_size=[32, 16],
                        stride_size=[3, 2],
                        lh_output_dim = 32,
                        activation='elu',
                        init_noise_std=1.0,
                        max_grad_norm=10,
                        is_privileged_obs=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_HIM, self).__init__()

        activation = get_activation(activation)

        self.num_short_obs = num_short_obs
        self.num_est_prob = num_est_prob
        self.lh_output_dim = lh_output_dim
        self.num_proprio_obs = num_single_obs

        mlp_input_dim_a = num_short_obs + num_est_prob + lh_output_dim
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Estimator
        # history_len = frame_stack(历史帧数)，作为 CNN 时间轴长度；CNN 真正的 in_channels 是 num_single_obs(在 Estimator_HIM 内部取 num_proprio_obs)
        self.estimator = Estimator_HIM( num_short_obs=num_short_obs,
                                        num_single_obs=num_single_obs,
                                        num_critic_obs=num_critic_obs,
                                        estimator_hidden_dims=estimator_hidden_dims,
                                        tar_hidden_dims=tar_hidden_dims,
                                        num_est_prob=num_est_prob,
                                        history_len=history_len,
                                        kernel_size=kernel_size,
                                        filter_size=filter_size,
                                        stride_size=stride_size,
                                        lh_output_dim=lh_output_dim,
                                        max_grad_norm=max_grad_norm,
                                        is_privileged_obs=is_privileged_obs)
        
        print(f'Estimator: {self.estimator.estimator}')
        print(f'Source Encoder: {self.estimator.long_history}')
        print(f'Target Encoder: {self.estimator.target_encoder}')

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs_history=None, **kwargs):
        with torch.no_grad():
            short_history = obs_history[:, -self.num_short_obs:]
            estimated_prob, latent = self.estimator(obs_history)  #latent在estimator里面做了归一化
        actor_obs = torch.cat((short_history, estimated_prob, latent), dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history):
        short_history = obs_history[...,-self.num_short_obs:]
        estimated_prob, latent = self.estimator(obs_history)
        actor_obs = torch.cat((short_history, estimated_prob, latent), dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value