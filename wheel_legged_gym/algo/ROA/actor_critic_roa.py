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
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .state_estimator import StateHistoryEncoder,MLP

class ActorCritic_ROA(nn.Module):
    def __init__(self,  num_actor_obs, # single
                        num_critic_obs, # single
                        num_actions,
                        actor_input_stack,
                        frame_stack,
                        c_frame_stack,
                        num_latent,

                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        priv_encoder_dims=[256, 128],
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("ActorCritic_ROA.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_ROA, self).__init__()

        # mlp_input_dim_a = num_actor_obs * actor_input_stack + num_latent
        # mlp_input_dim_c = num_critic_obs * c_frame_stack
        mlp_input_dim_a = num_actor_obs + num_latent # input for actor
        mlp_input_dim_c = num_critic_obs  # input for critic
        input_priv_encoder = num_critic_obs * c_frame_stack
        input_hist_encoder = num_actor_obs
        
        # for used
        self.num_actor_obs = num_actor_obs 
        self.num_critic_obs = num_critic_obs
        self.frame_stack = frame_stack
        self.c_frame_stack = c_frame_stack
        self.num_latent = num_latent
        # <><><><> Encoder <><><><>
        self.priv_encoder = MLP(input_priv_encoder, num_latent, priv_encoder_dims, activation)
        self.history_encoder = StateHistoryEncoder(input_hist_encoder, frame_stack, num_latent, activation)

        print(f"Priv Encoder MLP: {self.priv_encoder}")
        print(f"History Encoder MLP: {self.history_encoder}")

        # =========== policy function ===========
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

        # =========== valye function ===========
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

    # ！本体 + 历史
    def student_forward(self, obs_prop, obs_history):
        latent = self.history_encoder(obs_history)
        actions = self.actor(torch.cat([obs_prop, latent], dim=1))
        return actions


    # ！本体 + 特权
    def teacher_forward(self, obs_prop, obs_priv):
        latent = self.priv_encoder(obs_priv)
        actions = self.actor(torch.cat([obs_prop, latent], dim=1))
        return actions
    

    # parmas: 单帧prop obs
    def ts_forward(self, obs_prop, obs_priv, obs_history, hist_encoding=False):
        if hist_encoding:
            latent = self.history_encoder(obs_history)
        else:
            latent = self.priv_encoder(obs_priv)
        actions = self.actor(torch.cat([obs_prop, latent], dim=1))
        return actions

    # def critic_forward(self, obs_prop, obs_priv, obs_history):
    #     prop_and_priv = torch.cat([obs_prop, obs_priv], dim=1)
    #     output = self.critic(prop_and_priv)
    #     return output
    
    #v  
    def update_distribution(self, obs_prop, obs_priv, obs_history, hist_encoding):
        mean = self.ts_forward(obs_prop, obs_priv, obs_history, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)
    #v  
    def act(self, obs_prop, obs_priv, obs_history, hist_encoding, **kwargs):  
        self.update_distribution(obs_prop, obs_priv, obs_history, hist_encoding)
        return self.distribution.sample()
    #v  
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    #v
    def act_inference(self, obs_prop, obs_priv, obs_history, hist_encoding=False):
        actions_mean = self.ts_forward(obs_prop, obs_priv, obs_history, hist_encoding)
        return actions_mean
    #v
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value








