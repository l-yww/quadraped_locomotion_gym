import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .state_estimator import Estimator, MLP
class ActorCritic(nn.Module):
    def __init__(self,  teacher,
                        dagger_on,
                        num_obs_history,
                        num_actor_obs,
                        num_est_prob,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        estimator_hidden_dims=[256, 128, 256, 64],
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()
        self.dagger_on = dagger_on
        self.teacher = teacher

        ### NOTE: estimator mlp
        self.estimator = Estimator(num_obs_history, num_est_prob, activation, estimator_hidden_dims)


        self.num_est_prob = num_est_prob
        if not self.dagger_on and teacher:
            mlp_input_dim_a = num_actor_obs # priv inputs
        else:
            mlp_input_dim_a = num_actor_obs  + self.num_est_prob

        mlp_input_dim_c = num_critic_obs

        self.actor_input = mlp_input_dim_a
        self.encoder_input = num_obs_history
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
        print(f"estimator MLP: {self.estimator}")

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

    def update_distribution(self, obs_buff):
        mean = self.actor(obs_buff)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs_input, obs_history, teacher, **kwargs): 
        if not self.dagger_on and teacher:
            obs_buff = obs_input
        else:
            self.estimated_prob = self.estimator(obs_history)
            obs_buff = torch.cat([obs_input, self.estimated_prob], dim=-1)
        self.update_distribution(obs_buff)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # 推理
    def act_inference(self, obs_actor, obs_history):
        if not self.dagger_on and self.teacher:
            obs_buff = obs_actor
        else:
            self.estimated_prob = self.estimator.forward(obs_history)
            obs_buff = torch.cat([obs_actor, self.estimated_prob], dim=-1)
        action_mean = self.actor.forward(obs_buff)
        return action_mean
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value