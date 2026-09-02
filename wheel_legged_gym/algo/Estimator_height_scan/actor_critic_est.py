import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .state_estimator import estimator, MLP
class ActorCritic_Estimator(nn.Module):
    def __init__(self,  num_obs_history,
                        num_obs_input,
                        num_est_prob,
                        num_critic_obs,
                        num_actions,
                        num_height_scan_input = 77,         # height
                        num_height_scan_output = 16,        # height
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        estimator_hidden_dims=[256, 128, 256, 64],
                        height_scan_encoder_dims = [128,64], # height
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_Estimator, self).__init__()

        self.num_height_scan_input = num_height_scan_input
        self.num_height_scan_output = num_height_scan_output
        # height scan encoder for inputs 
        self.height_scan_encoder = MLP(num_height_scan_input, num_height_scan_output, height_scan_encoder_dims, activation)
        self.estimator = estimator(num_obs_history, num_est_prob, activation, estimator_hidden_dims)
        print(self.estimator)
        self.num_est_prob = num_est_prob
        mlp_input_dim_a = num_obs_input  + self.num_est_prob + self.num_height_scan_output*5
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
        print(f"height scan MLP: {self.height_scan_encoder}")

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

    def act(self, obs_input, obs_history, obs_height, **kwargs): 
        height_latent = self.height_scan_encoder.forward(obs_height)
        height_latent_multi = torch.cat([height_latent , height_latent, height_latent, height_latent, height_latent], dim=-1) # height_latent x 5
        self.estimated_prob = self.estimator(obs_history)
        obs_buff = torch.cat([obs_input, self.estimated_prob, height_latent_multi], dim=-1)
        self.update_distribution(obs_buff)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_input, obs_history, obs_height):
        height_latent = self.height_scan_encoder.forward(obs_height)
        height_latent_multi = torch.cat([height_latent , height_latent, height_latent, height_latent, height_latent], dim=-1) # height_latent x 5
        self.estimated_prob = self.estimator.forward(obs_history)
        obs_buff = torch.cat([obs_input, self.estimated_prob, height_latent_multi], dim=-1)
        action_mean = self.actor.forward(obs_buff)
        return action_mean
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value