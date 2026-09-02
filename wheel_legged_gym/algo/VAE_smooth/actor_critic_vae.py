import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import (
    get_activation,
    MultivariateGaussianDiagonalCovariance,
    init_orhtogonal,
)
from torch.nn.modules import rnn
class ActorCritic_VAE_Smooth(nn.Module):
    def __init__(self,  num_obs_history,
                        num_obs_input,
                        num_est_prob,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        encoder_hidden_dims=[256, 128, 64],
                        decoder_hidden_dims=[64, 128, 256],
                        vae_latent_dims=[3, 1, 16],
                        activation='elu',
                        init_noise_std=1.0,
                        cv=0.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_VAE_Smooth, self).__init__()
        activation = get_activation(activation)

        self.total_latent_dim = sum(vae_latent_dims)
        self.num_est_prob = num_est_prob
        self.num_obs_input = num_obs_input
        history_input_dim = num_obs_history
        mlp_input_dim_a = num_obs_input + self.total_latent_dim
        mlp_input_dim_c = num_critic_obs

        self.cv = nn.Parameter(torch.tensor(cv))

        print(f"vae_latent_dims: {vae_latent_dims}")
        print(f"cv: {self.cv}")
        print(f"num_obs_history: {num_obs_history}")
        print(f"num_obs_input: {num_obs_input}")
        print(f"num_critic_obs: {num_critic_obs}")

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

        # VAE Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_input_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)-1):  
            encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
            encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # VAE latent variables
        self.latent_dims = vae_latent_dims
        self.encode_mean_1 = nn.Linear(encoder_hidden_dims[-1], vae_latent_dims[0])
        self.encode_logvar_1 = nn.Linear(encoder_hidden_dims[-1], vae_latent_dims[0])
        self.encode_mean_2 = nn.Linear(encoder_hidden_dims[-1], vae_latent_dims[1])
        self.encode_logvar_2 = nn.Linear(encoder_hidden_dims[-1], vae_latent_dims[1])
        self.encode_mean_3 = nn.Linear(encoder_hidden_dims[-1], vae_latent_dims[2])
        self.encode_logvar_3 = nn.Linear(encoder_hidden_dims[-1], vae_latent_dims[2])

        # VAE Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(self.total_latent_dim, decoder_hidden_dims[0]))  
        decoder_layers.append(activation)
        for l in range(len(decoder_hidden_dims)-1):  
            decoder_layers.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
            decoder_layers.append(activation)
        decoder_layers.append(nn.Linear(decoder_hidden_dims[-1], num_obs_input)) 
        self.decoder = nn.Sequential(*decoder_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"VAE Encoder: {self.encoder}")
        print(f"VAE Decoder: {self.decoder}")
        
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
    
    def reparameterize(self, mean, logvar, cv):
        """实现重参数化技巧"""
        std = torch.exp(0.5 * logvar) * (1 - torch.tanh(cv)) 
        eps = torch.randn_like(std)
        return mean + eps * std

    def vae_forward(self, history_obs):
        """VAE的前向传播"""
        hidden = self.encoder(history_obs)
        mean1 = self.encode_mean_1(hidden)
        logvar1 = self.encode_logvar_1(hidden)
        mean2 = self.encode_mean_2(hidden)
        logvar2 = self.encode_logvar_2(hidden)
        mean3 = self.encode_mean_3(hidden)
        logvar3 = self.encode_logvar_3(hidden)

        z1 = self.reparameterize(mean1, logvar1, self.cv)
        z2 = self.reparameterize(mean2, logvar2, self.cv)
        z3 = self.reparameterize(mean3, logvar3, self.cv)
        z = torch.cat([z1, z2, z3], dim=-1)

        means = (mean1, mean2, mean3)
        logvars = (logvar1, logvar2, logvar3)
        
        return self.decoder(z), means, logvars, z
    
    @property
    def action_mean(self):
        if self.distribution is None:
            return None
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            return None
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            return None
        return self.distribution.entropy().sum(dim=-1)


    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs_history, **kwargs):
        num_obs_input = self.num_obs_input
        obs_input = obs_history[:, -num_obs_input:]
        _, _, _, z = self.vae_forward(obs_history)
        combined_obs = torch.cat([obs_input, z], dim=-1)
        self.update_distribution(combined_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    def act_inference(self, obs_history):
        num_obs_input = self.num_obs_input
        obs_input = obs_history[:, -num_obs_input:]
        _, _, _, z = self.vae_forward(obs_history)
        combined_obs = torch.cat([obs_input, z], dim=-1)
        action_mean = self.actor(combined_obs)
        return action_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
