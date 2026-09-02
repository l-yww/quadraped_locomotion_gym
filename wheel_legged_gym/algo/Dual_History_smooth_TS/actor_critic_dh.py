import logging
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from .state_estimator import estimator_TS
from .state_estimator import Conv2dHeadModel

logger = logging.getLogger(__name__)
""" 
    channel = single_obs_dims
    seq_len = frame stack
"""

class ActorCritic_DH_Smooth_TS(nn.Module):
    def __init__(self,  num_short_obs,
                        num_single_obs,
                        num_est_prob,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        estimator_hidden_dims=[128, 64],
                        # in_channels = 66,
                        history_len = 66,
                        kernel_size=[6, 4],
                        filter_size=[32, 16],
                        stride_size=[3, 2],
                        lh_output_dim=64,
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        encoder_class_name = "Conv2dHeadModel",
                        encoder_output_dim=32,
                        encoder_kwargs = None,
                        **kwargs):
        if kwargs:
            logger.warning("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_DH_Smooth_TS, self).__init__()

        self.num_short_obs = num_short_obs
        self.num_est_prob = num_est_prob
        self.lh_output_dim = lh_output_dim
        self.num_proprio_obs = num_single_obs
        self.encoder_output_dim = encoder_output_dim
        mlp_input_dim_a = num_short_obs + self.num_est_prob + self.lh_output_dim+self.encoder_output_dim
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
        
        # Estimator
        mlp_input_dim_e = num_short_obs
        est_layers = []
        est_layers.append(nn.Linear(mlp_input_dim_e, estimator_hidden_dims[0]))
        est_layers.append(activation)
        for l in range(len(estimator_hidden_dims)):
            if l == len(estimator_hidden_dims) - 1:
                est_layers.append(nn.Linear(estimator_hidden_dims[l], num_est_prob))
            else:
                est_layers.append(nn.Linear(estimator_hidden_dims[l], estimator_hidden_dims[l + 1]))
                est_layers.append(activation)
        self.estimator = nn.Sequential(*est_layers)
        # self.estimator = estimator(num_short_obs, num_est_prob, activation, estimator_hidden_dims)
        # print(self.estimator)

        #define long_history CNN
        long_history_layers = []
        in_channels = self.num_proprio_obs
        self.history_len = history_len
        cnn_output_dim = self.history_len
        for out_channels, kernel_size, stride_size in zip(filter_size, kernel_size, stride_size):
            long_history_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride_size))
            long_history_layers.append(nn.ReLU())
            cnn_output_dim = (cnn_output_dim - kernel_size + stride_size) // stride_size
            in_channels = out_channels
        cnn_output_dim *= out_channels
        long_history_layers.append(nn.Flatten())
        long_history_layers.append(nn.Linear(cnn_output_dim, 128))
        long_history_layers.append(nn.ELU())
        long_history_layers.append(nn.Linear(128, lh_output_dim))
        self.long_history = nn.Sequential(*long_history_layers)

        encoder_class=eval(encoder_class_name)
        self.encoder = encoder_class(**encoder_kwargs)

        # print(f"Actor MLP: {self.actor}")
        # print(f"Critic MLP: {self.critic}")
        # print(f"Estimator MLP: {self.estimator}")
        # print(f"long_history CNN: {self.long_history}")
        # print(f"Encoder: {self.encoder}")

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Estimator MLP: {self.estimator}")
        print(f"long_history CNN: {self.long_history}")
        print(f"Encoder: {self.encoder}")

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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, forward_depth, **kwargs):
        short_history = observations[...,-self.num_short_obs:]
        estimated_prob = self.estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.num_proprio_obs, self.history_len))
        if forward_depth is not None:
            encoder_output = self.encoder(forward_depth)
            actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history, encoder_output),dim=-1)
        else:
            # When forward_depth is not provided, use zero tensor as encoder output
            encoder_output = torch.zeros(observations.shape[0], self.encoder_output_dim, device=observations.device)
            actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history, encoder_output),dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, forward_depth):
        short_history = observations[...,-self.num_short_obs:]
        estimated_prob = self.estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.num_proprio_obs, self.history_len))
        if forward_depth is not None:
            encoder_output = self.encoder(forward_depth)
            actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history, encoder_output),dim=-1)
        else:
            # When forward_depth is not provided, use zero tensor as encoder output
            encoder_output = torch.zeros(observations.shape[0], self.encoder_output_dim, device=observations.device)
            actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history, encoder_output),dim=-1)
        action_mean = self.actor(actor_obs)
        return action_mean
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
        
    def est_loss_fn(self, estimator_latent, critic_obs, num_est_prob):
        prob_real = critic_obs[:, -num_est_prob:]
        loss = F.mse_loss(estimator_latent, prob_real)
        return loss