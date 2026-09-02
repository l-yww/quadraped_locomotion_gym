import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from .state_estimator import estimator

""" 
    channel = single_obs_dims
    seq_len = frame stack
"""

class ActorCritic_DH_Smooth_Mix(nn.Module):
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
                        leg_control_head_hidden_dims=[64, 32],
                        arm_control_head_hidden_dims=[64, 32],
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_DH_Smooth_Mix, self).__init__()

        self.num_arm_actions = 6
        self.num_leg_actions = num_actions - self.num_arm_actions

        self.num_short_obs = num_short_obs
        self.num_est_prob = num_est_prob
        self.lh_output_dim = lh_output_dim
        self.num_proprio_obs = num_single_obs
        mlp_input_dim_a = num_short_obs + self.num_est_prob + self.lh_output_dim
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_backbone_layers = []
        actor_backbone_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_backbone_layers.append(activation)
        for l in range(len(actor_hidden_dims) - 1):
            actor_backbone_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_backbone_layers.append(activation)
        self.actor_backbone = nn.Sequential(*actor_backbone_layers)

        actor_arm_layers = []
        actor_arm_layers.append(nn.Linear(actor_hidden_dims[-1], arm_control_head_hidden_dims[0]))
        actor_arm_layers.append(activation)
        for l in range(len(arm_control_head_hidden_dims) - 1):
            actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
            actor_arm_layers.append(activation)

        actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[-1], self.num_arm_actions))
        self.actor_arm_head = nn.Sequential(*actor_arm_layers)
        
        actor_leg_layers = []
        actor_leg_layers.append(nn.Linear(actor_hidden_dims[-1], leg_control_head_hidden_dims[0]))
        actor_leg_layers.append(activation)
        for l in range(len(leg_control_head_hidden_dims) - 1):
            actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
            actor_leg_layers.append(activation)
        actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[-1], self.num_leg_actions))
        self.actor_leg_head = nn.Sequential(*actor_leg_layers)


        # Value function
        critic_backbone_layers = []
        critic_backbone_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_backbone_layers.append(activation)
        for l in range(len(critic_hidden_dims) - 1):
            critic_backbone_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
            critic_backbone_layers.append(activation)
        self.critic_backbone = nn.Sequential(*critic_backbone_layers)

        critic_arm_layers = []
        critic_arm_layers.append(nn.Linear(critic_hidden_dims[-1], arm_control_head_hidden_dims[0]))
        critic_arm_layers.append(activation)
        for l in range(len(arm_control_head_hidden_dims) - 1):
            critic_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
            critic_arm_layers.append(activation)
        critic_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[-1], 1))
        self.critic_arm_head = nn.Sequential(*critic_arm_layers)

        critic_leg_layers = []
        critic_leg_layers.append(nn.Linear(critic_hidden_dims[-1], leg_control_head_hidden_dims[0]))
        critic_leg_layers.append(activation)
        for l in range(len(leg_control_head_hidden_dims) - 1):
            critic_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
            critic_leg_layers.append(activation)
        critic_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[-1], 1))
        self.critic_leg_head = nn.Sequential(*critic_leg_layers)
        
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

        print(f"Actor Backbone: {self.actor_backbone}")
        print(f"Actor Arm Head: {self.actor_arm_head}")
        print(f"Actor Leg Head: {self.actor_leg_head}")
        print(f"Critic Backbone: {self.critic_backbone}")
        print(f"Critic Arm Head: {self.critic_arm_head}")
        print(f"Critic Leg Head: {self.critic_leg_head}")
        print(f"Estimator MLP: {self.estimator}")
        print(f"long_history CNN: {self.long_history}")

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
        entropy = self.distribution.entropy()
        arm_entropy_sum = entropy[:, :self.num_arm_actions].sum(dim=-1, keepdim=True)
        leg_entropy_sum = entropy[:, self.num_arm_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([arm_entropy_sum, leg_entropy_sum], dim=-1)

    def update_distribution(self, observations):
        backbone_output = self.actor_backbone(observations)
        arm_mean = self.actor_arm_head(backbone_output)
        leg_mean = self.actor_leg_head(backbone_output)
        mean = torch.cat([arm_mean, leg_mean], dim=-1)
        std = self.std.unsqueeze(0).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        short_history = observations[...,-self.num_short_obs:]
        estimated_prob = self.estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.num_proprio_obs, self.history_len))
        actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history),dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):

        log_prob = self.distribution.log_prob(actions)
        arm_log_prob = log_prob[:, :self.num_arm_actions].sum(dim=-1, keepdim=True)
        leg_log_prob = log_prob[:, self.num_arm_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([arm_log_prob, leg_log_prob], dim=-1)

    def act_inference(self, observations):
        short_history = observations[...,-self.num_short_obs:]
        estimated_prob = self.estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.num_proprio_obs, self.history_len))
        actor_obs = torch.cat((short_history, estimated_prob, compressed_long_history),dim=-1)
        backbone_output = self.actor_backbone(actor_obs)
        arm_mean = self.actor_arm_head(backbone_output)
        leg_mean = self.actor_leg_head(backbone_output)
        action_mean = torch.cat([arm_mean, leg_mean], dim=-1)
        return action_mean
    
    def evaluate(self, critic_observations, **kwargs):
        backbone_output = self.critic_backbone(critic_observations)
        arm_value = self.critic_arm_head(backbone_output)
        leg_value = self.critic_leg_head(backbone_output)
        value = torch.cat([arm_value, leg_value], dim=-1)
        return value
        
    def est_loss_fn(self, estimator_latent, critic_obs, num_est_prob):
        prob_real = critic_obs[:, -num_est_prob:]
        loss = F.mse_loss(estimator_latent, prob_real)
        return loss