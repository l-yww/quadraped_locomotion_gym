"""PPO+AMP actor-critic with a direct forward height scan input."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

def get_activation(activation):
    if activation == 'elu':
        return nn.ELU()
    if activation == 'selu':
        return nn.SELU()
    if activation in ('relu', 'crelu'):
        return nn.ReLU()
    if activation == 'lrelu':
        return nn.LeakyReLU()
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'sigmoid':
        return nn.Sigmoid()
    raise ValueError(f'Unsupported activation: {activation}')


class Estimator_AMP_HeightScan(nn.Module):
    """Supervised short-history state estimator for the height-scan policy."""

    def __init__(self, num_short_obs, num_est_prob,
                 estimator_hidden_dims=(128, 64), learning_rate=1e-3,
                 max_grad_norm=10, activation='elu'):
        super().__init__()
        layers = [nn.Linear(num_short_obs, estimator_hidden_dims[0]), get_activation(activation)]
        for index in range(len(estimator_hidden_dims) - 1):
            layers.extend([
                nn.Linear(estimator_hidden_dims[index], estimator_hidden_dims[index + 1]),
                get_activation(activation),
            ])
        layers.append(nn.Linear(estimator_hidden_dims[-1], num_est_prob))
        self.estimator = nn.Sequential(*layers)
        self.num_short_obs = num_short_obs
        self.num_est_prob = num_est_prob
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs_history):
        return self.estimator(obs_history[..., -self.num_short_obs:])

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        estimate = self.forward(obs_history)
        target = next_critic_obs[:, -self.num_est_prob:].detach()
        loss = F.mse_loss(estimate, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item(), 0.0


class ActorCritic_AMP_HeightScan(nn.Module):
    """PPO+AMP policy that directly consumes the forward height scan.

    The former HIM contrastive latent branch is intentionally not used. The
    actor consumes the short observation history, including height scans, and
    a supervised estimate of the privileged state.
    """

    is_recurrent = False

    def __init__(self, num_short_obs, num_single_obs, num_est_prob, num_critic_obs, num_actions,
                 actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
                 estimator_hidden_dims=(128, 64), activation='elu', init_noise_std=1.0,
                 max_grad_norm=10, learning_rate=1e-3, **kwargs):
        super().__init__()
        if kwargs:
            print('ActorCritic_AMP_HeightScan.__init__ ignored arguments: '
                  + str(list(kwargs.keys())))

        self.num_short_obs = num_short_obs
        self.num_est_prob = num_est_prob
        self.num_proprio_obs = num_single_obs

        actor_hidden_dims = list(actor_hidden_dims)
        actor_layers = [
            nn.Linear(num_short_obs + num_est_prob, actor_hidden_dims[0]),
            get_activation(activation),
        ]
        for index in range(len(actor_hidden_dims)):
            if index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[index], num_actions))
            else:
                actor_layers.extend([
                    nn.Linear(actor_hidden_dims[index], actor_hidden_dims[index + 1]),
                    get_activation(activation),
                ])
        self.actor = nn.Sequential(*actor_layers)

        critic_hidden_dims = list(critic_hidden_dims)
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), get_activation(activation)]
        for index in range(len(critic_hidden_dims)):
            if index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[index], 1))
            else:
                critic_layers.extend([
                    nn.Linear(critic_hidden_dims[index], critic_hidden_dims[index + 1]),
                    get_activation(activation),
                ])
        self.critic = nn.Sequential(*critic_layers)
        self.estimator = Estimator_AMP_HeightScan(
            num_short_obs=num_short_obs,
            estimator_hidden_dims=estimator_hidden_dims,
            num_est_prob=num_est_prob,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            activation=activation,
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
        print('[ActorCritic_AMP_HeightScan] no-contrast: '
              f'actor_input_dim={num_short_obs + num_est_prob}')

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs_history=None, **kwargs):
        with torch.no_grad():
            short_history = obs_history[:, -self.num_short_obs:]
            estimated_prob = self.estimator(obs_history)
        self.update_distribution(torch.cat((short_history, estimated_prob), dim=-1))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history):
        short_history = obs_history[..., -self.num_short_obs:]
        estimated_prob = self.estimator(obs_history)
        return self.actor(torch.cat((short_history, estimated_prob), dim=-1))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
