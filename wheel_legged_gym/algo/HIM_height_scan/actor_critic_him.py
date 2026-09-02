import torch
import torch.nn as nn
from torch.distributions import Normal

from ..utils import get_activation
from .estimator_him import Estimator_HIM


class ActorCritic_HIM(nn.Module):
    """HIM policy using full proprioception-height-scan observation frames.

    Actor input is exactly ``latest_frame | predicted_base_height_and_velocity |
    history_dynamic_latent``.  Height scans are already included in every frame;
    there is deliberately no separate height encoder.
    """

    is_recurrent = False

    def __init__(
        self,
        num_short_obs,
        num_single_obs,
        num_est_prob,
        num_critic_obs,
        num_actions,
        num_proprio_per_frame=None,
        height_map_shape=None,
        critic_obs_layout=None,
        actor_height_in_obs=True,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        estimator_hidden_dims=(256, 256),
        tar_hidden_dims=(256, 256),
        history_len=5,
        lh_output_dim=32,
        activation="elu",
        init_noise_std=1.0,
        max_grad_norm=10.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic_HIM.__init__ ignored unused arguments: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        if num_short_obs != num_single_obs:
            raise ValueError(
                "The HIM height-scan Actor uses exactly the latest full frame; "
                "num_short_obs must equal num_single_obs."
            )
        if num_est_prob != 4:
            raise ValueError(
                "The HIM height-scan Actor expects [base_height, vx, vy, vz]."
            )

        self.num_short_obs = num_single_obs
        self.num_single_obs = num_single_obs
        self.num_est_prob = num_est_prob
        self.num_proprio_per_frame = num_single_obs
        # Kept for PPO symmetry helpers: this is the full frame, including map.
        self.num_proprio_obs = num_single_obs
        self.num_obs_history = history_len * num_single_obs
        self.history_len = history_len
        self.lh_output_dim = lh_output_dim
        self.actor_height_in_obs = actor_height_in_obs
        self.height_map_shape = tuple(height_map_shape) if height_map_shape is not None else None
        self.critic_obs_layout = critic_obs_layout
        self.observe_heightmap = False
        self.num_height_scan_input = 0
        self.height_encoder = None

        if self.actor_height_in_obs and self.height_map_shape is None:
            raise ValueError("height_map_shape is required when height scans are embedded in actor frames")

        activation_module = get_activation(activation)
        actor_input_dim = num_single_obs + num_est_prob + lh_output_dim
        self.actor = self._make_mlp(
            actor_input_dim, actor_hidden_dims, num_actions, activation_module
        )
        self.critic = self._make_mlp(
            num_critic_obs, critic_hidden_dims, 1, activation_module
        )
        self.estimator = Estimator_HIM(
            num_single_obs=num_single_obs,
            num_critic_obs=num_critic_obs,
            history_len=history_len,
            estimator_hidden_dims=estimator_hidden_dims,
            tar_hidden_dims=tar_hidden_dims,
            num_est_prob=num_est_prob,
            lh_output_dim=lh_output_dim,
            activation=activation,
            max_grad_norm=max_grad_norm,
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def _make_mlp(input_dim, hidden_dims, output_dim, activation):
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), activation))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

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

    def _actor_input(self, obs_history):
        if obs_history.shape[-1] != self.num_obs_history:
            raise ValueError(
                f"Expected {self.num_obs_history} stacked actor values, got {obs_history.shape[-1]}"
            )
        latest_frame = obs_history[..., -self.num_single_obs:]
        with torch.no_grad():
            estimated_state, dynamic_latent = self.estimator(obs_history)
        return torch.cat((latest_frame, estimated_state, dynamic_latent), dim=-1)

    def update_distribution(self, obs_history):
        actor_input = self._actor_input(obs_history)
        action_mean = self.actor(actor_input)
        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history):
        # Estimator outputs are detached by Estimator_HIM, but the Actor remains
        # differentiable here for PPO symmetry loss.
        actor_input = self._actor_input(obs_history)
        return self.actor(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
