import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utils import get_activation


class Estimator_HIM(nn.Module):
    """HIM encoder for a history of complete proprioception and height scans.

    The source encoder receives ``history_len`` full actor frames.  Every frame
    contains proprioception followed by the cached height scan. It predicts
    ``[base_height, vx, vy, vz]`` and a dynamic latent. The latent is aligned with a
    target representation learned from the privileged critic observation.
    """

    def __init__(
        self,
        num_single_obs,
        num_critic_obs,
        history_len,
        estimator_hidden_dims=(256, 256),
        tar_hidden_dims=(256, 256),
        num_est_prob=4,
        lh_output_dim=32,
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        num_prototype=64,
        temperature=3.0,
    ):
        super().__init__()
        if num_est_prob != 4:
            raise ValueError(
                "The HIM height-scan estimator predicts [base_height, vx, vy, vz], "
                "so num_est_prob must be 4."
            )

        self.num_single_obs = num_single_obs
        self.num_critic_obs = num_critic_obs
        self.history_len = history_len
        self.num_est_prob = num_est_prob
        self.lh_output_dim = lh_output_dim
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        activation_module = get_activation(activation)
        history_dim = history_len * num_single_obs
        self.encoder = self._make_mlp(
            history_dim,
            estimator_hidden_dims,
            num_est_prob + lh_output_dim,
            activation_module,
        )
        self.target_encoder = self._make_mlp(
            num_critic_obs,
            tar_hidden_dims,
            lh_output_dim,
            activation_module,
        )
        self.proto = nn.Embedding(num_prototype, lh_output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

    @staticmethod
    def _make_mlp(input_dim, hidden_dims, output_dim, activation):
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), activation))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def _check_history(self, obs_history):
        expected_dim = self.history_len * self.num_single_obs
        if obs_history.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} history values "
                f"({self.history_len} x {self.num_single_obs}), got {obs_history.shape[-1]}"
            )

    def forward(self, obs_history):
        self._check_history(obs_history)
        source = self.encoder(obs_history)
        estimated_state, latent = source[..., :self.num_est_prob], source[..., self.num_est_prob:]
        return estimated_state.detach(), F.normalize(latent, dim=-1, p=2).detach()

    def update(self, obs_history, next_critic_obs, lr=None):
        self._check_history(obs_history)
        if next_critic_obs.shape[-1] != self.num_critic_obs:
            raise ValueError(
                f"Expected {self.num_critic_obs} critic values, got {next_critic_obs.shape[-1]}"
            )
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        source = self.encoder(obs_history)
        predicted_state = source[..., :self.num_est_prob]
        source_latent = F.normalize(source[..., self.num_est_prob:], dim=-1, p=2)
        target_latent = F.normalize(self.target_encoder(next_critic_obs.detach()), dim=-1, p=2)

        with torch.no_grad():
            self.proto.weight.copy_(F.normalize(self.proto.weight, dim=-1, p=2))
            source_assignments = sinkhorn(source_latent @ self.proto.weight.T)
            target_assignments = sinkhorn(target_latent @ self.proto.weight.T)

        source_log_prob = F.log_softmax(
            (source_latent @ self.proto.weight.T) / self.temperature, dim=-1
        )
        target_log_prob = F.log_softmax(
            (target_latent @ self.proto.weight.T) / self.temperature, dim=-1
        )
        swap_loss = -0.5 * (
            source_assignments * target_log_prob
            + target_assignments * source_log_prob
        ).mean()

        # Critic observations always end in [base_height, vx, vy, vz].
        target_state = next_critic_obs[..., -self.num_est_prob:].detach()
        estimation_loss = F.mse_loss(predicted_state, target_state)
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    assignments = torch.exp(out / eps).T
    num_prototypes, batch_size = assignments.shape
    assignments /= assignments.sum()
    for _ in range(iters):
        assignments /= assignments.sum(dim=1, keepdim=True)
        assignments /= num_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True)
        assignments /= batch_size
    return (assignments * batch_size).T
