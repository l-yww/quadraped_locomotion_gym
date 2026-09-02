import torch
import torch.nn as nn
from torch import autograd


class AMPDiscriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 amp_reward_coef,
                 hidden_layer_sizes,
                 device,
                 task_reward_lerp=0.0,
                 style_reward_function="quad_mapping"):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        self.style_reward_function = style_reward_function
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None, style_reward_normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            if self.style_reward_function == "quad_mapping":
                style_reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            elif self.style_reward_function == "log_mapping":
                style_reward = -torch.log(torch.maximum(1 - 1 / (1 + torch.exp(-d)), torch.tensor(0.0001, device=self.device)))
            elif self.style_reward_function == "wasserstein_mapping":
                if style_reward_normalizer is not None:
                    style_reward = style_reward_normalizer.normalize(d.clone())
                    style_reward_normalizer.update(d)
                else:
                    style_reward = d
                style_reward = self.amp_reward_coef * style_reward
            else:
                raise ValueError("Unexpected style reward mapping specified")
            amp_reward = style_reward
            if self.task_reward_lerp >= 0:
                total_reward = self._lerp_reward(amp_reward, task_reward.unsqueeze(-1))
            else:
                total_reward = task_reward.unsqueeze(-1) + amp_reward
            self.train()
        return total_reward.squeeze(-1), amp_reward

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
