import torch.nn as nn
import torch
import numpy as np
from .utils import get_activation, check_cnnoutput
from torch.distributions import Normal
from torch.nn import functional
from torchsummary import summary

class Estimator(nn.Module):
    def __init__(
        self,
        num_normal_obs,
        num_est_prob=3,
        activation="elu",
        estimator_hidden_dims=[512, 256, 256],
        device = "cuda"
    ):
        super().__init__()
        self.device = device
        self.num_normal_obs = num_normal_obs

        self.mlp_est = MLP(
            num_normal_obs,
            num_est_prob,
            estimator_hidden_dims,
            activation,
        )


    def forward(self, obs_history):
        output= self.mlp_est(obs_history)
        # if output.shape[1] == 6:
        #     output[:, -2:] = torch.sigmoid(output[:, -2:]) # 如果有6dim输出,最后两维是概率，需要sigmoid
        return output

    def encoder_inference(self, obs_history):
        return self.mlp_est(obs_history)

    def loss_fn(self, estimator_latent, critic_obs, num_est_prob):
        """ 全都认为是回归问题 """
        prob_real = critic_obs[:, -num_est_prob:]
        est_loss = functional.mse_loss(estimator_latent, prob_real, reduction="none").mean(-1)

        abs_diff = (estimator_latent - prob_real).abs().mean(dim=0)
        diff_values = abs_diff.cpu().detach().numpy()
        diff_names = ["v_avg_diff_x", "v_avg_diff_y", "v_avg_diff_z",
                  "base_height_diff", "contact_prob_left_diff", "contact_prob_right_diff"]

        diff_dict = {name: val for name, val in zip(diff_names, diff_values)}

        return {
            "loss": est_loss.mean(),  # 标量
            **diff_dict
        }


# zsy remove params: num_history
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims, activation):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.encoder = nn.Sequential(*module)

    def forward(self, obs_history):
        # RS_obs_history = obs_history.reshape(obs_history.shape[0],-1)
        return self.encoder(obs_history)


