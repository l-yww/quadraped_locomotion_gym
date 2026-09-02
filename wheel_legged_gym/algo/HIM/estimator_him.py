import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


""" 
    channel = single_obs_dims
    seq_len = frame stack
"""

""" 
    channel = single_obs_dims
    seq_len = frame stack
"""

class Estimator_HIM(nn.Module):
    def __init__(self,
                 num_short_obs,
                 num_single_obs,
                 num_critic_obs,
                 estimator_hidden_dims=[128, 64],
                 tar_hidden_dims=[256, 128],
                 num_est_prob = 4,
                 history_len = 30,
                 kernel_size=[6, 4],
                 filter_size=[32, 16],
                 stride_size=[3, 2],
                 lh_output_dim = 32,
                 activation='elu',
                 learning_rate=1e-3,
                 num_prototype=16,
                 temperature=3.0,
                 is_privileged_obs=False,
                 max_grad_norm=10,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(Estimator_HIM, self).__init__()
        activation = get_activation(activation)

        self.num_single_obs = num_single_obs
        self.num_short_obs = num_short_obs
        self.num_est_prob = num_est_prob
        self.lh_output_dim = lh_output_dim
        self.num_proprio_obs = num_single_obs
        self.num_critic_obs = num_critic_obs
        self.is_privileged_obs = is_privileged_obs
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm

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
        print(f'Estimator: {self.estimator}')
       
        # Source Encoder
        # define long_history CNN —— 沿时间轴卷积：
        #   obs_history (B, history_len*T=30, num_proprio_obs=50) → view (B, T=30, C=50) → transpose (B, C=50, T=30)
        #   Conv1d 输入 (B, in_channels, length)：num_proprio_obs(50) 当通道，history_len(30) 当时间序列长度
        #   kernel 在 T=30 上滑动，提取时序特征
        long_history_layers = []
        cnn_in_channels = self.num_proprio_obs
        self.in_channels = cnn_in_channels
        self.history_len = history_len
        cnn_output_dim = self.history_len
        for out_channels, kernel_size, stride_size in zip(filter_size, kernel_size, stride_size):
            long_history_layers.append(nn.Conv1d(in_channels=cnn_in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride_size))
            long_history_layers.append(nn.ReLU())
            cnn_output_dim = (cnn_output_dim - kernel_size + stride_size) // stride_size
            cnn_in_channels = out_channels
        cnn_output_dim *= out_channels
        long_history_layers.append(nn.Flatten())
        long_history_layers.append(nn.Linear(cnn_output_dim, 128))
        long_history_layers.append(nn.ELU())
        long_history_layers.append(nn.Linear(128, lh_output_dim))
        self.long_history = nn.Sequential(*long_history_layers)
        print(f'Source Encoder: {self.long_history}')
        
        # Target Encoder
        ### 输入维度是特权观测信息的维度
        if self.is_privileged_obs:
            tar_input_dim = num_critic_obs
        else:
            tar_input_dim = num_single_obs
        tar_layers = []
        tar_layers.append(nn.Linear(tar_input_dim, tar_hidden_dims[0]))
        tar_layers.append(activation)
        for l in range(len(tar_hidden_dims)):
            if l == len(tar_hidden_dims) - 1:
                tar_layers.append(nn.Linear(tar_hidden_dims[l], lh_output_dim))
            else:
                tar_layers.append(nn.Linear(tar_hidden_dims[l], tar_hidden_dims[l + 1]))
                tar_layers.append(activation)
        self.target_encoder = nn.Sequential(*tar_layers)
        print(f'Target Encoder: {self.target_encoder}')

        # Prototype
        self.proto = nn.Embedding(num_prototype, lh_output_dim)
        print(f'Prototype Embedding: {self.proto}')

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

    def forward(self, obs_history):
        short_history = obs_history[...,-self.num_short_obs:]
        estimated_prob = self.estimator(short_history)
        # (B, T*history_len*C) → (B, T, C) → (B, C=num_proprio_obs, T=history_len)，时间轴 T 放到 Conv1d 的 length 维
        z = self.long_history(obs_history.view(-1, self.history_len, self.num_proprio_obs).transpose(1, 2).contiguous())
        z = F.normalize(z, dim=-1, p=2)
        return estimated_prob.detach(), z.detach()

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        # if self.is_privileged_obs:
        #     next_obs = next_critic_obs.detach()[:, -self.num_critic_obs:]
        # else:
        #     next_obs = next_critic_obs.detach()[:, -self.num_critic_obs:-self.num_critic_obs+self.num_single_obs]   #只包含特权obs和本体obs的重叠的部分
        body_obs = next_critic_obs.detach()[:, 3:self.num_single_obs]          # 跳过 commands 的本体状态
        lin_vel = next_critic_obs.detach()[:, -3:]                             # 末尾 lin_vel（3 维）
        next_obs = torch.cat((body_obs, lin_vel), dim=-1)                      # (num_single_obs-3)+3 = num_single_obs
        
        """" target_encoder, long_history 以及 Prototype 部分的更新 [利用 SwAV 对比学习] """
        z_s = self.long_history(obs_history.view(-1, self.history_len, self.num_proprio_obs).transpose(1, 2).contiguous())     # source: 沿时间轴卷积
        z_t = self.target_encoder(next_obs)                                                       # target

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

        """" Estimator 部分的更新 [利用特权观测信息监督] """
        short_history = obs_history[...,-self.num_short_obs:]
        estimated_prob = self.estimator(short_history)                  # estimator
        prob_real = next_critic_obs[:, -self.num_est_prob:].detach()    # next_obs 

        estimation_loss = F.mse_loss(estimated_prob, prob_real)

        """ [整体更新] """
        loss = swap_loss + estimation_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()

@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None