# ActorCritic_HIM 的前视高程图变体。
#
# 背景: 给 actor 加前视高程图观测(121 维, base_link 中心 x 1.0~2.0 前方)后,
#       actor 每帧 obs 维度 num_single_obs 从 45(纯本体) 变成 166(本体+高程图)。
#
# 问题: Estimator_HIM 把 num_single_obs 同时用于两件事:
#         (1) long_history CNN 的 in_channels —— 应当 = 166(历史帧确实含高程图);
#         (2) target_encoder 的输入维度 + update() 里 body_obs 的切片
#             `next_critic_obs[:, 3:num_single_obs]` —— 应当仍是 45(纯本体),
#             否则会把特权量(摩擦/电机/近身高程图)塞进 SwAV 对比目标, 污染 latent。
#       这里把 (2) 单独抽出 num_body_dim, (1) 继续用 num_single_obs。
#
# 不改 Estimator_HIM / ActorCritic_HIM 主类, 全部通过子类实现, 旧 task 不受影响。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .actor_critic_him import ActorCritic_HIM
from .estimator_him import Estimator_HIM, sinkhorn, get_activation


class Estimator_HIM_Heightmap(Estimator_HIM):
    """Estimator_HIM 变体: 区分历史帧维度(num_single_obs, 含高程图) 与本体维度(num_body_dim)。

    - long_history CNN: 仍用 num_single_obs 当通道(历史帧含高程图, 由父类 forward 处理, 不重写)。
    - target_encoder: 只编码本体状态 critic_obs[3:num_body_dim] + lin_vel, 维度 = num_body_dim。
      num_body_dim 缺省 = num_single_obs(向后兼容, 行为与 Estimator_HIM 完全一致)。
    - update(): body_obs 切到 num_body_dim, 避免对比目标混入特权量。
    """

    def __init__(self, *args, num_body_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_body_dim = num_body_dim if num_body_dim is not None else self.num_single_obs
        # target_encoder 在父类里按 num_single_obs 建好了; 若本体维度不同则按 num_body_dim 重建
        if self.num_body_dim != self.num_single_obs:
            self._rebuild_target_encoder(
                kwargs.get('tar_hidden_dims', [128, 64]),
                kwargs.get('activation', 'elu'),
            )

    def _rebuild_target_encoder(self, tar_hidden_dims, activation):
        act = get_activation(activation)
        layers = [nn.Linear(self.num_body_dim, tar_hidden_dims[0]), act]
        for l in range(len(tar_hidden_dims)):
            if l == len(tar_hidden_dims) - 1:
                layers.append(nn.Linear(tar_hidden_dims[l], self.lh_output_dim))
            else:
                layers.append(nn.Linear(tar_hidden_dims[l], tar_hidden_dims[l + 1]))
                layers.append(act)
        self.target_encoder = nn.Sequential(*layers)
        print(f'[Estimator_HIM_Heightmap] target_encoder 重建: input_dim={self.num_body_dim} '
              f'(num_single_obs={self.num_single_obs})')
        # 重建 optimizer 以纳入新的 target_encoder 参数
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def update(self, obs_history, next_critic_obs, lr=None):
        """与父类 update 一致, 仅 body_obs 切片改用 num_body_dim。"""
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        body_obs = next_critic_obs.detach()[:, 3:self.num_body_dim]          # 本体状态(不含特权/高程图)
        lin_vel = next_critic_obs.detach()[:, -3:]                           # 末尾 lin_vel(3 维)
        next_obs = torch.cat((body_obs, lin_vel), dim=-1)                    # (num_body_dim-3)+3 = num_body_dim

        # ---- SwAV 对比学习: source(历史 CNN) vs target(本体状态编码) ----
        z_s = self.long_history(obs_history.view(-1, self.history_len, self.num_proprio_obs).transpose(1, 2).contiguous())
        z_t = self.target_encoder(next_obs)

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

        # ---- Estimator 监督: 短历史 -> 预测 [base_height, lin_vel] ----
        short_history = obs_history[..., -self.num_short_obs:]
        estimated_prob = self.estimator(short_history)
        prob_real = next_critic_obs[:, -self.num_est_prob:].detach()

        estimation_loss = F.mse_loss(estimated_prob, prob_real)

        loss = swap_loss + estimation_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


class ActorCritic_HIM_Heightmap(ActorCritic_HIM):
    """ActorCritic_HIM + actor 前视高程图。

    与 ActorCritic_HIM 唯一区别: estimator 换成 Estimator_HIM_Heightmap,
    使 target_encoder 的对比目标只用本体状态(num_body_dim), 不含高程图/特权量。
    actor 仍通过 short_history 直接看到高程图(高程图已并入 num_single_obs)。
    act / act_inference / evaluate 全部继承父类, 无需重写。
    """

    def __init__(self, num_short_obs, num_single_obs, num_est_prob, num_critic_obs, num_actions,
                 num_body_dim=None, **kwargs):
        super().__init__(num_short_obs, num_single_obs, num_est_prob, num_critic_obs, num_actions, **kwargs)
        # 用 Heightmap 版 estimator 替换 super 建好的标准 estimator
        self.estimator = Estimator_HIM_Heightmap(
            num_short_obs=num_short_obs,
            num_single_obs=num_single_obs,
            num_critic_obs=num_critic_obs,
            estimator_hidden_dims=kwargs.get('estimator_hidden_dims', [128, 64]),
            tar_hidden_dims=kwargs.get('tar_hidden_dims', [128, 64]),
            num_est_prob=num_est_prob,
            history_len=kwargs.get('history_len', 30),
            kernel_size=kwargs.get('kernel_size', [6, 4]),
            filter_size=kwargs.get('filter_size', [32, 16]),
            stride_size=kwargs.get('stride_size', [3, 2]),
            lh_output_dim=kwargs.get('lh_output_dim', 32),
            activation=kwargs.get('activation', 'elu'),
            learning_rate=kwargs.get('learning_rate', 1e-3),
            max_grad_norm=kwargs.get('max_grad_norm', 10),
            is_privileged_obs=kwargs.get('is_privileged_obs', False),
            num_body_dim=num_body_dim,
        )
        print(f'[ActorCritic_HIM_Heightmap] num_single_obs={num_single_obs} '
              f'num_body_dim={self.estimator.num_body_dim} num_short_obs={num_short_obs}')
