"""wtw + AMP 融合环境：姿态调整 + 专家柔和先验。

继承链: QuadWtwAmpEnv_HIM → QuadWtwEnv_HIM → QuadEnv → LeggedRobotAMP → LeggedRobot
  - wtw 的全部姿态调整命令（12维：vx/vy/yaw/body_h/freq/.../pitch/roll）← QuadWtwEnv_HIM
  - AMP 判别器 + 专家数据 + 参考姿态初始化 ← LeggedRobotAMP

"""
import torch
from wheel_legged_gym.envs.quadruped_wtw_him_arm_fix.quadruped_wtw_him_arm_fix_env import QuadWtwEnv_HIM
from wheel_legged_gym.envs.quadruped_wtw_him_arm_fix.quadruped_wtw_him_arm_fix_config import QuadWtwCfg_HIM
from wheel_legged_gym.envs.amp_d1.legged_robot_amp import LeggedRobotAMP


class QuadWtwAmpEnv_HIM(QuadWtwEnv_HIM, LeggedRobotAMP):
    """wtw 姿态调整 + AMP 专家柔和先验，判别器对姿态免疫。"""

    def reset_idx(self, env_ids):
        """Override: wtw 的 reset_idx 不处理 terminal_amp_states，
        AMPRunner_HIM 需要它在 reset 时记录终止前的 amp obs。这里补上。"""
        # 先记录终止前的 amp obs（terminal_amp_states），再调 super 做 reset
        if len(env_ids) > 0:
            self.terminal_amp_states = self.get_amp_observations()[env_ids].clone()
        super().reset_idx(env_ids)

    def reset(self):
        """Override: wtw step 返回 7 个值（HIM 模式），LeggedRobotAMP 期望 8 个，这里对齐。"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        step_out = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return step_out[0], step_out[1]

    def step(self, actions):
        """Override: AMPRunner_HIM 期望 step 返回 8 个值（含 terminal_amp_states），
        wtw 原生返回 7 个。这里在 HIM 分支末尾追加 amp_states 补齐到 8 个。"""
        out = super().step(actions)
        if 'HIM' in self.train_cfg.runner_class_name and len(out) == 7:
            # wtw HIM 返回 7 个: obs, priv, rew, reset, extras, term_ids, term_priv
            # AMPRunner_HIM 期望 8 个: ..., term_ids, term_priv, terminal_amp_states
            terminal_amp = getattr(self, 'terminal_amp_states', None)
            return (*out, terminal_amp)
        return out

    def get_amp_observations(self):
        """姿态免疫判别器 obs。

        与基类 LeggedRobotAMP.get_amp_observations 的区别：
        - dof_pos → (dof_pos - default_dof_pos)：相对偏移，对 body_height/pitch 调整免疫
        - key_body_pos 改用相对 base 的高度差（去掉 base 高度的影响）

        判别器学到的"专家风格"是动作动态（速度 + 相对位置变化），
        不含绝对姿态信息，所以 wtw 调姿态时判别器不会惩罚。
        """
        key_body_pos_relative_to_base = self._get_key_body_pos() - self.base_pos.unsqueeze(1)
        # ★ 核心改动：dof_pos 用相对偏移（减 default），对姿态调整免疫
        dof_pos_relative = self.dof_pos - self.default_dof_pos
        return torch.cat((
            self.base_lin_vel,                                  # 3
            self.base_ang_vel,                                  # 3
            dof_pos_relative,                                   # num_dofs（相对偏移）
            self.dof_vel,                                       # num_dofs
            key_body_pos_relative_to_base.flatten(start_dim=1), # num_key_bodies * 3
        ), dim=-1)
