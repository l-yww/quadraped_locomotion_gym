"""悬空 sin 跟踪环境 —— 隔离关节伺服子系统 sim2real gap。

继承 QuadEnv（与 quadruped_wtw_him_arm_fix 同基类），自动复用 LeggedRobot._compute_torques
的完整力矩链（PD + 力矩-速度曲线 + 限幅 + motor offset/PD 抖动），保证 sim/real 一致。

仅重写：
- compute_ref_state : 12 关节各自 sin 参考轨迹
- compute_observations : 38 维 obs = dof_pos + dof_vel + last_action + sin + cos（无 IMU/cmd）
- _get_noise_scale_vec : 38 维 noise（只对 dof_pos/vel 加编码器噪声）
- check_termination : 悬空下只按长度截断，不因姿态/接触终止
- _post_physics_step_callback : 持续锁死 base 速度，防止积分漂移
- _reward_tracking_joint_pos : 唯一跟踪 reward（指数核）
"""
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch

from wheel_legged_gym.envs.quadruped.quadruped_env import QuadEnv
from wheel_legged_gym.envs.quadruped_joint_track.quadruped_joint_track_config import QuadJointTrackCfg


class QuadJointTrackEnv(QuadEnv):

    def __init__(self, cfg: QuadJointTrackCfg, train_cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.train_cfg = train_cfg

        # sin 参考参数 → 张量
        self.ref_amplitude = torch.tensor(cfg.ref.amplitude, device=self.device).unsqueeze(0)   # (1,12)
        self.ref_phase_offset = torch.tensor(cfg.ref.phase_offset, device=self.device).unsqueeze(0)
        self.ref_offset = torch.tensor(cfg.ref.offset, device=self.device).unsqueeze(0)
        self.cycle_time = cfg.ref.cycle_time
        # 各 env 独立的初始相位（reset 时随机化）
        self.init_phase = torch.zeros(self.num_envs, 1, device=self.device)

        self.ref_dof_vel = torch.zeros_like(self.dof_pos)
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

    # ------------------------------------------------------------------
    # 参考轨迹：θ_i(t) = default_i + offset_i + A_i*sin(2π·phase + φ_i)
    # phase = init_phase + episode_length_buf * dt / cycle_time
    # ------------------------------------------------------------------
    def _get_phase(self):
        # (N,1)：初始相位 + 时间推进
        t = self.episode_length_buf * self.dt
        return self.init_phase + (t / self.cycle_time).unsqueeze(1)

    def compute_ref_state(self):
        phase = self._get_phase()                                   # (N,1)
        arg = 2 * torch.pi * phase                                  # (N,1)
        sin_val = torch.sin(arg + self.ref_phase_offset)            # (N,12) broadcast
        self.ref_dof_pos = self.default_dof_pos + self.ref_offset + self.ref_amplitude * sin_val
        # 解析导数，供可选 vel 跟踪 / 分析
        omega = 2 * torch.pi / self.cycle_time
        self.ref_dof_vel = self.ref_amplitude * omega * torch.cos(arg + self.ref_phase_offset)

    # ------------------------------------------------------------------
    # 观测：38 维 = dof_pos(12) + dof_vel(12) + last_action(12) + sin(1) + cos(1)
    # 不喂目标 —— 策略从相位 sin/cos + 历史推断该跟踪什么
    # ------------------------------------------------------------------
    def compute_observations(self):
        self.compute_ref_state()
        phase = self._get_phase()                                   # (N,1)
        sin_obs = torch.sin(2 * torch.pi * phase)                  # (N,1)
        cos_obs = torch.cos(2 * torch.pi * phase)                  # (N,1)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos   # (N,12)
        dq = self.dof_vel * self.obs_scales.dof_vel                            # (N,12)

        # privileged = actor（悬空无特权需求）
        obs_buf = torch.cat([q, dq, self.actions, sin_obs, cos_obs], dim=-1)   # (N,38)
        self.privileged_obs_buf = obs_buf.clone()

        obs_now = obs_buf.clone()
        if self.add_noise:
            obs_now = obs_now + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i]
                                             for i in range(self.cfg.env.c_frame_stack)], dim=1)

        return self.obs_buf

    def _get_noise_scale_vec(self):
        """38 维 noise：仅 dof_pos/dof_vel 加编码器噪声，其余为 0。"""
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)  # 38
        self.add_noise = self.cfg.noise.add_noise
        ns = self.cfg.noise.noise_scales
        na = self.num_actions
        noise_vec[0:na] = ns.dof_pos * self.obs_scales.dof_pos        # dof_pos
        noise_vec[na:2 * na] = ns.dof_vel * self.obs_scales.dof_vel   # dof_vel
        # last_action(2na:3na)=0, sin/cos(36:38)=0
        return noise_vec

    # ------------------------------------------------------------------
    # 悬空处理：每步锁死 base 线/角速度，防止积分漂移导致腿甩
    # ------------------------------------------------------------------
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.cfg.asset.fix_base_link:
            self.root_states[:, 7:13] = 0   # 世界系线速度[7:10] + 角速度[10:13]
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # ------------------------------------------------------------------
    # 终止：悬空下只按 episode 长度截断
    # ------------------------------------------------------------------
    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_buf |= self.episode_length_buf >= self.max_episode_length
        # 不加 base 姿态/接触/高度终止

    # ------------------------------------------------------------------
    # reset：随机化初始相位（可选），避免相位跳变 reward 抖 + 提升泛化
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # 防御：基类 __init__ 链路中会先调一次 reset_idx，此时 init_phase 尚未创建
        if not hasattr(self, "init_phase") or len(env_ids) == 0:
            return
        if self.cfg.ref.random_init_phase:
            self.init_phase[env_ids] = torch.rand(len(env_ids), 1, device=self.device)

    # ------------------------------------------------------------------
    # 唯一跟踪 reward：指数核，误差越小越接近 1
    # ------------------------------------------------------------------
    def _reward_tracking_joint_pos(self):
        diff = self.dof_pos - self.ref_dof_pos                       # (N,12)
        err = torch.mean(torch.square(diff), dim=1)                  # (N,)
        sigma = self.cfg.rewards.tracking_sigma
        return torch.exp(-err / sigma)

    # 正则项（已在 config 中以小权重开启，这里提供实现）
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        return torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)

    # ------------------------------------------------------------------
    # 基类 pre_physics_step / post_physics_step 无条件调用以下速度跟踪 reward
    # （存到 rwd_*Prev 用于 pbrs）。悬空 + 无命令下速度跟踪无意义，且基类实现
    # 依赖 wtw 才有的 rewards 字段（tracking_sigma_lin_vel 等），这里 override 返回 0。
    # ------------------------------------------------------------------
    def _reward_tracking_lin_vel(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_tracking_ang_vel(self):
        return torch.zeros(self.num_envs, device=self.device)
