import torch
import random
import numpy as np
from isaacgym.torch_utils import torch_rand_float, quat_rotate_inverse
import random
import numpy as np
from isaacgym.torch_utils import torch_rand_float, quat_rotate_inverse
from wheel_legged_gym.envs.quadruped.quadruped_env import QuadEnv
from wheel_legged_gym.utils.math import get_scale_shift, quat_apply, quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

from wheel_legged_gym.utils.math import get_scale_shift, quat_apply, quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float



class QuadHIMEnv(QuadEnv):

    """quadruped(3 维命令) + PPO_HIM(estimator)，带机械臂URDF。
    """

    # ===== [步态时钟] =====
    # 步态参数全部从 cfg.gait.* 读取 (config 里 class gait 定义), env 不再硬编码默认值。
    # 若 config 未配置 class gait, 使用下方 GAIT_DEFAULTS 兜底 (仅用于旧 task 兼容)。
    GAIT_DEFAULTS = {
        "gait_freq": 2.0,       # 步频 Hz (固定模式)
        "freq_min": 0.8,        # 速度自适应下限 Hz
        "freq_max": 2.0,        # 速度自适应上限 Hz
        "step_stride": 0.30,    # 速度自适应目标跨距 m
        "gait_phase": 0.5,      # trot 对角同相
        "gait_offset": 0.0,
        "gait_bound": 0.0,
        "gait_duration": 0.5,   # 占空比 50%
        "kappa_gait_probs": 0.07,  # von mises 平滑参数 (rewards 段, 此处仅兜底)
    }

    def _init_gait_buffers(self):
        """惰性初始化步态时钟相关 buffer（避免改父类 _init_buffers）。
        在第一次 _step_contact_targets 前调用。
        """
        if hasattr(self, "gait_indices"):
            return
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices_tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device)

    def _step_contact_targets(self):
        """步态时钟（速度自适应步频版）。
        freq 不再是固定常数, 而是随前进速度变化: freq = |v_x| / step_stride, clamp 到 [fmin, fmax]。
        物理依据: 慢走慢踏、快走快踏; 爬楼时 policy 自然减速 → 步频自动降低 → 腾空时间长 → 抬腿更充分。
        这是隐式地形自适应: 不显式判断楼梯/平地, 而是通过速度间接区分 (爬楼必减速)。
        config 的 gait.gait_freq 现在作为 step_stride 的参考 (每周期目标跨距 = 训练台阶进深)。
        产出：clock_inputs（4 维 sin 相位，进 actor/critic obs）、
              desired_contact_states（4 维期望接触，给步态奖励用）。
        原地时 zero_cmd_mask=True，步态时钟停转、clock_inputs 归零、四脚全支撑。
        """
        self._init_gait_buffers()
        gait_cfg = getattr(self.cfg, "gait", None)
        d = self.GAIT_DEFAULTS
        phase = getattr(gait_cfg, "gait_phase", d["gait_phase"])
        offset = getattr(gait_cfg, "gait_offset", d["gait_offset"])
        bound = getattr(gait_cfg, "gait_bound", d["gait_bound"])
        duration = getattr(gait_cfg, "gait_duration", d["gait_duration"])
        kappa = getattr(self.cfg.rewards, "kappa_gait_probs", d["kappa_gait_probs"])

        # 步频模式 (config.gait.adaptive_freq 控制):
        #   True  → 速度自适应: freq = |v_x|/step_stride, clamp[freq_min, freq_max]
        #            爬楼时 policy 自然减速 → 步频自动降低 → 抬腿更充分 (隐式地形自适应)
        #   False → 固定步频: freq = gait_freq (传统方案, 所有 env 共用)
        # 切换模式 obs 维度/PPO 配置都不变, 仅 clock_inputs 节奏分布不同
        adaptive_freq = getattr(gait_cfg, "adaptive_freq", False)
        if adaptive_freq:
            step_stride = getattr(gait_cfg, "step_stride", d.get("step_stride", 0.30))
            fmin = getattr(gait_cfg, "freq_min", d.get("freq_min", 0.8))
            fmax = getattr(gait_cfg, "freq_max", d.get("freq_max", 2.0))
            v_x = self.base_lin_vel[:, 0].abs()
            freq = (v_x / step_stride).clamp(fmin, fmax)  # [num_envs] 张量, 每 env 不同
        else:
            freq_val = getattr(gait_cfg, "gait_freq", d["gait_freq"])
            freq = torch.full((self.num_envs,), float(freq_val), device=self.device)

        # freq 是每 env 张量, 直接用; 其余步态参数仍是标量 → broadcast 成 [num_envs]
        frequencies = freq
        phases = torch.full((self.num_envs,), float(phase), device=self.device)
        offsets = torch.full((self.num_envs,), float(offset), device=self.device)
        bounds = torch.full((self.num_envs,), float(bound), device=self.device)
        durations = torch.full((self.num_envs,), float(duration), device=self.device)

        # 原地检测：arm_him 无 smoothed_commands，直接用 commands[:,:3]
        zero_cmd_mask = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & \
                        (torch.abs(self.commands[:, 2]) < 0.1)

        gait_increment = self.dt * frequencies * (~zero_cmd_mask).float()
        self.gait_indices = torch.remainder(self.gait_indices + gait_increment, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,    # FL
            self.gait_indices + bounds,                       # FR
            self.gait_indices + offsets,                      # RL
            self.gait_indices + phases                        # RR
        ]
        self.foot_indices_tensor = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (0.5 / (1 - durations[swing_idxs]))

        clock_vals = torch.stack([torch.sin(2 * np.pi * foot_indices[i]) for i in range(4)], dim=1)
        self.clock_inputs = clock_vals * (~zero_cmd_mask).unsqueeze(1).float()

        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
        for i in range(4):
            foot_phase = torch.remainder(foot_indices[i], 1.0)
            desired_contact = (
                smoothing_cdf_start(foot_phase) *
                (1 - smoothing_cdf_start(foot_phase - 0.5)) +
                smoothing_cdf_start(foot_phase - 1) *
                (1 - smoothing_cdf_start(foot_phase - 0.5 - 1))
            )
            self.desired_contact_states[:, i] = torch.where(
                zero_cmd_mask, torch.ones_like(desired_contact), desired_contact
            )

    def compute_privileged_observations(self, env_ids):

        self.base_height_obs = self.base_height[env_ids].unsqueeze(1)

        privileged = torch.cat((
            self.commands[env_ids, :3] * self.commands_scale,
            self.dof_pos[env_ids] * self.obs_scales.dof_pos,
            self.dof_vel[env_ids] * self.obs_scales.dof_vel,
            self.actions[env_ids],
            self.base_ang_vel[env_ids] * self.obs_scales.ang_vel,
            self.projected_gravity[env_ids],
        ), dim=-1)

        if self.cfg.env.priv_observe_friction:
            s, sh = get_scale_shift(self.cfg.domain_rand.friction_range)
            privileged = torch.cat((privileged, (self.friction_coeffs[env_ids, 0:1] - sh) * s), dim=1)
        if self.cfg.env.priv_observe_restitution:
            s, sh = get_scale_shift(self.cfg.domain_rand.restitution_range)
            privileged = torch.cat((privileged, (self.restitutions[env_ids, 0:1] - sh) * s), dim=1)
        if self.cfg.env.priv_observe_payloads:
            s, sh = get_scale_shift(self.cfg.domain_rand.added_mass_range)
            privileged = torch.cat((privileged, (self.payloads[env_ids].unsqueeze(1) - sh) * s), dim=1)
        if self.cfg.env.priv_observe_inertia:
            s, sh = get_scale_shift(self.cfg.domain_rand.randomize_inertia_range)
            privileged = torch.cat((privileged, (self.inertia_scale[env_ids].unsqueeze(1) - sh) * s), dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            s, sh = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            privileged = torch.cat((privileged, (self.motor_strengths[env_ids] - sh) * s), dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            s, sh = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            privileged = torch.cat((privileged, (self.motor_offsets[env_ids] - sh) * s), dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            s, sh = get_scale_shift(self.cfg.domain_rand.com_displacement_range)
            privileged = torch.cat((privileged, (self.com_displacements[env_ids, :3] - sh) * s), dim=1)

        if self.cfg.env.priv_observe_heightmap:
            # 这里的核心修改：必须在调用前确保在这个时间步，你已经针对这一帧或者上一帧刷新了 measured_heights
            # 并严格使用 [env_ids] 进行切片
            # heights = torch.clip(self.root_states[env_ids, 2].unsqueeze(1) - 0.5 - self.measured_heights[env_ids], -1, 1.0)
            # privileged = torch.cat((privileged, heights * self.obs_scales.height_measurements), dim=1)
            base_ref = self.cfg.rewards.base_height_target
            heights = torch.clip(self.root_states[env_ids, 2].unsqueeze(1) - base_ref - self.measured_heights[env_ids], -1, 1.0)
            privileged = torch.cat((privileged, heights * self.obs_scales.height_measurements), dim=1)
            
        # ===== [步态时钟] privileged 也加 timing 信号（对齐 wtw） =====
        if getattr(self.cfg.env, "observe_timing_parameter", True):
            privileged = torch.cat((privileged, self.gait_indices[env_ids].unsqueeze(1)), dim=1)
        if getattr(self.cfg.env, "observe_clock_inputs", True):
            privileged = torch.cat((privileged, self.clock_inputs[env_ids]), dim=1)

        # 末尾 estimator 监督目标 [base_height(1), base_lin_vel(3)]
        privileged = torch.cat((privileged,
                                self.base_height_obs * self.obs_scales.height_measurements,
                                self.base_lin_vel[env_ids] * self.obs_scales.lin_vel), dim=1)

        # 此时得到的 privileged 本身就已经过滤好了，直接返回即可
        return privileged


    def compute_observations(self):
        """override 父类：在 actor obs 和 privileged obs 末尾追加步态时钟信号
        (gait_indices 1 维 + clock_inputs 4 维)，对齐 wtw_him_arm_fix。
        其余逻辑与父类 quadruped_env.compute_observations 一致。
        """
        self.base_height_obs = self.base_height.unsqueeze(1)
        base_ref = getattr(self.cfg.rewards, "base_height_target", 0.4)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - base_ref - self.measured_heights,
                -1, 1.0,
            )
        )
        # ---- privileged obs ----
        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
        ), dim=-1)

        if self.cfg.env.priv_observe_friction:
            s, sh = get_scale_shift(self.cfg.domain_rand.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                (self.friction_coeffs[:, 0].unsqueeze(1) - sh) * s), dim=1)
        if self.cfg.env.priv_observe_restitution:
            s, sh = get_scale_shift(self.cfg.domain_rand.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                (self.restitutions[:, 0].unsqueeze(1) - sh) * s), dim=1)
        if self.cfg.env.priv_observe_payloads:
            s, sh = get_scale_shift(self.cfg.domain_rand.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                (self.payloads.unsqueeze(1) - sh) * s), dim=1)
        if self.cfg.env.priv_observe_inertia:
            s, sh = get_scale_shift(self.cfg.domain_rand.randomize_inertia_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                (self.inertia_scale.unsqueeze(1) - sh) * s), dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            s, sh = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                (self.motor_strengths - sh) * s), dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            s, sh = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                (self.motor_offsets - sh) * s), dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            s, sh = get_scale_shift(self.cfg.domain_rand.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                (self.com_displacements[:, :3] - sh) * s), dim=1)
        if self.cfg.env.priv_observe_heightmap:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                heights * self.obs_scales.height_measurements), dim=1)
        # [步态时钟] privileged 加 timing 信号
        if getattr(self.cfg.env, "observe_timing_parameter", True):
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        if getattr(self.cfg.env, "observe_clock_inputs", True):
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.clock_inputs), dim=-1)
        # 末尾 estimator 监督目标 [base_height(1), base_lin_vel(3)]
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
            self.base_height_obs * self.obs_scales.height_measurements,
            self.base_lin_vel * self.obs_scales.lin_vel), dim=1)

        # ---- lagged buffers (延迟观测，arm_him domain_rand 关了延迟，走 else 分支) ----
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                    self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (self.num_envs,), device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        if self.cfg.domain_rand.add_imu_lag:
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                    self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (self.num_envs,), device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:, :3].clone()
            if self.cfg.env.projected_gravity:
                self.lagged_projected_gravity = self.lagged_imu[:, -3:].clone()
            else:
                self.lagged_base_euler_rpy = self.lagged_imu[:, -3:].clone()
        else:
            self.lagged_base_ang_vel = self.base_ang_vel[:, :3]
            self.lagged_base_euler_rpy = self.base_euler_rpy
            if self.cfg.env.projected_gravity:
                self.lagged_projected_gravity = self.projected_gravity

        lagged_q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        lagged_dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        # ---- actor obs ----
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            lagged_q,
            lagged_dq,
            self.actions,
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,
        ), dim=-1)
        if self.cfg.env.projected_gravity:
            obs_buf = torch.cat((obs_buf, self.lagged_projected_gravity * self.obs_scales.quat), dim=-1)
        else:
            obs_buf = torch.cat((obs_buf, self.lagged_base_euler_rpy[:, :2] * self.obs_scales.quat), dim=-1)
        # [步态时钟] actor obs 加 timing 信号
        if getattr(self.cfg.env, "observe_timing_parameter", True):
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        if getattr(self.cfg.env, "observe_clock_inputs", True):
            obs_buf = torch.cat((obs_buf, self.clock_inputs), dim=-1)
        obs_now = obs_buf.clone()

        if self.add_noise:
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise

        self.noised_q = obs_now[:, self.cfg.env.num_commands:self.cfg.env.num_commands + self.cfg.env.num_actions] / self.obs_scales.dof_pos
        self.noised_dq = obs_now[:, self.cfg.env.num_commands + self.cfg.env.num_actions:self.cfg.env.num_commands + 2*self.cfg.env.num_actions] / self.obs_scales.dof_vel
        self.noised_ang_vel = obs_now[:, self.cfg.env.num_commands + 3*self.cfg.env.num_actions:self.cfg.env.num_commands + 3*self.cfg.env.num_actions + 3] / self.obs_scales.ang_vel
        self.noised_euler_rpy = obs_now[:, self.cfg.env.num_commands + 3*self.cfg.env.num_actions + 3:self.cfg.env.num_commands + 3*self.cfg.env.num_actions + 5] / self.obs_scales.quat

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)


    # def _get_foot_heights(self):
    #     """采样 4 只脚正下方的地形高度(世界系 z)，供 base_height 用"支撑面"度量。
    #     plane→0(等价 base_height=root_z)；trimesh→脚下四点 min 插值查 height_samples。
    #     rigid_state 已是世界系，脚的 xy 无需再按 yaw 旋转。
    #     与 quadruped_wtw_him_arm_fix 一致，避免 patch 平均在楼梯顶/过渡处的虚高偏差。
    #     """
    #     if self.cfg.terrain.mesh_type == 'plane':
    #         return torch.zeros(self.num_envs, 4, device=self.device)
    #     points = self.rigid_state[:, self.feet_indices, :2] + self.terrain.cfg.border_size
    #     points = (points / self.terrain.cfg.horizontal_scale).long()
    #     px = torch.clip(points[:, :, 0], 0, self.height_samples.shape[0] - 2)
    #     py = torch.clip(points[:, :, 1], 0, self.height_samples.shape[1] - 2)
    #     h1 = self.height_samples[px, py]
    #     h2 = self.height_samples[px + 1, py]
    #     h3 = self.height_samples[px, py + 1]
    #     h = torch.min(h1, torch.min(h2, h3))
    #     # print("88888888888888")
    #     return h * self.terrain.cfg.vertical_scale

    def _post_physics_step_callback(self):
        """override：仅保留 heading 模式逻辑（3 维命令兼容）。
        base_height 度量回到父类 patch 平均（root_z - measured_heights.mean），
        与 HIMLoco / cowa_8dof_trigger / 0701(已验证效果好) 一致。
        之前改用"脚底支撑面 + min 插值"反而因 min 偏差带歪 base_height → 策略蹲低。
        """
        if not hasattr(self, "_use_heading_mode"):
            self._use_heading_mode = self.cfg.commands.heading_command
        # 临时关掉 heading_command，避免 super() 里访问 commands[:, 3] 越界（3 维命令）
        _saved_heading = self.cfg.commands.heading_command
        self.cfg.commands.heading_command = False
        super()._post_physics_step_callback()
        self.cfg.commands.heading_command = _saved_heading
        # ===== [步态时钟] 在 super() 之后调用，更新 clock_inputs / desired_contact_states =====
        # 关时钟时跳过: desired_contact_states 保持全 0 (配合两个 shaped 接触奖励权重=0)
        if getattr(self.cfg.env, "enable_gait_clock", True):
            self._step_contact_targets()
        # base_height 用父类 patch 平均（不覆盖）
        # heading 模式：实时从目标航向算 ang_vel_yaw（3 维命令兼容版）
        if self._use_heading_mode:
            if not hasattr(self, "target_heading"):
                self.target_heading = torch.zeros(self.num_envs, device=self.device)
            forward = quat_apply(self.base_quat, self.forward_vec)
            cur_heading = torch.atan2(forward[:, 1], forward[:, 0])
            yaw_max = self.command_ranges["ang_vel_yaw"][1]
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.target_heading - cur_heading), -yaw_max, yaw_max
            )

    def _resample_commands(self, env_ids):
        """override：3 维命令下 heading 模式兼容版。
        heading 模式采样目标航向角存进 self.target_heading（不写 commands[:, 3]，避免越界）；
        ang_vel_yaw 由 _post_physics_step_callback 实时从航向误差算出写进 commands[:, 2]。
        非 heading 模式走父类原逻辑。
        注意：用 self._use_heading_mode 判断，不用 cfg.commands.heading_command
        （callback 里会临时改 cfg，这里要独立于它）。
        """
        if len(env_ids) == 0:
            return
        if not hasattr(self, "_use_heading_mode"):
            self._use_heading_mode = self.cfg.commands.heading_command
        if not self._use_heading_mode:
            super()._resample_commands(env_ids)
            return
        # ---- heading 模式：自己实现，避免访问 commands[:, 3] ----
        if not hasattr(self, "target_heading"):
            self.target_heading = torch.zeros(self.num_envs, device=self.device)
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        # 采样目标航向角，存进 target_heading（不进 commands，3 维够用）
        self.target_heading[env_ids] = torch_rand_float(
            self.command_ranges["heading"][0], self.command_ranges["heading"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        # commands[:, 2] (ang_vel) 由 callback 实时算，这里先置 0
        self.commands[env_ids, 2] = 0.0
        # 1/3 直行：目标航向 = 当前航向（让这批 env 先直行）
        resample_nums = len(env_ids)
        if resample_nums > 0:
            half_env_list = random.sample(range(resample_nums), resample_nums // 3)
            half_ids = env_ids[half_env_list]
            forward = quat_apply(self.base_quat[half_ids], self.forward_vec[half_ids])
            cur_heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.target_heading[half_ids] = cur_heading
        # 小命令清零
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)

    # ============================================================================ #
    # ============================ Rewards Tools  ================================ #
    # ============================================================================ #
    # def _get_base_heights(self, env_ids=None):
    #     """ Samples heights of the terrain at required points around each robot.
    #         The points are offset by the base's position and rotated by the base's yaw

    #     Args:
    #         env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    #     Raises:
    #         NameError: [description]

    #     Returns:
    #         [type]: [description]
    #     """
    #     if self.cfg.terrain.mesh_type == 'plane':
    #         return self.root_states[:, 2].clone()
    #     elif self.cfg.terrain.mesh_type == 'none':
    #         raise NameError("Can't measure height with terrain mesh type 'none'")

    #     if env_ids:
    #         points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
    #     else:
    #         points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


    #     points += self.terrain.cfg.border_size
    #     points = (points/self.terrain.cfg.horizontal_scale).long()
    #     px = points[:, :, 0].view(-1)
    #     py = points[:, :, 1].view(-1)
    #     px = torch.clip(px, 0, self.height_samples.shape[0]-2)
    #     py = torch.clip(py, 0, self.height_samples.shape[1]-2)

    #     heights1 = self.height_samples[px, py]
    #     heights2 = self.height_samples[px+1, py]
    #     heights3 = self.height_samples[px, py+1]
    #     heights = torch.min(heights1, heights2)
    #     heights = torch.min(heights, heights3)
    #     # heights = (heights1 + heights2 + heights3) / 3

    #     base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

    #     return base_height

    def _resolve_diagonal_leg_indices(self):
        """解析对角线腿对 (FL<->RR, FR<->RL) 对应的 dof 索引,惰性缓存。

        依赖 self.dof_names(URDF 中的关节名,形如 'FL_thigh_joint')。
        返回: diag_pairs, list of (idx_legA, idx_legB) 逐关节配对,
              例如 FL_thigh 与 RR_thigh 一对、FL_calf 与 RR_calf 一对,另一对同理。
        """
        if hasattr(self, "_diag_dof_pairs"):
            return self._diag_dof_pairs

        # 腿前缀 -> 对角线对侧前缀
        opposite = {"FL": "RR", "FR": "RL", "RR": "FL", "RL": "FR"}
        name_to_idx = {n: i for i, n in enumerate(self.dof_names)}

        def leg_of(name):
            for leg in ("FL", "FR", "RL", "RR"):
                if name.startswith(leg):
                    return leg
            return None

        pairs = []
        seen = set()
        # 只配对周期性摆动关节 (thigh / calf),hip 控制侧向外摆,
        # 其对称是镜像 (FL_hip == -FR_hip) 而非相等,故排除。
        swing_joints = ("thigh", "calf")
        for name, idx in name_to_idx.items():
            leg = leg_of(name)
            if leg is None:
                continue
            if not any(f"_{j}_" in name for j in swing_joints):
                continue
            opp = opposite[leg]
            # 只从一侧出发避免重复 (FL->RR, FR->RL),取另一侧的同类关节
            if leg not in ("FL", "FR"):
                continue
            suffix = name[len(leg):]  # 如 '_thigh_joint'
            opp_name = opp + suffix
            if opp_name in name_to_idx:
                key = (name, opp_name)
                if key not in seen:
                    seen.add(key)
                    pairs.append((idx, name_to_idx[opp_name]))

        self._diag_dof_pairs = pairs
        return pairs

    # ============================================================================ #
    # ============================ Custom Rewards ================================ #
    # ============================================================================ #
    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        base_lin_acc = torch.norm(self.last_root_vel[:,0:3] - self.root_states[:, 7:10], dim=1) / self.cfg.sim.dt
        base_ang_acc = torch.norm(self.last_root_vel[:,3:6] - self.root_states[:, 10:13], dim=1) / self.cfg.sim.dt
        rew = base_lin_acc + 0.02 * base_ang_acc
        return rew

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.8 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    # ------------------------------------------------------------------------------#
    # --------------------------- tracking rewards ---------------------------------#
    # ------------------------------------------------------------------------------#
    def _get_standstill_weight(self, transition_speed=0.15):
        """平滑静止掩码: 1/(1 + cmd_norm/transition_speed)
        改用反比例函数: cmd=0→1.0, cmd=0.15→0.5, cmd=0.5→0.23
        """
        cmd_norm = torch.norm(self.smoothed_commands[:, :3], dim=1)
        return 1.0 / (1.0 + cmd_norm / transition_speed)

    # ------------------------------------------------------------------------------#
    # ------------------------- stand still rewards --------------------------------#
    # ------------------------------------------------------------------------------#
    def _stand_command_mask(self):
        threshold = getattr(self.cfg.rewards, "stand_command_threshold", 0.1)
        return torch.norm(self.commands[:, :3], dim=1) < threshold

    def _reward_stand_base_vel_penality(self):
        """当命令很小时，机器人base不应该有各个方向的速度"""
        # Penalize motion at zero commands
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * self._stand_command_mask().float()

    def _reward_stability(self):
        """当命令很小时,惩罚机器人速度、角速度、关节扭矩，以保持机器人静止时较为稳定"""
        velocity_error = torch.sum(torch.abs(self.base_lin_vel[:, :3]), dim=1)
        energy_cost = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1) * 0.01  # 能量惩罚系数
        stability_penalty = torch.sum(torch.abs(self.base_ang_vel[:, :3]), dim=1) * 0.2  # 身体角速度惩罚
        reward = (1.0 / (1.0 + velocity_error)) - energy_cost - stability_penalty
        return reward * self._stand_command_mask().float()

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * self._stand_command_mask().float()

    def _reward_dof_vel_stand_still(self):
        return torch.norm(self.dof_vel, dim=-1) * self._stand_command_mask().float()

    def _reward_stand_foot_vel(self):
        foot_vel = torch.norm(self.feet_velocities, dim=-1)
        return torch.sum(foot_vel, dim=1) * self._stand_command_mask().float()

    def _reward_stand_feet_air(self):
        threshold = getattr(self.cfg.rewards, "stand_contact_force_threshold", 1.0)
        contact = self.contact_forces[:, self.feet_indices, 2] > threshold
        return torch.sum((~contact).float(), dim=1) * self._stand_command_mask().float()

    # ------------------------------------------------------------------------------#
    # ------------------------- termination rewards --------------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    # ------------------------------------------------------------------------------#
    # --------------------------- tracking rewards ---------------------------------#
    # ------------------------------------------------------------------------------#

    # def _reward_tracking_lin_vel(self):
    #     # 同时计算 X 轴和 Y 轴的误差
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma_lin_vel)

    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw) — commands[:,2] = ang_vel_yaw
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_ang_vel)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    # def _reward_tracking_lin_vel_enhance(self):  #暂时没有参与
    #     # Tracking of linear velocity commands (x axes)
    #     lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
    #     return torch.exp(-lin_vel_error *  self.cfg.rewards.tracking_sigma_lin_vel / 10) - 1

    def _reward_default_joint_pos(self):
        rew = torch.norm(self.leg_pos, dim=1)
        if self.reward_scales["default_joint_pos"] < 0:
            return rew
        else:
            return torch.exp(-20 * rew)

    # ------------------------------------------------------------------------------#
    # ---------------------- common regularization rewards -------------------------#
    # ------------------------------------------------------------------------------#
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
        c_update = (lin_mismatch + ang_mismatch) / 2.
        return c_update

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_positive(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_rpy[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_base_height(self):
        # Penalize base height away from target — 目标用 base_height_target(0.39)，而非 yaw 命令
        scale = self.reward_scales.get("base_height", None)
        if scale is None:
            return torch.zeros_like(self.base_height, device=self.device)
        if scale < 0:
            return torch.abs(self.base_height - self.cfg.rewards.base_height_target)
        else:
            base_height_error = torch.square(self.base_height - self.cfg.rewards.base_height_target)
            return torch.exp(-200 * base_height_error)

    def _reward_base_height_stable(self):
        target = self.cfg.rewards.base_height_target
        error = abs(self.base_height - target)
        dead_zone = 0.02                              # 2cm 死区
        error = torch.clamp(error - dead_zone, min=0.0)
        return torch.square(error)                          # square 形式

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = self._get_base_heights()
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_action(self):
        # Penalize actions
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        return torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0] * self.cfg.rewards.soft_dof_pos_limit).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1] * self.cfg.rewards.soft_dof_pos_limit).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limits).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # Penalize torques above soft_torque_limit of the velocity-dependent torque envelope.
        lower_limit, upper_limit = self._torque_velocity_limits()
        soft_ratio = getattr(self.cfg.rewards, "soft_torque_limit", 0.8)
        soft_upper_limit = soft_ratio * upper_limit
        soft_lower_limit = soft_ratio * lower_limit
        upper_violation = (self.torques - soft_upper_limit).clip(min=0.)
        lower_violation = (soft_lower_limit - self.torques).clip(min=0.)
        return torch.sum(upper_violation + lower_violation, dim=1)

    def _reward_power(self):
        # Penalize torques
        return torch.sum(self.power, dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_acc = self.dof_acc_200hz
        return torch.sum(torch.square(dof_acc), dim=1)


    # def _reward_foot_stumble(self):
    #     """绊脚惩罚: 脚碰到垂直面 (水平力 vs 法向力) 越严重惩罚越大, 返回连续值。
    #     原实现 torch.any 返回 bool, 离散 0/-0.8 几乎无梯度; 阈值 5x 法向也过松。
    #     现: ratio = 水平力 / 法向力, 超过 1 即开始罚 (ratio-1 越大越重),
    #     4 脚求和, 避免单脚剧烈碰撞被掩盖。
    #     注意: 打滑 (棱角处脚横移) 和绊脚 (撞墙) 是不同物理现象, 此函数治绊脚;
    #     打滑需要 foot_slip 奖励 (基于脚切向速度) 单独处理。
    #     """
    #     horizontal_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)  # (N, 4)
    #     vertical_force = torch.abs(self.contact_forces[:, self.feet_indices, 2])              # (N, 4)
    #     ratio = horizontal_force / (vertical_force + 1.0)  # +1 避免除零
    #     return torch.sum(torch.clamp(ratio - 1.0, min=0.0), dim=1)  # (N,) 连续梯度

    def _reward_foot_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_active_foot_lift(self):
        """主动抬腿奖励（修复版）：
        奖励「脚相对 base 的抬升量」，仅在爬楼/上坡时激活，平地静默。
        修复点：
          1. 用 commands[:,0] 而非 smoothed_commands（arm_him 无 smoothed_commands）
          2. 用世界系 z 速度 root_states[:,8] 而非机体系 base_lin_vel[:,2]（抬头时机体系 z 会漏检）
          3. 用「脚相对 base 高度」foot_z_rel 而非世界系 foot_z（杜绝站在高处白拿奖励）
          4. 抬腿量 <0.05m 不给奖励（平地小碎步静默，不奖励贴地倒脚）
          5. 摆动相阈值 2N（避免擦地误判）
          6. 爬楼检测用世界系 z 速度 >0.15（滤平地行走振荡）+ 抬头 pitch
        """
        # 1. 摆动相 + 前进指令
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        is_swinging = (foot_forces < 2.0).float()
        is_moving_forward = (self.commands[:, 0] > 0.1).float().unsqueeze(1)
        # 2. 抬腿量 = 脚世界z - base世界z（相对高度，杜绝站高处刷分）
        foot_z_rel = self.rigid_state[:, self.feet_indices, 2] - self.root_states[:, 2].unsqueeze(1)
        # 抬腿量 <0.05 不奖励（平地小碎步静默），>0.25 封顶（贴合台阶高度）
        lift_reward = torch.clamp(foot_z_rel - 0.05, min=0.0, max=0.20)
        # 3. 爬楼检测：世界系 z 速度 >0.15（持续上坡，滤平地行走 ±0.1 振荡）或 抬头 pitch
        is_going_up = self.root_states[:, 8] > 0.15
        # pitch 符号按 URDF 约定：<0 抬头（需实测确认，见 play 打印 base_pitch）
        is_pitching_up = self.base_euler_rpy[:, 1] < -0.1
        is_climbing = (is_going_up | is_pitching_up).float().unsqueeze(1)
        # 4. 掩码相乘：摆动 + 前进 + 爬楼 才给抬腿奖励
        return torch.sum(lift_reward * is_swinging * is_moving_forward * is_climbing, dim=1)

    def _reward_active_foot_lift(self):
        """主动抬腿奖励（修复版）：
        奖励「脚相对 base 的抬升量」，仅在爬楼/上坡时激活，平地静默。
        修复点：
          1. 用 commands[:,0] 而非 smoothed_commands（arm_him 无 smoothed_commands）
          2. 用世界系 z 速度 root_states[:,8] 而非机体系 base_lin_vel[:,2]（抬头时机体系 z 会漏检）
          3. 用「脚相对 base 高度」foot_z_rel 而非世界系 foot_z（杜绝站在高处白拿奖励）
          4. 抬腿量 <0.05m 不给奖励（平地小碎步静默，不奖励贴地倒脚）
          5. 摆动相阈值 2N（避免擦地误判）
          6. 爬楼检测用世界系 z 速度 >0.15（滤平地行走振荡）+ 抬头 pitch
        """
        # 1. 摆动相 + 前进指令
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        is_swinging = (foot_forces < 2.0).float()
        is_moving_forward = (self.commands[:, 0] > 0.1).float().unsqueeze(1)
        # 2. 抬腿量 = 脚世界z - base世界z（相对高度，杜绝站高处刷分）
        foot_z_rel = self.rigid_state[:, self.feet_indices, 2] - self.root_states[:, 2].unsqueeze(1)
        # 抬腿量 <0.05 不奖励（平地小碎步静默），>0.25 封顶（贴合台阶高度）
        lift_reward = torch.clamp(foot_z_rel - 0.05, min=0.0, max=0.20)
        # 3. 爬楼检测：世界系 z 速度 >0.15（持续上坡，滤平地行走 ±0.1 振荡）或 抬头 pitch
        is_going_up = self.root_states[:, 8] > 0.15
        # pitch 符号按 URDF 约定：<0 抬头（需实测确认，见 play 打印 base_pitch）
        is_pitching_up = self.base_euler_rpy[:, 1] < -0.1
        is_climbing = (is_going_up | is_pitching_up).float().unsqueeze(1)
        # 4. 掩码相乘：摆动 + 前进 + 爬楼 才给抬腿奖励
        return torch.sum(lift_reward * is_swinging * is_moving_forward * is_climbing, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    # def _reward_default_pos(self):
    #     # Penalize motion at zero commands
    #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) #* (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_default_hip_pos(self):

        """Penalize hip deviation from default（而非绝对 0，兼容 default_joint_angles 内八偏置）."""
        joint_diff = torch.abs(self.dof_pos[:, 0] - self.default_dof_pos[:, 0]) + \
                     torch.abs(self.dof_pos[:, 3] - self.default_dof_pos[:, 3]) + \
                     torch.abs(self.dof_pos[:, 6] - self.default_dof_pos[:, 6]) + \
                     torch.abs(self.dof_pos[:, 9] - self.default_dof_pos[:, 9])
        # 1. 静止判定逻辑
        is_standstill = self._stand_command_mask().float()
        vy_cmd = torch.abs(self.commands[:, 1])
        wz_cmd = torch.abs(self.commands[:, 2])
        is_straight_forward = torch.exp(-(vy_cmd + wz_cmd) / 0.1)
        # 最终惩罚倍率 = (基础 1.0 + 静止奖励 2.0 + 爬坡重罚) * 直行豁免系数
        amplifier = (1.0 + 2.0 * is_standstill ) * is_straight_forward
        return joint_diff * amplifier


    def _reward_foot_clearance(self):
        """约束摆动相抬腿高度：脚在机体系 z 偏离目标高度 × 水平速度 惩罚。
        防止「高抬腿正步走」(脚抬过高且水平快移)和「拖地走」(脚过低)。
        与 HIMLoco cowa 一致。用 feet_positions/feet_velocities(世界系)转到机体系。
        """
        cur_footpos_translated = self.feet_positions - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_velocities - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_lateral_vel, dim=1)

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_vel_xy = torch.norm(self.feet_velocities[:, :, :2], dim=-1)
        slip = torch.clamp(foot_vel_xy - 0.03, min=0.0)
        slip = contact.float() * slip
        cmd_xy = torch.norm(self.commands[:, :2], dim=1)
        cmd_yaw = torch.abs(self.commands[:, 2])
        moving = ((cmd_xy > 0.10) | (cmd_yaw > 0.20)).float()

        return torch.sum(slip, dim=1) * moving

    # def _reward_foot_slip(self): #amp_d1_env版
      
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    #     foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
    #     rew = torch.sqrt(foot_speed_norm)
    #     rew *= contact
    #     return torch.sum(rew, dim=1)

    def _reward_diagonal_symmetry(self):
        """奖励对角线运动对称 (trot):对角线两腿的对应关节位置应同步。

        对每对对角线关节 (FL<->RR, FR<->RL),惩罚其关节位置差的平方,
        并用高斯核映射为 [0,1] 奖励,鼓励对角线两腿同相、与另一对反相的步态。
        零指令时不施加(避免在原地静止时强制小幅抖动)。
        """
        pairs = self._resolve_diagonal_leg_indices()
        if len(pairs) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        # 各对角线关节位置差
        diff_sq = torch.zeros(self.num_envs, device=self.device)
        for ia, ib in pairs:
            diff_sq = diff_sq + torch.square(self.dof_pos[:, ia] - self.dof_pos[:, ib])

        # 归一化为平均每关节误差后用高斯核
        mean_diff = diff_sq / float(len(pairs))
        sigma = self.cfg.rewards.tracking_sigma if hasattr(self.cfg.rewards, "tracking_sigma") else 0.25
        reward = torch.exp(-mean_diff / sigma)

        # 零指令豁免:无速度指令时不奖励对称步态
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        reward = reward * (cmd_norm > 0.1).float()
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        """Penalize unexpected contact forces during swing phase."""
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        """Penalize foot sliding during stance phase."""
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i] * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / 0.5)))
        return reward / 4

    def _reward_zero_action(self):
        """零命令时鼓励 action 输出接近 0 (静止). 物理: action=0 → joint 保持 default → base 静止."""
        action_norm_sq = torch.sum(self.actions ** 2, dim=-1)
        return torch.exp(-action_norm_sq / 0.1) * self._stand_command_mask().float()