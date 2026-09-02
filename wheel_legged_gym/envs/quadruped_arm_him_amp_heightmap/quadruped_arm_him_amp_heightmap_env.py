# quadruped_arm_him_amp + WTW 采样语义的 actor 前视高程图环境。
#
# 继承: QuadHIMAmpHeightmapEnv → QuadHIMAmpEnv → QuadHIMEnv → QuadEnv → LeggedRobotAMP → LeggedRobot
# AMP / HIM machinery 继承自 QuadHIMAmpEnv。本类只增加 actor 前视高程图，
# Critic 仍使用 HIM-AMP 原有的近身特权高程图。

import torch
from wheel_legged_gym.utils.math import quat_apply_yaw, get_scale_shift
from ..quadruped_arm_him_amp.quadruped_arm_him_amp_env import QuadHIMAmpEnv
from .quadruped_arm_him_amp_heightmap_config import QuadCfg_HIM_AMP_Heightmap, QuadCfgPPO_HIM_AMP_Heightmap


class QuadHIMAmpHeightmapEnv(QuadHIMAmpEnv):
    """HIM-AMP with a WTW-style actor height scan."""

    def _init_buffers(self):
        super()._init_buffers()

        expected_points = self.cfg.env.num_actor_height_points

        # Keep a dedicated actor buffer; its WTW-aligned grid is independent
        # from the HIM-AMP critic grid.
        self.actor_height_points = self._init_actor_height_points()
        if self.num_actor_height_points != expected_points:
            raise ValueError(
                "Actor height-point count does not match num_height_scan_points: "
                f"{self.num_actor_height_points} != {expected_points}"
            )
        self.actor_measured_heights = torch.zeros(
            self.num_envs, self.cfg.env.num_actor_height_points, device=self.device)
        self.actor_height_scan = torch.zeros_like(self.actor_measured_heights)
        # During _init_buffers(), LeggedRobot has created height_points but keeps
        # measured_heights as the scalar 0 until the first physics callback.
        # Allocate from the HIM critic grid rather than using zeros_like on that
        # not-yet-populated value.
        num_critic_height_points = (
            len(self.cfg.terrain.measured_points_x)
            * len(self.cfg.terrain.measured_points_y)
        )
        if self.num_height_points != num_critic_height_points:
            raise ValueError(
                "Critic height-point count does not match terrain.measured_points: "
                f"{self.num_height_points} != {num_critic_height_points}"
            )
        # HIM-AMP critic uses the original terrain.measured_points_x/y grid (17 x 11).
        self.critic_height_scan = torch.zeros(
            self.num_envs, num_critic_height_points, device=self.device
        )
        self.previous_height_scan = torch.zeros_like(self.actor_measured_heights)
        self.height_scan_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.height_episode_offset = torch.zeros(self.num_envs, device=self.device)
        self.height_yaw_noise = torch.zeros(self.num_envs, device=self.device)
        self.height_pitch_bias = torch.zeros(self.num_envs, device=self.device)
        self.height_roll_bias = torch.zeros(self.num_envs, device=self.device)

        x_points = torch.tensor(
            self.cfg.env.actor_measured_points_x, device=self.device
        )
        y_points = torch.tensor(
            self.cfg.env.actor_measured_points_y, device=self.device
        )
        grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing="ij")
        x_values = self.cfg.env.actor_measured_points_x
        y_values = self.cfg.env.actor_measured_points_y
        x_max = max(abs(x_values[0]), abs(x_values[-1]))
        y_max = max(abs(y_values[0]), abs(y_values[-1]))
        self._height_tilt_x_norm = (grid_x / x_max).reshape(-1)
        self._height_tilt_y_norm = (grid_y / y_max).reshape(-1)
        height_update_hz = float(getattr(self.cfg.domain_rand, "height_update_hz", 10.0))
        if height_update_hz <= 0.0:
            raise ValueError("height_update_hz must be positive")
        self.height_update_interval_steps = max(
            1, int(round(1.0 / (self.dt * height_update_hz)))
        )
        self._height_update_step = 0

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "actor_height_scan"):
            self.actor_measured_heights[env_ids] = 0.0
            self.actor_height_scan[env_ids] = 0.0
            self.critic_height_scan[env_ids] = 0.0
            self.previous_height_scan[env_ids] = 0.0
            self.height_scan_initialized[env_ids] = False
            self._resample_height_episode_state(env_ids)

    def _resample_height_episode_state(self, env_ids):
        """Sample the same per-episode height-sensor biases as the WTW task."""
        if len(env_ids) == 0:
            return
        cfg = self.cfg.domain_rand
        count = len(env_ids)

        def uniform(low, high):
            return torch.empty(count, device=self.device).uniform_(low, high)

        randomize_offset = getattr(cfg, "randomize_height_offset", False)
        randomize_yaw = getattr(cfg, "randomize_height_yaw", False)
        randomize_tilt = getattr(cfg, "randomize_height_roll_pitch", False)
        self.height_episode_offset[env_ids] = (
            uniform(*getattr(cfg, "height_offset_range", [0.0, 0.0]))
            if randomize_offset else 0.0
        )
        self.height_yaw_noise[env_ids] = (
            uniform(*getattr(cfg, "height_yaw_noise_range", [0.0, 0.0]))
            if randomize_yaw else 0.0
        )
        if randomize_tilt:
            self.height_pitch_bias[env_ids] = uniform(
                *getattr(cfg, "height_pitch_bias_range", [0.0, 0.0])
            )
            self.height_roll_bias[env_ids] = uniform(
                *getattr(cfg, "height_roll_bias_range", [0.0, 0.0])
            )
        else:
            self.height_pitch_bias[env_ids] = 0.0
            self.height_roll_bias[env_ids] = 0.0

    # ------------------------------------------------------------------ #
    # actor 前视高程图采样 (镜像父类 _init_height_points / _get_heights,
    # 但用独立的 actor_height_points 网格, 不影响近身 measured_heights)
    # ------------------------------------------------------------------ #
    def _init_actor_height_points(self):
        y = torch.tensor(self.cfg.env.actor_measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.env.actor_measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.num_actor_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_actor_height_points, 3,
                             device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_actor_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_actor_height_points,
                               device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_actor_height_points),
                                    self.actor_height_points[env_ids]) \
                     + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_actor_height_points),
                                    self.actor_height_points) \
                     + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_actor_heights_with_yaw_noise(self):
        yaw = self.height_yaw_noise
        yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        yaw_quat[:, 2] = torch.sin(yaw / 2.0)
        yaw_quat[:, 3] = torch.cos(yaw / 2.0)
        local_points = quat_apply_yaw(
            yaw_quat.repeat(1, self.num_actor_height_points).reshape(-1, 4),
            self.actor_height_points.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_actor_height_points, 3)
        world_points = quat_apply_yaw(
            self.base_quat.repeat(1, self.num_actor_height_points).reshape(-1, 4),
            local_points.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_actor_height_points, 3)
        world_points += self.root_states[:, :3].unsqueeze(1)
        grid_points = (
            world_points[:, :, :2] + self.terrain.cfg.border_size
        ) / self.terrain.cfg.horizontal_scale
        px = grid_points[:, :, 0].reshape(-1)
        py = grid_points[:, :, 1].reshape(-1)
        rows, cols = self.height_samples.shape
        px0 = torch.clip(px.floor().long(), 0, rows - 2)
        py0 = torch.clip(py.floor().long(), 0, cols - 2)
        px1 = (px0 + 1).clamp(max=rows - 1)
        py1 = (py0 + 1).clamp(max=cols - 1)
        heights = torch.min(
            torch.min(self.height_samples[px0, py0], self.height_samples[px0, py1]),
            torch.min(self.height_samples[px1, py0], self.height_samples[px1, py1]),
        )
        return heights.reshape(
            self.num_envs, self.num_actor_height_points
        ) * self.terrain.cfg.vertical_scale

    def _update_height_scan(self):
        """Update actor scan using WTW sampling semantics.

        The critic scan remains the original HIM-AMP 17 x 11 scan.
        """
        cfg = self.cfg.domain_rand
        base_ref = getattr(self.cfg.rewards, "base_height_target", 0.4)
        clean_terrain = self._get_actor_heights()
        self.actor_measured_heights.copy_(clean_terrain)
        clean = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - base_ref - clean_terrain,
            -1.0, 1.0,
        )
        critic_terrain = self._get_heights()
        critic_clean = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - base_ref - critic_terrain,
            -1.0, 1.0,
        )
        self.critic_height_scan.copy_(critic_clean)

        if getattr(cfg, "randomize_height_yaw", False):
            visible_terrain = self._get_actor_heights_with_yaw_noise()
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - base_ref - visible_terrain,
                -1.0, 1.0,
            )
        else:
            heights = clean.clone()

        if getattr(cfg, "randomize_height_offset", False):
            heights += self.height_episode_offset.unsqueeze(1)
        if getattr(cfg, "add_height_noise", False):
            if getattr(cfg, "add_height_gaussian_noise", False):
                heights += torch.randn_like(heights) * getattr(cfg, "height_gaussian_noise", 0.0)
            if getattr(cfg, "add_height_spike_noise", False):
                spike_mask = torch.rand_like(heights) < 0.05
                spike_amplitude = torch.empty(
                    self.num_envs, 1, device=self.device
                ).uniform_(*getattr(cfg, "height_spike_noise_range", [0.0, 0.0]))
                heights += spike_mask * torch.randn_like(heights) * spike_amplitude
        if getattr(cfg, "randomize_height_roll_pitch", False):
            heights += (
                self.height_pitch_bias.unsqueeze(1)
                * self._height_tilt_x_norm.unsqueeze(0)
                + self.height_roll_bias.unsqueeze(1)
                * self._height_tilt_y_norm.unsqueeze(0)
            )

        height_repeat_probability = float(getattr(cfg, "height_repeat_probability", 0.0))
        if height_repeat_probability > 0.0:
            repeat_mask = (
                torch.rand(self.num_envs, device=self.device)
                < height_repeat_probability
            )
            repeat_mask &= self.height_scan_initialized
            visible = torch.where(
                repeat_mask.unsqueeze(1), self.previous_height_scan, heights
            )
        else:
            visible = heights

        self.previous_height_scan.copy_(visible)
        self.height_scan_initialized[:] = True
        self.actor_height_scan.copy_(visible)

    def _post_physics_step_callback(self):
        # 父类更新地形/reward缓存和步态时钟。Actor/Critic 的观测扫描独立按 10 Hz 更新。
        super()._post_physics_step_callback()
        self._height_update_step += 1
        if self._height_update_step >= self.height_update_interval_steps:
            self._update_height_scan()
            self._height_update_step = 0

    def _get_noise_scale_vec(self):
        # 父类填充本体噪声，帧末尾的 54 维高程图在这里补齐噪声尺度。
        noise_vec = super()._get_noise_scale_vec()
        noise_scales = self.cfg.noise.noise_scales
        proprio_dim = self.cfg.env.num_body_dim   # 45
        noise_vec[proprio_dim:] = noise_scales.height_measurements * self.obs_scales.height_measurements
        return noise_vec

    # ------------------------------------------------------------------ #
    # 与父类 QuadHIMEnv.compute_observations 一致，在 Actor 帧末尾追加
    # 54 维前视高程图；Critic 保留 HIM-AMP 原来的 17 x 11 高程图。
    # ------------------------------------------------------------------ #
    def compute_observations(self):
        self.base_height_obs = self.base_height.unsqueeze(1)
        # Critic sees the clean scan; Actor sees the randomized 10 Hz scan.
        heights = self.critic_height_scan
        # ---- privileged obs (与父类一致, 近身高程图 + 特权量 + estimator 监督目标) ----
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
        if self.privileged_obs_buf.shape[1] != self.cfg.env.single_num_privileged_obs:
            raise RuntimeError(
                "Critic frame dimension mismatch: "
                f"{self.privileged_obs_buf.shape[1]} != "
                f"{self.cfg.env.single_num_privileged_obs}"
            )

        # ---- lagged buffers (延迟观测; arm_him domain_rand 关了延迟, 走 else 分支) ----
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

        # ---- actor obs (本体 45 维, 与父类一致) ----
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

        # WTW-aligned actor height scan: 6 x 9 = 54 values.
        actor_heights = self.actor_height_scan
        if actor_heights.shape[1] != self.cfg.env.num_height_scan_points:
            raise RuntimeError(
                "Actor height-scan dimension mismatch: "
                f"{actor_heights.shape[1]} != {self.cfg.env.num_height_scan_points}"
            )
        obs_buf = torch.cat((obs_buf, actor_heights * self.obs_scales.height_measurements), dim=-1)
        if obs_buf.shape[1] != self.cfg.env.num_single_obs:
            raise RuntimeError(
                "Actor frame dimension mismatch: "
                f"{obs_buf.shape[1]} != {self.cfg.env.num_single_obs}"
            )

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
        if self.obs_buf.shape[1] != self.cfg.env.num_observations:
            raise RuntimeError(
                "Actor history dimension mismatch: "
                f"{self.obs_buf.shape[1]} != {self.cfg.env.num_observations}"
            )
        if self.privileged_obs_buf.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                "Critic history dimension mismatch: "
                f"{self.privileged_obs_buf.shape[1]} != "
                f"{self.cfg.env.num_privileged_obs}"
            )

    def _reward_pre_motor_torque_limits(self):
        """Penalize raw PD commands near the WTW pre-efficiency hard clips."""
        if not getattr(self.cfg.control, "enable_pre_motor_torque_clip", True):
            return torch.zeros(self.num_envs, device=self.device)

        hip_indices = self.cfg.control.torque_vel_hip_indices
        calf_indices = self.cfg.control.torque_vel_calf_indices
        soft_ratio = self.cfg.rewards.pre_motor_torque_soft_ratio
        hip_soft_limit = self.cfg.control.pre_torque_vel_clip_hip * soft_ratio
        calf_soft_limit = self.cfg.control.pre_torque_vel_clip_calf * soft_ratio
        hip_excess = (
            torch.abs(self.torques_cmd[:, hip_indices]) - hip_soft_limit
        ).clip(min=0.0)
        calf_excess = (
            torch.abs(self.torques_cmd[:, calf_indices]) - calf_soft_limit
        ).clip(min=0.0)
        return torch.sum(hip_excess, dim=1) + torch.sum(calf_excess, dim=1)
