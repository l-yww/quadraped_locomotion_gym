# quadruped_arm_amp + actor 前视高程图环境。
#
# This task owns its reward implementations. Only generic simulation and AMP
# machinery comes from the direct quadruped/AMP base classes.
#   1. 采样 actor 前视高程图(base_link 为局部原点, x 0.5~1.0, y -0.4~0.4, 54 点)。
#   2. 把它并入 actor 每帧观测(num_single_obs 45 -> 99), 其余逻辑与父类一致。
# critic/特权观测使用同一前方网格，维度也为54点。

import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from wheel_legged_gym.utils.math import (
    quat_apply_yaw, get_scale_shift,
)
from ..quadruped.quadruped_env import QuadEnv
from ..amp_d1.legged_robot_amp import LeggedRobotAMP


class QuadAmpHeightmapEnv(QuadEnv, LeggedRobotAMP):
    """Standalone AMP heightmap environment."""

    def _init_gait_buffers(self):
        if hasattr(self, "gait_indices"):
            return
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.ones(self.num_envs, 4, device=self.device)

    def _step_contact_targets(self):
        self._init_gait_buffers()
        gait = getattr(self.cfg, "gait", None)
        if gait is None or not getattr(self.cfg.env, "enable_gait_clock", False):
            self.clock_inputs.zero_()
            self.desired_contact_states.fill_(1.0)
            return
        freq = float(getattr(gait, "gait_freq", 1.2))
        self.gait_indices = torch.remainder(
            getattr(self, "gait_indices", torch.zeros(self.num_envs, device=self.device))
            + self.dt * freq, 1.0
        )
        phase = torch.remainder(self.gait_indices[:, None] +
                                torch.tensor([0.5, 0.0, 0.0, 0.5], device=self.device), 1.0)
        self.clock_inputs = torch.sin(2.0 * np.pi * phase)
        self.desired_contact_states = (phase < float(getattr(gait, "gait_duration", 0.5))).float()

    def compute_privileged_observations(self, env_ids):
        """Build terminal critic observations from the current physical state."""
        if env_ids.numel() == 0:
            return self.privileged_obs_buf.new_empty((0, self.cfg.env.num_privileged_obs))
        privileged = torch.cat((
            self.commands[env_ids, :3] * self.commands_scale,
            self.dof_pos[env_ids] * self.obs_scales.dof_pos,
            self.dof_vel[env_ids] * self.obs_scales.dof_vel,
            self.actions[env_ids],
            self.base_ang_vel[env_ids] * self.obs_scales.ang_vel,
            self.projected_gravity[env_ids],
        ), dim=1)

        def append(value, bounds):
            scale, shift = get_scale_shift(bounds)
            return (value - shift) * scale

        if self.cfg.env.priv_observe_friction:
            privileged = torch.cat((privileged, append(self.friction_coeffs[env_ids, :1], self.cfg.domain_rand.friction_range)), dim=1)
        if self.cfg.env.priv_observe_restitution:
            privileged = torch.cat((privileged, append(self.restitutions[env_ids, :1], self.cfg.domain_rand.restitution_range)), dim=1)
        if self.cfg.env.priv_observe_payloads:
            privileged = torch.cat((privileged, append(self.payloads[env_ids, None], self.cfg.domain_rand.added_mass_range)), dim=1)
        if self.cfg.env.priv_observe_inertia:
            privileged = torch.cat((privileged, append(self.inertia_scale[env_ids, None], self.cfg.domain_rand.randomize_inertia_range)), dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            privileged = torch.cat((privileged, append(self.motor_strengths[env_ids], self.cfg.domain_rand.motor_strength_range)), dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            privileged = torch.cat((privileged, append(self.motor_offsets[env_ids], self.cfg.domain_rand.motor_offset_range)), dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            privileged = torch.cat((privileged, append(self.com_displacements[env_ids, :3], self.cfg.domain_rand.com_displacement_range)), dim=1)
        if self.cfg.env.priv_observe_heightmap:
            heights = torch.clip(
                self.root_states[env_ids, 2, None] - self.cfg.rewards.base_height_target - self.measured_heights[env_ids],
                -1.0, 1.0,
            )
            privileged = torch.cat((privileged, heights * self.obs_scales.height_measurements), dim=1)
        if getattr(self.cfg.env, "observe_timing_parameter", False):
            privileged = torch.cat((privileged, self.gait_indices[env_ids, None]), dim=1)
        if getattr(self.cfg.env, "observe_clock_inputs", False):
            privileged = torch.cat((privileged, self.clock_inputs[env_ids]), dim=1)
        return torch.cat((
            privileged,
            self.base_height[env_ids, None] * self.obs_scales.height_measurements,
            self.base_lin_vel[env_ids] * self.obs_scales.lin_vel,
        ), dim=1)

    def _init_buffers(self):
        super()._init_buffers()
        # actor 前视高程图采样点(机体系), 与父类近身 height_points 独立的一套网格
        self.actor_height_points = self._init_actor_height_points()
        self.actor_measured_heights = torch.zeros(
            self.num_envs, self.cfg.env.num_actor_height_points, device=self.device)

    # ------------------------------------------------------------------ #
    # actor 前视高程图采样 (镜像父类 _init_height_points / _get_heights,
    # 但用独立的 actor_height_points 网格, 不影响近身 measured_heights)
    # ------------------------------------------------------------------ #
    def _init_actor_height_points(self):
        y = torch.tensor(self.cfg.env.actor_measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.env.actor_measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
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

    def _post_physics_step_callback(self):
        # Base callback updates near terrain heights/base height; then update this task's scan.
        super()._post_physics_step_callback()
        self._step_contact_targets()
        self.actor_measured_heights = self._get_actor_heights()

    def _get_noise_scale_vec(self):
        # 父类返回 num_single_obs(99) 维: 前45(本体)已填，后54(高程图)为0。
        noise_vec = super()._get_noise_scale_vec()
        noise_scales = self.cfg.noise.noise_scales
        proprio_dim = self.cfg.env.num_body_dim   # 45
        noise_vec[proprio_dim:] = noise_scales.height_measurements * self.obs_scales.height_measurements
        return noise_vec

    # ------------------------------------------------------------------ #
    # 与父类 QuadHIMEnv.compute_observations 一致, 仅在 actor obs 帧末尾
    # 追加前视高程图(54维)。actor和critic使用同一网格。
    # ------------------------------------------------------------------ #
    def compute_observations(self):
        self.base_height_obs = self.base_height.unsqueeze(1)
        base_ref = getattr(self.cfg.rewards, "base_height_target", 0.4)
        # 近身高程图(特权观测用, 与父类完全一致)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - base_ref - self.measured_heights,
                -1, 1.0,
            )
        )
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

        # ===== actor 前视高程图 (54维, base_link 局部原点前方 0.5~1.0m) =====
        actor_heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - base_ref - self.actor_measured_heights,
            -1, 1.0,
        )
        obs_buf = torch.cat((obs_buf, actor_heights * self.obs_scales.height_measurements), dim=-1)
        # 至此 obs_buf = 本体(45) + 前视高程图(54) = 99 = num_single_obs

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


    # ---- task-local reward functions ---------------------------------
    def _stand_command_mask(self):
        threshold = getattr(self.cfg.rewards, "stand_command_threshold", 0.1)
        return torch.norm(self.commands[:, :3], dim=1) < threshold

    def _reward_tracking_lin_vel(self):
        error = (self.commands[:, :2] - self.base_lin_vel[:, :2]).square().sum(dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        error = (self.commands[:, 2] - self.base_ang_vel[:, 2]).square()
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_orientation(self):
        return self.projected_gravity[:, :2].square().sum(dim=1)

    def _reward_orientation_stand(self):
        error = self.projected_gravity[:, :2].square().sum(dim=1)
        return torch.exp(-error / 0.005) * self._stand_command_mask()

    def _reward_base_height(self):
        error = (self.base_height - self.cfg.rewards.base_height_target).abs()
        return error if self.reward_scales.get("base_height", 0.0) < 0 else torch.exp(-200.0 * error.square())

    def _reward_base_height_stand(self):
        error = (self.base_height - self.cfg.rewards.base_height_target).square()
        return torch.exp(-error / 0.002) * self._stand_command_mask()

    def _reward_stand_base_vel_penality(self):
        value = 5.0 * self.base_lin_vel[:, 0].square() + self.base_lin_vel[:, 1:3].square().sum(dim=1)
        return value * self._stand_command_mask()

    def _reward_stand_still(self):
        return (self.dof_pos - self.default_dof_pos).abs().sum(dim=1) * self._stand_command_mask()

    def _reward_dof_vel_stand_still(self):
        return self.dof_vel.norm(dim=1) * self._stand_command_mask()

    def _reward_stand_foot_vel(self):
        return self.feet_velocities.norm(dim=2).sum(dim=1) * self._stand_command_mask()

    def _reward_stand_feet_air(self):
        threshold = getattr(self.cfg.rewards, "stand_contact_force_threshold", 1.0)
        contact = self.contact_forces[:, self.feet_indices, 2] > threshold
        return (~contact).float().sum(dim=1) * self._stand_command_mask()

    def _reward_default_joint_pos(self):
        value = torch.norm(self.leg_pos, dim=1)
        return value if self.reward_scales.get("default_joint_pos", 0.0) < 0 else torch.exp(-20.0 * value)

    def _reward_default_hip_pos(self):
        indices = [0, 3, 6, 9]
        error = torch.abs(self.dof_pos[:, indices] - self.default_dof_pos[:, indices]).sum(dim=1)
        straight = torch.exp(-(self.commands[:, 1].abs() + self.commands[:, 2].abs()) / 0.1)
        return error * (1.0 + 2.0 * self._stand_command_mask()) * straight

    def _reward_lin_vel_z(self):
        return self.base_lin_vel[:, 2].square()

    def _reward_ang_vel_xy(self):
        return self.base_ang_vel[:, :2].square().sum(dim=1)

    def _reward_base_acc(self):
        linear = (self.last_root_vel[:, :3] - self.root_states[:, 7:10]).norm(dim=1) / self.cfg.sim.dt
        angular = (self.last_root_vel[:, 3:6] - self.root_states[:, 10:13]).norm(dim=1) / self.cfg.sim.dt
        return linear + 0.02 * angular

    def _reward_torques(self):
        return self.torques.square().sum(dim=1)

    def _reward_action(self):
        return self.actions.square().sum(dim=1)

    def _reward_action_rate(self):
        return (self.last_actions - self.actions).square().sum(dim=1)

    def _reward_action_smoothness(self):
        return (self.actions + self.last_last_actions - 2.0 * self.last_actions).square().sum(dim=1)

    def _reward_collision(self):
        force = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=2)
        return (force > 0.1).float().sum(dim=1)

    def _reward_dof_pos_limits(self):
        lower = -(self.dof_pos - self.dof_pos_limits[:, 0] * self.cfg.rewards.soft_dof_pos_limit).clip(max=0.0)
        upper = (self.dof_pos - self.dof_pos_limits[:, 1] * self.cfg.rewards.soft_dof_pos_limit).clip(min=0.0)
        return (lower + upper).sum(dim=1)

    def _reward_dof_vel_limits(self):
        return (self.dof_vel.abs() - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limits).clip(min=0.0).sum(dim=1)

    def _torque_velocity_limits(self):
        upper = self.torque_limits.unsqueeze(0).expand_as(self.torques).clone()
        lower = -upper.clone()

        def apply(indices, max_vel, vel_1, max_torque):
            indices = list(indices)
            slope = max_torque / max(max_vel - vel_1, 1e-6)
            velocity = self.dof_vel[:, indices]
            dynamic_upper = torch.clamp(-slope * (velocity - max_vel), -max_torque, max_torque)
            dynamic_lower = torch.clamp(-slope * (velocity + max_vel), -max_torque, max_torque)
            invalid = velocity.abs() > max_vel + max_torque / slope
            dynamic_upper = torch.where(invalid, torch.zeros_like(dynamic_upper), dynamic_upper)
            dynamic_lower = torch.where(invalid, torch.zeros_like(dynamic_lower), dynamic_lower)
            static = self.torque_limits[indices].unsqueeze(0)
            upper[:, indices] = torch.minimum(dynamic_upper, static)
            lower[:, indices] = torch.maximum(dynamic_lower, -static)

        apply(getattr(self.cfg.control, "torque_vel_hip_indices", [0, 1, 3, 4, 6, 7, 9, 10]),
              getattr(self.cfg.control, "torque_vel_hip_max_vel", 21.0),
              getattr(self.cfg.control, "torque_vel_hip_vel_1", 7.28),
              getattr(self.cfg.control, "torque_vel_hip_max_torque", 200.0))
        apply(getattr(self.cfg.control, "torque_vel_calf_indices", [2, 5, 8, 11]),
              getattr(self.cfg.control, "torque_vel_calf_max_vel", 13.0),
              getattr(self.cfg.control, "torque_vel_calf_vel_1", 6.6),
              getattr(self.cfg.control, "torque_vel_calf_max_torque", 330.0))
        return lower, upper

    def _reward_torque_limits(self):
        lower, upper = self._torque_velocity_limits()
        ratio = getattr(self.cfg.rewards, "soft_torque_limit", 0.8)
        upper_error = (self.torques - ratio * upper).clip(min=0.0)
        lower_error = (ratio * lower - self.torques).clip(min=0.0)
        return (upper_error + lower_error).sum(dim=1)

    def _reward_power(self):
        return self.power.sum(dim=1)

    def _reward_dof_vel(self):
        return self.dof_vel.square().sum(dim=1)

    def _reward_dof_acc(self):
        return self.dof_acc_200hz.square().sum(dim=1)

    def _reward_foot_stumble(self):
        horizontal = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        vertical = self.contact_forces[:, self.feet_indices, 2].abs()
        return torch.any(horizontal > 5.0 * vertical, dim=1)

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        speed = self.feet_velocities[:, :, :2].norm(dim=2)
        moving = ((self.commands[:, :2].norm(dim=1) > 0.1) | (self.commands[:, 2].abs() > 0.2))
        return (torch.clamp(speed - 0.03, min=0.0) * contact).sum(dim=1) * moving

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first = (self.feet_air_time > 0.0) & filt
        self.feet_air_time += self.dt
        reward = ((self.feet_air_time - 0.5) * first).sum(dim=1)
        reward *= self.commands[:, :2].norm(dim=1) > 0.1
        self.feet_air_time *= ~filt
        return reward

    def _reward_foot_clearance(self):
        pos = self.feet_positions - self.root_states[:, None, :3]
        vel = self.feet_velocities - self.root_states[:, None, 7:10]
        quat = self.base_quat[:, None].expand(-1, pos.shape[1], -1)
        pos_body = quat_rotate_inverse(quat, pos)
        vel_body = quat_rotate_inverse(quat, vel)
        error = (pos_body[:, :, 2] - self.cfg.rewards.clearance_height_target).square()
        speed = vel_body[:, :, :2].square().sum(dim=2).sqrt()
        return (error * speed).sum(dim=1)

    def _reward_zero_action(self):
        return torch.exp(-self.actions.square().sum(dim=1) / 0.1) * self._stand_command_mask()

    def _reward_diagonal_symmetry(self):
        # FL<->RR and FR<->RL thigh/calf pairs; hips are mirror joints.
        pairs = ((1, 10), (2, 11), (4, 7), (5, 8))
        error = sum((self.dof_pos[:, left] - self.dof_pos[:, right]).square() for left, right in pairs)
        return torch.exp(-(error / len(pairs)) / self.cfg.rewards.tracking_sigma) * (
            self.commands[:, :2].norm(dim=1) > 0.1
        )

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_keep_balance(self):
        return torch.ones(self.num_envs, device=self.device)