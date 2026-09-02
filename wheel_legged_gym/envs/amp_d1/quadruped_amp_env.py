from .amp_d1_config import AmpD1Cfg, AmpD1CfgPPO

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import random
from .legged_robot_amp import LeggedRobotAMP

from wheel_legged_gym.utils.terrain import  Terrain
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift

class QuadAmpEnv(LeggedRobotAMP):

    def __init__(self, cfg: AmpD1Cfg, train_cfg: AmpD1CfgPPO, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        
    def _init_buffers(self):
        super()._init_buffers()        
        self.noised_q = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.noised_dq = torch.zeros_like(self.dof_vel)
        self.noised_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.noised_euler_rpy = torch.zeros_like(self.base_euler_rpy)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales

        num_commands = self.cfg.env.num_commands
        noise_vec[:num_commands] = 0
        noise_vec[num_commands : num_commands + self.num_actions] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[num_commands + self.num_actions : num_commands + 2 * self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[num_commands + 2 * self.num_actions : num_commands + 3 * self.num_actions] = 0
        noise_vec[num_commands + 3 * self.num_actions : num_commands + 3 * self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        if self.cfg.env.projected_gravity:
            noise_vec[num_commands + 3 * self.num_actions + 3 : num_commands + 3 * self.num_actions + 3 + 3] = noise_scales.gravity * self.obs_scales.gravity
        else:
            noise_vec[num_commands + 3 * self.num_actions + 3 : num_commands + 3 * self.num_actions + 3 + 2] = noise_scales.quat * self.obs_scales.quat
        return noise_vec
    
    def compute_observations(self):
        key_body_pos_relative_to_base = self._get_key_body_pos() - \
                self.base_pos.unsqueeze(1)
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
        )
        self.privileged_obs_buf = torch.cat(( # 
            self.commands[:, :3] * self.commands_scale,  # 3
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            key_body_pos_relative_to_base.flatten(start_dim=1),
            # self.base_euler_rpy[:,:] * self.obs_scales.quat,
            self.projected_gravity, # 重力投影
        ), dim=-1)

        if self.cfg.env.priv_observe_friction:
             # 1
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (self.friction_coeffs[:, 0].unsqueeze(1)
                                                    - friction_coeffs_shift) * friction_coeffs_scale),dim=1)
        
        if self.cfg.env.priv_observe_restitution:
            # 1
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(1) 
                                                 - restitutions_shift) * restitutions_scale),dim=1)

        if self.cfg.env.priv_observe_payloads:
            # 1
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),dim=1)

        if self.cfg.env.priv_observe_inertia:
            # 1
            inertia_scale, inertia_shift = get_scale_shift(self.cfg.domain_rand.randomize_inertia_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.inertia_scale.unsqueeze(1) - inertia_shift) * inertia_scale),dim=1)

        if self.cfg.env.priv_observe_motor_strength:
            # 6
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
        
        if self.cfg.env.priv_observe_motor_offset:
            # 6
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1) 
        if self.cfg.env.priv_observe_com_displacement:
            # 3
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.domain_rand.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.com_displacements[:, :3] - com_displacements_shift) * com_displacements_scale), dim=1)
        if self.cfg.env.priv_observe_heightmap:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 heights * self.obs_scales.height_measurements), dim=1)        
        # 始终将需要预测的值量放在最后（若有EST网络）
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                             self.base_height_obs *self.obs_scales.height_measurements,
                                            self.base_lin_vel * self.obs_scales.lin_vel, 
                                            ), dim=1)

        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
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
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            if self.cfg.env.projected_gravity == True:
                self.lagged_projected_gravity = self.lagged_imu[:,-3:].clone()
            else:
                self.lagged_base_euler_rpy = self.lagged_imu[:,-3:].clone()
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            if self.cfg.env.projected_gravity == True:
                self.lagged_projected_gravity = self.projected_gravity
            else:
                self.lagged_base_euler_rpy = self.base_euler_rpy

        lagged_q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        lagged_dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        obs_buf = torch.cat((
            self.commands[:, :3]  * self.commands_scale,
            lagged_q,
            lagged_dq,
            self.actions,
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,
        ), dim=-1)
        if self.cfg.env.projected_gravity:
            obs_buf = torch.cat((
                obs_buf,
                self.lagged_projected_gravity * self.obs_scales.quat,
            ),dim=-1)
        else:
            obs_buf = torch.cat((
                obs_buf,
                self.lagged_base_euler_rpy[:,:2] * self.obs_scales.quat,
            ),dim=-1)
        obs_now = obs_buf.clone()

        if self.add_noise:
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise
    
        self.noised_q = obs_now[:, self.cfg.env.num_commands:self.cfg.env.num_commands+self.cfg.env.num_actions] / self.obs_scales.dof_pos
        self.noised_dq = obs_now[:, self.cfg.env.num_commands+self.cfg.env.num_actions : self.cfg.env.num_commands+2*self.cfg.env.num_actions] / self.obs_scales.dof_vel
        imu_start = self.cfg.env.num_commands + 3 * self.cfg.env.num_actions
        self.noised_ang_vel = obs_now[:, imu_start:imu_start + 3] / self.obs_scales.ang_vel
        if self.cfg.env.projected_gravity:
            self.noised_euler_rpy = obs_now[:, imu_start + 3:imu_start + 6] / self.obs_scales.quat
        else:
            self.noised_euler_rpy = obs_now[:, imu_start + 3:imu_start + 5] / self.obs_scales.quat

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0



# ==================================================================================================================== #
# ================================================ Custom Rewards Function ================================================== #
# ==================================================================================================================== #
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
    # ------------------------- stand still rewards --------------------------------# 
    # ------------------------------------------------------------------------------# 
    def _reward_stand_base_vel_penality(self):
        """当命令很小时，机器人不应该有各个方向的速度"""
        # Penalize motion at zero commands
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stability(self):
        """当命令很小时,惩罚机器人速度、角速度、关节扭矩，以保持机器人静止时较为稳定"""
        velocity_error = torch.sum(torch.abs(self.base_lin_vel[:, :3]), dim=1)
        energy_cost = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1) * 0.01  # 能量惩罚系数
        stability_penalty = torch.sum(torch.abs(self.base_ang_vel[:, :3]), dim=1) * 0.2  # 身体角速度惩罚
        reward = (1.0 / (1.0 + velocity_error)) - energy_cost - stability_penalty
        return reward * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    # ------------------------------------------------------------------------------# 
    # ------------------------- termination rewards --------------------------------# 
    # ------------------------------------------------------------------------------# 
    def _reward_dof_vel_stand_still(self):
        # 当指令很小的时候，惩罚关节乱动
        cmd_norm = torch.norm(self.commands[:, :3], dim=1) < 0.2  # 是否静止指令
        dof_vel_norm = torch.norm(self.dof_vel, dim=-1) # 关节速度大小
        return dof_vel_norm * cmd_norm.float()

    def _reward_dof_pos_stand_still(self):
        # Penalize position deviation at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.2)

    # ------------------------------------------------------------------------------# 
    # ------------------------- termination rewards --------------------------------# 
    # ------------------------------------------------------------------------------# 
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    def _reward_foot_slip(self):
      
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    # ------------------------------------------------------------------------------# 
    # --------------------------- tracking rewards ---------------------------------# 
    # ------------------------------------------------------------------------------# 
    def _reward_tracking_lin_vel(self):
        commands_lin_vel = self.commands[:,0]
        lin_vel_error = torch.square(commands_lin_vel - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error * (self.cfg.rewards.tracking_sigma_lin_vel))
    
    def _reward_tracking_lin_vel_enhance(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error *  self.cfg.rewards.tracking_sigma_lin_vel / 10) - 1

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_ang_vel)

    def _reward_default_joint_pos(self):
        rew = torch.norm(self.leg_pos, dim=1)
        if self.reward_scales["default_joint_pos"] < 0:
            return rew
        else:
            return torch.exp(-20 * rew)
    
    # ------------------------------------------------------------------------------# 
    # ---------------------- common regularization rewards -------------------------# 
    # ------------------------------------------------------------------------------#    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
 
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
        # Penalize base height away from target
        scale = self.reward_scales.get("base_height", None)
        if scale is None:
            return torch.zeros_like(self.base_height, device=self.device)
        
        if scale < 0:
            return torch.abs(self.base_height - self.commands[:, 2])
        else:
            base_height_error = torch.square(self.base_height - self.commands[:, 2])
            return torch.exp(-200 * base_height_error)   
        
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
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_power(self):
        # Penalize torques
        return torch.sum(self.power, dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_acc = self.dof_acc_200hz
        return torch.sum(torch.square(dof_acc), dim=1)
