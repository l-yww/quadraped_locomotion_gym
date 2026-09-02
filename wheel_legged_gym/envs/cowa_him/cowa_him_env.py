# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
from wheel_legged_gym.envs.cowa_him.cowa_him_config import CowaCfg_HIM, CowaCfgPPO_HIM
from wheel_legged_gym.envs.cowa.cowa_env import CowaEnv
import torch
from wheel_legged_gym.utils.math import get_scale_shift


class CowaEnv_HIM(CowaEnv):
    def __init__(self, cfg: CowaCfg_HIM, train_cfg: CowaCfgPPO_HIM,  sim_params, physics_engine, sim_device, headless):
            super().__init__(cfg, train_cfg, sim_params, physics_engine, sim_device, headless)
            self.cfg = cfg
            self.train_cfg = train_cfg

    def compute_privileged_observations(self, env_ids):
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        privileged_obs_buf = torch.cat(( # 
            self.commands[:, :3] * self.commands_scale,  # 3
            self.leg_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            # self.base_euler_rpy[:,:] * self.obs_scales.quat,
            self.projected_gravity, # 重力投影
        ), dim=-1)

        if self.cfg.env.priv_observe_friction:
            # 1
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                    (self.friction_coeffs[:, 0].unsqueeze(1)
                                                    - friction_coeffs_shift) * friction_coeffs_scale),dim=1)
        
        if self.cfg.env.priv_observe_restitution:
            # 1
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.restitutions[:, 0].unsqueeze(1) 
                                                - restitutions_shift) * restitutions_scale),dim=1)

        if self.cfg.env.priv_observe_payloads:
            # 1
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),dim=1)

        if self.cfg.env.priv_observe_inertia:
            # 1
            inertia_scale, inertia_shift = get_scale_shift(self.cfg.domain_rand.randomize_inertia_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.inertia_scale.unsqueeze(1) - inertia_shift) * inertia_scale),dim=1)

        if self.cfg.env.priv_observe_motor_strength:
            # 6
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            # 6
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1) 
        if self.cfg.env.priv_observe_com_displacement:
            # 3
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.domain_rand.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                (self.com_displacements[:, :3] - com_displacements_shift) * com_displacements_scale), dim=1)
        # 始终将需要预测的值量放在最后（若有EST网络）
        privileged_obs_buf = torch.cat((privileged_obs_buf,
                                        self.base_height_obs * self.obs_scales.height_measurements,
                                        self.base_lin_vel * self.obs_scales.lin_vel, 
                                    ), dim=1)
        return privileged_obs_buf[env_ids]