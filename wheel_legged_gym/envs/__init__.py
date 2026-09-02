# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from wheel_legged_gym.utils.task_registry import task_registry
from wheel_legged_gym.utils.task_registry_stage import task_registry_stage
import os

# -------------------------- 轮足 -------------------------- #

""" 标准版 """
from .cowa.cowa_config import CowaCfg, CowaCfgPPO
from .cowa.cowa_env import CowaEnv
task_registry.register(
    "cowa", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" 标准版-[commands和prop_obs分开存储] """
from .cowa_prop.cowa_prop_config import CowaCfg, CowaCfgPPO
from .cowa_prop.cowa_prop_env import CowaEnv
task_registry.register(
    "cowa_prop", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" 空中跟随sin曲线 """
from .cowa_net_sin.cowa_net_sin_config import CowaCfg_Net_Sin, CowaCfgPPO_Net_Sin
from .cowa_net_sin.cowa_net_sin_env import CowaEnv_net_sin
task_registry.register(
    "cowa_net_sin", CowaEnv_net_sin, CowaCfg_Net_Sin(), CowaCfgPPO_Net_Sin()
)


""" 空中跟随sin曲线10dof """
from .cowa_net_sin_10dof.cowa_net_sin_10dof_config import CowaCfg_Net_Sin, CowaCfgPPO_Net_Sin
from .cowa_net_sin_10dof.cowa_net_sin_10dof_env import CowaEnv_net_sin
task_registry.register(
    "cowa_net_sin_10dof", CowaEnv_net_sin, CowaCfg_Net_Sin(), CowaCfgPPO_Net_Sin()
)

""" 悬空 sin 跟踪：隔离关节伺服子系统 sim2real gap（12 dof，无 IMU/cmd） """
from .quadruped_joint_track.quadruped_joint_track_config import QuadJointTrackCfg, QuadJointTrackCfgPPO
from .quadruped_joint_track.quadruped_joint_track_env import QuadJointTrackEnv
task_registry.register(
    "quadruped_joint_track", QuadJointTrackEnv, QuadJointTrackCfg(), QuadJointTrackCfgPPO()
)

""" EST """
from .cowa_est.cowa_est_config import CowaCfg_EST, CowaCfgPPO_EST
from .cowa_est.cowa_est_env import CowaEnv_EST
task_registry.register(
    "cowa_est", CowaEnv_EST, CowaCfg_EST(), CowaCfgPPO_EST()
)

""" Dual History """
from .cowa_dh.cowa_dh_config import CowaCfg_DH, CowaCfgPPO_DH
from .cowa_dh.cowa_dh_env import CowaEnv_DH
task_registry.register(
    "cowa_dh", CowaEnv_DH, CowaCfg_DH(), CowaCfgPPO_DH()
)

""" Dual History 10dof """
from .cowa_10dof.cowa_10dof_config import CowaCfg, CowaCfgPPO
from .cowa_10dof.cowa_10dof_env import CowaEnv
task_registry.register(
    "cowa_10dof", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History 10dof teacher"""
from .cowa_10dof_teacher.cowa_10dof_teacher_config import CowaCfg, CowaCfgPPO
from .cowa_10dof_teacher.cowa_10dof_teacher_env import CowaEnv
task_registry.register(
    "cowa_10dof_teacher", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History 10dof distill depth camera"""
from .cowa_10dof_distill_depth_camera.cowa_10dof_distill_depth_camera_config import CowaCfg, CowaCfgPPO
from .cowa_10dof_distill_depth_camera.cowa_10dof_distill_depth_camera_env import CowaEnv
task_registry.register(
    "cowa_10dof_distill_depth_camera", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History 10dof switch"""
from .cowa_10dof_switch.cowa_10dof_switch_config import CowaCfg, CowaCfgPPO
from .cowa_10dof_switch.cowa_10dof_switch_env import CowaEnv
task_registry.register(
    "cowa_10dof_switch", CowaEnv, CowaCfg(), CowaCfgPPO()
)


""" Dual History wbc """
from .cowa_wbc.cowa_wbc_config import CowaCfg, CowaCfgPPO
from .cowa_wbc.cowa_wbc_env import CowaEnv
task_registry.register(
    "cowa_wbc", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History wbc stage"""
from .cowa_wbc_stage.cowa_wbc_stage_config import CowaCfg, CowaCfgPPO
from .cowa_wbc_stage.cowa_wbc_stage_env import CowaEnv
task_registry_stage.register(
    "cowa_wbc_stage", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History wbc mix advantages"""
from .cowa_wbc_mix_advantages.cowa_wbc_mix_advantages_config import CowaCfg, CowaCfgPPO
from .cowa_wbc_mix_advantages.cowa_wbc_mix_advantages_env import CowaEnv
task_registry.register(
    "cowa_wbc_mix_advantages", CowaEnv, CowaCfg(), CowaCfgPPO()
)


""" Dual History arm """
from .cowa_arm.cowa_arm_config import CowaCfg, CowaCfgPPO
from .cowa_arm.cowa_arm_env import CowaEnv
task_registry.register(
    "cowa_arm", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History arm force"""
from .cowa_arm_force.cowa_arm_force_config import CowaCfg, CowaCfgPPO
from .cowa_arm_force.cowa_arm_force_env import CowaEnv
task_registry.register(
    "cowa_arm_force", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History 8dof Trigger"""
from .cowa_8dof_trigger.cowa_8dof_trigger_config import CowaCfg, CowaCfgPPO
from .cowa_8dof_trigger.cowa_8dof_trigger_env import CowaEnv
task_registry.register(
    "cowa_8dof_trigger", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History 10dof height maps"""
from .cowa_10dof_height_maps.cowa_10dof_height_maps_config import CowaCfg, CowaCfgPPO
from .cowa_10dof_height_maps.cowa_10dof_height_maps_env import CowaEnv
task_registry.register(
    "cowa_10dof_height_maps", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" Dual History 10dof P3O """
from .cowa_10dof_p3o.cowa_10dof_p3o_config import CowaCfg, CowaCfgPPO
from .cowa_10dof_p3o.cowa_10dof_p3o_env import CowaEnv
task_registry.register(
    "cowa_10dof_p3o", CowaEnv, CowaCfg(), CowaCfgPPO()
)

"""VAE 10dof """
from .cowa_10dof_vae.cowa_10dof_vae_config import CowaCfg, CowaCfgPPO
from .cowa_10dof_vae.cowa_10dof_vae_env import CowaEnv
task_registry.register(
    "cowa_10dof_vae", CowaEnv, CowaCfg(), CowaCfgPPO()
)


""" HIM + DH """
from .cowa_him.cowa_him_config import CowaCfg_HIM, CowaCfgPPO_HIM
from .cowa_him.cowa_him_env import CowaEnv_HIM
task_registry.register(
    "cowa_him", CowaEnv_HIM, CowaCfg_HIM(), CowaCfgPPO_HIM()
)

# -------------------------- 有机械臂 没有lidar -------------------------- #

# <><><> estimator <><><>
from .cowa_w_arm_est_add_arm.cowa_w_arm_est_add_arm_config import CowaCfg_Arm, CowaCfgPPO_Arm
from .cowa_w_arm_est_add_arm.cowa_w_arm_est_add_arm_env import CowaFreeEnv_Arm
task_registry.register(
    "cowa_w_arm_est_add_arm", CowaFreeEnv_Arm, CowaCfg_Arm(), CowaCfgPPO_Arm()
)

# <><><> him <><><>
from .cowa_w_arm_him_add_arm.cowa_w_arm_him_add_arm_config import CowaCfg_Arm, CowaCfgPPO_Arm
from .cowa_w_arm_him_add_arm.cowa_w_arm_him_add_arm_env import CowaFreeEnv_Arm
task_registry.register(
    "cowa_w_arm_him_add_arm", CowaFreeEnv_Arm, CowaCfg_Arm(), CowaCfgPPO_Arm()
)

# <><><> roa <><><>
from .cowa_w_arm_roa_add_arm.cowa_w_arm_roa_add_arm_config import CowaCfg_Arm, CowaCfgPPO_Arm
from .cowa_w_arm_roa_add_arm.cowa_w_arm_roa_add_arm_env import CowaFreeEnv_Arm
task_registry.register(
    "cowa_w_arm_roa_add_arm", CowaFreeEnv_Arm, CowaCfg_Arm(), CowaCfgPPO_Arm()
)

# -------------------------- 没有机械臂 有lidar -------------------------- #
## roa + height_l -> hist_encoder
from .cowa_wo_arm_roa_terrain_origin.cowa_wo_arm_roa_terrain_config import CowaCfg, CowaCfgPPO
from .cowa_wo_arm_roa_terrain_origin.cowa_wo_arm_roa_terrain_env import CowaFreeEnv
task_registry.register(
    "cowa_wo_arm_roa_terrain_origin", CowaFreeEnv, CowaCfg(), CowaCfgPPO()
)

## roa height_l -> actor
from .cowa_wo_arm_roa_terrain.cowa_wo_arm_roa_terrain_config import CowaCfg_2, CowaCfgPPO_2
from .cowa_wo_arm_roa_terrain.cowa_wo_arm_roa_terrain_env import CowaFreeEnv_2
task_registry.register(
    "cowa_wo_arm_roa_terrain", CowaFreeEnv_2, CowaCfg_2(), CowaCfgPPO_2()
)


## PPO+height
from .cowa_wo_arm_ppo_terrain.cowa_wo_arm_ppo_terrain_config import CowaCfg, CowaCfgPPO
from .cowa_wo_arm_ppo_terrain.cowa_wo_arm_ppo_terrain_env import CowaFreeEnv
task_registry.register(
    "cowa_wo_arm_ppo_terrain", CowaFreeEnv, CowaCfg(), CowaCfgPPO()
)

## EST+height
from .cowa_wo_arm_est_terrain.cowa_wo_arm_est_terrain_config import CowaCfg, CowaCfgPPO
from .cowa_wo_arm_est_terrain.cowa_wo_arm_est_terrain_env import CowaFreeEnv
task_registry.register(
    "cowa_wo_arm_est_terrain", CowaFreeEnv, CowaCfg(), CowaCfgPPO()
)


## EST+height+feet-pos
from .cowa_wo_arm_est_terrain_2.cowa_wo_arm_est_terrain_2_config import CowaCfg, CowaCfgPPO
from .cowa_wo_arm_est_terrain_2.cowa_wo_arm_est_terrain_2_env import CowaFreeEnv
task_registry.register(
    "cowa_wo_arm_est_terrain_2", CowaFreeEnv, CowaCfg(), CowaCfgPPO()
)

## EST+height but like hz success models' configs
from .cowa_wo_arm_ts_terrain.cowa_wo_arm_ts_terrain_config import CowaCfg, CowaCfgPPO
from .cowa_wo_arm_ts_terrain.cowa_wo_arm_ts_terrain_env import CowaFreeEnv
task_registry.register(
    "cowa_wo_arm_ts_terrain", CowaFreeEnv, CowaCfg(), CowaCfgPPO()
)

## tron1_blind_est_ts
from .tron1_blind_est_ts.tron1_blind_est_ts_config import Tron1Cfg, Tron1CfgPPO
from .tron1_blind_est_ts.tron1_blind_est_ts_env import Tron1FreeEnv
task_registry.register(
    "tron1_trigger", Tron1FreeEnv, Tron1Cfg(), Tron1CfgPPO()
)
from .cowa_8dof.cowa_8dof_config import CowaCfg, CowaCfgPPO
from .cowa_8dof.cowa_8dof_env import CowaEnv
task_registry.register(
    "cowa_8dof", CowaEnv, CowaCfg(), CowaCfgPPO()
)

""" 4dof版 """
from .cowa_4dof.cowa_4dof_config import CowaCfg, CowaCfgPPO
from .cowa_4dof.cowa_4dof_env import CowaEnv
task_registry.register(
    "cowa_4dof", CowaEnv, CowaCfg(), CowaCfgPPO()
)

# -------------------------- 四足机器人 -------------------------- #
from .quadruped.quadruped_config import QuadCfg, QuadCfgPPO
from .quadruped.quadruped_env import QuadEnv
task_registry.register(
    "quadruped", QuadEnv, QuadCfg(), QuadCfgPPO()
)

""" trot_plane版 """
from .quadruped_wtw.quadruped_wtw_config import QuadWtwCfg, QuadWtwCfgPPO
from .quadruped_wtw.quadruped_wtw_env import QuadWtwEnv
task_registry.register(
    "quadruped_wtw", QuadWtwEnv, QuadWtwCfg(), QuadWtwCfgPPO()
)

""" trot_slope版 """
from .quadruped_wtw_slope.quadruped_wtw_slope_config import QuadWtwCfg, QuadWtwCfgPPO
from .quadruped_wtw_slope.quadruped_wtw_slope_env import QuadWtwEnv
task_registry.register(
    "quadruped_wtw_slope", QuadWtwEnv, QuadWtwCfg(), QuadWtwCfgPPO()
)

""" MoE CTS版 """
from .quadruped_moe_cts.quadruped_moe_cts_config import QuadMoECTSCfg, QuadMoECTSCfgPPO
from .quadruped_moe_cts.quadruped_moe_cts_env import QuadMoECTSEnv
task_registry.register(
    "quadruped_moe_cts", QuadMoECTSEnv, QuadMoECTSCfg(), QuadMoECTSCfgPPO()
)

""" 无wtw moe_cts版 """
from .quadruped_mc.quadruped_mc_config import QuadMCCfg, QuadMCCfgPPO
from .quadruped_mc.quadruped_mc_env import QuadMCEnv
task_registry.register(
    "quadruped_mc", QuadMCEnv, QuadMCCfg(), QuadMCCfgPPO()
)


# -------------------------- 四足机器人(带机械臂) -------------------------- #
""" trot_homie版 """
from .quadruped_wtw_arm.quadruped_wtw_arm_config import QuadWtwCfg, QuadWtwCfgPPO
from .quadruped_wtw_arm.quadruped_wtw_arm_env import QuadWtwEnv
task_registry.register(
    "quadruped_wtw_arm", QuadWtwEnv, QuadWtwCfg(), QuadWtwCfgPPO()
)

""" trot_homie_dh版 """
from .quadruped_wtw_arm_dh.quadruped_wtw_arm_dh_config import QuadWtwCfg, QuadWtwCfgPPO
from .quadruped_wtw_arm_dh.quadruped_wtw_arm_dh_env import QuadWtwEnv
task_registry.register(
    "quadruped_wtw_arm_dh", QuadWtwEnv, QuadWtwCfg(), QuadWtwCfgPPO()
)

""" wtw_arm_fix版 """
from .quadruped_wtw_arm_fix.quadruped_wtw_arm_fix_config import QuadWtwCfg, QuadWtwCfgPPO
from .quadruped_wtw_arm_fix.quadruped_wtw_arm_fix_env import QuadWtwEnv
task_registry.register(
    "quadruped_wtw_arm_fix", QuadWtwEnv, QuadWtwCfg(), QuadWtwCfgPPO()
)

""" wtw_arm_him_fix版 """
from .quadruped_wtw_him_arm_fix.quadruped_wtw_him_arm_fix_config import QuadWtwCfg_HIM, QuadWtwCfgPPO_HIM
from .quadruped_wtw_him_arm_fix.quadruped_wtw_him_arm_fix_env import QuadWtwEnv_HIM
task_registry.register(
    "quadruped_wtw_him_arm_fix", QuadWtwEnv_HIM, QuadWtwCfg_HIM(), QuadWtwCfgPPO_HIM()
)

"""WTW HIM with an optional 10 Hz actor height scan."""
from .quadruped_wtw_him_arm_fix_height_scan.quadruped_wtw_him_arm_fix_height_scan_config import (
    QuadWtwCfg_HIM as QuadWtwHeightScanCfg,
    QuadWtwCfgPPO_HIM as QuadWtwHeightScanCfgPPO,
)
from .quadruped_wtw_him_arm_fix_height_scan.quadruped_wtw_him_arm_fix_height_scan_env import (
    QuadWtwEnv_HIM as QuadWtwHeightScanEnv,
)
task_registry.register(
    "quadruped_wtw_him_arm_fix_height_scan",
    QuadWtwHeightScanEnv,
    QuadWtwHeightScanCfg(),
    QuadWtwHeightScanCfgPPO(),
)

""" wtw + AMP 融合：姿态调整 + 专家柔和先验（判别器姿态免疫）"""
from .quadruped_wtw_him_arm_fix_amp.quadruped_wtw_him_arm_fix_amp_config import QuadWtwAmpCfg_HIM, QuadWtwAmpCfgPPO_HIM
from .quadruped_wtw_him_arm_fix_amp.quadruped_wtw_him_arm_fix_amp_env import QuadWtwAmpEnv_HIM
task_registry.register(
    "quadruped_wtw_him_arm_fix_amp", QuadWtwAmpEnv_HIM, QuadWtwAmpCfg_HIM(), QuadWtwAmpCfgPPO_HIM()
)

""" quadruped_arm_him 版 """
from .quadruped_arm_him.quadruped_arm_him_config import QuadCfg_HIM ,QuadCfgPPO_HIM
from .quadruped_arm_him.quadruped_arm_him_env import QuadHIMEnv
task_registry.register(
    "quadruped_arm_him", QuadHIMEnv, QuadCfg_HIM(), QuadCfgPPO_HIM()
)

""" walking版 """
from .amp_d1.walking_d1.walking_d1_config import WalkingD1Cfg, WalkingD1CfgPPO
from .amp_d1.walking_d1.walking_d1_env import WalkingD1Env
task_registry.register(
    "walking_d1", WalkingD1Env, WalkingD1Cfg(), WalkingD1CfgPPO()
)
""" walking_amp版 """
from .amp_d1.amp_d1_config import AmpD1Cfg, AmpD1CfgPPO
from .amp_d1.amp_d1_env import AmpD1Env
task_registry.register(
    "amp_d1", AmpD1Env, AmpD1Cfg(), AmpD1CfgPPO()
)

""""quadruped_arm_him_amp版"""
from .quadruped_arm_him_amp.quadruped_arm_him_amp_config import QuadCfg_HIM_AMP,QuadCfgPPO_HIM_AMP
from .quadruped_arm_him_amp.quadruped_arm_him_amp_env import QuadHIMAmpEnv
task_registry.register(
    "quadruped_arm_him_amp",QuadHIMAmpEnv,QuadCfg_HIM_AMP(),QuadCfgPPO_HIM_AMP()
)

""""quadruped_arm_him_amp + WTW 前视高程图版 (x 0.5~1.0, y -0.5~0.5, 66 点)"""
from .quadruped_arm_him_amp_heightmap.quadruped_arm_him_amp_heightmap_config import (
    QuadCfg_HIM_AMP_Heightmap, QuadCfgPPO_HIM_AMP_Heightmap)
from .quadruped_arm_him_amp_heightmap.quadruped_arm_him_amp_heightmap_env import QuadHIMAmpHeightmapEnv
task_registry.register(
    "quadruped_arm_him_amp_heightmap", QuadHIMAmpHeightmapEnv,
    QuadCfg_HIM_AMP_Heightmap(), QuadCfgPPO_HIM_AMP_Heightmap()
)

"""quadruped_arm_amp + actor 前视高程图版。"""
from .quadruped_arm_amp_heightmap.quadruped_arm_amp_heightmap_config import (
    QuadCfg_AMP_Heightmap, QuadCfgPPO_AMP_Heightmap)
from .quadruped_arm_amp_heightmap.quadruped_arm_amp_heightmap_env import QuadAmpHeightmapEnv
task_registry.register(
    "quadruped_arm_amp_heightmap", QuadAmpHeightmapEnv,
    QuadCfg_AMP_Heightmap(), QuadCfgPPO_AMP_Heightmap()
)

# -------------------------- 四轮足机器人 -------------------------- #

"""test版"""
from .quadwheel.quadwheel_config import QuadwheelCfg, QuadwheelCfgPPO
from .quadwheel.quadwheel_env import QuadwheelEnv
task_registry.register(
    "quadwheel", QuadwheelEnv, QuadwheelCfg(), QuadwheelCfgPPO()
)
