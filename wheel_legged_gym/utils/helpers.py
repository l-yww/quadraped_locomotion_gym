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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR

def get_joint_names(DOF: int, WLR_index: str = None):
    if DOF == 4:
        return [
            "left_hip_pitch_joint",
            "left_wheel_joint",
            "right_hip_pitch_joint",
            "right_wheel_joint",
        ]
    elif DOF == 6:
        if WLR_index == 'v2_arm':
            return [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
        else:
            return [
                "left_hip_pitch_joint",
                "left_knee_pitch_joint",
                "left_wheel_joint",
                "right_hip_pitch_joint",
                "right_knee_pitch_joint",
                "right_wheel_joint",
            ]
    elif DOF == 8:
        return [
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_pitch_joint",
            "left_wheel_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_pitch_joint",
            "right_wheel_joint",
        ]
    elif DOF == 10:
        return [
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_pitch_joint",
            "left_wheel_joint",
            "left_foot_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_pitch_joint",
            "right_wheel_joint",
            "right_foot_joint",
        ]
    elif DOF == 16:
        return [
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_pitch_joint",
            "left_wheel_joint",
            "left_foot_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_pitch_joint",
            "right_wheel_joint",
            "right_foot_joint",

            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]

def get_quadruped_joint_names():
    return [
            "FL_hip_joint",
            "RL_hip_joint",
            "FR_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "RL_thigh_joint",
            "FR_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "RL_calf_joint",
            "FR_calf_joint",
            "RR_calf_joint",
        ]

def get_quadruped_arm_joint_names():
    return [
            "FL_hip_joint",
            "RL_hip_joint",
            "FR_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "RL_thigh_joint",
            "FR_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "RL_calf_joint",
            "FR_calf_joint",
            "RR_calf_joint",
            "arm17",
            "arm18",
            "arm19",
            "arm20",
            "arm21",
            "arm22",
        ]

def get_quadwheel_joint_names():
    return [
            "fl_hipx_joint",   # FL hip
            "fr_hipx_joint",   # FR hip
            "hl_hipx_joint",   # RL hip
            "hr_hipx_joint",   # RR hip
            "fl_hipy_joint",   # FL thigh
            "fr_hipy_joint",   # FR thigh
            "hl_hipy_joint",   # RL thigh
            "hr_hipy_joint",   # RR thigh
            "fl_knee_joint",   # FL calf
            "fr_knee_joint",   # FR calf
            "hl_knee_joint",   # RL calf
            "hr_knee_joint",   # RR calf
            "fl_wheel_joint",  # FL wheel
            "fr_wheel_joint",  # FR wheel
            "hl_wheel_joint",  # RL wheel
            "hr_wheel_joint",  # RR wheel
        ]
   

def get_stiffness_damping(DOF: int, WLR_index: str = None):
    # 定义各关节默认参数
    base_stiffness = {
        "hip_roll": 110.0,
        "hip_pitch": 75.0,
        "knee": 160.0,
        "wheel": 0.0,
        "foot": 60.0,

        # 在第二阶段load机械臂的policy的时候可以适当降低pd
        "joint1": 20.0,
        "joint2": 20.0,
        "joint3": 20.0,
        "joint4": 20.0,
        "joint5": 20.0,
        "joint6": 20.0,

    }
    base_damping = {
        "hip_roll": 5.0,
        "hip_pitch": 4.0,
        "knee": 6.0,
        "wheel": 3.0,
        "foot": 5.0,

        # 在第二阶段load机械臂的policy的时候可以适当降低pd
        "joint1": 1.5,
        "joint2": 1.5,
        "joint3": 1.5,
        "joint4": 1.5,
        "joint5": 1.5,
        "joint6": 1.5,
    }

    # 根据 DOF 获取关节名
    joint_names = get_joint_names(DOF, WLR_index)

    # 遍历 joint_names，按名字类别分配 stiffness/damping
    stiffness = {}
    damping = {}
    for j in joint_names:
        if "hip_roll" in j:
            stiffness[j] = base_stiffness["hip_roll"]
            damping[j] = base_damping["hip_roll"]
        elif "hip_pitch" in j:
            stiffness[j] = base_stiffness["hip_pitch"]
            damping[j] = base_damping["hip_pitch"]
        elif "knee" in j:
            stiffness[j] = base_stiffness["knee"]
            damping[j] = base_damping["knee"]
        elif "wheel" in j:
            stiffness[j] = base_stiffness["wheel"]
            damping[j] = base_damping["wheel"]
        elif "foot" in j:  
            stiffness[j] = base_stiffness["foot"]
            damping[j] = base_damping["foot"]
        elif "joint1" in j:
            stiffness[j] = base_stiffness["joint1"]
            damping[j] = base_damping["joint1"]
        elif "joint2" in j:
            stiffness[j] = base_stiffness["joint2"]
            damping[j] = base_damping["joint2"]
        elif "joint3" in j:
            stiffness[j] = base_stiffness["joint3"]
            damping[j] = base_damping["joint3"]
        elif "joint4" in j:
            stiffness[j] = base_stiffness["joint4"]
            damping[j] = base_damping["joint4"]
        elif "joint5" in j:
            stiffness[j] = base_stiffness["joint5"]
            damping[j] = base_damping["joint5"]
        elif "joint6" in j:
            stiffness[j] = base_stiffness["joint6"]
            damping[j] = base_damping["joint6"]
        else:
            # 万一以后有新关节，就给默认 0
            stiffness[j] = 0.0
            damping[j] = 0.0

    return stiffness, damping


# # ================================== dog1 辨识参数 ======================================== #
# def get_quadruped_default_joint_friction(quad_index: str):
#     # Joint order:  FL_hip, FL_thigh, FL_calf, 
#     #               FR_hip, FR_thigh, FR_calf,    
#     #               RL_hip, RL_thigh, RL_calf, 
#     #               RR_hip, RR_thigh, RR_calf
#     joint_friction = [
#             0.0054971277713775635, 6.631016731262207e-06, 0.05982851982116699,
#             0.007290467619895935, 0.015472427010536194, 0.07318446040153503,
#             0.004257142543792725, 0.006278917193412781, 0.04780678451061249,
#             0.005522802472114563, 0.0010340213775634766, 0.02277398109436035,
#         ]
#     return joint_friction

# def get_quadruped_default_joint_damping(quad_index: str):
#     # Joint order:  FL_hip, FL_thigh, FL_calf, 
#     #               FR_hip, FR_thigh, FR_calf,    
#     #               RL_hip, RL_thigh, RL_calf, 
#     #               RR_hip, RR_thigh, RR_calf
#     joint_damping = [
#             1.2516975402832031e-05, 4.172325134277344e-06, 5.364418029785156e-06,
#             3.457069396972656e-05, 7.748603820800781e-06, 1.7881393432617188e-06,
#             4.172325134277344e-06, 7.152557373046875e-06, 3.5762786865234375e-06,
#             5.364418029785156e-06, 2.384185791015625e-06, 2.384185791015625e-06,
#         ]
#     return joint_damping

# def get_quadruped_default_joint_armature(quad_index: str):
#     # Joint order:  FL_hip, FL_thigh, FL_calf, 
#     #               FR_hip, FR_thigh, FR_calf,    
#     #               RL_hip, RL_thigh, RL_calf, 
#     #               RR_hip, RR_thigh, RR_calf
#     joint_armature = [
#             0.004417330026626587, 2.086162567138672e-07, 0.07663074135780334,
#             1.4007091522216797e-06, 3.2782554626464844e-07, 0.07500061392784119,
#             4.559755325317383e-06, 1.4007091522216797e-06, 0.07970243692398071,
#             7.450580596923828e-07, 3.5762786865234375e-07, 0.05527627468109131,
#         ]
#     return joint_armature
# # ================================== dog1 辨识参数 ======================================== #


# # ================================== dog2 辨识参数 v1 ======================================== #
# def get_quadruped_default_joint_friction(quad_index: str):

#     joint_friction = [
#             0.0166633278131485, 0.026315897703170776, 0.22496037185192108, 
#             0.01694570481777191, 0.023310795426368713, 0.21034517884254456, 
#             0.019308865070343018, 0.025806203484535217, 0.1560538411140442, 
#             0.018144428730010986, 0.029282137751579285, 0.15188813209533691]
#     return joint_friction

# def get_quadruped_default_joint_damping(quad_index: str):

#     joint_damping = [
#             0.5471593141555786, 0.312039852142334, 6.556510925292969e-05, 
#             0.1258307695388794, 0.364263653755188, 1.8477439880371094e-05, 
#             0.6220364570617676, 0.4942166805267334, 8.225440979003906e-05, 
#             0.0005000829696655273, 0.44378459453582764, 0.0001823902130126953]
#     return joint_damping

# def get_quadruped_default_joint_armature(quad_index: str):

#     joint_armature = [
#             8.553266525268555e-06, 1.1920928955078125e-06, 0.08518049120903015, 
#             2.5570392608642578e-05, 1.2814998626708984e-06, 0.08498609066009521, 
#             1.7970800399780273e-05, 1.6093254089355469e-06, 0.07017886638641357, 
#             0.02426353096961975, 1.6093254089355469e-06, 0.0852460265159607]
#     return joint_armature
# # ================================== dog2 辨识参数 v1 ======================================== #


def get_quadruped_default_joint_friction(quad_index: str):
    # Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    #              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
    # 按 quad_index 选择对应台狗的辨识参数
    if quad_index == '1':
        joint_friction = [   # dog1 辨识参数
            0.0054971277713775635, 6.631016731262207e-06, 0.05982851982116699,
            0.007290467619895935, 0.015472427010536194, 0.07318446040153503,
            0.004257142543792725, 0.006278917193412781, 0.04780678451061249,
            0.005522802472114563, 0.0010340213775634766, 0.02277398109436035]
    else:
        joint_friction = [   # dog2 辨识参数 v2(默认)
            0.013015717267990112, 0.012962326407432556, 0.10828112065792084,
            0.010571613907814026, 0.010165184736251831, 0.0678667277097702,
            0.014302417635917664, 0.018830284476280212, 0.05405837297439575,
            0.007468432188034058, 0.019104883074760437, 0.0686643123626709]
    return joint_friction

def get_quadruped_default_joint_damping(quad_index: str):
    # Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    #              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
    if quad_index == '1':
        joint_damping = [   # dog1 辨识参数
            1.2516975402832031e-05, 4.172325134277344e-06, 5.364418029785156e-06,
            3.457069396972656e-05, 7.748603820800781e-06, 1.7881393432617188e-06,
            4.172325134277344e-06, 7.152557373046875e-06, 3.5762786865234375e-06,
            5.364418029785156e-06, 2.384185791015625e-06, 2.384185791015625e-06]
    else:
        joint_damping = [   # dog2 辨识参数 v2(默认)
            0.18128454685211182, 1.3113021850585938e-05, 3.0994415283203125e-05,
            0.00018537044525146484, 7.748603820800781e-05, 2.384185791015625e-05,
            0.13528823852539062, 5.3048133850097656e-05, 4.112720489501953e-05,
            4.172325134277344e-05, 8.881092071533203e-05, 1.8477439880371094e-05]
    return joint_damping

def get_quadruped_default_joint_armature(quad_index: str):
    # Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    #              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
    if quad_index == '1':
        joint_armature = [   # dog1 辨识参数
            0.004417330026626587, 2.086162567138672e-07, 0.07663074135780334,
            1.4007091522216797e-06, 3.2782554626464844e-07, 0.07500061392784119,
            4.559755325317383e-06, 1.4007091522216797e-06, 0.07970243692398071,
            7.450580596923828e-07, 3.5762786865234375e-07, 0.05527627468109131]
    else:
        joint_armature = [   # dog2 辨识参数 v2(默认)
            2.682209014892578e-06, 7.63237476348877e-05, 0.09383302927017212,
            0.032879263162612915, 5.155801773071289e-06, 0.09398800134658813,
            6.973743438720703e-06, 0.0034439563751220703, 0.0872204601764679,
            0.04760277271270752, 9.894371032714844e-06, 0.0929601788520813]
    return joint_armature


def get_quadruped_arm_default_joint_friction(quad_index: str):
    # Joint order:  FL_hip, FL_thigh, FL_calf, 
    #               FR_hip, FR_thigh, FR_calf,    
    #               RL_hip, RL_thigh, RL_calf, 
    #               RR_hip, RR_thigh, RR_calf
    joint_friction = [
            0.0054971277713775635, 6.631016731262207e-06, 0.05982851982116699,
            0.007290467619895935, 0.015472427010536194, 0.07318446040153503,
            0.004257142543792725, 0.006278917193412781, 0.04780678451061249,
            0.005522802472114563, 0.0010340213775634766, 0.02277398109436035,
            0, 0, 0, 0, 0, 0,
        ]
    return joint_friction

def get_quadruped_arm_default_joint_damping(quad_index: str):
    # Joint order:  FL_hip, FL_thigh, FL_calf, 
    #               FR_hip, FR_thigh, FR_calf,    
    #               RL_hip, RL_thigh, RL_calf, 
    #               RR_hip, RR_thigh, RR_calf
    joint_damping = [
            1.2516975402832031e-05, 4.172325134277344e-06, 5.364418029785156e-06,
            3.457069396972656e-05, 7.748603820800781e-06, 1.7881393432617188e-06,
            4.172325134277344e-06, 7.152557373046875e-06, 3.5762786865234375e-06,
            5.364418029785156e-06, 2.384185791015625e-06, 2.384185791015625e-06,
            0, 0, 0, 0, 0, 0,
        ]
    return joint_damping

def get_quadruped_arm_default_joint_armature(quad_index: str):
    # Joint order:  FL_hip, FL_thigh, FL_calf, 
    #               FR_hip, FR_thigh, FR_calf,    
    #               RL_hip, RL_thigh, RL_calf, 
    #               RR_hip, RR_thigh, RR_calf
    joint_armature = [
            0.004417330026626587, 2.086162567138672e-07, 0.07663074135780334,
            1.4007091522216797e-06, 3.2782554626464844e-07, 0.07500061392784119,
            4.559755325317383e-06, 1.4007091522216797e-06, 0.07970243692398071,
            7.450580596923828e-07, 3.5762786865234375e-07, 0.05527627468109131,
            0, 0, 0, 0, 0, 0,
        ]
    return joint_armature

def get_quadwheel_default_joint_friction(quad_index: str):
    # Joint order:  FL_hip, FL_thigh, FL_calf, 
    #               FR_hip, FR_thigh, FR_calf,    
    #               RL_hip, RL_thigh, RL_calf, 
    #               RR_hip, RR_thigh, RR_calf
    joint_friction = [
                0,0,0,0,
                0,0,0,0,
                0,0,0,0,
                0,0,0,0,
        ]
    return joint_friction

def get_quadwheel_default_joint_damping(quad_index: str):
    # Joint order:  FL_hip, FL_thigh, FL_calf, 
    #               FR_hip, FR_thigh, FR_calf,    
    #               RL_hip, RL_thigh, RL_calf, 
    #               RR_hip, RR_thigh, RR_calf
    joint_damping = [
                0,0,0,0,
                0,0,0,0,
                0,0,0,0,
                0,0,0,0,
        ]
    return joint_damping

def get_quadwheel_default_joint_armature(quad_index: str):
    # Joint order:  FL_hip, FL_thigh, FL_calf, 
    #               FR_hip, FR_thigh, FR_calf,    
    #               RL_hip, RL_thigh, RL_calf, 
    #               RR_hip, RR_thigh, RR_calf
    joint_armature = [
                0,0,0,0,
                0,0,0,0,
                0,0,0,0,
                0,0,0,0,
        ]
    return joint_armature

def get_adam_default_joint_friction(adam_index: str):
    joint_friction = [0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0]
    return joint_friction

def get_adam_default_joint_damping(adam_index: str):
    joint_damping = [0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0]
    return joint_damping

def get_adam_default_joint_armature(adam_index: str):
    joint_armature = [0.13426, 0.281573, 0.23409, 0.13426, 0.0549, 0.0549,
                             0.13426, 0.281573, 0.23409, 0.13426, 0.0549, 0.0549,
                             0.23409, 0.23409, 0.23409,
                             0.1578807, 0.1578807, 0.0423963, 0.0423963,
                             0.03, 0.03, 0.03,
                             0.1578807, 0.1578807, 0.0423963, 0.0423963,
                             0.03, 0.03, 0.03]
    return joint_armature

def get_default_joint_friction(WLR_index: str, DOF: int):
    friction_8dof = None
    friction_10dof = None
    friction_16dof = None
    friction_6dof_arm = None  
    
    if WLR_index == '1':
        friction_8dof = [0.050, 0.045, 0.08, 0.0,
                         0.050, 0.045, 0.2,  0.0]   # lrk + wh
    elif WLR_index == '2':
        friction_8dof = [0.050, 0.045, 0.08, 0.0,
                         0.050, 0.045, 0.2,  0.0]
    elif WLR_index == '3':
        friction_8dof = [0.030, 0.045, 0.25, 0.0,
                         0.030, 0.045, 0.15, 0.0]  # lq + wh
    elif WLR_index == 'v2':
        friction_8dof = [0.08, 0.08, 0.12, 0.10930828005075455,
                         0.08, 0.08, 0.03, 0.03416772186756134]  # lrk (12.22)
        # friction_8dof = [0.07169139385223389, 0.07957252860069275, 0.12, 0.10930828005075455,
        #                  0.07725889980792999, 0.08428050577640533, 0.03, 0.03416772186756134]  # wh (3.31)
    elif WLR_index == 'v2_10dof':
        friction_10dof = [0.08, 0.08, 0.12, 0.2, 0.10930828005075455, 
                         0.08, 0.08, 0.03, 0.2, 0.03416772186756134]  # lrk (12.22)
        # friction_10dof = [0.07169139385223389, 0.07957252860069275, 0.12, 0.16581368446350098, 0.10930828005075455, 
        #                  0.07725889980792999, 0.08428050577640533, 0.03, 0.17369762063026428, 0.03416772186756134]  # wh (3.31)
    elif WLR_index == 'v2_16dof':
        friction_16dof = [0.05350407958030701, 0.07973851263523102, 0.08691060543060303, 0.13093066215515137, 0.12061440199613571, 0.1921817660331726, \
                         0.08, 0.08, 0.12, 0.2, 0.10930828005075455, 
                         0.08, 0.08, 0.03, 0.2, 0.03416772186756134]  
        # friction_16dof = [0.05350407958030701, 0.07973851263523102, 0.08691060543060303, 0.13093066215515137, 0.12061440199613571, 0.1921817660331726, \
        #                     0.07169139385223389, 0.07957252860069275, 0.12, 0.16581368446350098, 0.10930828005075455, \
        #                     0.07725889980792999, 0.08428050577640533, 0.03, 0.17369762063026428, 0.03416772186756134]  # wh (3.31)
    elif WLR_index == 'v2_arm':  # 
        friction_6dof_arm = [0.05350407958030701, 0.07973851263523102, 0.08691060543060303, 0.13093066215515137, 0.12061440199613571, 0.1921817660331726]  # wh (3.31) 
    else:
        raise ValueError(f"Unsupported WLR_index={WLR_index}")

    # 根据 DOF 选取索引
    if DOF == 8:
        return friction_8dof
    elif DOF == 6:
        if WLR_index == 'v2_arm':
            return friction_6dof_arm
        else:
            indices = [1, 2, 3, 5, 6, 7]
            return [friction_8dof[i] for i in indices]
    elif DOF == 4:
        indices = [1, 3, 5, 7]
        return [friction_8dof[i] for i in indices]
    elif DOF == 10:
        return friction_10dof
    elif DOF == 16:
        return friction_16dof
    else:
        raise ValueError(f"Unsupported DOF={DOF}")


def get_default_joint_damping(WLR_index: str, DOF: int):
    damping_8dof = None
    damping_10dof = None
    damping_16dof = None
    damping_6dof_arm = None  
    
    # 8dof 完整表
    if WLR_index == '1':
        damping_8dof = [8, 12, 5, 0.15,
                         8, 12, 5, 0.15]    # lrk + wh
    elif WLR_index == '2':
        damping_8dof = [8, 12, 8, 0.03,
                         8, 12, 8, 0.03,]
    elif WLR_index == '3':
        damping_8dof = [13, 12, 8, 0.015,
                         13, 12, 8, 0.015,]  # lq + wh
    elif WLR_index == 'v2':
        damping_8dof = [8, 10, 6, 0.09479373693466187,
                         8, 16, 4, 0.03290677070617676]  # lrk (12.22)
        # damping_8dof = [11.566535949707031, 8.256438255310059, 6, 0.09479373693466187,
        #                  12.644949913024902, 8.765789031982422, 4, 0.03290677070617676]  # wh (3.31)
    elif WLR_index == 'v2_10dof':
        damping_10dof = [8, 10, 6, 5, 0.09479373693466187,
                         8, 16, 4, 5, 0.03290677070617676]  # lrk (12.22)
        # damping_10dof = [11.566535949707031, 8.256438255310059, 6, 1.586683988571167, 0.09479373693466187,
        #                  12.644949913024902, 8.765789031982422, 4, 1.4645415544509888, 0.03290677070617676]  # wh (3.31)
    elif WLR_index == 'v2_16dof':
        damping_16dof = [8.4097318649292, 8.934985160827637, 4.122097015380859, 0.17706364393234253, 1.4889142513275146, 1.6342681646347046, \
                         8, 10, 6, 5, 0.09479373693466187,
                         8, 16, 4, 5, 0.03290677070617676]  
        # damping_16dof = [8.4097318649292, 8.934985160827637, 4.122097015380859, 0.17706364393234253, 1.4889142513275146, 1.6342681646347046, \
        #                  11.566535949707031, 8.256438255310059, 6, 1.586683988571167, 0.09479373693466187, \
        #                  12.644949913024902, 8.765789031982422, 4, 1.4645415544509888, 0.03290677070617676]  # wh (3.31)
    elif WLR_index == 'v2_arm': 
        damping_6dof_arm = [8.4097318649292, 8.934985160827637, 4.122097015380859, 0.17706364393234253, 1.4889142513275146, 1.6342681646347046]  # wh (3.31) 
    else:
        raise ValueError(f"Unsupported WLR_index={WLR_index}")

    # 根据 DOF 选取索引
    if DOF == 8:
        return damping_8dof
    elif DOF == 6:
        if WLR_index == 'v2_arm':
            return damping_6dof_arm
        else:
            indices = [1, 2, 3, 5, 6, 7]
            return [damping_8dof[i] for i in indices]
    elif DOF == 4:
        indices = [1, 3, 5, 7]
        return [damping_8dof[i] for i in indices]
    elif DOF == 10:
        return damping_10dof
    elif DOF == 16:
        return damping_16dof
    else:
        raise ValueError(f"Unsupported DOF={DOF}")


def get_default_joint_armature(WLR_index: str, DOF: int):
    armature_8dof = None
    armature_10dof = None
    armature_16dof = None
    armature_6dof_arm = None 
    
    # 8dof 完整表
    if WLR_index == '1':
        armature_8dof = [96 * 120 ** 2 * 1e-7, 96 * 120 ** 2 * 1e-7, 0.2, 0.08,\
                         96 * 120 ** 2 * 1e-7, 96 * 120 ** 2 * 1e-7, 0.4, 0.08] # lrk + wh
    elif WLR_index == '2':
        armature_8dof = [96 * 120 ** 2 * 1e-7, 96 * 120 ** 2 * 1e-7, 0.5, 0.08,
                         96 * 120 ** 2 * 1e-7, 96 * 120 ** 2 * 1e-7, 0.5, 0.08]
    elif WLR_index == '3':
        armature_8dof = [96 * 120 ** 2 * 1e-7, 96 * 120 ** 2 * 1e-7, 0.5, 0.08,
                         96 * 120 ** 2 * 1e-7, 96 * 120 ** 2 * 1e-7, 0.5, 0.08] # lq + wh
    elif WLR_index == 'v2':     
        armature_8dof = [96 * 100 ** 2 * 1e-7, 96 * 100 ** 2 * 1e-7, 0.5, 0.058210402727127075,\
                         96 * 100 ** 2 * 1e-7, 96 * 100 ** 2 * 1e-7, 0.5, 0.0047006309032440186] # lrk (12.22)
        # armature_8dof = [0.7551929950714111, 0.3254449963569641, 0.5, 0.058210402727127075,\
        #                  0.6589693427085876, 0.3143749237060547, 0.5, 0.0047006309032440186]  # wh (3.31)
    elif WLR_index == 'v2_10dof':     
        armature_10dof = [96 * 100 ** 2 * 1e-7, 96 * 100 ** 2 * 1e-7, 0.5, 0.5, 0.058210402727127075, \
                         96 * 100 ** 2 * 1e-7, 96 * 100 ** 2 * 1e-7, 0.5, 0.5, 0.0047006309032440186] # lrk (12.22)
        # armature_10dof = [0.7551929950714111, 0.3254449963569641, 0.5, 0.14375269412994385, 0.058210402727127075, \
        #                  0.6589693427085876, 0.3143749237060547, 0.5, 0.1443158984184265, 0.0047006309032440186]  # wh (3.31)
    elif WLR_index == 'v2_16dof':
        armature_16dof = [0.49286651611328125, 0.454853355884552, 0.28742125630378723, 0.004195909481495619, 0.0050822035409510136, 0.044422734528779984, \
                         96 * 100 ** 2 * 1e-7, 96 * 100 ** 2 * 1e-7, 0.5, 0.5, 0.058210402727127075, \
                         96 * 100 ** 2 * 1e-7, 96 * 100 ** 2 * 1e-7, 0.5, 0.5, 0.0047006309032440186] 
        # armature_16dof = [0.49286651611328125, 0.454853355884552, 0.28742125630378723, 0.004195909481495619, 0.0050822035409510136, 0.044422734528779984, \
        #                  0.7551929950714111, 0.3254449963569641, 0.5, 0.14375269412994385, 0.058210402727127075, \
        #                  0.6589693427085876, 0.3143749237060547, 0.5, 0.1443158984184265, 0.0047006309032440186]  # wh (3.31)
    elif WLR_index == 'v2_arm':  
        armature_6dof_arm = [0.49286651611328125, 0.454853355884552, 0.28742125630378723, 0.004195909481495619, 0.0050822035409510136, 0.044422734528779984] # wh (3.31)
    else:
        raise ValueError(f"Unsupported WLR_index={WLR_index}")

    # 根据 DOF 选取索引
    if DOF == 8:
        return armature_8dof
    elif DOF == 6:
        if WLR_index == 'v2_arm':
            return armature_6dof_arm
        else:
            indices = [1, 2, 3, 5, 6, 7]
            return [armature_8dof[i] for i in indices]
    elif DOF == 4:
        indices = [1, 3, 5, 7]
        return [armature_8dof[i] for i in indices]
    elif DOF == 10:
        return armature_10dof
    elif DOF == 16:
        return armature_16dof
    else:
        raise ValueError(f"Unsupported DOF={DOF}")


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "anymal_c_flat",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {
            "name": "--env_device",
            "type": int,
            "default": "0",
            "help": "Device used by the gym env (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {"name": "--exptid", "type": str, "default": "", "help": "exptid"},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.env_device
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, "memory_a"):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(
            f"hidden_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )
        self.register_buffer(
            f"cell_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
