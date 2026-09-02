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


from wheel_legged_gym.envs.cowa_w_arm_him_add_arm_point.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import random
from wheel_legged_gym.envs.cowa_w_arm_him_add_arm_point.legged_robot import LeggedRobot

from wheel_legged_gym.utils.terrain import  Terrain
# from collections import deque
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
import numpy as np
from scipy.interpolate import make_interp_spline
from IPython import embed; eee = embed
class CowaFreeEnv_Arm(LeggedRobot):
    '''
    CowaFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05   # 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()


# ================================================================================
#                          rand arm curriculumn zsy                              =
# dof都是加上default后的 ===========================================================

    def  _get_phase(self):
        cycle_time = 10
        phase = self.episode_length_buf * self.dt / cycle_time # 20s / 5 = 4s
        return phase


    def  _get_phase_add_stop_time(self):
        cycle_time = 20
        phase = self.episode_length_buf * self.dt / cycle_time # 20s / 5 = 4s
        self.last_phase = self.arm_stop_time * self.dt / cycle_time
        manbaout = torch.nonzero((self.episode_length_buf >= self.arm_stop_time), as_tuple=True)[0] 
        phase[manbaout] = self.last_phase[manbaout]
        return phase


    def  _get_random_phase(self):
        """
        随机化cycle time 10 ~ 40
        """
        phase = self.episode_length_buf * self.dt / self.random_cycle_time # 20s / 5 = 4s
        return phase



    def _cal_max_scale(self, arm_indice, max):
        """
        計算scale值，和机械臂的default pos有关
        """
        if arm_indice == 16:
            scale = (max + 2.55) / 2.
        elif arm_indice == 17:
            scale = (max -3.14) / -2.
        return scale



    # reset继承
    def reset_idx(self, env_ids):  
        self.Curriculum_random_init(env_ids)
        super().reset_idx(env_ids) 
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        self.ref_buff[env_ids,:6] = self.default_dof_pos[env_ids, :6].clone()  # 键盘遥操ref_buff清空 


    # 对每一个关节角度使用clip限制
    def Curriculum_clip_limit_joint(self):
        for i in range (3):
            self.dof_pos_arm_set[:,i] = torch.clip(self.dof_pos_arm_set[:,i], self.dof_pos_limits[i,0], self.dof_pos_limits[i,1])


    def Curriculum_random_init_buffers(self):
        self.dof_pos_arm_set        = torch.zeros_like(self.dof_pos_arm, dtype=torch.float32,device=self.device) # 机械臂的关节角度
        self.joint_pos_target_buff  = torch.zeros_like(self.dof_pos, dtype=torch.float32,device=self.device) #用于测试torque的暂存变量
        self.arm_mod_choice         = torch.zeros(self.num_envs, 1,  dtype=torch.float32,device=self.device)         
        self.rand_arm_sin_amplitude = torch.zeros(self.num_envs, 3,  dtype=torch.float32,device=self.device) 
        self.amp_0_to_1             = torch.zeros(self.num_envs,   dtype=torch.float32,device=self.device)
        self.joint15_noise          = torch.zeros(self.num_envs,   dtype=torch.float32,device=self.device)  # 幅度噪声
        self.joint16_noise          = torch.zeros(self.num_envs,   dtype=torch.float32,device=self.device)  # 
        self.joint17_noise          = torch.zeros(self.num_envs,   dtype=torch.float32,device=self.device)  # 
        self.fourier_frequency      = torch.zeros(self.num_envs, 3,  dtype=torch.float32,device=self.device)  # 
        self.FK_rand_angles         = torch.zeros(self.num_envs, 3,  dtype=torch.float32,device=self.device) 
        self.rand_phase             = torch.zeros(self.num_envs,     dtype=torch.float32,device=self.device)
        # 随机化episode_length_buff的进程,随机到达一定的时间后机械臂就不动了 根episode_length
        self.arm_stop_time          = torch.zeros(self.num_envs,     dtype=torch.long,   device=self.device)
        self.last_phase             = torch.zeros(self.num_envs,     dtype=torch.long,   device=self.device)
        self.random_cycle_time      = torch.zeros(self.num_envs,     dtype=torch.float32,device=self.device)
        self.target_pos             = torch.zeros(self.num_envs,  3, dtype=torch.float32,device=self.device) # LQ


    def Curriculum_random_init(self, env_ids):   
        # 清零
        self.amp_0_to_1[env_ids]             = torch.rand((len(env_ids), ),  dtype=torch.float32,device=self.device)  # 0~1
        self.arm_mod_choice[env_ids]         = torch.rand((len(env_ids), 1),  dtype=torch.float32,device=self.device)               #  0~1
        self.rand_arm_sin_amplitude[env_ids] = torch.rand((len(env_ids), 3),  dtype=torch.float32,device=self.device)* 2 - 1       # -1~1 
        self.joint15_noise[env_ids]          = (torch.rand((len(env_ids), ),  dtype=torch.float32,device=self.device)* 2 - 1) * 0.3              
        self.joint16_noise[env_ids]          = (torch.rand((len(env_ids), ),  dtype=torch.float32,device=self.device)* 2 - 1) * 0.15       
        self.joint17_noise[env_ids]          = (torch.rand((len(env_ids), ),  dtype=torch.float32,device=self.device)* 2 - 1) * 0.2    
        self.fourier_frequency[env_ids]      = torch.rand((len(env_ids), 3),  dtype=torch.float32,device=self.device) + 1          # 傅里叶三个随机化频率 1 ～ 2
        self.FK_rand_angles[env_ids]         = torch.rand((len(env_ids), 3),  dtype=torch.float32,device=self.device) * 2 - 1       # 随机化机械臂的FK关节角度 -1~1
        self.rand_phase[env_ids]             = torch.rand((len(env_ids),  ),  dtype=torch.float32,device=self.device) * 0.5         # 0~0.5
        # print("randomize arm sampings --------------------- ")
        self.arm_stop_time[env_ids]          = torch.randint(5,   int(self.max_episode_length),    (len(env_ids), ),  dtype=torch.long,  device=self.device) # 20/0.02 = 1000  125~1000  0.5s ~ 2s
        self.random_cycle_time[env_ids]      = torch.rand((len(env_ids),  ),  dtype=torch.float32,device=self.device) * 10 + 5              # 5 ~ 15


    ## ============ 机械臂随机化课程选择 ==========
    # 一、默认姿态
    def Curriculum_default(self,choice_ids):
        if choice_ids.numel() == 0: #判断为空
            return 
        self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()
        # print("1")



    # 二、给定角度,做一个前向动力学
    def Curriculum_fk(self, choice_ids,scale = 2): #随机给定三个关节角度，对机械臂进行正解 
        if choice_ids.numel() == 0: #判断为空
            return 
        self.dof_pos_arm_set[choice_ids,0:3] = 1.5 * self.FK_rand_angles[choice_ids,0:3] * scale + self.default_dof_pos[choice_ids, 0:3]
        # print("2")


    # 三、期望根据关键点计算机械臂关节角度
    def Curriculum_ik(self):
        raise NotImplementedError


    # 3.fourier的连续轨迹 
    # def Curriculum_fourier(self,choice_ids):
    #     if choice_ids.numel() == 0: #判断为空
    #         return 
    #     phase = self._get_phase()   
    #     self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()  # 机械臂的默认位置
    #     sin_pos0 = torch.sin(1 * self.fourier_frequency[choice_ids,0] * torch.pi * phase)    
    #     sin_pos1 = torch.sin(2 * self.fourier_frequency[choice_ids,1] * torch.pi * phase)  
    #     sin_pos2 = torch.sin(3 * self.fourier_frequency[choice_ids,2] * torch.pi * phase)  
    #     max_amp_temp = 2.5
    #     for i in range(3):         
    #         # self.dof_pos_arm_set[choice_ids,i:i+1]
    #         self.dof_pos_arm_set[choice_ids,i] = (self.fourier_amplitude_0[choice_ids,0] * sin_pos0 + self.fourier_amplitude_1[choice_ids,0] * sin_pos1 + self.fourier_amplitude_2[choice_ids,0] * sin_pos2) / max_amp_temp



    # 【3】.正弦的连续轨迹 
    # def Curriculum_sin(self,choice_ids):
    #     if choice_ids.numel() == 0:
    #         return 
    #     """ 对机械臂做一个参考轨迹，这是轨迹函数，用于测试下肢的鲁棒性 zsy add
    #     """
    #     phase = self._get_phase() 
    #     self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()  # 机械臂的默认位置
    #     sin0 = torch.sin(0.8*self.fourier_frequency[choice_ids,0] * torch.pi * phase[choice_ids])    # torch.Size([10])
    #     sin1 = torch.sin(1.0*self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids])
    #     sin2 = torch.sin(1.0*self.fourier_frequency[choice_ids,2] * torch.pi * phase[choice_ids])
    #     # import ipdb; ipdb.set_trace()  
    #     # print('dim', self.ref_dof_pos.size())  
    #     scale_1 = self.rand_arm_sin_amplitude[choice_ids,0] * 2.0  #随机化幅度 -2 ～ 2  # torch.Size([10])
    #     scale_2 = self.rand_arm_sin_amplitude[choice_ids,1] * 2.0  #随机化幅度 -2 ～ 2
    #     scale_3 = self.rand_arm_sin_amplitude[choice_ids,2] * 2.0  #随机化幅度 -2 ～ 2
    #     # import ipdb; ipdb.set_trace()    
    #     # left foot stance phase set to default joint pos        
    #     # sin_pos_arm[sin_pos_arm < 0] = 0    
    #     self.dof_pos_arm_set[choice_ids, 0] += sin0 * scale_1  
    #     self.dof_pos_arm_set[choice_ids, 1] += sin1 * scale_2 
    #     self.dof_pos_arm_set[choice_ids, 2] += sin2 * scale_3
    #     # print("3")



    # 四、2dof运动，default->pick
    def Curriculum_curve_2_straight(self,choice_ids):
        """
        16关节和17关节频率相同；16关节的幅度较大，17关节的幅度较小，模拟真实情况；cycle_time随机化；
        """
        if choice_ids.numel() == 0: #判断为空
            return 
        phase = self._get_random_phase()
        self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()  # 机械臂的默认位置
        # 幅度都是1    
        # sin_pos_joint15 = torch.sin(self.fourier_frequency[choice_ids,0] * torch.pi * phase)
        # 初始值为-2.55 按照-2.55 -> -1.55 -> -0.55 变化角度
        """让15和16关节同频"""
        amp_scale_16joint = (self._cal_max_scale(16, 0.0) + self.joint16_noise)[choice_ids]
        amp_scale_17joint = (self._cal_max_scale(17, 1.5) + self.joint17_noise)[choice_ids]
        sin_pos_joint16 = amp_scale_16joint * (torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi/2) + 1.0)  # 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> 1 -> 2    
        # 初始值为3.14 按照3.14 -> 2.14 -> 1.14 变化角度 ，与上一个相位相同，但幅度相反
        sin_pos_joint17 = amp_scale_17joint * ( -(torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi/2) + 1.0)  )# 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> -1 -> -2    
        # self.dof_pos_arm_set[choice_ids, 0] += sin0 * scale_1  
        self.dof_pos_arm_set[choice_ids, 0] = 0
        self.dof_pos_arm_set[choice_ids, 1] += sin_pos_joint16
        self.dof_pos_arm_set[choice_ids, 2] += sin_pos_joint17




    # 五、相较{四}添加yaw的dof：3dof运动，default->pick
    def Curriculum_curve_2_straight_add_yaw(self,choice_ids):
        if choice_ids.numel() == 0: #判断为空
            return 
        phase = self._get_random_phase()  
        self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()  # 机械臂的默认位置
        # 15 joint 限制只能向前的方向摆动    
        # 16 joint 初始值为-2.55 按照-2.55 -> -1.55 -> -0.55 变化角度
        # 17 joint 初始值为3.14 按照3.14 -> 2.14 -> 1.14 变化角度 ，与上一个相位相同，但幅度相反
        amp_scale_16joint = (self._cal_max_scale(16, 0.0) + self.joint16_noise)[choice_ids]
        amp_scale_17joint = (self._cal_max_scale(17, 1.5) + self.joint17_noise)[choice_ids]
        sin_pos_joint15 = 0.6 * self.amp_0_to_1[choice_ids] * torch.sin(self.fourier_frequency[choice_ids,0] * torch.pi * phase[choice_ids])
        sin_pos_joint16 = amp_scale_16joint * (torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi/2) + 1.0)  # 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> 1 -> 2    
        sin_pos_joint17 = amp_scale_17joint * ( -(torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi/2) + 1.0)  )# 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> -1 -> -2    
        # print(":::",amp_scale_17joint)
        # self.dof_pos_arm_set[choice_ids, 0] += sin0 * scale_1  
        self.dof_pos_arm_set[choice_ids, 0] += sin_pos_joint15
        self.dof_pos_arm_set[choice_ids, 1] += sin_pos_joint16
        self.dof_pos_arm_set[choice_ids, 2] += sin_pos_joint17




    # 六、增加超时，电机就不再转动了，相当于正弦没做完一个周期，只完成了一部分, 3dof运动，default->pick
    def Curriculum_curve_2_straight_add_stop_time(self,choice_ids):  
        if choice_ids.numel() == 0: #判断为空
            return 
        phase = self._get_phase_add_stop_time()     
        self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()  # 机械臂的默认位置     
        # 15 joint 限制只能向前的方向摆动    
        # 16 joint 初始值为-2.55 按照-2.55 -> -1.55 -> -0.55 变化角度
        # 17 joint 初始值为3.14 按照3.14 -> 2.14 -> 1.14 变化角度 ，与上一个相位相同，但幅度相反
        amp_scale_16joint = (self._cal_max_scale(16, 0.0) + self.joint16_noise)[choice_ids]
        amp_scale_17joint = (self._cal_max_scale(17, 1.5) + self.joint17_noise)[choice_ids]
        sin_pos_joint15 = 0.6 * self.amp_0_to_1[choice_ids] * torch.sin(self.fourier_frequency[choice_ids,0] * torch.pi * phase[choice_ids])
        sin_pos_joint16 = amp_scale_16joint * (torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi/2) + 1.0)  # 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> 1 -> 2    
        sin_pos_joint17 = amp_scale_17joint * ( -(torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi/2) + 1.0)  )# 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> -1 -> -2    
        # self.dof_pos_arm_set[choice_ids, 0] += sin0 * scale_1  
        self.dof_pos_arm_set[choice_ids, 0] += sin_pos_joint15
        self.dof_pos_arm_set[choice_ids, 1] += sin_pos_joint16
        self.dof_pos_arm_set[choice_ids, 2] += sin_pos_joint17





    # 七、随机化相移
    def Curriculum_curve_2_straight_rand_phase(self,choice_ids):  
        if choice_ids.numel() == 0: #判断为空
            return 
        phase = self._get_phase_add_stop_time()     
        self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()  # 机械臂的默认位置     
        # 15 joint 限制只能向前的方向摆动    
        # 16 joint 初始值为-2.55 按照-2.55 -> -1.55 -> -0.55 变化角度
        # 17 joint 初始值为3.14 按照3.14 -> 2.14 -> 1.14 变化角度 ，与上一个相位相同，但幅度相反
        amp_scale_16joint = (self._cal_max_scale(16, 0.0) + self.joint16_noise)[choice_ids]
        amp_scale_17joint = (self._cal_max_scale(17, 1.5) + self.joint17_noise)[choice_ids]
        sin_pos_joint15 = 0.6 * self.amp_0_to_1[choice_ids] * torch.sin(self.fourier_frequency[choice_ids,0] * torch.pi * phase[choice_ids])
        sin_pos_joint16 = amp_scale_16joint * (torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi * self.rand_phase[choice_ids] ) + 1.0)  # 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> 1 -> 2    
        sin_pos_joint17 = amp_scale_17joint * ( -(torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi * self.rand_phase[choice_ids]) + 1.0)  )# 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> -1 -> -2    
        # self.dof_pos_arm_set[choice_ids, 0] += sin0 * scale_1  
        self.dof_pos_arm_set[choice_ids, 0] += sin_pos_joint15
        self.dof_pos_arm_set[choice_ids, 1] += sin_pos_joint16
        self.dof_pos_arm_set[choice_ids, 2] += sin_pos_joint17
        # print("4")


    # 八、固定轨迹 - 一个很小的扰动 - 实机
    def Curriculum_curve_2_straight_fix_tracks(self,  choice_ids):  
        if choice_ids.numel() == 0: #判断为空
            return 

        phase = self._get_random_phase()     
        self.dof_pos_arm_set[choice_ids] = self.default_dof_pos[choice_ids,0:6].clone()  # 机械臂的默认位置     
        # 15 joint 限制只能向前的方向摆动    
        # 16 joint 初始值为-2.55 按照-2.55 -> -1.55 -> -0.55 变化角度   
        # 17 joint 初始值为3.14 按照3.14 -> 2.14 -> 1.14 变化角度 ，与上一个相位相同，但幅度相反   
        amp_scale_16joint = (self._cal_max_scale(16, 0.0) + (self.amp_0_to_1 * 2 - 1) * 0.15)[choice_ids] # 土0.15 8度
        amp_scale_17joint = (self._cal_max_scale(17, 1.5) + (self.amp_0_to_1 * 2 - 1) * 0.15)[choice_ids] # 土0.15 8度
        # sin_pos_joint15 = 0.6 * self.amp_0_to_1[choice_ids] * torch.sin(self.fourier_frequency[choice_ids,0] * torch.pi * phase[choice_ids]) 
        sin_pos_joint16 = amp_scale_16joint * (torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi * self.rand_phase[choice_ids] ) + 1.0)  # 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> 1 -> 2    
        sin_pos_joint17 = amp_scale_17joint * ( -(torch.sin(self.fourier_frequency[choice_ids,1] * torch.pi * phase[choice_ids] - torch.pi * self.rand_phase[choice_ids]) + 1.0)  )# 移相 加偏执，从折叠状态伸展，sin幅度为 0  -> -1 -> -2    
        # self.dof_pos_arm_set[choice_ids, 0] += sin0 * scale_1  
        self.dof_pos_arm_set[choice_ids, 0] += 0
        self.dof_pos_arm_set[choice_ids, 1] += sin_pos_joint16
        self.dof_pos_arm_set[choice_ids, 2] += sin_pos_joint17
        # print("4")


    # 九、 线性插值
    def Curriculum_spline_phase_control(self, choice_ids):
        """
        从随机目标位姿 → 默认位姿，使用二阶样条插值。
        """
        if choice_ids.numel() == 0:
            return
        cur_phases = torch.clamp(self._get_random_phase()[choice_ids], 0.0, 1.0)
        cur_phases_np = cur_phases.detach().cpu().numpy()
        default_pose = self.default_dof_pos[choice_ids[0], 0:3].detach().cpu().numpy()
        target_pose = self.target_pos[choice_ids[0], :].detach().cpu().numpy()
        mid_pose = (default_pose + target_pose) / 2.0
        t = np.array([0.0, 0.5, 1.0])
        control_points = np.stack([default_pose, mid_pose, target_pose], axis=0)
        spline_fn = make_interp_spline(t, control_points, k=2)
        interpolated_poses = spline_fn(cur_phases_np)
        self.dof_pos_arm_set[choice_ids, :3] = torch.tensor(interpolated_poses, dtype=torch.float32, device=self.device)



    # <><><><><><><><><><><><><><> 根据随机数选择不同的随机化函数 <><><><><><><><><><><><><><>  
    def Curriculum_choice_arm(self ,weight = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0,])): #在这里更改权重，分别对应三种不同的课程
        """
        把arm_mod_choice里值挑出来
        其中Curriculum_choice_arm函数里的xxx_ids都是和Curriculum_choice_arm_default_pos函数里的xxx_ids一一相对应的
        """
        if 1:
            Weight_sum = torch.tensor([torch.sum(weight[: i + 1])for i in range(len(weight))])
            a_ids = torch.nonzero((self.arm_mod_choice < Weight_sum[0]), as_tuple=True)[0]
            b_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice < Weight_sum[1] , self.arm_mod_choice > Weight_sum[0])), as_tuple=True)[0]
            c_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice < Weight_sum[2] , self.arm_mod_choice > Weight_sum[1])), as_tuple=True)[0]
            d_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice < Weight_sum[3] , self.arm_mod_choice > Weight_sum[2])), as_tuple=True)[0]
            # e_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice < Weight_sum[4] , self.arm_mod_choice > Weight_sum[3])), as_tuple=True)[0]
            # f_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice < Weight_sum[5] , self.arm_mod_choice > Weight_sum[4])), as_tuple=True)[0]
            # e_ids = torch.nonzero((self.arm_mod_choice > Weight_sum[3]), as_tuple=True)[0]
            # print(a_ids,b_ids,c_ids,d_ids)  
            self.Curriculum_default(a_ids)   
            # self.Curriculum_curve_2_straight(b_ids)  
            # self.Curriculum_curve_2_straight_add_yaw(c_ids)  
            self.Curriculum_curve_2_straight_fix_tracks(b_ids)   
            self.Curriculum_curve_2_straight_add_stop_time(c_ids) 
            # self.Curriculum_curve_2_straight_rand_phase(e_ids) 
            self.Curriculum_spline_phase_control(d_ids)    
            # self.Curriculum_spline_phase_control(e_ids)    
            # self.Curriculum_fk(f_ids)   


            ### clip关节 防-超-限
            self.Curriculum_clip_limit_joint() #clip关节  
        # 测试，默认角度  
        if 0: # 直接给默认角度  
            self.dof_pos_arm_set[:, 0:6] = self.default_dof_pos[:, 0:6].clone()
        ref_buff = self.dof_pos_arm_set.clone()
        return ref_buff


    # <><><><><><><><><><><><><><> 根据随机数选择不同的初始位姿 <><><><><><><><><><><><><><>
    def Curriculum_choice_arm_default_pos(self, env_ids, weight = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0,])): #在这里更改权重，分别对应三种不同的课程
        """
        随机化不同的default角度,env_ids用在reset_dofs选择要reset的关节部分
        把arm_mod_choice被reset的部分的值挑出来
        """
        if 1:
            Weight_sum = torch.tensor([torch.sum(weight[: i + 1])for i in range(len(weight))])
            a_ids = torch.nonzero((self.arm_mod_choice[env_ids] < Weight_sum[0]), as_tuple=True)[0]
            b_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice[env_ids] < Weight_sum[1] , self.arm_mod_choice[env_ids] > Weight_sum[0])), as_tuple=True)[0]
            c_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice[env_ids] < Weight_sum[2] , self.arm_mod_choice[env_ids] > Weight_sum[1])), as_tuple=True)[0]
            d_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice[env_ids] < Weight_sum[3] , self.arm_mod_choice[env_ids] > Weight_sum[2])), as_tuple=True)[0]
            # e_ids = torch.nonzero((torch.logical_and(self.arm_mod_choice[env_ids] < Weight_sum[4] , self.arm_mod_choice[env_ids] > Weight_sum[3])), as_tuple=True)[0]


            self.default_pose_reset(a_ids)
            self.default_pose_reset(b_ids)
            self.default_pose_reset(c_ids)
            self.default_pose_rise(d_ids)
            # self.default_pose_drop(e_ids)
            
            # self.default_pose_random(c_ids)
            # self.default_pose_random(d_ids)



    def default_pose_random(self, choice_ids):
        """
        随机化初始default值
        """
        if choice_ids.numel() == 0:
            return
        self.dof_pos_arm[choice_ids, :3] = self.raw_default_dof_pos[:3]
        # 遍历 choice_ids，逐个判断是否需要重置
        for idx in choice_ids:
            # 随机偏移
            self.dof_pos_arm[idx,0:1] += (torch.rand(1, device=self.device) - 0.5) * 0.6    # ±0.3
            self.dof_pos_arm[idx,1:2] += torch.rand(1, device=self.device) * 2.78           # + [0, 2.4]
            self.dof_pos_arm[idx,2:3] -= torch.rand(1, device=self.device) * 2           # - [0, 1.3]
        # self.dof_pos_arm[choice_ids,0:3] = torch.tensor([[0.0, -0.24, 1.87]], device = self.device)
        self.default_dof_pos[choice_ids,0:3] = self.dof_pos_arm[choice_ids,0:3]



    def default_pose_reset(self,choice_ids):
        """
        reset固定值
        """
        self.default_dof_pos[choice_ids,:] = self.raw_default_dof_pos



    def default_pose_rise(self, choice_ids):
        """
        上升的default
        """
        if choice_ids.numel() == 0:
            return
        self.default_dof_pos[choice_ids,:3] = torch.tensor([[0.0, 0.27, 1.29]], device = self.device)
        self.dof_pos_arm[choice_ids,0:3] = torch.tensor([[0.0, 0.27, 1.29]], device = self.device)
        self.target_pos[choice_ids,0:1] = (torch.rand(1, device=self.device) - 0.5) * 0.6    # ±0.3
        self.target_pos[choice_ids,1:2] = torch.rand(1, device=self.device) * 2.78  - 2.55         # + [0, 2.4]
        self.target_pos[choice_ids,2:3] = - torch.rand(1, device=self.device) * 2  + 3.14        # - [0, 1.3]



    def default_pose_drop(self, choice_ids):
        """
        下降的default
        """
        if choice_ids.numel() == 0:
            return
        self.default_dof_pos[choice_ids,:] = self.raw_default_dof_pos
        self.dof_pos_arm[choice_ids,0:3] = self.raw_default_dof_pos[:3]
        self.target_pos[choice_ids,0:1] = (torch.rand(1, device=self.device) - 0.5) * 0.6    # ±0.3
        self.target_pos[choice_ids,1:2] = torch.rand(1, device=self.device) * 2.78 - 2.55       # + [0, 2.4]
        self.target_pos[choice_ids,2:3] = -torch.rand(1, device=self.device) * 2 + 3.14     # - [0, 1.3]


# =========================================================================================================

    # TODO
    def _get_noise_scale_vec(self, cfg):
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

        noise_vec[:3] = 0
        noise_vec[3:7] = noise_scales.dof_pos  * self.obs_scales.dof_pos
        noise_vec[7:13] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[13:19] = 0 
        noise_vec[19:22] = noise_scales.ang_vel  * self.obs_scales.ang_vel
        noise_vec[22:25] = noise_scales.quat  * self.obs_scales.quat
        noise_vec[25:31] = noise_scales.dof_pos * self.obs_scales.dof_pos   # 机械臂扰动加噪 6
        noise_vec[31:37] = noise_scales.dof_vel * self.obs_scales.dof_pos   # 机械臂扰动加噪 6 
        return noise_vec 

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay

        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return super().step(actions)

    # for him use
    def compute_privileged_observations(self):
            # compute the critic_obs ,not change critic_history

            self.base_height_obs = self.base_height.unsqueeze(1)
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            privileged_termination_obs = torch.cat((
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.dof_pos[:, [0,1,3,4]] - self.default_dof_pos[:, [0+6,1+6,3+6,4+6]]) * self.obs_scales.dof_pos,  # 4         
                self.dof_vel * self.obs_scales.dof_vel,  # 6  
                self.actions,  # 6 
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
                self.dof_pos_arm * self.obs_scales.dof_pos,   # 6 zsy  # NOTE 没加延迟 
                # self.dof_vel_arm * self.obs_scales.dof_vel,   # 6 zsy  # NOTE 没加延迟 
                # self.projected_gravity, # 重力投影    
                self.dof_acc * self.obs_scales.dof_acc, # 6 
                self.torques[:,6:] * self.obs_scales.torques, # 6self  
                (self.body_mass - self.body_mass.mean()).view(self.num_envs, 1),    # 1
                self.base_com, # 3  
                # self.total_base_com, # 3 # NOTE zsy 全局质心带机械臂  
                self.default_dof_pos[:,6:] - self.raw_default_dof_pos[6:], # 6
                # self.default_dof_pos
                self.p_gains[:,6:] / 20.0, # 6
                self.d_gains[:,6:] / 0.2, # 6
                # self.ref_dof_pos[:, [0,1,3,4]] - self.dof_pos[:, [0,1,3,4]],
                # self.ref_dof_vel[:, [2,5]] - self.dof_vel[:, [2,5]],
                # sin_pos_obs,  
                # self.arm_2_root_pos # 3 机械臂相对于身体的位置 
            ), dim=-1)
            privileged_termination_obs = torch.cat((privileged_termination_obs,
                                                heights), dim=1)
            # NOTE: estimator的预测变量 ---> new 
            privileged_termination_obs = torch.cat((privileged_termination_obs,
                                        self.base_lin_vel * self.obs_scales.lin_vel,                                # 3
                                        self.base_height_obs * self.obs_scales.height_measurements,                 # 1
                                        self.arm_2_root_pos * self.obs_scales.arm_2_base_scales,) ,dim=1)           # 3
                                        # self.ee_arm_2_root_pos * self.obs_scales.arm_2_base_scales) ,dim=1)       # 3 
            if self.cfg.env.c_frame_stack > 1:           
                return torch.cat([torch.cat([self.critic_history[i+1] for i in range(self.cfg.env.c_frame_stack-1)], dim=1),
                                privileged_termination_obs], dim=1)
            return privileged_termination_obs


    # TODO
    def compute_observations(self):
        # phase = self._get_phase()
        # sin_pos_obs = torch.sin(2*torch.pi*phase).unsqueeze(1)
        # self.compute_ref_state_ARM()
        self.base_height_obs = self.base_height.unsqueeze(1)

        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,  # 3
            (self.dof_pos[:, [0,1,3,4]] - self.default_dof_pos[:, [0+6,1+6,3+6,4+6]]) * self.obs_scales.dof_pos,  # 4         
            self.dof_vel * self.obs_scales.dof_vel,  # 6  
            self.actions,  # 6 
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.dof_pos_arm * self.obs_scales.dof_pos,   # 6 zsy  # NOTE 没加延迟 
            # self.dof_vel_arm * self.obs_scales.dof_vel,   # 6 zsy  # NOTE 没加延迟 
            # self.projected_gravity, # 重力投影    
            self.dof_acc * self.obs_scales.dof_acc, # 6 
            self.torques[:,6:] * self.obs_scales.torques, # 6self  
            (self.body_mass - self.body_mass.mean()).view(self.num_envs, 1),    # 1
            self.base_com, # 3  
            # self.total_base_com, # 3 # NOTE zsy 全局质心带机械臂  
            self.default_dof_pos[:,6:] - self.raw_default_dof_pos[6:], # 6
            # self.default_dof_pos
            self.p_gains[:,6:] / 20.0, # 6
            self.d_gains[:,6:] / 0.2, # 6
            # self.ref_dof_pos[:, [0,1,3,4]] - self.dof_pos[:, [0,1,3,4]],
            # self.ref_dof_vel[:, [2,5]] - self.dof_vel[:, [2,5]],
            # sin_pos_obs,  
            # self.arm_2_root_pos # 3 机械臂相对于身体的位置 
            
        ), dim=-1)

        # 加入额外的privileged
        # if self.cfg.env.priv_observe_friction:
        #      # 1
        #     friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                             (self.friction_coeffs[:, 0].unsqueeze(1)
        #                                             - friction_coeffs_shift) * friction_coeffs_scale),dim=1)
        
        # if self.cfg.env.priv_observe_restitution:
        #     # 1
        #     restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.restitutions[:, 0].unsqueeze(1) 
        #                                          - restitutions_shift) * restitutions_scale),dim=1)

        # if self.cfg.env.priv_observe_base_mass:
        #     # 1
        #     payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),dim=1)

        # if self.cfg.env.priv_observe_com_displacement:
        #     # 3
        #     com_displacements_scale, com_displacements_shift = get_scale_shift(
        #         self.cfg.domain_rand.com_displacement_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.com_displacements - com_displacements_shift) * com_displacements_scale), dim=1)
        # if self.cfg.env.priv_observe_motor_strength:
        #     # 6
        #     motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)

        # if self.cfg.env.priv_observe_motor_offset:
        #     # 6
        #     motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1) 
        # if self.cfg.env.priv_observe_gravity:
        #     # 3
        #     gravity_scale, gravity_shift = get_scale_shift(self.cfg.domain_rand.gravity_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.gravities - gravity_shift) / gravity_scale), dim=1)

        # NOTE: 始终将需要预测的值量放在最后（若有预测网络）
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                            heights), dim=1) #77
        # NOTE: estimator的预测变量 --->
        # self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                     self.base_lin_vel * self.obs_scales.lin_vel, 
        #                                     self.base_height_obs * self.obs_scales.height_measurements), dim=1) #4
        # NOTE: estimator的预测变量 ---> new
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                    self.base_lin_vel * self.obs_scales.lin_vel,                                # 3
                                    self.base_height_obs * self.obs_scales.height_measurements,                 # 1
                                    self.arm_2_root_pos * self.obs_scales.arm_2_base_scales,) ,dim=1)           # 3
                                    # self.ee_arm_2_root_pos * self.obs_scales.arm_2_base_scales) ,dim=1)       # 3 
        # agibot 
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
            self.lagged_dof_pos_arm = self.dof_lag_buffer_arm[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]   # NOTE zsy add 机械臂的观测值延时
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
            self.lagged_dof_vel_arm = self.dof_vel_lag_buffer_arm[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]   # NOTE zsy add 机械臂的观测值延时
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_pos_arm = self.dof_pos_arm
            self.lagged_dof_vel = self.dof_vel
            self.lagged_dof_vel_arm = self.dof_vel_arm

        if self.cfg.domain_rand.add_imu_lag:
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
            self.lagged_projected_gravity = self.lagged_imu[:,-3:].clone()
        # no imu lag       
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz
            self.lagged_projected_gravity = self.projected_gravity

        q = self.lagged_dof_pos * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel
        q_arm = self.lagged_dof_pos_arm * self.obs_scales.dof_pos 
        dq_arm = self.lagged_dof_vel_arm * self.obs_scales.dof_vel
        # ==================================     
        obs_buf = torch.cat((  
            self.commands[:, :3]  * self.commands_scale,   # 3 
            q[:, [0,1,3,4]], #4 
            dq,  # 6D    
            self.actions,   # 6D
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
            # self.lagged_projected_gravity,  #3
            q_arm,   # 6 zsy      
            # dq_arm,  # 6 zsy    
        ), dim=-1)
        obs_now = obs_buf.clone()

        if self.cfg.domain_rand.randomize_obs_delay:    
            obs_now = self.hist_obs.popleft().to(self.device)
            self.hist_obs.append(obs_buf)

        if self.add_noise:  
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)



# ================================================ Rewards Humanoid Gym ================================================== #
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        diff[:,hip_roll_yaw_pitch_indices] *= 3
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_leg_roll_joint_pos_outside(self):
        """
        防止leg roll外阔
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        r = 0
        diff_left_leg_roll = joint_pos[:, 0] - pos_target[:, 0]
        diff_right_leg_roll = joint_pos[:, 6] - pos_target[:, 6]
        mask_diff_left_leg_roll = diff_left_leg_roll <= 0
        mask_diff_right_leg_roll = diff_right_leg_roll >= 0
        diff_left_leg_roll[mask_diff_left_leg_roll] = 0
        diff_right_leg_roll[mask_diff_right_leg_roll] = 0
        
        temp_left = -1 * torch.exp(-2 * torch.norm(diff_left_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_left_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_left
        temp_right = -1 * torch.exp(-2 * torch.norm(diff_right_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_right_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_right
        return r

    def _reward_leg_roll_joint_pos_inside(self):
        """
        防止leg roll内收
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        r = 0
        diff_left_leg_roll = joint_pos[:, 0] - pos_target[:, 0]
        diff_right_leg_roll = joint_pos[:, 6] - pos_target[:, 6]
        mask_diff_left_leg_roll = diff_left_leg_roll >= 0
        mask_diff_right_leg_roll = diff_right_leg_roll <= 0
        diff_left_leg_roll[mask_diff_left_leg_roll] = 0
        diff_right_leg_roll[mask_diff_right_leg_roll] = 0
        
        temp_left = -1 * torch.exp(-2 * torch.norm(diff_left_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_left_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_left
        temp_right = -1 * torch.exp(-2 * torch.norm(diff_right_leg_roll, dim=-1)) + 0.2 * torch.norm(diff_right_leg_roll, dim=-1).clamp(0, 0.5) + 0.6
        r += temp_right

        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        self.foot_dist = foot_dist.unsqueeze(1)
        fd = self.cfg.rewards.min_feet_dist
        max_fd = self.cfg.rewards.max_feet_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_fd, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        knee_pos = self.rigid_state[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        self.knee_dist = knee_dist.unsqueeze(1)
        fd = self.cfg.rewards.min_knee_dist
        max_df = self.cfg.rewards.max_knee_dist # / 1.5
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2 # 只要是在[fd, max_df]，奖励就拿满

    def _reward_enhance_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # print(self.contact_forces[0, self.feet_indices, :])
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)
    
    def _reward_feet_xy_contact_forces(self):
        """惩罚足部水平方向的接触力超出范围的部分"""
        # 提取x/y方向接触力 [envs, feet, xy]
        xy_contact_forces = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=-1) 
        return torch.sum((xy_contact_forces - 10).clip(0, 20), dim=1)/20
        

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.04, 0, 50)
        return torch.exp(-yaw_roll * 100)


    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
        c_update = (lin_mismatch + ang_mismatch) / 2.
        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    # only consider vel_x
    def _reward_tracking_lin_vel_x(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_x_error = torch.square(self.commands[:,0] - self.base_lin_vel[:,0])
        if self.cfg.rewards.tracking_vel_hard:
            return torch.exp(-torch.abs(self.commands[:,0] - self.base_lin_vel[:,0]) * 20)
        if self.cfg.rewards.tracking_vel_enhance:
            return 0.2 * torch.exp(-lin_vel_x_error * self.cfg.rewards.tracking_sigma_vel_x) + 0.8 * torch.exp(-lin_vel_x_error * 6 * self.cfg.rewards.tracking_sigma_vel_x)
        else:
            return torch.exp(-lin_vel_x_error * self.cfg.rewards.tracking_sigma_vel_x)
        # return torch.exp(-lin_vel_x_error * self.cfg.rewards.tracking_sigma) + torch.exp(-lin_vel_x_error * 10 * self.cfg.rewards.tracking_sigma) - 1
    
    # def _reward_tracking_ang_vel(self):
    #     """
    #     Tracks angular velocity commands for yaw rotation.
    #     Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
    #     """   
    
    #     ang_vel_error = torch.square(
    #         self.commands[:, 1] - self.base_ang_vel[:, 2])
    #     # print('ang_vel :', self.base_ang_vel[:, 2])
    #     if self.cfg.rewards.tracking_vel_enhance:
    #         return 0.2 *torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_vel_y) + 0.8 *torch.exp(-ang_vel_error * 6 * self.cfg.rewards.tracking_sigma_vel_y)
    #     else:
    #         return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma_vel_y)

    def _reward_feet_height_smoothness(self):
        """
        计算feet height的平滑程度
        """
        r = torch.sum(torch.square(self.feet_height - self.last_feet_z), dim=1)
        return r

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
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)        
    
    
        #原本写法
    # def _reward_action_smoothness(self):
    #     """
    #     Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
    #     This is important for achieving fluid motion and reducing mechanical stress.
    #     """
    #     hip_knee_indices = [0,1,3,4]
    #     wheel_indices = [2,5]
    #     hip_knee_term_1 = torch.sum(torch.square(
    #         self.last_actions[:,hip_knee_indices]- self.actions[:,hip_knee_indices]), dim=1)
    #     hip_knee_term_2 = torch.sum(torch.square(
    #         self.actions[:,hip_knee_indices] + self.last_last_actions[:,hip_knee_indices] - 2 * self.last_actions[:,hip_knee_indices]), dim=1)
    #     hip_knee_term_3 = 0.2 * torch.sum(torch.abs(self.actions[:,hip_knee_indices]), dim=1)
    #     hip_knee_term = hip_knee_term_1 + hip_knee_term_2 + hip_knee_term_3
        
    #     wheel_term_1 = torch.sum(torch.square(
    #         self.last_actions[:,wheel_indices]- self.actions[:,wheel_indices]), dim=1)
    #     wheel_term_2 = torch.sum(torch.square(
    #         self.actions[:,wheel_indices] + self.last_last_actions[:,wheel_indices] - 2 * self.last_actions[:,wheel_indices]), dim=1)
    #     wheel_term_3 = 0.05 * torch.sum(torch.abs(self.actions[:,wheel_indices]), dim=1)
    #     wheel_term = wheel_term_1 + wheel_term_2 + wheel_term_3

    #     return hip_knee_term + wheel_term
    
    # 2025.3.6 xlh
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions- self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)

        return term_1 + term_2 + term_3
    def _reward_stand_still_vel_penality(self):
        """当命令很小时，机器人不应该有各个方向的速度"""
        # Penalize motion at zero commands
        term_x = 5 * torch.square(self.base_lin_vel[:, 0])
        term_y_z = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1)
        return (term_x + term_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_base_pos_penality(self):
        """当命令很小时，机器人不应该有各个方向的位置变化"""
        # Penalize motion at zero commands
        diff_x = 5 * torch.square(self.base_pos[:, 0] - self.base_pos_init[:, 0])
        diff_y_z = torch.sum(torch.square(self.base_pos[:, 1:3] - self.base_pos_init[:, 1:3]), dim=1)
        return (diff_x + diff_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_still_base_pos(self):
        """当命令很小时，机器人不应该有各个方向的位置变化"""
        # Penalize motion at zero commands
        diff = torch.norm(self.base_pos[:, :2] - self.base_pos_init[:, :2], dim=1)
        # return () * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return torch.exp(-100 * diff) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    # ------------------------- Rewards Unitree Gym --------------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # print(self.commands[0, 2], self.base_height[0])
        if self.reward_scales["base_height"] < 0:
            return torch.abs(self.base_height - self.commands[:, 2])
        else:
            base_height_error = torch.square(self.base_height - self.commands[:, 2])
            return torch.exp(-1000 * base_height_error)   #xlh: 张chenyang版本，原-200
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques[:,6:]), dim=1)

    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        mask = 1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1)
        mask[:,0] = mask[:,0]*2 # xlh: 增大base碰撞的惩罚权重,索引0是根据penalize_contacts_on中base的索引
        return torch.sum(mask, dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:, :2] - self.dof_pos_limits[6:2+6, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, :2] - self.dof_pos_limits[6:2+6, 1]).clip(
            min=0.0
        )
        out_of_limits += -(self.dof_pos[:, 3:5] - self.dof_pos_limits[3+6:5+6, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos[:, 3:5] - self.dof_pos_limits[3+6:5+6, 1]).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits[6:]*0.8).clip(min=0.), dim=1) ## 0.8zhangchenyang参数

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques[:,6:]) - self.torque_limits[6:]*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    # def _reward_tracking_lin_vel(self):
    #     # 原本写法
    #     lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
    #     return 0.8 * torch.exp(-10 * lin_vel_error) + 0.2 * torch.exp(-40 * lin_vel_error)
    def _reward_tracking_lin_vel(self):
        # 张chenyang版本写法，tracking_sigma = 0.25
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_enhance(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # -------------------------------------------------- wl
    def _reward_nominal_state(self):
        # return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        if self.reward_scales["nominal_state"] < 0:
            return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        else:
            ang_diff = torch.square(self.theta0[:, 0] - self.theta0[:, 1])
            return torch.exp(-ang_diff / 0.1)

    def _reward_power(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques[:,6:] * self.dof_vel), dim=1)

    def _reward_same_foot_z_position(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :3]
        foot_z_dist = foot_pos[:, 0, 2] - foot_pos[:, 1, 2]
        foot_dist = torch.abs(foot_z_dist)
        # fd = self.cfg.rewards.min_feet_z_dist
        max_fd = self.cfg.rewards.max_feet_z_dist
        # d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_fd, 0, 0.5)
        return torch.exp(-torch.abs(d_max) * 100) 
    
    # def _reward_same_foot_z_position(self):
    #     foot_pos = self.rigid_state[:, self.feet_indices, :3]
    #     foot_z_dist = foot_pos[:, 0, 2] - foot_pos[:, 1, 2]
    #     foot_dist = foot_z_dist
    #     fd = self.cfg.rewards.min_feet_z_dist
    #     max_fd = self.cfg.rewards.max_feet_z_dist
    #     d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
    #     d_max = torch.clamp(foot_dist - max_fd, 0, 0.5)
    #     return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_wheel_adjustment(self):
        # 鼓励使用轮子的滑动克服前后的倾斜，奖励轮速和倾斜方向一致的情况，并要求轮速方向也一致
        incline_x = self.projected_gravity[:, 0]
        # mean velocity
        wheel_x_mean = (self.foot_velocities[:, 0, 0] + self.foot_velocities[:, 1, 0]) / 2
        # 两边轮速不一致的情况，不给奖励
        wheel_x_invalid = (self.foot_velocities[:, 0, 0] * self.foot_velocities[:, 1, 0]) < 0
        wheel_x_mean[wheel_x_invalid] = 0.0
        wheel_x_mean = wheel_x_mean.reshape(-1)  
        reward = (incline_x * wheel_x_mean) > 0
        return reward


    def _reward_wheel_vel(self):
        # Penalize dof velocities
        # left_wheel_vel = self.commands[:,0]/2 - self.commands[:,1]
        # right_wheel_vel = self.commands[:,0]/2 + self.commands[:,1]
        # return torch.sum(torch.square(self.dof_vel[:, 2] - left_wheel_vel) + torch.square(self.dof_vel[:, 5]) - right_wheel_vel)
        return torch.sum(torch.square(self.dof_vel[:, [2, 5]]), dim=1)

    def _reward_wheel_acc(self):
        # Penalize dof accelerations 
        return torch.sum(torch.square((self.last_dof_vel[:, 2:3] - self.dof_vel[:, 2:3]) / self.dt), dim=1) + torch.sum(torch.square((self.last_dof_vel[:, 5:6] - self.dof_vel[:, 5:6]) / self.dt), dim=1)

    ### 训练空中跟随sin
    def _reward_nominal_state(self):
        # return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        if self.reward_scales["nominal_state"] < 0:
            return torch.square(self.theta0[:, 0] - self.theta0[:, 1])
        else:
            ang_diff = torch.square(self.theta0[:, 0] - self.theta0[:, 1])
            return torch.exp(-ang_diff / 0.1)
        

    def _reward_dof_vel(self):
        # zhangchenyang版本
        return torch.sum(torch.square(self.dof_vel), dim=1)
    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.dof_vel[:, :2]), dim=1) + torch.sum(
    #         torch.square(self.dof_vel[:, 3:5]), dim=1
    #     )


    def _reward_ref_hip_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[0,3]] - pos_target[:,[0,3]]) * 1
        # hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        # diff[:,hip_roll_yaw_pitch_indices] *= 3
        # r = torch.exp(-50 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.8)
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r
    
    def _reward_ref_knee_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = (joint_pos[:,[1,4]] - pos_target[:,[1,4]]) * 1
        # hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        # diff[:,hip_roll_yaw_pitch_indices] *= 3
        # r = torch.exp(-50 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.8)
        r = torch.exp(-10 * torch.sum(torch.abs(diff), dim=1))
        return r
    
    def _reward_ref_wheel_vel(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_vel = self.dof_vel.clone()
        vel_target = self.ref_dof_vel.clone()
        diff = (joint_vel[:,[2,5]] - vel_target[:,[2,5]]) * 0.5
        # hip_roll_yaw_pitch_indices = [0,1,2,6,7,8]
        # diff[:,hip_roll_yaw_pitch_indices] *= 3
        # r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r = torch.exp(-0.5 * torch.sum(torch.abs(diff), dim=1))
        return r

# <><><><><><><><><><> zsy's reward design <><><><><><><><><><><><><><><><>
    # 根据质心的偏移惩罚
    def _reward_total_com(self):
        raise NotImplementedError


    # 希望机器人hip角度不会变化较大 
    def _reward_hip_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:,[0,3]]), dim=1)


    def _reward_default_hip_knee_pos(self):
        """
        鼓勵機器人hip關節的電機擺動的幅度不要太大
        """
        joint_diff = self.dof_pos - self.default_dof_pos[:,6:]
        left_hip_knee = joint_diff[:, :2]
        right_hip_knee = joint_diff[:, 6:8]
        bugg = torch.norm(left_hip_knee, dim=1) + torch.norm(right_hip_knee, dim=1)
        bugg = torch.clamp(bugg - 0.04, 0, 50)
        return torch.exp(-bugg * 100)


    def _reward_keep_self_origin_pos(self):
        """
        世界座標系下機器人的位置和默認origin的distance的差距的懲罰,鼓勵機器人保持在原來的位置
        """
        distance_error = torch.norm(torch.abs(self.root_states[:, 0:2]) - torch.abs(self.env_origins[:, 0:2]), dim=-1)
        r = torch.exp(-distance_error * 10)
        return r


    def _reward_stand_still_default_pos(self):
        """
        鼓勵靜止的時候關節保持默認的狀態
        """
        wheel_indices = [2, 5]
        # hip_indices = [0, 3]
        dof_err = self.dof_pos - self.default_dof_pos[:, 6:]
        dof_err[:, wheel_indices] = 0.0  # 輪子不考慮
        # dof_err[:, hip_indices] = 0.0    # hip不考虑
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)


    def _reward_penarlized_too_low_height(self):
        """
        懲罰過低的高度，防止坐下來獲得所有獎勵
        """
        return -100. * torch.ones_like(self.base_height) * (self.base_height < 0.2)


    def _reward_penarlized_too_yaw_flip(self):
        """
        懲罰過大的傾斜角度，如yaw等,防止機器人用電池靠在地上
        """
        return -100. * torch.ones_like(self.base_height) * (torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) > 0.5)



    def _reward_orientation_yaw_roll(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = 3 * torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        # print("<><><>",torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1))
        return quat_mismatch 



    def _reward_base_pitch_ang_vel_constraint(self):
        """
        鼓勵機器人base繞y軸的角速度不會過快
        """
        r = torch.sum(torch.square(self.base_ang_vel[:, 1]), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        print("--",r)
        return r


    # def _reward_stand_still_wheel_pos_penality(self):
    #     """当命令很小时，机器人的轮子不应该有各个方向的位置变化"""
    #     # Penalize motion at zero commands
    #     diff_x = 5 * torch.square(self.wheel_pos[:, 0, :] - self.wheel_pos_init[:, 0])
    #     diff_y_z = torch.sum(torch.square(self.base_pos[:, 1:3] - self.base_pos_init[:, 1:3]), dim=1)
    #     return (diff_x + diff_y_z) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)




    # def _reward_stand_still_wheel_pos_penality(self):
    #     """当命令很小时，机器人不应该有各个方向的位置变化"""
    #     # Penalize motion at zero commands
    #     diff_lr = torch.norm(self.wheel_pos[..., :2]  - self.wheel_pos_init[..., :2], dim=-1)  #init = tensor([0.1481, 0.2796, 0.3470], device='cuda:0')
    #     diff_lr2 = torch.norm(diff_lr, dim=-1)
    #     # print(self.wheel_pos[[1,2,3],0,:], self.wheel_pos_init[[0,1,2],0,:])  
    #     # return () * (torch.norm(self.commands[:, :2], dim=1) < 0.1)  
    #     return 2 * torch.exp(-5 * (diff_lr2)) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # @@@
    def _reward_stand_still_proj_dist(self):
        """当命令很小时,鼓励机器人的双轮中心点到base投影距离始终在一定距离左右，不会发生很大的变化"""
        # Penalize motion at zero commands  
        mean_wheel_pos_x = torch.mean(self.wheel_pos[:,:,0], dim=1)   # 两轮中心点
        mean_wheel_pos_y = torch.mean(self.wheel_pos[:,:,1], dim=1)
        mean_wheel_pos_z = torch.mean(self.wheel_pos[:,:,2], dim=1)
        mean_wheel_pos = torch.stack((mean_wheel_pos_x, mean_wheel_pos_y, mean_wheel_pos_z), dim=-1) # num x 3

        proj_base_2_proj_wheel_mid = torch.norm(mean_wheel_pos[..., :2]  - self.base_pos[..., :2], dim=-1)  #init = tensor([0.1481, 0.2796, 0.3470], device='cuda:0')
        diff = proj_base_2_proj_wheel_mid
        # print(proj_base_2_proj_wheel_mid) 
        # print(self.wheel_pos[[1,2,3],0,:], self.wheel_pos_init[[0,1,2],0,:])  
        # return () * (torch.norm(self.commands[:, :2], dim=1) < 0.1)  
        x1 = 0.02
        x2 = 0.05
        x_mid = (x1 + x2) / 2
        if proj_base_2_proj_wheel_mid < x1 or proj_base_2_proj_wheel_mid > x2:
            r = -10 
        else:
            r =  2 * torch.exp(-100 * (proj_base_2_proj_wheel_mid - x_mid)) 
        return r * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

