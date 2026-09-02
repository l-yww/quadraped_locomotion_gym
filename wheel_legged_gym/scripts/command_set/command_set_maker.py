import os
import numpy as np
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import time
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from wheel_legged_gym.utils import Logger, Logger_pd, Logger_foot, Logger_wbc, Logger_arm

# import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import json
import copy
from wheel_legged_gym.utils.Integrated import Integrated_EST_policy, Integrated_RMA_policy, Integrated_HIM_policy, Integrated_ROA_policy, Integrated_PPO_policy, Integrated_TS_policy, Integrated_EST_Plane_policy, Integrated_DH_policy, Integrated_Prop_policy, Integrated_VAE_policy, Integrated_DH_Map_policy

import tkinter as tk
from tkinter import ttk
import threading

# ==================== play =======================
def play(args):
    if USE_JOYSTICK:
        # 启动滑块 GUI 在后台线程（daemon=True 确保主线程退出时自动结束）
        slider_thread = threading.Thread(target=slider_window, daemon=True)
        slider_thread.start()
    args.run_name = args.task
    print('args.task-------------------------',args.task)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.sim.max_gpu_contact_pairs = 2**10

    env_cfg.terrain.en_fix_step_height = True
    env_cfg.terrain.en_fix_slope = True
    # env_cfg.terrain.draw_scan_dots = True

    env_cfg.commands.push_gripper_stators = False
    env_cfg.commands.randomize_gripper_force_gains = False

    env_cfg.commands.tracking_ee_start_step = 0
    env_cfg.commands.force_start_step = 0

    env_cfg.asset.fix_base_link = HANG_ON 
    env_cfg.mode.use_net = USE_NET

    if args.control_test:
        env_cfg.control.control_test = True   # true for keyboard else for curriculumn
    else:
        env_cfg.control.control_test = False   # true for keyboard else for curriculumn 

    stop_state_log = 250 # number of steps before plotting states
    env_cfg.env.episode_length_s = 25
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s

    env_cfg.terrain.track_test = False #True
    env_cfg.terrain.track_units =[
                                    # "rough",
                                    # "sloped",
                                    # "pyramid sloped",
                                    # "sloped obstacle",
                                    "stairs",
                                    # "pyramid stairs",
                                    # "obstacles",
                                    # "wave",
                                    # "gap",
                                    # "stone pillars"
                                ]

    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.terrain_proportions =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] #[0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0, 0, 0.1, 0.1]
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5

    env_cfg.init_state.rand_init_dof = False

    '-----观测噪声----'
    env_cfg.noise.add_noise = True and RANDOM_ON
    '-----扰动、外界摩擦和恢复系数----'
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_rigids_after_start = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    '-----质量、质心、转动惯量----'
    env_cfg.domain_rand.randomize_base_mass = True and RANDOM_ON
    env_cfg.domain_rand.randomize_inertia = True and RANDOM_ON
    env_cfg.domain_rand.randomize_com_displacement = True and RANDOM_ON
    '-----电机能力----'
    env_cfg.domain_rand.randomize_motor_strength = True and RANDOM_ON
    env_cfg.domain_rand.randomize_PD_factor = True and RANDOM_ON
    '-----编码器的随机偏置----'
    env_cfg.domain_rand.randomize_motor_offset = True and RANDOM_ON
    env_cfg.domain_rand.randomize_default_dof_pos = False
    '-----固定延迟模拟----'
    env_cfg.domain_rand.fixed_action_delay = False
    env_cfg.domain_rand.fixed_torque_delay = False
    env_cfg.domain_rand.fixed_obs_delay = False
    '-----随机延迟模拟----'
    env_cfg.domain_rand.add_dof_lag = True and RANDOM_ON
    env_cfg.domain_rand.add_imu_lag = True and RANDOM_ON
    env_cfg.domain_rand.add_action_lag = True and RANDOM_ON
    '-----电机的阻尼摩擦特性----'
    env_cfg.domain_rand.randomize_joint_friction = True and RANDOM_ON
    env_cfg.domain_rand.randomize_joint_damping = True and RANDOM_ON
    env_cfg.domain_rand.randomize_joint_armature = True and RANDOM_ON

    '-----电高度图的随机化----'
    env_cfg.domain_rand.add_height_lag = True and RANDOM_ON
    env_cfg.domain_rand.height_lag_timesteps_range = [20, 100] 
    env_cfg.domain_rand.randomize_height_lag_timesteps = True and RANDOM_ON
    env_cfg.domain_rand.randomize_height_lag_timesteps_perstep = False

    env_cfg.domain_rand.add_height_noise = True and RANDOM_ON
    env_cfg.domain_rand.add_height_gaussian_noise = True and RANDOM_ON
    env_cfg.domain_rand.height_gaussian_noise = 0.02
    env_cfg.domain_rand.add_height_spike_noise = True and RANDOM_ON
    env_cfg.domain_rand.height_spike_noise_range = [0.1, 0.5]

    env_cfg.domain_rand.randomize_height_offset = True and RANDOM_ON
    env_cfg.domain_rand.height_offset_range = [-0.02, 0.02]
    env_cfg.domain_rand.randomize_height_rotation = True and RANDOM_ON
    env_cfg.domain_rand.height_rotation_roll_range = [-0.2, 0.2]
    env_cfg.domain_rand.height_rotation_pitch_range = [-0.2, 0.2]
    env_cfg.domain_rand.height_rotation_yaw_range = [-0.2, 0.2]
    env_cfg.commands.use_pose_commands_curriculum = use_pose_commands_curriculum

    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    def sample_arm_poses(env, sample_num=100):
        k = 0  # 已采样有效数量
        dof_pos = torch.zeros((env.num_envs, 6), device=env.device)  # 6个关节位置
        target_tensor = torch.zeros((sample_num, 6+3+4), device=env.device)  # 6(dof)+3(pos)+4(quat)
        
        while k < sample_num:
            start_time = time.time()
            # 1. 批量采样6个关节位置（在限位内均匀随机）
            
            for i in range(6):
                dof_pos[:, i] = (env.dof_pos_limits[i, 1] - env.dof_pos_limits[i, 0]) * \
                                torch.rand(env.num_envs, device=env.device) + env.dof_pos_limits[i, 0]
            
            # 2. 调用采样函数，检测碰撞
           
            pos, quat, ls_collision = env.command_set_sample(dof_pos)
            # print(ls_collision)
            
            # 3. 碰撞/越界则跳过当前采样
            if ls_collision:
                continue  # 无需k-1，直接跳过即可
            
            # 4. 拼接有效数据（修复维度匹配：dof_pos需取单环境数据）
            # 若num_envs=1，直接拼接；若多环境，取第一个有效环境
            target_data = torch.cat([pos.unsqueeze(0), quat.unsqueeze(0), dof_pos[0:1]], dim=1)
            
            # 5. 填充到目标张量（处理最后一批数据不足的情况）
            remaining = sample_num - k
            if target_data.shape[0] > remaining:
                target_tensor[k:k+remaining] = target_data[:remaining]
            else:
                target_tensor[k:k+target_data.shape[0]] = target_data
            
            # 6. 更新已采样数量
            k += target_data.shape[0]
            print(f"已采样：{k}/{sample_num}")
        return target_tensor
    def save_to_pt(tensor_data, save_path):
        """
        保存PyTorch张量（无需转NumPy）
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 若张量在GPU，先转CPU
        if tensor_data.is_cuda:
            tensor_data = tensor_data.cpu()
        torch.save(tensor_data, save_path)
        print(f"PT张量已保存到：{save_path}")

    # 调用保存（直接用原始torch张量，无需转numpy）
    # target_tensor_pt = torch.zeros((sample_num,13), device="cuda:0")  # 你的原始张量
    # save_to_pt(target_tensor_pt, "./arm_dataset/arm_sample_data.pt")

    # 读取验证
    # pt_data = torch.load("./arm_dataset/arm_sample_data.pt")
    command_set = sample_arm_poses(env,sample_num = 50000)
    save_to_pt(command_set,'wheel_legged_gym/scripts/command_set/command_set.pt')


    
if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    USE_NET = True         # False为跟随ref
    EXPORT_POLICY = True    # 导出model
    RENDER = True          # 保存视频
    FIX_COMMAND = True
    USE_JOYSTICK = False
    MOVE_CAMERA = False
    args = get_args()    
    args.play_flag = True   
    HANG_ON = True #False         # 机器人挂天上
    RANDOM_ON = False        # 开启随机化
    args.control_test = not FIX_COMMAND
    use_pose_commands_curriculum = False
    args.task= 'cowa_arm' #'cowa_dh'
    args.experiment_name = args.task
    args.load_run = "Mar16_13-44-31_arm_pose_command_range_increase_wbc_w_curriculum"
    args.checkpoint = -1
    args.headless = False
    args.num_envs = 1

    cuda = 1
    args.sim_device=f"cuda:{cuda}"
    args.rl_device=f"cuda:{cuda}"
    play(args)