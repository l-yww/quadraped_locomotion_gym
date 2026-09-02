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


from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Cowa_Num:
    WLR_index = '1'

class CowaCfg(LeggedRobotCfg):
    class mode:
        use_net = True

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66        # long history
        short_frame_stack = 5   # short history
        actor_input_stack = 5   # 输入给actor的
        c_frame_stack = 1
        num_est_prob = 3 + 1    # vel_xyz, height预测的信息的总维度

        num_actions = 4
        num_single_obs = 3 + 3*num_actions - 2 + 6 # cmd + dof pos[w/o wheel] + dof vel + action + imu
        
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + num_est_prob
        projected_gravity = False

        priv_observe_friction = False
        if priv_observe_friction:
            single_num_privileged_obs += 1

        priv_observe_restitution = False
        if priv_observe_restitution:
            single_num_privileged_obs += 1

        priv_observe_payloads = True
        if priv_observe_payloads:
            single_num_privileged_obs += 1

        priv_observe_inertia = True
        if priv_observe_inertia:
            single_num_privileged_obs += 1

        priv_observe_motor_strength = True
        if priv_observe_motor_strength:
            single_num_privileged_obs += num_actions

        priv_observe_motor_offset = True
        if priv_observe_motor_offset:
            single_num_privileged_obs += num_actions

        priv_observe_com_displacement = True
        if priv_observe_com_displacement:
            single_num_privileged_obs += 3

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        
    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_deployed/urdf/cowa_4rad_wo_arm_4dof.urdf"     
        name = "wheel_legged_robotxxxx"  # actor name
        foot_name = "wheel"
        foot_radius = 0.11
        knee_name = "knee"
        hip_name = "hip"
        penalize_contacts_on = ["hip", "knee", "battery", "base", "hand_center"]
        terminate_after_contacts_on = ["hip", "knee", "battery", "base", "hand_center"]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = False
        rand_init_dof_range = 0.2 # [rad]
        default_joint_angles = {  # target angles when action = 0.0
            "left_hip_pitch_joint": 0,
            "left_wheel_joint": 0,
            "right_hip_pitch_joint": 0,
            "right_wheel_joint": 0                    
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        WLR_index = Cowa_Num.WLR_index
        # PD Drive parameters:
        if WLR_index == "1":
            """ No1 """
            stiffness = {"hip": 75.0, "wheel": 0}  # [N*m/rad]
            damping = {"hip": 4, "wheel": 3}  # [N*m*s/rad]
        elif WLR_index == "2":
            """ No2 """
            stiffness = {"hip": 110.0, "knee": 210, "wheel": 0}  # [N*m/rad]
            damping = {"hip": 8, "knee": 8, "wheel": 3}  # [N*m*s/rad]
        elif WLR_index == "3":
            """ No3 """
            stiffness = {"hip": 75.0, "wheel": 0}  # [N*m/rad]
            damping = {"hip": 4, "wheel": 3}  # [N*m*s/rad]

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 100Hz
        pos_action_scale = 0.25
        vel_action_scale = 2.

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

        cycle_time = 3.0 # ref pos 的 cycle time

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 500 Hz

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = True

        push_robots = use_random
        push_interval_s = 8
        max_push_vel_xy = 0.1  # 0.2
        max_push_ang_vel = 0.1

        action_noise = 0.0 # 0.02
        action_delay = 0. # 0.1

        rand_interval_s = 10    # 每隔rand_interval_s会重置一次随机化
        randomize_rigids_after_start = False     #控制link的friction和restituion的随机化开关
        randomize_friction = use_random             # xxw True
        friction_range = [0.2, 1.6]
        
        randomize_restitution = use_random           
        restitution_range = [0, 1.0]           

        # --------------- 随机化 base_link 质量 & 转动惯量 ----------------- #
        randomize_base_mass = use_random #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-2, 5]                  # 加负载

        randomize_inertia = use_random    
        randomize_inertia_range = [0.8, 1.2]

        # --------------- 随机化 质心位置 ----------------- #
        randomize_com_displacement = use_random      
        com_displacement_range = [-0.08, 0.08]  # base link com的随机化范围
        randomize_each_link = False
        link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)
        
        # --------------- 随机化电机能力 ----------------- #
        randomize_motor_strength = use_random      
        motor_strength_range = [0.8, 1.2]      

        randomize_PD_factor = use_random #             
        Kp_factor_range = [0.8, 1.2]            
        Kd_factor_range = [0.8, 1.2]

        # --------------- randomize_motor_offset与randomize_default_dof_pos含义相同，均模拟关节角度的固定误差 ----------------- #
        randomize_motor_offset = use_random # 目前是使用torque的offset
        default_motor_offset = [0, 0,\
                                0, 0]
        motor_offset_range = [-0.03, 0.03] # 仅针对No2，其他的轮足均无该问题

        randomize_default_dof_pos = False # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos_range = [-0.03, 0.03]

        # ------------------- 延迟模拟 -------------------------- #
        '维护队列，固定延迟'
        '固定延迟中,timesteps是按照Policy的频率'
        # action传到PD控制器的延迟
        fixed_action_delay = False
        action_delay_steps = 2       # 10 * steps ms
        # PD控制器到电机扭矩实际达到torque值的延迟
        fixed_torque_delay = False
        torque_delay_steps = 2       # 2 * steps ms
        # 编码器和IMU传回Policy的延迟
        fixed_obs_delay = False
        obs_delay_steps = 1

        '维护Tensor,随机范围延迟'
        '随机延迟中,timesteps是按照PD的频率'
        # action延迟
        add_action_lag = True and use_random
        randomize_lag_timesteps = True and use_random
        randomize_lag_timesteps_perstep = True and use_random
        lag_timesteps_range = [3, 11]       # 2ms * steps
        # 编码器延迟
        add_dof_lag = True and use_random
        randomize_dof_lag_timesteps = True and use_random # [1]False，固定取延迟的最大值; [2]True,随机范围
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 2]    # 2ms * steps
        # IMU延迟
        add_imu_lag = True and use_random
        randomize_imu_lag_timesteps = True and use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 2]    # 2ms * steps

        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #       
        WLR_index = Cowa_Num.WLR_index
        
        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        if WLR_index == '1':
            default_joint_friction = [0.045, 0.0, 0.045, 0.0]
        elif WLR_index == '2':
            default_joint_friction = [0.045, 0.15, 0.02, 0.045, 0.25, 0.02,]
        elif WLR_index == '3':
            default_joint_friction = [0.045, 0.0, 0.045, 0.0]
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = use_random
        joint_1_friction_range = [0.2, 1.2]
        joint_2_friction_range = [0.2, 1.2]
        joint_3_friction_range = [0.2, 1.2]
        joint_4_friction_range = [0.2, 1.2]
        joint_5_friction_range = [0.2, 1.2]
        joint_6_friction_range = [0.2, 1.2]


        if WLR_index == '1':
            default_joint_damping = [12, 0.15, 12, 0.15] # wh No1
        elif WLR_index == '2':
            default_joint_damping = [12., 8, 0.03, 12., 6, 0.03] # wh No2
        elif WLR_index == '3':
            default_joint_damping = [12, 0.015, 12, 0.015] #  wh No3 # [12, 8, 0.015]
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        joint_1_damping_range = [0.2, 1.2]
        joint_2_damping_range = [0.2, 1.2]
        joint_3_damping_range = [0.2, 1.2]
        joint_4_damping_range = [0.2, 1.2]
        joint_5_damping_range = [0.2, 1.2]
        joint_6_damping_range = [0.2, 1.2]

        if WLR_index == '1':
            default_joint_armature = [96 * 120 ** 2 * 1e-7, 0.08,\
                                      96 * 120 ** 2 * 1e-7, 0.08] # wh No1
        elif WLR_index == '2':
            default_joint_armature = [96 * 120 ** 2 * 1e-7, 0.5, 0.08,\
                                      96 * 120 ** 2 * 1e-7, 0.5, 0.08] # wh No2 还未确定 0.5, 0.3, 0.12
        elif WLR_index == '3':
            default_joint_armature = [96 * 120 ** 2 * 1e-7, 0.08,\
                                      96 * 120 ** 2 * 1e-7, 0.08] # wh No3
        randomize_joint_armature = use_random
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        joint_1_armature_range = [0.2, 1.2]
        joint_2_armature_range = [0.2, 1.2]
        joint_3_armature_range = [0.2, 1.2]
        joint_4_armature_range = [0.2, 1.2]
        joint_5_armature_range = [0.2, 1.2]
        joint_6_armature_range = [0.2, 1.2]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            height = [0.34, 0.34]
            heading = [-3.14, 3.14]

    class rewards:
        only_positive_rewards = False 
        tracking_sigma_lin_vel = 20
        tracking_sigma_ang_vel = 20
        base_height_target = 0.34
        max_contact_force = 100  
        soft_dof_pos_limit = (
            1  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 0.8
        soft_dof_vel_limits = 0.8
        clip_single_reward = 1
        min_feet_dist = 0.64
        max_feet_dist = 0.645
        max_feet_z_dist = 0.1

        class scales:
            ################# termination rewards ##################
            # termination = -1.
            # keep_balance = 1.0 
            #################   tracking rewards  ##################
            " 速度跟踪 "
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0
            # tracking_lin_vel_pbrs = 0.4 / 60.
            # tracking_ang_vel_pbrs = 0.4 / 60.
            #################    style rewards    ######################
            " 机体姿态 "
            base_height = -20
            # tracking_base_height_pbrs = 0.4 / 60.
            orientation = -20
            # orientation_positive = 0.5
            " 脚部姿态 "
            same_foot_x_position = -1
            feet_distance = 0.2
            # default_joint_pos = -0.1
            # nominal_foot_position = 0.5
            ############## normalized rewards #####################
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            dof_vel = -5e-5             # 若有wheel需要注意
            dof_acc = -5e-7             # 若有wheel需要注意
            torques = -1e-5
            wheel_acc = -5e-6
            # torques = -1e-30
            torque_limits = -0.05
            power = -2e-5
            action_rate = -0.1
            action_smoothness = -0.1
            base_acc = -1e-1
            collision = -2.0
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            stand_base_vel_penality = -4
            stand_wheel_vel_penality = -0.3
            # opposite_wheel_vel = -1
            # opposite_vel = -2
            # stability = -1
         
    class normalization:
        class obs_scales:
            lin_vel = 10.0
            ang_vel = 2.
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            torques = 0.05
            quat = 1
            gravity = 1

        clip_observations = 100.
        clip_actions = 20.

class CowaCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_DH'   # OnPolicyRunnerEstimator

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        # short_history estimator
        estimator_hidden_dims=[128, 64]
        long_history_type = 'cnn'
        # for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   # long history output dim
        in_channels = CowaCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.0e-4  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        # estimator para
        mlp_learning_rate = 5.e-4
        num_adaptation_module_substeps = 1

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_DH'
        algorithm_class_name = 'PPO_DH'
        num_steps_per_env = 48  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa'
        run_name = ''
        # Load and resume
        resume = False
        load_run = ""  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt

