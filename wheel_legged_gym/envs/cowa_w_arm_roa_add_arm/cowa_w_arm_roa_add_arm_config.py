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

# zsy modified ---->

from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


# class CowaCfg_Father_Arm(LeggedRobotCfg):
class CowaCfg_Arm(LeggedRobotCfg):
    """
    Configuration class for the Cowa robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim  
        num_est_prob = 9            # vel_xyz + arm_point_xyz + ee_pos
        frame_stack = 10            # actor-history
        actor_input_stack = 1       # 当前actor的输入维度 -> actor
        c_frame_stack = 1           # critic-history NOTE: Fixed 1 do not change bestly
        num_latent = 20             # The dim of latents "z"
        
        # single
        num_single_obs = 25 + 12   # 25 prop obs + 12 obs from arm joints     
        single_num_privileged_obs = num_single_obs + 12  + 1 + 3 + 6 + 6 + 6 + 77 + num_est_prob

        # frames x singles
        ### Q:why use num_observations to house so big??  A:To easily fetch parameters or metrics
        ### NOTE: Do not change below
        num_observations = int((frame_stack + 1) * num_single_obs + single_num_privileged_obs)  #prop + priv + hist
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)      #300
        
        num_actions = 6  
        num_envs = 4096  
        episode_length_s = 25 # episode length in seconds     
        use_ref_actions = False
        fail_to_terminal_time_s = 0.5


        # privileged obs
        # priv_observe_friction = False  # 1
        # priv_observe_base_mass = False # 1
        # priv_observe_restitution = False #1
        # priv_observe_com_displacement = False    # 3
        
        # priv_observe_motor_strength = False  # 6
        # priv_observe_motor_offset = False    # 6 感觉还是不要了比较好
        # priv_observe_gravity = False    # 3
        # priv_observe_measure_heights = False # 187


    class safety:
        # safety factors
        pos_limit = 1
        vel_limit = 1
        torque_limit = 1    #0.85 xxx 1
        # acc_limit = 0.6
        # dof_acc_limits_ratio = 6

    class asset(LeggedRobotCfg.asset):
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged_withoutknee.urdf"  
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged.urdf"  
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged_rea.urdf"     
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_deployed/urdf/cowa_jiaozhun_w_arms_right.urdf"  

        name = "cowa_robot"   
        offset = 0. 
        l1 = 0.25
        l2 = 0.25
        foot_name = "wheel"
        penalize_contacts_on = ["base", "hip", "knee", "battery", "hand_center"] ## 加上机械臂末端的碰撞惩罚
        terminate_after_contacts_on = ["base", "hip", "knee", "battery", "hand_center"]

        disable_gravity = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False #
        replace_cylinder_with_capsule = False
        fix_base_link = False
        fix_base_link_height = 1.8  # fix the base of the robot at the height
        Curriculum_test = False #用来设置compute torques时是否使用机械臂的轨迹

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'  
        mesh_type = 'trimesh'
        curriculum = True   
        track_test = False # 测试柏林噪声是否添加成功
        add_perlin_noise = False # 开启时需要将机器人初始位姿提高，避免机器人初始姿态陷入地面
        # rough terrain only:
        measure_heights = True
        static_friction = 1
        dynamic_friction = 1
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 0  # starting curriculum state  wende 改的，本来是10
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        # terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]   #wende 改的，本来是[0, 0, 1, 0., 0., 0, 0] 
        restitution = 0.      
        measured_points_x = [
            -0.5,
            -0.4, 
            -0.3, 
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 0.6m x 1m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        slope_treshold = 0.2  #0.3

    # TODO
    class noise:
        add_noise = True
        noise_level = 1    # scales other values

        class noise_scales:
            dof_pos = 0.1
            dof_vel = 0.3   #0.5
            ang_vel = 0.2   #0.2
            lin_vel = 0.1   #0.1
            gravity = 0.04  #0.05
            quat = 0.08     #0.1
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = False 
        default_joint_angles = {  # target angles when action = 0.0
            "left_hip_pitch_joint": 0.,
            "left_knee_pitch_joint": 0.,
            "left_wheel_joint": 0.,
            "right_hip_pitch_joint": 0.,
            "right_knee_pitch_joint": 0.,
            "right_wheel_joint": 0.,
            
            "joint15":0.,
            "joint16":-2.55,
            "joint17":3.14,
            "joint18":0.,
            "joint19":0.,
            "joint20":0.,
        }


    # TODO control_type,pos_action_scale,vel_action_scale
    class control(LeggedRobotCfg.control):
        control_test = False # zsy add 用于机械臂键盘控制的测试，默认False即可，可在args里修改
        curriculumn_arm_select = [0.1, 0.1, 0.4, 0.4]
        
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:    
        stiffness = {"hip": 110.0, "knee": 210, "wheel": 0,   "joint1": 150,"joint2": 150,"joint3": 150,"joint4": 50,"joint5": 50,"joint6": 50}  # [N*m/rad]
        damping   = {"hip": 8,     "knee": 8,   "wheel": 3,   "joint1":5,   "joint2":5,   "joint3":5,   "joint4":5,  "joint5":5,  "joint6":5}  # [N*m*s/rad]
        # 机械臂不用pd
        # stiffness = {"hip": 110.0, "knee": 210, "wheel": 0, }  # [N*m/rad]
        # damping   = {"hip": 8,     "knee": 8,   "wheel": 3, }  # [N*m*s/rad]
        
        # action scale: target angle = actionScale * action + defaultAngle   
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT   
        decimation = 5  # 100Hz   
        pos_action_scale = 1.5  
        vel_action_scale = 10.0  
        
        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = True   # NOTE： 原来为False
        ratio = 0.9
        projected_gravity = False
        # feedforward_force = 6 # 42.46kg / 2
        
    class sim(LeggedRobotCfg.sim):
        web_vis = False
        port = 6001      # zmp的端口，如果训练启动卡住，则这个端口被占用，更改以下就行
        web_vis_envs = 1 #用于控制web可视化的机器人个数，建议不要超过5个，性能影响很大
        keep_default_viewer = False

        dt = 0.002   # 500HZ train
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10                                  # xxw
            solver_type = 1  # 0: pgs, 1: tgs   
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2



    class domain_rand(LeggedRobotCfg.domain_rand):
        # TODO randomize_default_dof_pos, randomize_action_delay
        push_robots = True         # NOTE: zsy修改原来为True
        push_interval_s = 8
        max_push_vel_xy = 0.05  # 0.2

        action_noise = 0.02      #   NOTE: 加噪聲
        action_delay = 0.1
        # <><><><><><><><><><><><><><>
        rand_interval_s = 10    ## 
        randomize_rigids_after_start = True  
        randomize_friction = True             # xxw True
        friction_range = [0.1, 2]
        randomize_base_mass = True #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-5, 5]              # 加负载
        randomize_restitution = True            # TODO
        restitution_range = [0, 1.0]            # 加到priv里的东西

        # <><><><><><><><><><><><><><>
        randomize_com_displacement = True       # 加到priv里的东西
        com_displacement_range = [-0.05, 0.05]  # base link com的随机化范围
        randomize_each_link = False
        link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)
        
        randomize_inertia = True    
        randomize_inertia_range = [0.8, 1.2]

        randomize_motor_strength = True      
        motor_strength_range = [0.9, 1.1]      

        randomize_PD_factor = True #             
        Kp_factor_range = [0.9, 1.1]            
        Kd_factor_range = [0.9, 1.1]

        randomize_motor_offset = True           #   目前是使用torque的offset
        default_motor_offset = [0,0,0,0,0,0,
                                0, 0.0, 0,\
                                0, 0.0, 0,\
                                ]
        motor_offset_range = [-0.03, 0.03]

        randomize_default_dof_pos = False     # zsy改的 NOTE:原來爲True  # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos_range = [-0.1, 0.1]

        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0

        randomize_gravity = False # 建议不加
        gravity_range = [-1.0, 1.0]         #

        # randomize_lag_timesteps = True  # NOTE：原来为Flase    # 模拟delay，对于lag用于给历史的action
        # lag_timesteps = 2       #2~4ms walk these ways 加固定action延迟

        randomize_torque_delay = False     
        torque_delay_steps = 2

        # randomize_obs_delay = False #用队列加固定obs延迟
        # obs_delay_steps = 1
        # <><><><><><><><><><><><><><><><> lag <><><><><><><><><><><><><><><><><<>><>><>
        # add action lags
        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = True
        lag_timesteps_range = [3, 7]       #10ms~34ms

        # --no need to be so big
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 2]        # 1~4ms 


        add_imu_lag = True # 现在是euler，需要projected gravity                    # 这个是 imu 的延迟
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False         # 不常用always False
        imu_lag_timesteps_range = [5, 11]       # 实际5 ~ 35
        # <><><><><><><><><><><><><><><><><> RD <><><><><><><><><><><><><><><><<>><>><>
        randomize_coulomb_friction = True
        joint_stick_friction_range = [0.1, 0.2]
        joint_coulomb_friction_range = [0.0, 0.0]
        
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False

        default_joint_friction = [0,0,0,0,0,0,\
                                 0.1, 0.002, 0.02, \
                                 0.1, 0.002, 0.02,]  # NOTE：减小摩擦力
        joint_friction_range = [0.8, 1.2]
        # joint_friction_range = [1.5, 1.5]
        joint_1_friction_range = [0.9, 1.1]
        joint_2_friction_range = [0.9, 1.1]
        joint_3_friction_range = [0.9, 1.1]
        joint_4_friction_range = [0.9, 1.1]
        joint_5_friction_range = [0.9, 1.1]
        joint_6_friction_range = [0.9, 1.1]


        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False      # 原来为True zsy
        default_joint_damping = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,\
                                1.8, 4, 0.03, \
                                 1, 4, 0.03, ]
        joint_damping_range = [0.8, 1.2]
        joint_1_damping_range = [0.8, 1.2]
        joint_2_damping_range = [0.8, 1.2] 
        joint_3_damping_range = [0.8, 1.2]
        joint_4_damping_range = [0.8, 1.2]
        joint_5_damping_range = [0.8, 1.2]
        joint_6_damping_range = [0.8, 1.2]


        randomize_joint_armature = True 
        randomize_joint_armature_each_joint = False
        default_joint_armature = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, \
                                  0.138096, 0.08, 0.08, \
                                  0.138096, 0.08, 0.08, ] 
        joint_armature_range = [0.8, 1.2]     # Factor
        joint_1_armature_range = [0.95, 1.05]
        joint_2_armature_range = [0.95, 1.05]
        joint_3_armature_range = [0.95, 1.05]
        joint_4_armature_range = [0.95, 1.05]
        joint_5_armature_range = [0.9, 1.1]
        joint_6_armature_range = [0.9, 1.1]
    # <><><><><><><><><><><><><><><><><><><><><><><><><><>
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [-0.01, 0.01]  # min max [m/s]
            # ang_vel_yaw = [-0.01, 0.01]  # min max [rad/s]
            # height = [0.329, 0.332]
            # # height = [0.30, 0.34]
            # heading = [-3.14, 3.14]
            lin_vel_x = [-0.8, 0.8]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            height = [0.25, 0.4]
            heading = [-3.14, 3.14]


    class rewards:
        base_height_target = 0.36         #wh 直立1.1   曲膝 1.06  pos = [0.0, 0.0, 1.36]  # 14 dof版本, 
        only_positive_rewards = False 
        # tracking_sigma = 4    # vel = 0.5 对应 20; vel = 1 对应 4; vel = 1.5 对应 2; vel = 2 对应 1; vel = 3 对应 0.5
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_vel_x = 20
        tracking_sigma_ang_vel = 20
        tracking_vel_enhance = False
        tracking_vel_hard = False    # 非常严苛
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized  
        )  
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        max_contact_force = 100  # Forces above this value are penalized xxx 1400
        clip_single_reward = 1

        min_feet_dist = 0.64
        max_feet_dist = 0.645
        max_feet_z_dist = 0.05
        min_feet_z_dist = -0.05
        base_lin_acc_limit = 0.8

        class scales:
            # ====== task reward ==============  
            feet_distance = 0.2
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0
            base_height = 1.0         # 1.
            # wheel_adjustment = 0.5                # NOTE: 模仿燃坤加的
            # keep_self_origin_pos = 2.0            # NOTE: 鼓勵機器人保持在原始位置處
            # stand_still_joint_default = 0.4       # NOTE: 鼓勵機器人保持在原位置時關節角度保持默認  
            # penarlized_too_low_height = 1.0       # NOTE：過低的高度直接給-100的懲罰
            # penarlized_too_yaw_flip = 1.0         # NOTE: 防止機器人傾倒過大伏在地面上不動
            stand_still_joint_default = 0.2
            same_foot_z_position = 10.5
            # ====== constraint reward ======== 
            nominal_state = -0.05         # NOTE: 感觉没什么用
            lin_vel_z = -0.1e-3
            ang_vel_xy = -0.05
            dof_vel = -5e-4
            dof_acc = -5e-7
            # wheel_acc = -1e-4           # NOTE： zsy -1e-7  #惩罚过大的加速度
            # wheel_vel = -1e-4           # NOTE： zsy add 懲罰過大的輪子的速度,擴大二倍
            torques = -1e-7  
            power = -1e-8
            action_rate = -0.2
            action_smoothness = -0.5 # smoothness  #-0.04
            # ====== panerlized reward ========   
            collision = -20.0
            dof_pos_limits = -0.1 
            dof_vel_limits = -0.1 
            stand_still_vel_penality = -20.0 #惩罚静止速度   
            orientation = -10.0     
            
                        




    class normalization:
        class obs_scales:
            lin_vel = 10.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torques = 0.05
            quat = 1
            # zsy add
            arm_2_base_scales = 1.0    #[0.1751,-0.0111,0.4968]
        clip_observations = 100.
        clip_actions = 100.
        


# =========================== ROA ===============================
class CowaCfgPPO_Arm(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunnerROA'   # DWLOnPolicyRunner

    class policy():
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        priv_encoder_dims = [256, 128] 

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

        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000] #if not RESUME else [0, 1, 1000, 1000]


    class runner:
        policy_class_name = 'ActorCritic_ROA'
        algorithm_class_name = 'PPO_ROA'
        num_steps_per_env = 40  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 500  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa-est-success'
        run_name = 'ppo'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt


