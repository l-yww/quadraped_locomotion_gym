# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class mode:
        use_net = True
        
    class env:
        frame_stack = 5        # long history
        actor_input_stack = 5   # 输入给actor的
        num_est_prob = 4        # vel_xyz, height预测的信息的总维度
        c_frame_stack = 1
        num_single_obs = 25
        num_envs = 4096
        num_observations = 27
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 6
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        fail_to_terminal_time_s = 0.2
        dof_vel_use_pos_diff = True

    class safety:
        # 影响 pos_limit, vel_limit, torque_limit 的 reward 计算
        #* torque_limit 会影响torque下发的限制
        pos_limit = 1
        vel_limit = 1
        torque_limit = 1

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        en_fix_step_height = False
        en_fix_slope = False
        track_test = False # 测试柏林噪声是否添加成功
        add_perlin_noise = False #True # 开启时需要将机器人初始位姿提高，避免机器人初始姿态陷入地面
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False #True
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
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [plane ,smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0, 0, 0.1, 0.1]
        # terrain difficulty params (read by terrain.make_terrain):
        slope_max = 0.466            # 斜坡坡度系数，difficulty=1 时 slope_max=tan(25°)
        rough_height_min = 0.01      # 坑洼起伏下限 [m]（difficulty=0）
        rough_height_max = 0.03      # 坑洼起伏上限 [m]（difficulty=1）
        rough_slope_scale = 0.5      # 坑洼地形基底坡度 = slope*this（0=纯平地+坑洼）
        pit_depth_min = 0.06         # 凹坑深度下限 [m]
        pit_depth_max = 0.075        # 凹坑深度上限 [m]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        
        timeout_at_border = False
        
        x_init_range = 1.
        y_init_range = 1.
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = True
        teleport_thresh = 2.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 6 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        curriculum_threshold = 0.7
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.06] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        rand_init_dof = False
        rand_init_dof_range = 0.1 # [rad]

    class control:
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        Kp_scale = 30

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9
        action_clip_mode = 0 # see GymDofDriveModeFlags (0 is none, 1 is clip action, 2 is soft_clip action)

    class depth:
        use_camera = False #True

        position = [0.27, 0, 0.03]  # front camera
        y_angle = [-5, 5]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0


    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        wheel_name = "None"
        knee_name = "None"
        hip_name = "None"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fix the base of the robot
        fix_base_link_height = 1.8  # fix the base of the robot at the height
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.0 # 0.0
        thickness = 0.01


    class domain_rand:
        use_random = True

        push_robots = use_random
        push_interval_s = 8     # 每隔 8s 推机器人一次
        max_push_vel_xy = 0.1   # 推机器人的最大速度【会用冲量转化为Force】
        max_push_ang_vel = 0.1  # 推机器人时的最大角速度【应转为Torque] #! 目前没有用

        action_noise = 0.0
        action_delay = 0.

        rand_interval_s = 10    # 每隔rand_interval_s会重置一次随机化
        randomize_rigids_after_start = False         # 开启后,会每 10s 随机化关节质量质心参数

        randomize_friction = use_random             #* 仅初始化时一次随机
        friction_range = [0.1, 1.2]
        
        randomize_restitution = use_random          #* 仅初始化时一次随机
        restitution_range = [0, 1.0]

        # --------------- 随机化 base_link 质量 & 转动惯量 ----------------- #
        randomize_base_mass = use_random            #? 每 10s 刷新一次
        added_mass_range = [-3, 5]

        randomize_inertia = use_random              #? 每 10s 刷新一次
        randomize_inertia_range = [0.9, 1.1]

        # --------------- 随机化 质心位置 ----------------- #
        randomize_com_displacement = use_random     #? 每 10s 刷新一次  
        com_displacement_range = [-0.06, 0.06]
        
        # --------------- 随机化电机能力 ----------------- #
        randomize_motor_strength = use_random       #? 每 10s 刷新一次
        motor_strength_range = [0.9, 1.1]

        randomize_PD_factor = use_random            #? 每 10s 刷新一次
        Kp_factor_range = [0.9, 1.1]
        Kd_factor_range = [0.9, 1.1]

        # --------------- randomize_motor_offset与randomize_default_dof_pos 含义基本相同，均模拟关节角度的固定误差 ----------------- #
        randomize_motor_offset = use_random # 目前是使用torque的offset   #? 每 10s 刷新一次
        motor_offset_range = [-0.03, 0.03]

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
        obs_delay_steps = 1          # 10 * steps ms

        '维护Tensor,随机范围延迟'
        '随机延迟中,timesteps是按照PD的频率'
        # action延迟
        add_action_lag = use_random
        randomize_lag_timesteps = use_random        # [1]False，固定取延迟的最大值; [2]True,随机范围
        randomize_lag_timesteps_perstep = False     # [1]False，在episode中延迟固定; [2]True,每个step的延迟都随机
        lag_timesteps_range = [3, 11]   # 6~22ms
        # 编码器延迟
        add_dof_lag = use_random
        randomize_dof_lag_timesteps = use_random    
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 2] # 0~4ms
        # IMU延迟
        add_imu_lag = use_random
        randomize_imu_lag_timesteps = use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 2] # 0~4ms
        
        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #       
        '修改仿真器的关节物理参数，模拟摩擦和阻尼'

        """ 电机摩擦 """
        default_joint_friction = [0, 0, 0, 0, 0, 0]
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = False
        joint_1_friction_range = [0.8, 1.2]
        joint_2_friction_range = [0.8, 1.2]
        joint_3_friction_range = [0.8, 1.2]
        joint_4_friction_range = [0.8, 1.2]
        joint_5_friction_range = [0.8, 1.2]
        joint_6_friction_range = [0.8, 1.2]

        """ 电机阻尼 """
        default_joint_damping = [0, 0, 0, 0, 0, 0]
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        joint_1_damping_range = [0.8, 1.2]
        joint_2_damping_range = [0.8, 1.2]
        joint_3_damping_range = [0.8, 1.2]
        joint_4_damping_range = [0.8, 1.2]
        joint_5_damping_range = [0.8, 1.2]
        joint_6_damping_range = [0.8, 1.2]

        """ 电机转子转动惯量 """
        default_joint_armature = [0, 0, 0, 0, 0, 0]
        randomize_joint_armature = use_random 
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        joint_1_armature_range = [0.8, 1.2]
        joint_2_armature_range = [0.8, 1.2]
        joint_3_armature_range = [0.8, 1.2]
        joint_4_armature_range = [0.8, 1.2]
        joint_5_armature_range = [0.8, 1.2]
        joint_6_armature_range = [0.8, 1.2]

    class rewards:
        class scales:
            termination = -0.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25
        max_contact_force = 100. # forces above this value are penalized
        clip_single_reward = 1

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
            torques = 0.02
            dist = 4
            gravity = 1
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.0
            ang_vel = 0.1
            lin_vel = 0.1
            gravity = 0.05
            quat = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [0, -3, 1]  # [m] [10, 0, 6]   xxw
        lookat = [10, 5, 1.5]  # [m]
        draw_commands = False # for debugger
        draw_base_com = False # for view base com
        debug_viz = False # for view scan dot
        class commands:
            color = [0.1, 0.8, 0.1] # rgb
            size = 0.5

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
