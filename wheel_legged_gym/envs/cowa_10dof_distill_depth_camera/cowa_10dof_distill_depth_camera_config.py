import numpy as np
from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_joint_names, get_stiffness_damping,\
                                            get_default_joint_friction, get_default_joint_damping, get_default_joint_armature

class Cowa_Num:
    WLR_index = 'v2_10dof'
    DOF = 10

class CowaCfg(LeggedRobotCfg):
    class mode:
        use_net = True

    class depth:
        use_warp = True
        use_camera = True  # 启用深度相机

        position = [0.24, -0.0175, 0.12]  # 使用sensor.forward_camera的位置
        y_angle = [21.2, 24.6]  # 对应rotation中的pitch
        z_angle = [-5.7, 5.7]
        x_angle = [-5.7, 5.7]

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (120, 160)  # 对应sensor.forward_camera.resolution
        resized = (48, 64)  # 对应sensor.forward_camera.resized_resolution
        horizontal_fov = 88  # 对应sensor.forward_camera.horizontal_fov中间值
        buffer_len = 2

        # near_clip = 0.05  # 对应near_plane
        # far_clip = 3.0  # 对应depth_range[1]
        # dis_noise = 0.0

    class sensor:
        class forward_camera:
            obs_components = ["forward_depth"]
            resolution = [int(480/4), int(640/4)]  # 原始分辨率 (480/4, 640/4)
            position = dict(
                mean=[0.27, 0.0, 0.03],
                std=[0.0, 0.0, 0.0]
            )
            rotation = dict(
                lower=[-0.1, 0.37, -0.1],
                upper=[0.1, 0.43, 0.1]
            )
            resized_resolution = [48, 64]
            output_resolution = [48, 64]
            horizontal_fov = [86, 90]
            crop_top_bottom = [int(48/4), 0]
            crop_left_right = [int(28/4), int(36/4)]
            near_plane = 0.05
            depth_range = [0.0, 3.0]
            latency_range = [0.08, 0.142]  # 80~142ms 延迟
            latency_resampling_time = 5.0
            refresh_duration = 1/10  # 10Hz 刷新率

    class env(LeggedRobotCfg.env):

        projected_gravity = False   # [True] projected_gravity; [False] Euler Angle
        
        height_map = False  # 是否将测量的地形高度作为obs的一部分输入给policy，True则输入，False则不输入
        

        # change the observation dim
        frame_stack = 66        # long history的帧数
        short_frame_stack = 5   # short history的帧数
        c_frame_stack = 1       # 输入给critic的帧数
        num_est_prob = 3 + 1   # vel_xyz, height预测的信息的总维度
        num_actions = Cowa_Num.DOF
        num_height_obs_len=7*11 # 7*11=77, 7是测量点的数量，11是每个测量点的特征数量（x,y,z, height, normal_x, normal_y, normal_z, contact_state, contact_force_x, contact_force_y, contact_force_z）

        if projected_gravity:
            num_single_obs = 2 + 3 + 3*num_actions - 2 + 3 + 3  # cmd + dof pos[w/o wheel] + dof vel + action + imu
        else:
            num_single_obs = 2 + 3 + 3*num_actions - 2 + 3 + 2  # cmd + dof pos[w/o wheel] + dof vel + action + imu
        
        #TODO：height_map 噪声先不加，后续可以考虑加上
        if height_map:
            num_single_obs += num_height_obs_len

        num_observations = int(frame_stack * num_single_obs)

        if projected_gravity:
            single_num_privileged_obs = num_single_obs + num_est_prob 
        else:
            single_num_privileged_obs = num_single_obs + num_est_prob + 1 

        single_num_privileged_obs+=num_height_obs_len

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

        class teacher_env:
            projected_gravity = False   # [True] projected_gravity; [False] Euler Angle
            height_map = True # 是否将测量的地形高度作为obs的一部分输入给policy，True则输入，False则不输入
            # change the observation dim
            frame_stack = 66        # long history的帧数
            short_frame_stack = 5   # short history的帧数
            c_frame_stack = 1       # 输入给critic的帧数
            num_est_prob = 3 + 1   # vel_xyz, height预测的信息的总维度
            num_actions = Cowa_Num.DOF
            num_height_obs_len=7*11 # 7*11=77, 7是测量点的数量，11是每个测量点的特征数量（x,y,z, height, normal_x, normal_y, normal_z, contact_state, contact_force_x, contact_force_y, contact_force_z）

            if projected_gravity:
                num_single_obs = 2 + 3 + 3*num_actions - 2 + 3 + 3  # cmd + dof pos[w/o wheel] + dof vel + action + imu
            else:
                num_single_obs = 2 + 3 + 3*num_actions - 2 + 3 + 2  # cmd + dof pos[w/o wheel] + dof vel + action + imu
            
            #TODO：height_map 噪声先不加，后续可以考虑加上
            if height_map:
                num_single_obs += num_height_obs_len

            num_observations = int(frame_stack * num_single_obs)



        
    class asset(LeggedRobotCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_v2/cowa_wheel_legged_v2/urdf/wheel_v2_10dof_wo_arm_simplify_cylinder.urdf"
        name = "wheel_legged_robotxxxx"  # actor name
        foot_name = "foot"
        wheel_name = "wheel"
        wheel_radius = 0.11
        knee_name = "knee"
        hip_name = "hip"
        penalize_contacts_on = ["hip", "knee", "base"]
        terminate_after_contacts_on = ["hip", "knee", "base"]
        replace_cylinder_with_capsule = True #False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # terrain types: [plane ,smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0.0, 0.0] #[0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0.0, 0.0]
        slope_treshold = 0.1 # slopes above this threshold will be corrected to vertical surfaces
        measure_heights = True
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

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = True
        rand_init_dof_range = 0.1 # [rad]
        joint_names = get_joint_names(Cowa_Num.DOF, Cowa_Num.WLR_index)
        default_joint_angles = {name: 0.0 for name in joint_names}

    class control(LeggedRobotCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness, damping = get_stiffness_damping(Cowa_Num.DOF, Cowa_Num.WLR_index)

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 100Hz 
        # decimation = 4 # 50Hz
        pos_action_scale = 0.25
        vel_action_scale = 2.

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

        cycle_time = 0.8 #2 #3.0 # ref pos 的 cycle time
        offset = 0.5

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 500Hz 
        # dt = 1/200  # 200Hz 

        body_measure_points = { # transform are related to body frame
            "left_foot": dict(
                x= [i for i in np.arange(-0.11, -0.06, 0.04)],
                y= [-0.12],
                z= [-0.09, -0.045 ,0.0, 0.045, 0.09],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
            "right_foot": dict(
                x= [i for i in np.arange(-0.11, -0.06, 0.04)],
                y= [0.12],
                z= [-0.09, -0.045 ,0.0, 0.045, 0.09],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = True  #第一阶段False，第二阶段True

        push_robots = use_random
        push_interval_s = 8
        max_push_vel_xy = 0.2  # 0.2
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
        motor_offset_range = [-0.05, 0.05] # 仅针对No2，其他的轮足均无该问题

        randomize_default_dof_pos = False # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos_range = [-0.05, 0.05]

        # ------------------- 延迟模拟 -------------------------- #
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
        DOF = Cowa_Num.DOF

        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        default_joint_friction = get_default_joint_friction(WLR_index, DOF)
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.5, 1.5]

        default_joint_damping = get_default_joint_damping(WLR_index, DOF)
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.5, 1.5]

        default_joint_armature = get_default_joint_armature(WLR_index, DOF)
        randomize_joint_armature = use_random 
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.5, 1.5]

    class viewer:
        ref_env = 0
        pos = [0, -3, 1]  # [m] [10, 0, 6]   xxw
        lookat = [10, 5, 1.5]  # [m]
        draw_commands = False # for debugger
        draw_base_com = False # for view base com
        debug_viz = True # for view scan dot
        class commands:
            color = [0.1, 0.8, 0.1] # rgb
            size = 0.5

        draw_volume_sample_points = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            # lin_vel_x = [0.1, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            # ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]
            height = [0.37, 0.37]
            # height = [0.45, 0.45]
            heading = [-3.14, 3.14]

    class rewards:
        only_positive_rewards = False 
        tracking_sigma_lin_vel = 20
        tracking_sigma_ang_vel = 20
        target_feet_height_min = 0.005 #0.02
        target_feet_height_max = 0.03
        clearance_height_target = 0.05
        max_contact_force = 500  
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 0.95
        soft_dof_vel_limits = 0.95
        clip_single_reward = 1
        min_wheel_dist = 0.637 #0.64
        max_wheel_dist = 0.643 #0.645
        min_feet_dist = 0.44 #0.48
        max_feet_dist = 0.54
        max_collision_xy_force_threshold = 50

        class scales:
            #################   tracking rewards  ##################
            " 速度跟踪 "
            tracking_lin_vel = 5.5
            tracking_ang_vel = 3.0
            #################    style rewards    ######################
            contact = 2.0
            feet_air_time = 1.0
            feet_clearance = -1
            no_fly = 0.75
            " 机体姿态 "
            base_height = -50 
            orientation = -10 
            " 脚部姿态 "
            feet_distance = -1
            default_hip_roll = -1
            feet_slip = -0.25
            ############## normalized rewards #####################
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            dof_vel = -5e-5             
            dof_acc = -5e-7            
            torques = -1e-6 
            wheel_acc = -5e-6
            wheel_vel = -0.1
            torque_limits = -0.05
            power = -2e-5
            action_rate = -0.001 
            action_smoothness = -0.1 
            collision = -2.0
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            stand_still = -0.4
            # feet_contact_forces = -0.01
            feet_ground_parallel = -2
            # feet_stumble = -2
            # feet_contact_safety = -2 

    class normalization:
        class obs_scales:
            lin_vel = 10.0
            ang_vel = 2.
            dof_pos = 1.0
            dof_vel = 0.05  #? 感觉有些太小
            height_measurements = 5.0
            quat = 1
            gravity = 1
            forces = 0.1
            forward_depth = 1.0

        clip_observations = 100.
        clip_actions = 20.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.01
            ang_vel = 0.1
            gravity = 0.01
            forward_depth = 0.0
        class forward_depth:
            # 基础噪声模拟配置（简化版本）
            stereo_min_distance = 0.0  # 不使用立体相机噪声
            stereo_far_distance = 0.0
            stereo_far_noise_std = 0.0
            stereo_near_noise_std = 0.0
            stereo_full_block_artifacts_prob = 0.0
            stereo_full_block_values = []
            stereo_full_block_height_mean_std = [0, 0]
            stereo_full_block_width_mean_std = [0, 0]
            stereo_half_block_spark_prob = 0.0
            stereo_half_block_value = 0.0
            sky_artifacts_prob = 0.0
            sky_artifacts_far_distance = 0.0
            sky_artifacts_values = []
            sky_artifacts_height_mean_std = [0, 0]
            sky_artifacts_width_mean_std = [0, 0]

class CowaCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_DH_Smooth_TS'   # OnPolicyRunnerEstimator

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.1
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        # short_history estimator
        estimator_hidden_dims=[128, 64]
        # long history type
        long_history_type = 'cnn' # 'cnn' or 'mlp'
        # for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   # long history output dim
        in_channels = CowaCfg.env.frame_stack
        # for long_history mlp only
        long_history_hidden_dims = [1024, 512, 256]

        encoder_class_name = "Conv2dHeadModel"
        encoder_output_dim = 128
        class encoder_kwargs:
            image_shape = [1, 48, 64]
            channels = [16, 32, 32]
            kernel_sizes = [5, 4, 3]
            strides = [2, 2, 1]
            hidden_sizes = [128]
            output_size = 128
            use_maxpool = True
            nonlinearity = "LeakyReLU"



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

        use_ppo=True

        # train net sin 需要注释
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1 #对于踏步运动，第一阶段0.05或者0.1均可，效果都还可以

        # use_flip = True
        # symmetry_scale = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_DH_Smooth_TS' #'ActorCritic_DH'
        algorithm_class_name = 'PPO_DH_Smooth_TS' #'PPO_DH'
        
        teacher_policy_class_name = 'ActorCritic_DH_Smooth' #'ActorCritic_DH'
        teacher_policy_checkpoint_path = 'cowa_10dof/exported/Mar06_14-20-54_cowa_10dof_stair_no_modified_baseheight_modify_reward/polices/Mar06_14-20-54_cowa_10dof_stair_no_modified_baseheight_modify_reward.pt' # 预训练教师模型的路径，如果不使用教师模型进行蒸馏，则保持为空字符串

        dagger_on = True

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