from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_joint_names, get_stiffness_damping,\
                                            get_default_joint_friction, get_default_joint_damping, get_default_joint_armature

class Cowa_Num:
    WLR_index = 'v2_10dof'
    DOF = 10

class CowaCfg(LeggedRobotCfg):
    class mode:
        use_net = True

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66        # long history的帧数
        short_frame_stack = 5   # short history的帧数
        c_frame_stack = 1       # 输入给critic的帧数
        num_est_prob = 3 + 1 + 2   # vel_xyz, height, feet_height预测的信息的总维度
        num_actions = Cowa_Num.DOF
        num_single_obs = 2 + 3 + 3*num_actions - 2 + 6 -1 # cmd + dof pos[w/o wheel] + dof vel + action + imu
        
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + num_est_prob + 1 + 77

        projected_gravity = False   # [True] projected_gravity; [False] Euler Angle

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
        terrain_proportions = [0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0.0, 0.0]
        slope_treshold = 0.1 # slopes above this threshold will be corrected to vertical surfaces

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = True
        rand_init_dof_range = 0.1 # [rad]
        joint_names = get_joint_names(Cowa_Num.DOF)
        default_joint_angles = {name: 0.0 for name in joint_names}

    class control(LeggedRobotCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness, damping = get_stiffness_damping(Cowa_Num.DOF)

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 100Hz
        pos_action_scale = 0.25
        vel_action_scale = 2.

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

        cycle_time = 0.8 #2 #3.0 # ref pos 的 cycle time

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 500 Hz

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = False #True  #第一阶段False，第二阶段True

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

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_x = [-0.6, 0.6]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            # height = [0.37, 0.37]
            height = [0.42, 0.42]
            # height = [0.45, 0.45]
            heading = [-3.14, 3.14]

    class rewards:
        only_positive_rewards = False 
        tracking_sigma_lin_vel = 20
        tracking_sigma_ang_vel = 20
        target_feet_height_min = 0.02 #0.02 #0.02
        target_feet_height_max = 0.20 #0.03
        max_contact_force = 500  
        soft_dof_pos_limit = (
            1  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 0.95
        soft_dof_vel_limits = 0.95
        clip_single_reward = 1
        min_wheel_dist = 0.637 #0.64
        max_wheel_dist = 0.643 #0.645
        min_feet_dist = 0.48
        max_feet_dist = 0.54

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
            joint_pos = 1.5 #3.0
            feet_air_time = 1.0 #2.0
            feet_contact_number = 1.0 #2.0
            feet_clearance = 2.0 #1.0 #2.0
            " 机体姿态 "
            base_height = -10 #-5 #-4 #-2 #-1
            orientation = -10 #-5 #-2
            " 脚部姿态 "
            # same_foot_x_position = -2 #-1
            # wheel_distance = 0.2
            feet_distance = 0.2
            default_hip_roll = -5 #-2 #-1
            # contact_no_vel = -0.5
            # default_joint_pos = -0.2 #-0.5
            # nominal_foot_position = 0.5
            ############## normalized rewards #####################
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            dof_vel = -5e-5             # 若有wheel需要注意
            dof_acc = -5e-7             # 若有wheel需要注意
            torques = -1e-6 #-1e-5
            wheel_acc = -5e-6
            wheel_vel = -0.1
            torque_limits = -0.05
            # dof_vel_limits = -0.05
            power = -2e-5
            action_rate = -0.1 #-0.05 #-0.01
            action_smoothness = -0.1 #-0.05 #-0.01
            # base_acc = -1e-2 #-1e-1
            collision = -2.0
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            stand_still = -5 #-1.0
            feet_contact_forces = -0.02 #-0.01
            # feet_pitch_level = -0.1 #-1
            # stand_base_vel_penality = -1 #-4.0
            # stand_wheel_vel_penality = -0.3

    class costs:
        class scales:
            feet_pitch_level = -0.1
            contact_no_vel = -0.5

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

        clip_observations = 100.
        clip_actions = 20.

class CowaCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_DH_Smooth_P3O' #'OnPolicyRunner_DH_Smooth'   # OnPolicyRunnerEstimator

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
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
        advantage_weight=0.5
        cost_advantage_weight=0.5

        # train net sin 需要注释
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1 #对于踏步运动，第一阶段0.05或者0.1均可，效果都还可以

        # p3o para
        cost_value_loss_coef = 1.0
        
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_DH_Smooth_P3O' #'ActorCritic_DH_Smooth' #'ActorCritic_DH'
        algorithm_class_name = 'PPO_DH_Smooth_P3O' #'PPO_DH_Smooth' #'PPO_DH'
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