from wheel_legged_gym.envs.quadwheel.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_quadwheel_joint_names, get_quadwheel_default_joint_friction, get_quadwheel_default_joint_damping, get_quadwheel_default_joint_armature


class Cowa_Num:
    quad_index = '1'
    DOF = 16

class QuadwheelCfg(LeggedRobotCfg):
    class mode:
        use_net = True

    class env(LeggedRobotCfg.env):
        # change the observation dim
        num_actions = Cowa_Num.DOF
        num_commands = 3
        projected_gravity = True # use projected_gravity or [roll, pitch]

        num_single_obs = num_commands + 3 * num_actions # cmd + dof pos + dof vel + action
        if projected_gravity:
            num_single_obs += 6
        else:
            num_single_obs += 5

        frame_stack = 1        # long history
        actor_input_stack = 1   # 输入给actor的
        c_frame_stack = 1
        num_envs = 4096
        
        num_observations = int(frame_stack * num_single_obs)
        observe_heights = False  # actor observe height_map
        if observe_heights:
            num_single_obs += 13 * 7
            num_observations = int(frame_stack * num_single_obs)

        single_num_privileged_obs = num_single_obs + 3 + 1 # vel + base_height

        priv_observe_friction = True
        if priv_observe_friction:
            single_num_privileged_obs += 1

        priv_observe_restitution = True
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

        priv_observe_heightmap = True
        if priv_observe_heightmap:
            single_num_privileged_obs += 13 * 7

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    class asset(LeggedRobotCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_quadwheel_v1/urdf/cowa_quadwheel.urdf"
        name = "cowa_quadruped"  # actor name
        foot_name = "wheel"
        wheel_name ="wheel"
        wheel_radius = 0.18 # m
        penalize_contacts_on = ["hipx", "hipy","knee","base"]
        terminate_after_contacts_on = []
        

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
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
        measure_heights = True #True
        measured_points_x = [
            -0.6,
            -0.45,
            -0.3,
            -0.15,
            0.0,
            0.15,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 12.
        terrain_width = 12.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [plane ,smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain type names for each column (0-9), matching make_terrain + terrain_proportions in base config
        # proportions: [0.2, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05, 0.1]
        # col 0: flat (0.2), col 1-4: slope/rough (0.2), col 5: stairs_up (0.2), col 6: stairs_down (0.2)
        # col 7: discrete_obstacles, col 8: sloped_obstacle, col 9: wave
        terrain_proportions = [0.2, 0.05, 0.05, 0.05, 0.05, 0.25, 0.25, 0.0, 0.0, 0.1]
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
        # terrain type names for each column (0-9), matching make_terrain + terrain_proportions in base config
        # proportions: [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0, 0, 0.1, 0.1]
        # col 0-1: flat pyramid_sloped, col 2-3: slope, col 4: pyramid_sloped_rough
        # col 5-6: stairs, col 7: discrete_obstacles, col 8: sloped_obstacle, col 9: wave
        terrain_type_names = [
            "pyramid_sloped_flat",  # col 0
            "pyramid_sloped_flat",  # col 1
            "pyramid_sloped",       # col 2
            "pyramid_sloped",       # col 3
            "pyramid_sloped_rough", # col 4
            "stairs_up",            # col 5
            "stairs_down",          # col 6
            "discrete_obstacles",   # col 7
            "sloped_obstacle",      # col 8
            "wave",                 # col 9
        ]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = True
        rand_init_dof_range = 0.3 # [rad]
        joint_names = get_quadwheel_joint_names()
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'fl_hipx_joint': 0.0,   # [rad]
            'fl_hipy_joint': -0.6,   # [rad]
            'fl_knee_joint': 1.0,  # [rad]
            'fl_wheel_joint': -0.0,   # [rad]

            'fr_hipx_joint': 0.0,     # [rad]
            'fr_hipy_joint': -0.6,   # [rad]
            'fr_knee_joint': 1.0,     # [rad]
            'fr_wheel_joint': 0.0,   # [rad]

            'hl_hipx_joint': 0.0,   # [rad]
            'hl_hipy_joint': 0.6,    # [rad]
            'hl_knee_joint': -1.0,  # [rad]
            'hl_wheel_joint': 0.0,    # [rad]

            'hr_hipx_joint': 0.0,   # [rad]
            'hr_hipy_joint': 0.6,    # [rad]
            'hr_knee_joint': -1.0,  # [rad]
            'hr_wheel_joint': 0.0,    # [rad]
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'hipx_joint': 80.,'hipy_joint': 80.,'knee_joint': 80.,'wheel_joint': 0.}  # [N*m/rad]
        damping = {'hipx_joint': 2.0,'hipy_joint': 2.0,'knee_joint': 2.0,'wheel_joint': 0.6}     # [N*m*s/rad]

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 50Hz
        # mixed control: leg joints use position control, wheel joints use velocity control
        pos_action_scale = 0.25  # leg position target = action * pos_action_scale
        vel_action_scale = 5.0   # wheel velocity target = action * vel_action_scale

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200 Hz

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = False

        push_robots = use_random
        push_interval_s = 8
        max_push_vel_xy = 0.2  # 0.2
        max_push_ang_vel = 0.1

        rand_interval_s = 10    # 每隔rand_interval_s会重置一次随机化
        randomize_rigids_after_start = False     #控制link的friction和restituion的随机化开关
        randomize_friction = use_random             # xxw True
        friction_range = [0.2, 2]
        
        randomize_restitution = use_random           
        restitution_range = [0, 1.0]           

        # --------------- 随机化 base_link 质量 & 转动惯量 ----------------- #
        randomize_base_mass = use_random #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-2, 2]                  # 加负载

        randomize_inertia = use_random    
        randomize_inertia_range = [0.8, 1.2]

        # --------------- 随机化 质心位置 ----------------- #
        randomize_com_displacement = use_random      
        com_displacement_range = [-0.05, 0.05]  # base link com的随机化范围
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
        randomize_default_dof_pos_range = [-0.03, 0.03]

        # ------------------- 延迟模拟 -------------------------- #
        '维护Tensor,随机范围延迟'
        '随机延迟中,timesteps是按照PD的频率'
        # action延迟
        add_action_lag = use_random
        randomize_lag_timesteps = use_random
        randomize_lag_timesteps_perstep = use_random
        lag_timesteps_range = [0, 4]       # 5ms * steps
        # 编码器延迟
        add_dof_lag = use_random
        randomize_dof_lag_timesteps = use_random # [1]False，固定取延迟的最大值; [2]True,随机范围
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1]    # 5ms * steps
        # IMU延迟
        add_imu_lag = use_random
        randomize_imu_lag_timesteps = use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 1]    # 5ms * steps

        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #       
        DOF = Cowa_Num.DOF

        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        default_joint_friction = get_quadwheel_default_joint_friction(Cowa_Num.quad_index)
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.5, 1.5]

        default_joint_damping = get_quadwheel_default_joint_damping(Cowa_Num.quad_index)
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.5, 1.5]

        default_joint_armature = get_quadwheel_default_joint_armature(Cowa_Num.quad_index)
        randomize_joint_armature = use_random 
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.5, 1.5]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.2
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        class ranges:
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class rewards:
        only_positive_rewards = False
        base_height_target = 0.52
        soft_dof_pos_limit = (
            0.9  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 0.9
        soft_dof_vel_limits = 0.95
        tracking_sigma = 0.25

        class scales:
            # normal
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            # TIN
            # tracking_lin_vel_x = 0.75
            # tracking_lin_vel_y = 0.75
            # tracking_ang_vel = 0.75

            lin_vel_z = -1
            ang_vel_xy = -0.05  # 惩罚机器人在X轴和Y轴上的角速度 对应现象为遏制机器人左右晃动和前后晃动
            orientation = -0.5  # 强烈鼓励机器人与初始姿态的基座方向一致
            # wheel_spin = -0.1   # 抑制空转
            dof_acc = -2.5e-7
            joint_power = -2.0e-5
            joint_reference = -0.03
            action_rate = -0.01
            action_smoothness = -0.01
            

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1
            gravity = 1
            forces = 0.1
            body_height_cmd = 1.0
            gait_freq_cmd = 1.0
            gait_phase_cmd = 1.0
            footswing_height_cmd = 1.0
            body_pitch_cmd = 1.0
            body_roll_cmd = 1.0
            gait_duration_cmd = 1.0

        clip_observations = 100.
        clip_actions = 20.

class QuadwheelCfgPPO(LeggedRobotCfgPPO):
    # MoE CTS 
    # seed = 10
    # runner_class_name = 'OnPolicyRunner_MoE_CTS'

    # class policy(LeggedRobotCfgPPO.policy):
    #     init_noise_std = 1.0
    #     actor_hidden_dims = [256, 128, 64]
    #     critic_hidden_dims = [256, 128, 64]

    #     # MoE_CTS: 编码器和 MoE 相关参数
    #     latent_dim = 32                         # teacher/student 编码器输出的特征维度
    #     norm_type = 'l2norm'                     # 特征归一化方式
    #     expert_num = 8                           # MoE 专家数量
    #     teacher_encoder_hidden_dims = [512, 256] # teacher 编码器 MLP
    #     student_encoder_hidden_dims = [512, 256, 256] # student 编码器 MLP

    # class algorithm(LeggedRobotCfgPPO.algorithm):
    #     value_loss_coef = 1.0
    #     use_clipped_value_loss = True
    #     clip_param = 0.2
    #     entropy_coef = 0.01
    #     num_learning_epochs = 5
    #     num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    #     learning_rate = 5.0e-4  # 5.e-4
    #     schedule = "adaptive"  # could be adaptive, fixed
    #     gamma = 0.99
    #     lam = 0.95
    #     desired_kl = 0.01
    #     max_grad_norm = 1.0

    #     # MoE_CTS: 训练过程相关
    #     teacher_env_ratio = 0.75               # 75% 环境用 privileged obs，25% 用 history
    #     student_encoder_learning_rate = 1e-3   # student 编码器学习率
    #     load_balance_coef = 0.01               # MoE 负载均衡系数

    # # MoE_CTS: student 编码器需要的历史窗口长度
    # history_length = 5

    # class runner(LeggedRobotCfgPPO.runner):
    #     policy_class_name = 'ActorCriticMoECTS'
    #     algorithm_class_name = 'MoECTS'
    #     num_steps_per_env = 24  # per iteration
    #     max_iterations = 1000001  # number of policy updates        #  xxw

    #     # logging
    #     save_interval = 100  # Please check for potential savings every `save_interval` iterations.
    #     experiment_name = 'quadwheel'
    #     run_name = ''
    #     # Load and resume
    #     resume = False
    #     load_run = -1  # -1 = last run
    #     checkpoint = -1 # -1 = last saved model
    #     resume_path = '/home/cowa'  # updated from load_run and chkpt

    runner_class_name = 'OnPolicyRunner'   # OnPolicyRunnerEstimator
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

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

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 500  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa'
        run_name = ''
        # Load and resume
        resume = False
        load_run = ""  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt