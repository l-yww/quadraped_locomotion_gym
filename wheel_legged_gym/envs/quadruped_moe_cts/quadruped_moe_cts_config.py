from wheel_legged_gym.envs.quadruped.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_quadruped_joint_names, get_quadruped_default_joint_friction, get_quadruped_default_joint_damping, get_quadruped_default_joint_armature


class Cowa_Num:
    quad_index = '1'
    DOF = 12

class QuadMoECTSCfg(LeggedRobotCfg):
    class mode:
        use_net = True

    class env(LeggedRobotCfg.env):
        # change the observation dim
        num_actions = Cowa_Num.DOF
        num_commands = 12
        projected_gravity = True # use projected_gravity or [roll, pitch]

        observe_timing_parameter = True  # add gait_indices to obs
        observe_clock_inputs = True       # add clock_inputs (4 sin waves) to obs
        timing_obs_dim = 0
        if observe_timing_parameter:
            timing_obs_dim += 1
        if observe_clock_inputs:
            timing_obs_dim += 4

        num_single_obs = num_commands + 3 * num_actions + timing_obs_dim # cmd + dof pos + dof vel + action + timing
        if projected_gravity:
            num_single_obs += 6
        else:
            num_single_obs += 5

        frame_stack = 5        # long history
        actor_input_stack = 5   # 输入给actor的
        c_frame_stack = 1
        num_envs = 4096
        
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + 3 + 1 # vel + base_height

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

        priv_observe_heightmap = False
        if priv_observe_heightmap:
            single_num_privileged_obs += 7 * 11

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        observe_gait_commands = True
        
    class asset(LeggedRobotCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_quadruped_v1/urdf/cowa_quadruped.urdf"
        name = "cowa_quadruped"  # actor name
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        joint_friction = [
            0.0054971277713775635, 6.631016731262207e-06, 0.05982851982116699,
            0.007290467619895935, 0.015472427010536194, 0.07318446040153503,
            0.004257142543792725, 0.006278917193412781, 0.04780678451061249,
            0.005522802472114563, 0.0010340213775634766, 0.02277398109436035,
        ]
        joint_damping = [
            1.2516975402832031e-05, 4.172325134277344e-06, 5.364418029785156e-06,
            3.457069396972656e-05, 7.748603820800781e-06, 1.7881393432617188e-06,
            4.172325134277344e-06, 7.152557373046875e-06, 3.5762786865234375e-06,
            5.364418029785156e-06, 2.384185791015625e-06, 2.384185791015625e-06,
        ]
        joint_armature = [
            0.004417330026626587, 2.086162567138672e-07, 0.07663074135780334,
            1.4007091522216797e-06, 3.2782554626464844e-07, 0.07500061392784119,
            4.559755325317383e-06, 1.4007091522216797e-06, 0.07970243692398071,
            7.450580596923828e-07, 3.5762786865234375e-07, 0.05527627468109131,
        ]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
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
            "stairs_up",       # col 5
            "stairs_down",       # col 6
            "discrete_obstacles",   # col 7
            "sloped_obstacle",      # col 8
            "wave",                 # col 9
        ]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = True
        rand_init_dof_range = 0.3 # [rad]
        joint_names = get_quadruped_joint_names()
        default_joint_angles = {name: 0.0 for name in joint_names}

    class control(LeggedRobotCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint': 160.0}  # [N*m/rad]
        damping = {'joint': 5.0}     # [N*m*s/rad]

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 50Hz
        action_scale = 0.25

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200 Hz

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = True

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
        added_mass_range = [-2, 5]                  # 加负载

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
        add_action_lag = True and use_random
        randomize_lag_timesteps = True and use_random
        randomize_lag_timesteps_perstep = True and use_random
        lag_timesteps_range = [0, 4]       # 5ms * steps
        # 编码器延迟
        add_dof_lag = True and use_random
        randomize_dof_lag_timesteps = True and use_random # [1]False，固定取延迟的最大值; [2]True,随机范围
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1]    # 5ms * steps
        # IMU延迟
        add_imu_lag = True and use_random
        randomize_imu_lag_timesteps = True and use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 1]    # 5ms * steps

        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #       
        DOF = Cowa_Num.DOF

        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        default_joint_friction = get_quadruped_default_joint_friction(Cowa_Num.quad_index)
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.5, 1.5]

        default_joint_damping = get_quadruped_default_joint_damping(Cowa_Num.quad_index)
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.5, 1.5]

        default_joint_armature = get_quadruped_default_joint_armature(Cowa_Num.quad_index)
        randomize_joint_armature = use_random 
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.5, 1.5]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1
        num_commands = 12
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.5, 1.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-1, 1]   # min max [rad/s]
            heading = [-3.14, 3.14]
            body_height_cmd = [0.39, 0.39]

            limit_vel_x = [-5.0, 5.0]
            limit_vel_y = [-0.6, 0.6]
            limit_vel_yaw = [-5.0, 5.0]
            limit_body_height = [-0.01, 0.01]
            limit_gait_phase = [0.0, 1.0]
            limit_gait_offset = [0.0, 1.0]
            limit_gait_bound = [0.0, 1.0]
            limit_gait_frequency = [2.0, 4.0]
            limit_gait_duration = [0.5, 0.5]
            limit_footswing_height = [0.03, 0.35]
            limit_body_pitch = [-0., 0.]
            limit_body_roll = [-0.0, 0.0]
            limit_aux_reward_coef = [0.0, 0.01]
            limit_compliance = [0.0, 1.0]
            limit_stance_width = [0.20, 0.55]
            limit_stance_length = [0.45, 0.65]

            gait_phase_cmd_range = [0.5, 0.5]   # trot: diagonal legs in phase
            gait_offset_cmd_range = [0.0, 0.0]   # trot
            gait_bound_cmd_range = [0.0, 0.0]    # trot
            gait_frequency_cmd_range = [1.0, 4.0]
            gait_duration_cmd_range = [0.35, 0.65]
            footswing_height_range = [0.03, 0.35]
            body_pitch_range = [-0.5, 0.5]
            body_roll_range = [-0.0, 0.0]
            stance_width_range = [0.30, 0.35]
            stance_length_range = [0.34, 0.38]

    class rewards:
        only_positive_rewards = False
        tracking_sigma_lin_vel = 20
        tracking_sigma_ang_vel = 20
        base_height_target = 0.39
        max_contact_force = 600  
        soft_dof_pos_limit = (
            0.9  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 0.8
        soft_dof_vel_limits = 0.9
        kappa_gait_probs = 0.07
        tracking_sigma = 0.25
        

        class scales:
            ################# termination rewards ##################
            # termination = -1.
            # keep_balance = 1.0 
            #################   tracking rewards  ##################
            " 速度跟踪 "
            tracking_lin_vel = 5
            tracking_ang_vel = 5
            #################    style rewards    ######################
            " 机体姿态 "
            base_height = -5.0
            orientation = -5
            # orientation_control = 3.0

            tracking_contacts_shaped_force = 3.0
            # tracking_contacts_shaped_force_both = 3.0

            tracking_contacts_shaped_vel = 3.0
            feet_clearance_cmd_linear = -1.0
            ############## normalized rewards #####################
            " 静止稳定 "
            stand_base_vel_penality = -1.0
            stand_stability = -1.0
            stand_all_feet_contact = 2.0
            dof_pos_symmetry=-1
            default_hip_pos=-1.0
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            dof_vel = -5e-5             # 若有wheel需要注意
            dof_acc = -5e-7             # 若有wheel需要注意
            torques = -1e-5
            torque_limits = -0.05
            # dof_vel_limits = -0.05
            power = -2e-5
            action_rate = -0.01
            action_smoothness = -0.01
            base_acc = -1e-2
            collision = -2.0
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            feet_contact_forces=-0.01

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

class QuadMoECTSCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_MoE_CTS'   # OnPolicyRunnerEstimator

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

        # MoE_CTS: 编码器和 MoE 相关参数
        latent_dim = 32                         # teacher/student 编码器输出的特征维度
        norm_type = 'l2norm'                     # 特征归一化方式，l2norm 或 simnorm
        expert_num = 8                           # MoE 专家数量，越多表达能力越强但计算更慢
        teacher_encoder_hidden_dims = [512, 256] # teacher 编码器的 MLP 结构（看 privileged obs）
        student_encoder_hidden_dims = [512, 256, 256] # student 编码器的 MLP 结构（看 proprioception history）

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

        # MoE_CTS: 训练过程相关
        teacher_env_ratio = 0.75               # 75% 环境用 privileged obs（teacher），25% 用 history（student）
        student_encoder_learning_rate = 1e-3   # student 编码器学习率，比主网络高一点让它更快模仿 teacher
        load_balance_coef = 0.01               # MoE 负载均衡系数，防止 gating 只激活少数专家

    # MoE_CTS: student 编码器需要的历史窗口长度，和 frame_stack 一致
    history_length = 5

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCriticMoECTS'
        algorithm_class_name = 'MoECTS'
        num_steps_per_env = 48  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'quadruped_moe_cts'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt