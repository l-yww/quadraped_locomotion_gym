from wheel_legged_gym.envs.quadruped.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_quadruped_joint_names, get_quadruped_default_joint_friction, get_quadruped_default_joint_damping, get_quadruped_default_joint_armature


class Cowa_Num:
    quad_index = '1'
    DOF = 12

class QuadWtwCfg(LeggedRobotCfg):
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

    class terrain(LeggedRobotCfg.terrain):
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
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [plane ,smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0]
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
            lin_vel_x = [-0.5, 1.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5] # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]   # min max [rad/s]
            heading = [-3.14, 3.14]
            body_height_cmd = [0.4, 0.4]
            gait_phase_cmd_range = [0.5, 0.5]   # trot: diagonal legs in phase
            gait_offset_cmd_range = [0.0, 0.0]   # trot
            gait_bound_cmd_range = [0.0, 0.0]    # trot
            gait_frequency_cmd_range = [2.0, 2.0]
            gait_duration_cmd_range = [0.5, 0.5]
            footswing_height_range = [0.3, 0.3]
            body_pitch_range = [-0, 0]
            body_roll_range = [-0.0, 0.0]

    class rewards:
        only_positive_rewards = False
        tracking_sigma_lin_vel = 20
        tracking_sigma_ang_vel = 20
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
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1
            #################    style rewards    ######################
            " 机体姿态 "
            base_height = -5 #-10
            orientation = -10 #-20

            tracking_contacts_shaped_force = -1.0 #3.0
            tracking_contacts_shaped_vel = -1.0 #3.0
            feet_clearance_cmd_linear = -0.5 #-2.0 #-5.0
            ############## normalized rewards #####################
            " 静止稳定 "
            stand_base_vel_penality = -1.0
            stand_all_feet_contact = 2.0

            dof_pos_symmetry= -1.0
            default_hip_pos= -0.5 #-1.0
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            dof_vel = -5e-5             
            dof_acc = -5e-7            
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
            feet_contact_forces= -0.01

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

class QuadWtwCfgPPO(LeggedRobotCfgPPO):
    seed = 10
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
        algorithm_class_name = 'PPO_SYM'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'quadruped_wtw'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt
