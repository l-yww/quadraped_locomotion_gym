# from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.envs.quadruped.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_quadruped_joint_names, get_quadruped_default_joint_friction, get_quadruped_default_joint_damping, get_quadruped_default_joint_armature

class Cowa_Num:
    quad_index = '1'
    DOF = 12

class QuadCfg_HIM(LeggedRobotCfg):
    class mode:
        use_net = True


    class env(LeggedRobotCfg.env):
        # change the observation dim
        num_actions = Cowa_Num.DOF
        num_commands = 3
        fail_to_terminal_time_s = 0.2  
        projected_gravity = True # use projected_gravity or [roll, pitch]
        # ===== [步态时钟] timing 信号开关（对齐 wtw_him_arm_fix） =====
        # 恢复时钟: 设 enable_gait_clock=True 并把两个 shaped 接触奖励权重改回 -2.0。
        enable_gait_clock = False
        observe_timing_parameter = enable_gait_clock    # 观测加 gait_index（1 维）
        observe_clock_inputs = enable_gait_clock        # 观测加 clock_inputs（4 维 sin 相位）
        timing_obs_dim = 0
        if observe_timing_parameter:
            timing_obs_dim += 1
        if observe_clock_inputs:
            timing_obs_dim += 4

        num_single_obs = num_commands + 3 * num_actions # cmd + dof pos + dof vel + action
        if projected_gravity:
            num_single_obs += 6
        else:
            num_single_obs += 5
        num_single_obs += timing_obs_dim   # +5（gait_index + clock_inputs）

        frame_stack = 30        # long history
        short_frame_stack = 10  # HIM(ActorCritic_HIM)需要：actor & estimator 的短历史帧数
        actor_input_stack = 10  # 输入给actor的
        c_frame_stack = 1
        num_envs = 4096

        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + 3 + 1 # vel + base_height

        priv_observe_friction = True
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

        priv_observe_heightmap = True   # 开启高程图特权观测(爬楼梯时 critic/estimator 可见地形)
        if priv_observe_heightmap:
            single_num_privileged_obs += 17 * 11

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        
    class asset(LeggedRobotCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_quadruped_w_arm/urdf/cowa_quadruped_arm.urdf"
        name = "cowa_quadruped"  # actor name
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter


    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'              # 地形类型：none/plane/heightfield/trimesh
        en_fix_step_height = False         # 固定台阶高度（调试用）
        en_fix_slope = False               # 固定坡度（调试用）
        track_test = False                 # 测试柏林噪声
        add_perlin_noise = False           # 添加柏林噪声粗糙度
        horizontal_scale = 0.1             # 水平分辨率 [m]
        vertical_scale = 0.005             # 垂直分辨率 [m]
        border_size = 25                   # 边界大小 [m]
        curriculum = True                  # 地形课程学习（难度逐步增加）
        static_friction = 1.0              # 静摩擦系数
        dynamic_friction = 1.0             # 动摩擦系数
        restitution = 0.                   # 恢复系数（弹性）

        # 高度测量（用于特权观测和奖励）
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False                   # 单选地形类型
        terrain_kwargs = None              # 地形参数
        max_init_terrain_level = 0         # 初始课程等级 
        terrain_length = 8.0               # 单块地形长度 [m]
        terrain_width = 8.0                # 单块地形宽度 [m]
        num_rows = 10                      # 地形行数（难度等级）
        num_cols = 10                      # 地形列数（类型数）
        step_height_max = 0.25             # 楼梯最大台阶高度 25cm
        slope_max = 0.6415                 # 斜坡坡度系数，difficulty=1 时 slope_max=tan(30°)
        rough_height_min = 0.01            # 坑洼起伏下限 [m]（difficulty=0）
        rough_height_max = 0.10            # 坑洼起伏上限 [m]（difficulty=1）
        rough_slope_scale = 0.5            # 坑洼地形基底坡度 = slope*this（0=纯平地+坑洼）
        pit_depth_min = 0.01               # 凹坑深度下限 [m]
        pit_depth_max = 0.10               # 凹坑深度上限 [m]

        # 地形类型比例：[平地, 平滑坡, 粗糙坡, 下楼梯, 上楼梯, 离散, 梅花桩, 沟槽, 斜面障碍, 波浪]
        terrain_proportions = [0.1, 0.2, 0.15, 0.2, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0]   

        # Trimesh 参数
        slope_treshold = 0.75              # 超过此阈值的斜坡修正为垂直面
        timeout_at_border = False          # 出界不重置

        # 初始化范围
        x_init_range = 1.0                  # X 初始化范围 [m]
        y_init_range = 1.0                  # Y 初始化范围 [m]
        yaw_init_range = 0.0                # 初始偏航角范围 [rad]
        x_init_offset = 0.0                 # X 偏移
        y_init_offset = 0.0                 # Y 偏移
        teleport_robots = True              # 超出范围传送到中心
        teleport_thresh = 2.0               # 传送阈值 [m]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]      # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]
        rand_init_dof = True
        rand_init_dof_range = 0.20  # [rad]
        joint_names = get_quadruped_joint_names()
        default_joint_angles = {name: 0.0 for name in joint_names}
        # default_joint_angles['FL_hip_joint'] = -0.04   # 主动内八偏置
        # default_joint_angles['FR_hip_joint'] =  0.04
        # default_joint_angles['RL_hip_joint'] = -0.04
        # default_joint_angles['RR_hip_joint'] =  0.04

    class control(LeggedRobotCfg.control):

        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint': 160.0}  # [N*m/rad]
        # stiffness = {'joint': 120.0}  # [N*m/rad]
        # stiffness = {'joint': 130.0}  # [N*m/rad]
        # stiffness = {'joint': 140.0}  # [N*m/rad]
        damping = {'joint': 5.0}     # [N*m*s/rad]
        # damping = {'joint': 4.5}     # [N*m*s/rad]
        decimation = 4  # 50Hz
        action_scale = 0.25 
        action_smoothness = False
        ratio = 0.9

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200Hz

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = True

        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.2  
        max_push_ang_vel = 0.1

        rand_interval_s = 10                     # 每隔rand_interval_s 重置一次随机化
        randomize_rigids_after_start = False     # 控制link的friction和restituion的随机化开关

        randomize_friction = use_random          # 地面摩擦
        friction_range = [0.5, 1.0]
        
        randomize_restitution = use_random       # 随机化恢复系数 
        restitution_range = [0, 0.3]             # 1.0 = 完全弹性碰撞

        randomize_base_mass = False              # 附加质量
        added_mass_range = [-2, 2]               # 加负载

        randomize_inertia = False                # 惯量	
        randomize_inertia_range = [0.8, 1.2]

        # --------------- 随机化 质心位置 ----------------- #
        randomize_com_displacement = False      
        com_displacement_range = [-0.02, 0.03]      # base link com的随机化范围
        randomize_each_link = False
        link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)
        
        # --------------- 随机化电机能力 ----------------- #
        # 对齐 cowa: 保留 motor_strength / PD_factor 随机化
        randomize_motor_strength = False
        # motor_strength_range = [0.8, 1.2]      # 对齐 cowa base [0.9,1.1]
        motor_strength_hip_range = [0.8, 1.0]     # hip 电机
        motor_strength_thigh_range = [0.8, 1.0]   # thigh 电机
        motor_strength_calf_range = [0.7, 1.0]     # calf 电机

        randomize_PD_factor = False
        Kp_factor_range = [0.9, 1.1]            # 对齐 cowa base [0.9,1.1] 附近
        Kd_factor_range = [0.9, 1.1]

        # --------------- randomize_motor_offset与randomize_default_dof_pos含义相同，均模拟关节角度的固定误差 ----------------- #
        randomize_motor_offset = False   #编码器零点误差
        motor_offset_range = [-0.05, 0.05] 
        randomize_default_dof_pos = False 
        randomize_default_dof_pos_range = [-0.01, 0.01]

        # ------------------- 延迟模拟 (每步加[0,1]随机延迟)-------------------------- #
        # action延迟
        add_action_lag = False
        randomize_lag_timesteps = False
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [0, 1]        # 5ms * steps

        # 编码器延迟
        add_dof_lag = False
        randomize_dof_lag_timesteps = False
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1]    # 5ms * steps

        # IMU延迟
        add_imu_lag = False
        randomize_imu_lag_timesteps = False
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 1]    # 5ms * steps

        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #
        # 对齐 cowa: cowa 无 joint_friction/damping/armature 随机化，全部关闭
        DOF = Cowa_Num.DOF

        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        default_joint_friction = get_quadruped_default_joint_friction(Cowa_Num.quad_index)
        randomize_joint_friction = False
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = False
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.8, 1.2]

        default_joint_damping = get_quadruped_default_joint_damping(Cowa_Num.quad_index)
        randomize_joint_damping = False
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = False
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.8, 1.2]

        default_joint_armature = get_quadruped_default_joint_armature(Cowa_Num.quad_index)
        randomize_joint_armature = False
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = False
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.8, 1.2]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 3
        resampling_time = 5.0    # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:

            lin_vel_x = [-1.0, 1.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            height = [0.395, 0.395]
            heading = [-3.14, 3.14]


    # ===== [步态时钟] =====
    class gait:
        adaptive_freq = False           # 速度-步频自适应 (新地形 25~35cm 进深更窄, 需要慢频抬腿)
        gait_freq = 1.2                 # 固定步频 Hz (adaptive_freq=False 时用)
        freq_min = 0.8                  # 速度自适应下限 (Hz), 防止过慢抖动
        freq_max = 2.0                  # 速度自适应上限 (Hz), 平地最大步频
        step_stride = 0.30              # 速度自适应目标跨距 (m), ≈训练台阶中位进深
        # 步态参数 (固定/自适应模式都用)
        gait_phase = 0.5                # trot 对角同相
        gait_offset = 0.0
        gait_bound = 0.0
        gait_duration = 0.5             # 占空比 50%

    class rewards:
        only_positive_rewards = False
        # tracking_sigma_lin_vel = 20
        # tracking_sigma_ang_vel = 20
        # tracking_sigma_lin_vel = 10
        # tracking_sigma_ang_vel = 10
        tracking_sigma = 0.25
        base_height_target = 0.395
        clearance_height_target = -0.20
        kappa_gait_probs = 0.07   # von mises 步态接触平滑参数
        max_contact_force = 100  
        soft_dof_pos_limit = (
            1.0  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 1.0
        soft_dof_vel_limits = 1.0

        class scales:

            #################   tracking rewards  ##################
            " 速度跟踪 "
            tracking_lin_vel = 2.8
            tracking_ang_vel = 1.5
            " 爬楼梯 "
            foot_stumble = -1.0
            foot_slip = -1.0

            #################    style rewards    ######################
            " 机体姿态 "
            base_height = -1.5 
            orientation = -1.5
            stand_still = -1.5
            default_hip_pos = -1.0
            diagonal_symmetry = 0.0

            ############## normalized rewards #####################
            lin_vel_z = -0.01
            ang_vel_xy = -0.01
            base_acc = -0.01
            dof_vel = -0.0        
            dof_acc = -0.0          
            torques = -0.0         
            power = -0.0           
            action_rate = -0.03
            action_smoothness = -0.03
            torque_limits  = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            collision = -1.0

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

        clip_observations = 100.0
        clip_actions = 20.0

class QuadCfgPPO_HIM(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_HIM'   # OnPolicyRunnerEstimator

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        # actor_hidden_dims = [256, 128, 64]
        # critic_hidden_dims = [256, 128, 64]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        history_len = QuadCfg_HIM.env.frame_stack # CNN 时间轴长度(=历史帧数=30)，必须 = env.frame_stack；传给 ActorCritic_HIM → Estimator_HIM.history_len


    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4    # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.0e-4  # 5.e-4
        schedule = "adaptive"   # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        symmetry_scale = 0.0  # 开启对称性 loss（flip_actor_obs/flip_critic_obs/flip_actions 已按 num_single_obs 自适应：45=quadruped，59=wtw）

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_HIM'
        algorithm_class_name = 'PPO_HIM'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1000001  # number of policy updates        

        # logging
        save_interval = 500  
        experiment_name = 'quadruped_arm_him'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  
        checkpoint = -1
        resume_path = '/home/cowa'  