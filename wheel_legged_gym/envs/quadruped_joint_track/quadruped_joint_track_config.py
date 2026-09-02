from wheel_legged_gym.envs.quadruped.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_quadruped_joint_names, get_quadruped_default_joint_friction, get_quadruped_default_joint_damping, get_quadruped_default_joint_armature


class Cowa_Num:
    """机器人常量（与 quadruped_wtw_him_arm_fix 保持一致）"""
    quad_index = '2'
    DOF = 12            # 4腿 × 3关节(hip/thigh/calf)


class QuadJointTrackCfg(LeggedRobotCfg):
    """悬空 sin 跟踪环境，用于隔离关节伺服子系统的 sim2real gap。

    设计：
    - fix_base_link=True，机器人悬空，去除接触/平衡/命令
    - obs = dof_pos(12) + dof_vel(12) + last_action(12) + sin(1) + cos(1) = 38
    - 参考轨迹：12 关节各自 offset_i + A_i*sin(2πf·t + φ_i)，频率统一
    - reward：唯一跟踪项 tracking_joint_pos（指数核）
    - 力矩链完整复用 wtw 的 _compute_torques（PD + 力矩-速度曲线 + 限幅），保证 sim/real 一致
    """

    class mode(LeggedRobotCfg.mode):
        use_net = True   # True=策略推理；False=直接把 ref_dof_pos 作位置目标下发 PD（开环验证用）

    # =========================================================================
    # 环境配置
    # =========================================================================
    class env(LeggedRobotCfg.env):
        num_actions = Cowa_Num.DOF        
        num_commands = 0                  
        projected_gravity = False         

        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_gait_commands = False
        timing_obs_dim = 0

        # 单帧 obs = dof_pos(12) + dof_vel(12) + last_action(12) + sin(1) + cos(1) = 38
        num_single_obs = 3 * num_actions + 2     # 38
        frame_stack = 5
        actor_input_stack = 5
        c_frame_stack = 1
        num_envs = 4096

        num_observations = int(frame_stack * num_single_obs)
        # 悬空下无需特权观测，critic 与 actor 同维
        single_num_privileged_obs = num_single_obs
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    # =========================================================================
    # sin 参考轨迹参数（compute_ref_state 使用）
    # =========================================================================
    class ref:
        cycle_time = 2.0                 # sin 周期 [s]，全局频率 f = 1/cycle_time

        # 各关节幅值 [rad]，形状相似、幅值不同
        amplitude = [
            0.2, 0.3, 0.3,
            0.2, 0.3, 0.3,
            0.2, 0.3, 0.3,
            0.2, 0.3, 0.3,
        ]
        # 各关节相位偏置 [rad]；trot 式对角分组：FL+RR 同相，FR+RL 反相
        # phase_offset = [
        #     0.0, 0.0, 0.0,
        #     3.14159265, 3.14159265, 3.14159265,
        #     3.14159265, 3.14159265, 3.14159265,
        #     0.0, 0.0, 0.0,
        # ]

        phase_offset = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]

        # 各关节直流偏置 [rad]（叠加在 default_dof_pos 之上）
        offset = [0.0] * 12
        # reset 时随机化初始相位，提升泛化、避免相位跳变 reward 抖
        random_init_phase = True


    class asset(LeggedRobotCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_quadruped_arm_v1/urdf/cowa_quadruped_arm_v1_fix_arm.urdf"
        name = "cowa_quadruped"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        fix_base_link = True              # ← 悬空核心
        fix_base_link_height = 0.3        # 抬高避免腿碰地

    # =========================================================================
    # 地形：悬空下用最简平面即可（不接触）
    # =========================================================================
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1
        num_cols = 1
        x_init_range = 0.
        y_init_range = 0.
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.5]            # 悬空高度（fix_base_link 会再加 fix_base_link_height）
        rot = [0.0, 0.0, 0.0, 1.0]
        joint_names = get_quadruped_joint_names()
        default_joint_angles = {name: 0.0 for name in joint_names}

    # =========================================================================
    # 控制：完整复用 wtw 的 PD + 力矩-速度曲线，保证 sim2real 力矩链一致
    # =========================================================================
    class control(LeggedRobotCfg.control):
        dof_velocity_limits = [20, 20, 10, 20, 20, 10, 20, 20, 10, 20, 20, 10]

        torque_vel_hip_indices = [0, 1, 3, 4, 6, 7, 9, 10]
        torque_vel_hip_max_vel = 20
        torque_vel_hip_max_torque = 200.0
        torque_vel_hip_vel_1 = 7.28

        torque_vel_calf_indices = [2, 5, 8, 11]
        torque_vel_calf_max_vel = 12
        torque_vel_calf_max_torque = 330.0
        torque_vel_calf_vel_1 = 6.6

        control_type = "P"
        stiffness = {'joint': 160.0}
        damping = {'joint': 5.0}
        # stiffness = {'joint': 120.0}
        # damping = {'joint': 3.5}
        # action_scale: target = default + action × action_scale。
        action_scale = 0.20
        decimation = 4                
        action_smoothness = False

    class sim(LeggedRobotCfg.sim):
        dt = 0.005                       

    # =========================================================================
    # 奖励：唯一跟踪项 + 极小正则
    # =========================================================================
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_joint_pos = 1.0     # 主跟踪项（单步上限 1）
            action_rate = -0.05         
            action_smoothness = -0.05 
            dof_vel_limits = -0.001      
            dof_pos_limits = -0.001     
        only_positive_rewards = True    
        # tracking 指数核容忍方差。
        # 0.01 太紧 → 误差带极窄 → 策略高频微调 action → PD(stiffness=160)把微调放大成
        # vel/torque 高频振荡 → 腿抖（pos 平滑 reward 抓不到，但 vel/torque 残差大）。
        # 放松到 0.05：策略不需高频微调，腿平滑。代价 tracking 0.95→0.85，但 0.95 是硬抠出的有害精度。
        tracking_sigma = 0.01
        soft_dof_pos_limit = 0.9       
        soft_dof_vel_limits = 0.9       


    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            dof_pos = 1.0
            dof_vel = 0.05
        clip_observations = 100.
        clip_actions = 20.


    class domain_rand(LeggedRobotCfg.domain_rand):
        # 与 quadruped_wtw_him_arm_fix 完全对齐，保证 sim/real 力矩链一致。
        # 悬空下 push/接触类随机化无物理意义，但保留开关与 wtw 一致（不影响悬空关节跟踪）。
        use_random = True

        # ---- 外力扰动（悬空无意义，关）----
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.1

        # ---- 刚体属性随机化 ----
        rand_interval_s = 10
        randomize_rigids_after_start = False

        randomize_friction = False
        friction_range = [0.5, 1.2]
        randomize_restitution = False
        restitution_range = [0, 0.3]

        # ---- 基座质量和惯量 ----
        randomize_base_mass = False
        added_mass_range = [-2, 2]
        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]

        # ---- 质心偏移 ----
        randomize_com_displacement = False
        com_displacement_range = [-0.02, 0.02]
        randomize_each_link = False
        link_com_displacement_range_factor = 0.02

        # ---- 电机力矩 ----
        randomize_motor_strength = use_random
        motor_strength_range = [0.8, 1.2]
        randomize_PD_factor = False
        Kp_factor_range = [0.9, 1.1]
        Kd_factor_range = [0.9, 1.1]

        # ---- 编码器偏置 ----
        randomize_motor_offset = use_random
        motor_offset_range = [-0.05, 0.05]
        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.03, 0.03]

        # ---- 延迟模拟 ----
        add_action_lag = use_random
        randomize_lag_timesteps = use_random
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [0, 1]

        add_dof_lag = use_random
        randomize_dof_lag_timesteps = use_random
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1]

        add_imu_lag = False
        randomize_imu_lag_timesteps = use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 1]

        # ---- 关节物理属性（与 wtw 一致，开 each_joint + per-joint range）----
        DOF = Cowa_Num.DOF
        default_joint_friction = get_quadruped_default_joint_friction(Cowa_Num.quad_index)
        randomize_joint_friction = use_random
        joint_friction_range = [0.9, 1.1]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.9, 1.1]

        default_joint_damping = get_quadruped_default_joint_damping(Cowa_Num.quad_index)
        randomize_joint_damping = use_random
        joint_damping_range = [0.9, 1.1]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.9, 1.1]

        default_joint_armature = get_quadruped_default_joint_armature(Cowa_Num.quad_index)
        randomize_joint_armature = use_random
        joint_armature_range = [0.9, 1.1]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.9, 1.1]


class QuadJointTrackCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner'   # 普通 PPO runner（非 HIM），避免 HIM 的 obs 布局硬编码
    policy_class_name = 'ActorCritic'
    algorithm_class_name = 'PPO'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3         # 跟踪任务不需大探索，0.5 太大导致误差降不下去
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001         # 跟踪任务目标明确，不需熵正则；0.01 会让 std 从 0.3 涨到 0.65+，
                                     # 带噪声 rollout reward 掉到 0.57（deterministic 仍 0.85）。压到 0.001 止血
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 5.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 1000
        save_interval = 10
        experiment_name = 'quadruped_joint_track'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
