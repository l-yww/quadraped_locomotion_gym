from wheel_legged_gym.envs.quadruped.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_quadruped_joint_names, get_quadruped_default_joint_friction, get_quadruped_default_joint_damping, get_quadruped_default_joint_armature


class Cowa_Num:
    """机器人常量"""
    quad_index = '2'   # 四足型号索引
    DOF = 12            # 自由度：4腿 × 3关节(hip/thigh/calf)


class QuadWtwCfg_HIM(LeggedRobotCfg):
    """
    WTW 框架环境配置
    =================
    继承自 LeggedRobotCfg，覆盖观测、命令、步态、斜坡地形等参数。
    """

    class mode:
        """运行模式：策略推理 vs 跟随参考轨迹"""
        use_net = True  # True=策略推理, False=跟随ref轨迹

    # =========================================================================
    # 环境配置
    # =========================================================================
    class env(LeggedRobotCfg.env):
        # ---------- 观测维度 ----------
        short_frame_stack = 5  #10            # short history的帧数  actor & est
        num_est_prob = 4                  # HIM estimator: [base_height, vx, vy, vz]
        num_actions = Cowa_Num.DOF        # 动作维度 = 12
        num_commands = 12                 # 命令维度（含步态参数）
        projected_gravity = True          # True=投影重力[3维], False=欧拉角[2维]

        # 步态时序信号
        observe_timing_parameter = True   # 观测中加入 gait_index（1 维）
        observe_clock_inputs = True       # 观测中加入 clock_inputs（4 维 sin 信号）
        timing_obs_dim = 0
        if observe_timing_parameter:
            timing_obs_dim += 1           # gait_index
        if observe_clock_inputs:
            timing_obs_dim += 4           # 4 条腿的 sin 相位

        # 本体单帧观测维度 = 12命令 + 3×12关节 + 5/6(IMU) + 5(时序)
        num_proprio_obs = num_commands + 3 * num_actions + timing_obs_dim  # = 12 + 36 + 5 = 53
        if projected_gravity:
            num_proprio_obs += 6          # ang_vel[3] + proj_gravity[3] = 6
        else:
            num_proprio_obs += 5          # ang_vel[3] + euler[2] = 5

        actor_observe_heightmap = True  # Actor observes the 6 x 9 height scan
        num_height_scan_points = 6 * 9
        num_height_scan_input = (
            num_height_scan_points if actor_observe_heightmap else 0
        )
        num_single_obs = num_proprio_obs + num_height_scan_input

        # 帧堆叠
        frame_stack = short_frame_stack    # 仅保留 5 帧短历史，供 Actor 与状态估计器使用
        c_frame_stack = 1                  # 输入给 Critic 的帧数
        num_envs = 4096                    # 并行环境数

        num_observations = int(frame_stack * num_single_obs)
        actor_obs_dim = num_observations

        # 特权观测（Critic 可见，Actor 不可见，用于非对称训练）
        single_num_privileged_obs = num_proprio_obs + 3 + 1  # + 线速度[3] + 体高[1]

        # ---------- 特权观测开关 ----------
        priv_observe_friction = True     # 地面摩擦系数  False
        if priv_observe_friction:
            single_num_privileged_obs += 1

        priv_observe_restitution = False  # 地面恢复系数
        if priv_observe_restitution:
            single_num_privileged_obs += 1

        priv_observe_payloads = True      # 负载质量
        if priv_observe_payloads:
            single_num_privileged_obs += 1

        priv_observe_inertia = True       # 转动惯量缩放
        if priv_observe_inertia:
            single_num_privileged_obs += 1

        priv_observe_motor_strength = True  # 电机力矩缩放（12 维）
        if priv_observe_motor_strength:
            single_num_privileged_obs += num_actions

        priv_observe_motor_offset = True  # 编码器偏置（12 维）
        if priv_observe_motor_offset:
            single_num_privileged_obs += num_actions

        priv_observe_com_displacement = True  # 质心偏移（3 维）
        if priv_observe_com_displacement:
            single_num_privileged_obs += 3

        priv_observe_heightmap = True # Critic observes the clean height scan
        if priv_observe_heightmap:
            single_num_privileged_obs += num_height_scan_points

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        observe_gait_commands = True      # 启用步态命令观测

    # =========================================================================
    # 机器人模型配置
    # =========================================================================
    class asset(LeggedRobotCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_quadruped_arm_v1/urdf/cowa_quadruped_arm_v1_fix_arm.urdf"
        #file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_quadruped_arm_v1/urdf/cowa_quadruped_arm_v1_fix_arm2.urdf"
        name = "cowa_quadruped"              # Actor 名称
        foot_name = "foot"                    # 足端 link 名称前缀
        foot_radius = 0.045                    # URDF 足端球碰撞半径 [m]
        penalize_contacts_on = ["thigh", "calf", "base_link"]   # 碰撞惩罚部位
        terminate_after_contacts_on = ["base_link"]              # 碰撞终止部位

    # =========================================================================
    # 地形配置
    # =========================================================================
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
        measured_points_x = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        # True: 四足足底局部支撑面；False: 原始高程图采样点均值。
        use_support_plane_base_height = False #True
        support_plane_max_slope = 2.0  # local support-plane slope limit (about 63 degrees)
        selected = False                   # 单选地形类型
        terrain_kwargs = None              # 地形参数 
        max_init_terrain_level = 1         # 初始课程等级 5
        terrain_kwargs = None              # 地形参数 
        max_init_terrain_level = 1         # 初始课程等级 5
        terrain_length = 8.                # 单块地形长度 [m]
        terrain_width = 8.                 # 单块地形宽度 [m]
        num_rows = 20 #10                      # 地形行数（难度等级）
        num_cols = 10                      # 地形列数（类型数）
        step_height_max = 0.20   # 先易后难,默认 0.275 偏大
        # 地形类型比例：[平面, 平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散]
        # terrain_proportions = [0.3, 0.35, 0.35, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0]
        terrain_proportions = [0.1, 0.1, 0.1, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]

        # Trimesh 参数
        slope_treshold = 0.75              # 超过此阈值的斜坡修正为垂直面
        timeout_at_border = False          # 出界不重置

        # 初始化范围
        x_init_range = 1.                  # X 初始化范围 [m]
        y_init_range = 1.                  # Y 初始化范围 [m]
        yaw_init_range = 0.                # 初始偏航角范围 [rad]
        x_init_offset = 0.                 # X 偏移
        y_init_offset = 0.                 # Y 偏移
        teleport_robots = True             # 超出范围传送到中心
        teleport_thresh = 2.0              # 传送阈值 [m]

    # =========================================================================
    # 初始状态
    # =========================================================================
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]              # 初始位置 x,y,z [m]（z=0.4 离地高度）
        rot = [0.0, 0.0, 0.0, 1.0]         # 初始旋转 x,y,z,w [四元数]（水平）
        lin_vel = [0.0, 0.0, 0.0]          # 初始线速度 [m/s]
        ang_vel = [0.0, 0.0, 0.0]          # 初始角速度 [rad/s]
        rand_init_dof = True                # 初始化时随机关节角度
        rand_init_dof_range = 0.25           # 随机范围 ±0.3 [rad]
        joint_names = get_quadruped_joint_names()
        # 默认关节角度全为 0（sim2sim 部署时 default_angles 也必须全 0）
        default_joint_angles = {name: 0.0 for name in joint_names}

    # =========================================================================
    # 控制配置
    # =========================================================================
    class control(LeggedRobotCfg.control):
        control_type = "P"                  # 控制模式：P=位置, V=速度, T=力矩
        # PD 控制器参数（Isaac Gym 内部 PD）
        stiffness = {'joint': 130.0}        # 刚度 [N·m/rad]
        damping = {'joint': 4.5}            # 阻尼 [N·m·s/rad]  5
        # 控制频率：decimation=4, dt=0.005 → 策略频率 50Hz, 仿真频率 200Hz
        decimation = 4
        action_scale = 0.25                 # 动作缩放：target_q = default_q + action * 0.25
        # 转速-扭矩包络使用的关节分组
        torque_vel_hip_indices = [0, 1, 3, 4, 6, 7, 9, 10]
        torque_vel_calf_indices = [2, 5, 8, 11]
        # 在 motor_strength 效率缩放之前执行的独立硬限幅
        enable_pre_motor_torque_clip = True  # False: 同时关闭硬限幅及其 reward
        pre_torque_vel_clip_hip = 100.0 #60.0       # hip + thigh [Nm]
        pre_torque_vel_clip_calf = 200.0 #180.0     # calf [Nm]
        # 动作平滑（低通滤波）
        action_smoothness = True
        ratio = 0.9                         # 平滑因子 action = 0.9*new + 0.1*old

    # =========================================================================
    # 仿真配置
    # =========================================================================
    class sim(LeggedRobotCfg.sim):
        dt = 0.005                          # 仿真步长 = 5ms = 200Hz

    # =========================================================================
    # 域随机化
    # =========================================================================
    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = True

        # ---- 外力扰动 ---- 
        push_robots = False                   # 随机推机器人
        push_interval_s = 8                   # 推力间隔 [s]
        max_push_vel_xy = 0.2                 # 最大推力速度 [m/s]
        max_push_ang_vel = 0.1                # 最大推力角速度 [rad/s]

        # ---- 刚体属性随机化 ----
        rand_interval_s = 10                # 随机化间隔 [s]
        randomize_rigids_after_start = False  # 运行时重新随机化刚体属性
        randomize_friction = use_random      # 随机化摩擦
        friction_range = [0.5, 1.2]

        randomize_restitution = use_random    # 随机化恢复系数
        restitution_range = [0, 0.3]

        # ---- 基座质量和惯量 ----
        randomize_base_mass = use_random      # 随机化质量
        added_mass_range = [-2, 2]            # 附加质量范围 [kg]

        randomize_inertia = use_random        # 随机化惯量
        randomize_inertia_range = [0.8, 1.2]  # 惯量缩放范围

        # ---- 质心偏移 ----
        randomize_com_displacement = use_random
        com_displacement_range = [-0.02, 0.05]   # base link 质心偏移 [m]

        randomize_each_link = False              # 不单独随机化每个 link 的质心
        link_com_displacement_range_factor = 0.02

        # ---- 电机力矩 ----
        randomize_motor_strength = True
        # motor_strength_range = [0.4, 0.8]    # 力矩缩放（兜底默认，没配下面三组时用这个）
        # 按关节类型分别设 motor_strength（配了覆盖兜底值，hip=[0,3,6,9], thigh=[1,4,7,10], calf=[2,5,8,11]）
        motor_strength_hip_range = [0.8, 1.0]     # hip 电机
        motor_strength_thigh_range = [0.8, 1.0]   # thigh 电机
        motor_strength_calf_range = [0.6, 0.85]     # calf 电机
        # motor_strength_hip_range = [0.4, 1.0]     # hip 电机
        # motor_strength_thigh_range = [0.4, 1.0]   # thigh 电机
        # motor_strength_calf_range = [0.4, 1.0]     # calf 电机

        randomize_PD_factor = use_random     # 随机化 PD 增益
        Kp_factor_range = [0.75, 1.2]         # Kp 缩放
        Kd_factor_range = [0.75, 1.2]         # Kd 缩放

        # ---- 编码器偏置（模拟安装误差）----
        randomize_motor_offset = False #use_random
        # motor_offset_range = [-0.1, 0.1]          # 偏置范围 [rad]
        motor_offset_range = [-0.05, 0.05]      # 偏置范围 [rad]
        randomize_default_dof_pos = False    # 随机化默认关节位置
        randomize_default_dof_pos_range = [-0.03, 0.03]

        # ---- 延迟模拟 ----
        # 动作延迟（策略输出到执行的时间差）
        add_action_lag =  False #use_random
        randomize_lag_timesteps = False #use_random
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [0, 1]         # 0~4 步 × 5ms = 0~20ms 延迟

        # 编码器延迟（关节传感器读数滞后）
        add_dof_lag = False #use_random
        randomize_dof_lag_timesteps = False #use_random
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1]     # 0~1 步 = 0~5ms
        # dof_lag_timesteps_range = [2, 5]     # 0~1 步 = 0~5ms

        # IMU 延迟
        add_imu_lag = use_random
        randomize_imu_lag_timesteps = use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 1]     # 0~1 步 = 0~5ms

        # ----- 高程图感知建模 -----
        # 高程图独立于 200 Hz 本体传感器：只在每 20 个物理步刷新一次。
        height_update_hz = 10
        randomize_height_offset = True and use_random
        height_offset_range = [-0.05, 0.05]  #[-0.03, 0.03]
        add_height_noise = True and use_random
        add_height_gaussian_noise = True and use_random
        height_gaussian_noise = 0.03
        add_height_spike_noise = True and use_random
        height_spike_noise_range = [0.05, 0.2]
        randomize_height_yaw = True and use_random
        height_yaw_noise_range = [-0.2, 0.2]
        randomize_height_roll_pitch = True and use_random
        height_pitch_bias_range = [-0.02, 0.02]
        height_roll_bias_range = [-0.02, 0.02]
        height_repeat_probability = 0.2 if use_random else 0.0

        # ---- 关节物理属性（模拟电机阻尼/摩擦特性）----
        DOF = Cowa_Num.DOF

        # 关节摩擦
        default_joint_friction = get_quadruped_default_joint_friction(Cowa_Num.quad_index)
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.8, 1.2]

        # 关节阻尼
        default_joint_damping = get_quadruped_default_joint_damping(Cowa_Num.quad_index)
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.8, 1.2]

        # 关节电感（转动惯量）
        default_joint_armature = get_quadruped_default_joint_armature(Cowa_Num.quad_index)
        randomize_joint_armature = use_random
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.8, 1.2]

    # =========================================================================
    # 命令配置
    # =========================================================================
    class commands(LeggedRobotCfg.commands):
        curriculum = True           
        max_curriculum = 1 
        num_commands = 12                    # 12 维命令（见下方 ranges）
        resampling_time = 10.                # 命令重采样间隔 [s]
        heading_command = False               # True=航向角模式, False=角速度模式

        class ranges:
            """命令取值范围（固定值 = 训练时始终使用该值）、
            [  0] lin_vel_x:        0.0 m/s 前进
            [  1] lin_vel_y:        0.0      侧移
            [  2] ang_vel_yaw:      0.0      转向
            [  3] body_height:      0.4 m    目标体高
            [  4] gait_freq:        2.0 Hz   trot 步频
            [  5] gait_phase:       0.5      trot 对角腿同相
            [  6] gait_offset:      0.0      trot
            [  7] gait_bound:       0.0      trot
            [  8] gait_duration:    0.5      占空比 50%
            [  9] footswing_height: 0.2 m    抬脚高度
            [ 10] body_pitch:       0.0      身体俯仰
            [ 11] body_roll:        0.0      身体滚转
            """

            lin_vel_x = [-0.5, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]
            body_height_cmd = [0.40, 0.40]
            gait_phase_cmd_range = [0.5, 0.5]     
            gait_offset_cmd_range = [-0.0, 0.0]
            gait_bound_cmd_range = [-0.0, 0.0]
            gait_frequency_cmd_range = [1.5, 1.5]  #[2.0, 2.0]
            gait_duration_cmd_range = [0.5, 0.5]   
            footswing_height_range = [0.08, 0.08]    
            body_pitch_range = [-0.0, 0.0]
            body_roll_range = [-0.0, 0.0]

    # =========================================================================
    # 奖励配置
    # =========================================================================
    class rewards:
        only_positive_rewards = False        # False=允许负奖励（惩罚）
        max_contact_force = 600              # 最大接触力 [N] 600
        soft_dof_pos_limit = 0.9             # 关节限位软阈值（URDF 限位的 90%）
        soft_torque_limit = 0.8              # 力矩软阈值
        pre_motor_torque_soft_ratio = 0.9    # 100/200 Nm 硬限幅的奖励软阈值
        soft_dof_vel_limits = 0.9            # 速度软阈值
        kappa_gait_probs = 0.07              # 步态概率平滑参数
        tracking_sigma = 0.1                # 通用跟踪 sigma
        adaptive_clearance_margin = 0.03
        adaptive_clearance_max = 0.40
        clearance_overheight_tolerance = 0.03
        virtual_collision_radius = 0.06
        virtual_collision_clearance = 0.02
        virtual_collision_sigma = 0.05
        virtual_collision_weight = 0.2 

        class scales:

            tracking_lin_vel = 3                     # 线速度跟踪（指数奖励）
            tracking_ang_vel = 3                     # 角速度跟踪
            base_height = -2                     # 体高跟踪（惩罚）
            orientation = -5                     # 身体倾斜惩罚         
            feet_clearance = -5                    # 摆动期足端高度轨迹误差惩罚
            default_hip_pos = -2.0  # -1.0 #
            foot_stumble = -1.0    
            stand_still = -5
            stand_base_vel_penality = -4.0 #-2

            # ---- 正则化惩罚 ----
            lin_vel_z = -0.1 #-1.0                 # Z 向速度
            ang_vel_xy = -0.1 #-1.0                # roll/pitch 角速度
            # dof_vel = -5e-5                  # 关节速度
            dof_acc = -2.5e-7                  # 关节加速度
            torques = -1e-6                  # 力矩大小
            power = -2e-5                    # 功率
            torque_limits = -1.0             # 力矩超限
            pre_motor_torque_limits = -0.2   # 效率前 100/200 Nm 限幅惩罚
            action_rate = -0.05              # 动作变化率
            action_smoothness = -0.05        # 动作平滑度
            base_acc = -1e-2                 # 基座加速度
            collision = -2.0                # 碰撞
            dof_pos_limits = -1.0            # 关节位置超限 
            dof_vel_limits = -1.0            # 关节速度超限

    # =========================================================================
    # 观测归一化
    # =========================================================================
    class normalization:
        class obs_scales:
            lin_vel = 2.0                   # 线速度缩放
            ang_vel = 0.25                  # 角速度缩放
            dof_pos = 1.0                   # 关节位置缩放（不缩放）
            dof_vel = 0.05                  # 关节速度缩放
            height_measurements = 5.0        # 高度测量缩放
            quat = 1                        # 四元数/投影重力缩放（不缩放）
            gravity = 1                     # 重力缩放
            forces = 0.1                    # 接触力缩放
            # 命令缩放（与 commands_scale 一致）
            body_height_cmd = 1.0
            gait_freq_cmd = 1.0
            gait_phase_cmd = 1.0
            footswing_height_cmd = 1.0
            body_pitch_cmd = 1.0
            body_roll_cmd = 1.0
            gait_duration_cmd = 1.0

        clip_observations = 100.             # 观测裁剪上限
        clip_actions = 100.                   # 动作裁剪上限

    class viewer(LeggedRobotCfg.viewer):
        draw_clean_heightmap = False
        draw_randomized_heightmap = False
        randomized_heightmap_env_id = 0

class QuadWtwCfgPPO_HIM(LeggedRobotCfgPPO):

    seed = 10
    runner_class_name = 'OnPolicyRunner_HIM_HeightScan'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0           # 初始探索噪声标准差
        actor_hidden_dims = [512, 256, 128]  # Actor MLP
        critic_hidden_dims = [512, 256, 128] # Critic MLP
        estimator_hidden_dims = [512, 256, 128]
        tar_hidden_dims = [512, 256, 128]
        history_len = 5                      # 5 帧完整 [proprio | height scan] 历史

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0               # Value Loss 系数
        use_clipped_value_loss = True        # PPO value clip
        clip_param = 0.2                    # PPO 裁剪参数 ε
        entropy_coef = 0.01                 # 熵正则系数
        num_learning_epochs = 5             # 每次更新迭代轮数
        num_mini_batches = 4                # Mini-batch 数量
        learning_rate = 5.0e-4              # 学习率
        schedule = "adaptive"               # 自适应 KL 调度
        gamma = 0.99                        # 折扣因子
        lam = 0.95                          # GAE λ 参数
        desired_kl = 0.01                   # 目标 KL 散度
        max_grad_norm = 1.0                 # 梯度裁剪
        symmetry_scale = 0.01                # 传给 PPO_HIM；0 表示关闭对称损失做消融

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_HIM'
        algorithm_class_name = 'PPO_HIM'   # 替换 'PPO'
        num_steps_per_env = 24             # 每次迭代每环境步数
        max_iterations = 1000001           
        # 日志与保存
        save_interval = 200                
        save_interval = 200                
        experiment_name = 'quadruped_wtw_him_arm_fix_height_scan'
        run_name = ''
        resume = False                    
        load_run = -1                  
        checkpoint = -1                    
        resume = False                    
        load_run = -1                  
        checkpoint = -1                    
        resume_path = '/home/cowa'
