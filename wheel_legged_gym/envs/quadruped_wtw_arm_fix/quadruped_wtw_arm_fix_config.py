"""
quadruped_wtw_slope 训练配置文件
======================================
基于 WTW (Walk These Ways) 框架的 COWA 四足机器人斜坡/地形训练配置。

核心特点：
- 12 维命令（速度 + 步态参数），支持步态控制
- 59 维单帧观测（12命令 + 36关节状态 + 6IMU + 5时序）
- 5 帧历史堆叠（295 维输入 Actor）
- Trimesh 斜坡地形，课程学习

观测顺序（每帧 59 维）：
  [0:12]   命令 (vx, vy, yaw, body_h, freq, phase, offset, bound, dur, swing, pitch, roll)
  [12:24]  关节位置 (q - default_q) * dof_pos_scale
  [24:36]  关节速度 dq * dof_vel_scale
  [36:48]  上一步动作（策略原始输出） 
  [48:51]  机体角速度 * ang_vel_scale
  [51:54]  投影重力方向（重力在机体系中的表示）
  [54]     步态相位索引 gait_index [0,1]
  [55:59]  步态时钟 sin 信号（4 条腿各 1 个）

关节顺序（训练）：
  hips:  FL → RL → FR → RR
  thighs: FL → RL → FR → RR
  calves: FL → RL → FR → RR
"""

from wheel_legged_gym.envs.quadruped.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_quadruped_joint_names, get_quadruped_default_joint_friction, get_quadruped_default_joint_damping, get_quadruped_default_joint_armature


class Cowa_Num:
    """机器人常量"""
    quad_index = '1'   # 四足型号索引
    DOF = 12            # 自由度：4腿 × 3关节(hip/thigh/calf)


class QuadWtwCfg(LeggedRobotCfg):
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

        # 单帧观测维度 = 12命令 + 3×12关节 + 5/6(IMU) + 5(时序)
        num_single_obs = num_commands + 3 * num_actions + timing_obs_dim  # = 12 + 36 + 5 = 53
        if projected_gravity:
            num_single_obs += 6           # ang_vel[3] + proj_gravity[3] = 6
        else:
            num_single_obs += 5           # ang_vel[3] + euler[2] = 5
        # 最终 num_single_obs = 53 + 6 = 59

        # 帧堆叠
        frame_stack = 5                   # 观测历史帧数（输入给策略的总帧数）
        actor_input_stack = 5             # 输入给 Actor 的帧数（通常 = frame_stack）
        c_frame_stack = 1                 # 输入给 Critic 的帧数
        num_envs = 4096                   # 并行环境数

        # 总观测维度 = 5 帧 × 59 维 = 295
        num_observations = int(frame_stack * num_single_obs)

        # 特权观测（Critic 可见，Actor 不可见，用于非对称训练）
        single_num_privileged_obs = num_single_obs + 3 + 1  # + 线速度[3] + 体高[1]

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

        priv_observe_heightmap = False    # 高程图（7×11=77 维）
        if priv_observe_heightmap:
            single_num_privileged_obs += 7 * 11

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
        penalize_contacts_on = ["thigh", "calf", "base_link"]   # 碰撞惩罚部位
        terminate_after_contacts_on = ["base_link"]              # 碰撞终止部位

    # =========================================================================
    # 地形配置
    # =========================================================================
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'              # 地形类型：none/plane/heightfield/trimesh
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
        measured_points_x = [              # 测量点 X 坐标（前后 0.5m）
            -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        ]
        measured_points_y = [              # 测量点 Y 坐标（左右 0.3m）
            -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3,
        ]
        selected = False                   # 单选地形类型
        terrain_kwargs = None              # 地形参数
        max_init_terrain_level = 5         # 初始课程等级
        terrain_length = 8.                # 单块地形长度 [m]
        terrain_width = 8.                 # 单块地形宽度 [m]
        num_rows = 10                      # 地形行数（难度等级）
        num_cols = 10                      # 地形列数（类型数）

        # 地形类型比例：[平面, 平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散]
        # terrain_proportions = [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0]
        terrain_proportions = [0., 0., 0., 0.0, 1.0, 0.0, 0, 0, 0.0, 0.0]

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
        pos = [0.0, 0.0, 0.4]              # 初始位置 x,y,z [m]（z=0.4 离地高度）
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
        stiffness = {'joint': 160.0}        # 刚度 [N·m/rad]
        damping = {'joint': 5.0}            # 阻尼 [N·m·s/rad]
        # 控制频率：decimation=4, dt=0.005 → 策略频率 50Hz, 仿真频率 200Hz
        decimation = 4
        action_scale = 0.25                 # 动作缩放：target_q = default_q + action * 0.25
        # 动作平滑（低通滤波）
        action_smoothness = False
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
        use_random = True                   # 总开关

        # ---- 外力扰动 ---- 
        push_robots = False                 # 随机推机器人
        push_interval_s = 8                 # 推力间隔 [s]
        max_push_vel_xy = 0.2               # 最大推力速度 [m/s]
        max_push_ang_vel = 0.1              # 最大推力角速度 [rad/s]

        # ---- 刚体属性随机化 ----
        rand_interval_s = 10                # 随机化间隔 [s]
        randomize_rigids_after_start = False  # 运行时重新随机化刚体属性
        randomize_friction = use_random      # 随机化摩擦
        friction_range = [0.2, 2]
        randomize_restitution = use_random   # 随机化恢复系数
        restitution_range = [0, 1.0]

        # ---- 基座质量和惯量 ----
        randomize_base_mass = use_random     # 随机化质量
        added_mass_range = [-2, 2]           # 附加质量范围 [kg]
        randomize_inertia = use_random       # 随机化惯量
        randomize_inertia_range = [0.8, 1.2] # 惯量缩放范围

        # ---- 质心偏移 ----
        randomize_com_displacement = use_random
        #com_displacement_range = [-0.05, 0.05]  # base link 质心偏移 [m]
        com_displacement_range = [-0.05, 0.15]  # base link 质心偏移 [m]
        randomize_each_link = False          # 不单独随机化每个 link 的质心
        link_com_displacement_range_factor = 0.02

        # ---- 电机力矩 ----
        randomize_motor_strength = use_random
        motor_strength_range = [0.8, 1.2]    # 力矩缩放
        randomize_PD_factor = use_random      # 随机化 PD 增益
        Kp_factor_range = [0.8, 1.2]         # Kp 缩放
        Kd_factor_range = [0.8, 1.2]         # Kd 缩放

        # ---- 编码器偏置（模拟安装误差）----
        randomize_motor_offset = use_random
        motor_offset_range = [-0.05, 0.05]   # 偏置范围 [rad]
        randomize_default_dof_pos = False    # 不随机化默认关节位置
        randomize_default_dof_pos_range = [-0.03, 0.03]

        # ---- 延迟模拟 ----
        # 动作延迟（策略输出到执行的时间差）
        add_action_lag = True and use_random
        randomize_lag_timesteps = True and use_random
        randomize_lag_timesteps_perstep = True and use_random
        lag_timesteps_range = [0, 4]         # 0~4 步 × 5ms = 0~20ms 延迟

        # 编码器延迟（关节传感器读数滞后）
        add_dof_lag = True and use_random
        randomize_dof_lag_timesteps = True and use_random
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 1]     # 0~1 步 = 0~5ms

        # IMU 延迟
        add_imu_lag = True and use_random
        randomize_imu_lag_timesteps = True and use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 1]     # 0~1 步 = 0~5ms

        # ---- 关节物理属性（模拟电机阻尼/摩擦特性）----
        DOF = Cowa_Num.DOF

        # 关节摩擦
        default_joint_friction = get_quadruped_default_joint_friction(Cowa_Num.quad_index)
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.5, 1.5]

        # 关节阻尼
        default_joint_damping = get_quadruped_default_joint_damping(Cowa_Num.quad_index)
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.5, 1.5]

        # 关节电感（转动惯量）
        default_joint_armature = get_quadruped_default_joint_armature(Cowa_Num.quad_index)
        randomize_joint_armature = use_random
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.5, 1.5]

    # =========================================================================
    # 命令配置
    # =========================================================================
    class commands(LeggedRobotCfg.commands):
        curriculum = False                   # 命令课程（逐步增大速度范围）
        max_curriculum = 1
        num_commands = 12                    # 12 维命令（见下方 ranges）
        resampling_time = 10.                # 命令重采样间隔 [s]
        heading_command = False               # True=航向角模式, False=角速度模式

        class ranges:
            """命令取值范围（固定值 = 训练时始终使用该值）、
            [  0] lin_vel_x:        1.5 m/s 前进
            [  1] lin_vel_y:        0.0      侧移
            [  2] ang_vel_yaw:      0.0      转向
            [  3] body_height:      0.4 m    目标体高
            [  4] gait_freq:        2.0 Hz   trot 步频
            [  5] gait_phase:       0.5      trot 对角腿同相
            [  6] gait_offset:      0.0      trot
            [  7] gait_bound:       0.0      trot
            [  8] gait_duration:    0.5      占空比 50%
            [  9] footswing_height: 0.3 m    抬脚高度
            [ 10] body_pitch:       0.0      身体俯仰
            [ 11] body_roll:        0.0      身体滚转
            """
            # lin_vel_x = [-1.0, 1.0]
            # lin_vel_y = [-0.5, 0.5]
            # ang_vel_yaw = [-0.5, 0.5]
            # heading = [-3.14, 3.14]
            # body_height_cmd = [0.2, 0.5]
            lin_vel_x = [0, 0]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]
            body_height_cmd = [0.2, 0.5]
            # 固定为 trot 步态参数
            gait_phase_cmd_range = [0.5, 0.5]      # 对角腿同相
            gait_offset_cmd_range = [-0.0, 0.0]
            gait_bound_cmd_range = [-0.0, 0.0]
            gait_frequency_cmd_range = [2.0, 2.0]  # 2Hz
            gait_duration_cmd_range = [0.5, 0.5]   # 50% 占空比，让狗学会在跑和走之间过渡
            footswing_height_range = [0.05, 0.05]    # 0.05m 抬脚
            body_pitch_range = [-0.5, 0.3]
            body_roll_range = [-0.0, 0.0]

    # =========================================================================
    # 奖励配置
    # =========================================================================
    class rewards:
        only_positive_rewards = False        # False=允许负奖励（惩罚）
        tracking_sigma_lin_vel = 20          # 线速度跟踪 sigma
        tracking_sigma_ang_vel = 20          # 角速度跟踪 sigma
        #base_height_target = 0.40
        max_contact_force = 600              # 最大接触力 [N]
        soft_dof_pos_limit = 0.9             # 关节限位软阈值（URDF 限位的 90%）
        soft_torque_limit = 0.8              # 力矩软阈值
        soft_dof_vel_limits = 0.9            # 速度软阈值
        kappa_gait_probs = 0.07              # 步态概率平滑参数
        tracking_sigma = 0.25                # 通用跟踪 sigma

        class scales:

            """奖励权重（正=奖励, 负=惩罚）"""
            # ---- 跟踪奖励 ----
            tracking_lin_vel = 2                    # 线速度跟踪（指数奖励）
            tracking_ang_vel = 4                    # 角速度跟踪

            # ---- 姿态/步态风格奖励 ----
            base_height = -10.0 #-10                # 体高跟踪（惩罚）
            #orientation = -10                      # 身体倾斜惩罚
            orientation_control = 8.0
            pitch_agility = 2.0                     # <--- 新增：主动奖励快速调 Pitch
            pitch_tracking_penalty = -10.0
            tracking_contacts_shaped_force = -2.0   # 摆动相接触力惩罚
            tracking_contacts_shaped_vel = -2.0     # 支撑相滑动惩罚
            #feet_clearance_cmd_linear = -3.0       # 抬腿高度跟踪惩罚

            feet_clearance_cmd_exp = 1.5            # 奖励运动抬腿
            trot_diagonal_symmetry_positive = 1.5   # 奖励运动对称姿势

            # ---- 静止稳定 ----
            #stand_base_vel_penality = -2.0   # 零命令时速度惩罚
            #stand_stability = -2.0
            stand_all_feet_contact = 2.0     # 零命令时四脚着地奖励
            stand_absolute_stable = 3.0
            stand_feet_slip_penalty = -5.0  # 如果依然有轻微滑动，可以直接到 -8.0 或 -10.0


            # stand_posture_positive = 2.0      
            # stand_all_feet_contact = 2.0    
            stand_action_freeze = 3.0        

            # ---- 对称性/默认姿态 ----
            dof_pos_symmetry = -5.0           # 左右关节对称惩罚
            default_hip_pos = -0.05           # hip 偏离 0 惩罚 -1.0

            # ---- 正则化惩罚 ----
            lin_vel_z = -0.3                 # Z 向速度
            ang_vel_xy = -0.3                # roll/pitch 角速度
            dof_vel = -5e-5                  # 关节速度
            dof_acc = -5e-7                  # 关节加速度
            # torques = -1e-5                  # 力矩大小
            # power = -2e-5                    # 功率
            torques = -5e-6
            power = -1e-5
            torque_limits = -0.05            # 力矩超限
            # action_rate = -0.01              # 动作变化率
            # action_smoothness = -0.01        # 动作平滑度
            # base_acc = -1e-2                 # 基座加速度
            action_rate = -0.005
            action_smoothness = -0.005
            base_acc = -5e-3        # 允许机身瞬间产生较高的角加速度去完成 Pitch
            collision = -10.0                 # 碰撞
            dof_pos_limits = -0.1            # 关节位置超限
            dof_vel_limits = -0.1            # 关节速度超限
            feet_contact_forces = -0.05      # 足端接触力过大 -0.01

    # =========================================================================
    # 观测归一化
    # =========================================================================
    class normalization:
        class obs_scales:
            """观测缩放因子（原始值 → 归一化值）
            训练观测 = raw_value * obs_scale
            例如：ang_vel=2 rad/s → 2 * 0.25 = 0.5（归一化）
            """
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
        clip_actions = 20.                   # 动作裁剪上限


class QuadWtwCfgPPO(LeggedRobotCfgPPO):
    """
    PPO 训练配置
    =============
    vanilla PPO + ActorCritic 网络，标准 50Hz 控制。
    """
    seed = 10
    runner_class_name = 'OnPolicyRunner'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0 #1.0                # 初始探索噪声标准差
        actor_hidden_dims = [256, 128, 64]  # Actor MLP：[295→256→128→64→12]
        critic_hidden_dims = [256, 128, 64] # Critic MLP：[privileged_obs→256→128→64→1]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0               # Value Loss 系数
        use_clipped_value_loss = True        # PPO value clip
        clip_param = 0.2                    # PPO 裁剪参数 ε
        entropy_coef = 0.01#0.01            # 熵正则系数
        num_learning_epochs = 5             # 每次更新迭代轮数
        num_mini_batches = 4                # Mini-batch 数量
        learning_rate = 5.0e-4              # 学习率
        schedule = "adaptive"               # 自适应 KL 调度
        gamma = 0.99                        # 折扣因子
        lam = 0.95                          # GAE λ 参数
        desired_kl = 0.01                   # 目标 KL 散度
        max_grad_norm = 1.0                 # 梯度裁剪

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        #algorithm_class_name = 'PPO'
        algorithm_class_name = 'PPO_SYM'   # 替换 'PPO'
        symmetry_scale = 0.01             # 加到 algorithm 段（默认 0.01）
        num_steps_per_env = 24              # 每次迭代每环境步数
        max_iterations = 1000001            # 总迭代次数

        # 日志与保存
        save_interval = 200                 # 每 100 次迭代保存一次
        experiment_name = 'quadruped_wtw_arm_fix'
        run_name = ''
        resume = False                      # 从 checkpoint 恢复
        load_run = -1                       # -1 = 最后一个 run
        checkpoint = -1                     # -1 = 最后一个模型
        resume_path = '/home/cowa'
