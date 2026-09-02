# HIM + AMP 融合环境配置
# 继承链: QuadCfg_HIM_AMP → QuadCfg_HIM → LeggedRobotCfg
#          QuadCfgPPO_HIM_AMP → QuadCfgPPO_HIM → LeggedRobotCfgPPO
# 本文件只覆盖 AMP 相关字段:
#   - reward 函数与系数(scales)完全继承自 QuadCfg_HIM,不在此重写
#   - AMP 不改 env reward,仅在 runner 层叠加 style reward: r=(1-lerp)*style + lerp*task
# 注册名: 'quadruped_arm_him_amp' (envs/__init__.py:356)
from ..quadruped_arm_him.quadruped_arm_him_config import QuadCfg_HIM,QuadCfgPPO_HIM
from ..amp_d1.amp_d1_config import AmpD1Cfg,AmpD1CfgPPO
from wheel_legged_gym.algo.PPO_AMP.symmetry import compute_symmetric_states_d1
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import glob


class QuadCfg_HIM_AMP(QuadCfg_HIM):

    class asset(QuadCfg_HIM.asset):
        
        # AMP 判别器/参考姿态初始化需要追踪的关键 body: 四个脚
        key_body_names = "foot" 

    class env(QuadCfg_HIM.env):

        # amp_motion_files = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/d1_data_v4/*.pkl") #专家动捕文件路径
        amp_motion_files = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/d1_data_v1/*.pkl") #专家动捕文件路径
        num_est_prob = 4  # vel_xyz(3) + base_height(1)，estimator 输出维度

    class control(QuadCfg_HIM.control):
        # 覆盖训练用 DOF limit 和电机 torque-speed 包络。
        # 顺序: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
        dof_velocity_limits = [
            20, 20, 10,
            20, 20, 10,
            20, 20, 10,
            20, 20, 10,
        ]

        torque_vel_hip_indices = [0, 1, 3, 4, 6, 7, 9, 10]
        torque_vel_hip_max_vel = 20
        torque_vel_hip_vel_1 = 7.28
        torque_vel_hip_max_torque = 150.0 #200

        torque_vel_calf_indices = [2, 5, 8, 11]
        torque_vel_calf_max_vel = 12
        torque_vel_calf_vel_1 = 6.6
        torque_vel_calf_max_torque = 250.0 #330

        # 独立力矩限幅（在 t-w 曲线 clip 之前生效）
        # 对齐真机减速器输出端能力：hip/thigh ≤ 60Nm，calf ≤ 180Nm
        # pre_torque_vel_clip_hip = 60.0      # hip+thigh 独立限幅
        # pre_torque_vel_clip_calf = 180.0    # calf 独立限幅

    class rewards(QuadCfg_HIM.rewards):
        # torque_limits reward starts at 80% of the velocity-dependent torque envelope.
        soft_torque_limit = 0.8
        # dof_vel_limits 惩罚从 0.9 * limit 开始，小腿约 9 rad/s 开始扣分，给 10 rad/s 留余量。
        soft_dof_vel_limits = 0.9

        class scales(QuadCfg_HIM.rewards.scales):

            stand_base_vel_penality = -0.5   # 主项：身体停住
            stand_foot_vel = -0.5            # 主项：脚停住
            stand_feet_air = -0.5            # 辅助：脚别悬空
            dof_vel_stand_still = -0.5       # 防抖，小权重
            stand_still = -0.5     

            base_height = -1.0
            orientation = -1.0
            foot_slip   = -1.0 
            foot_stumble = -1.0

            torque_limits  = -0.0   # 转矩限制
            dof_vel_limits = -0.0   # 转速限制

    class init_state(QuadCfg_HIM.init_state):
        # 参考姿态初始化: 一部分 reset 把机器人摆成专家动捕帧的姿态
        # 由 LeggedRobotAMP.reset_idx 实现 (_reset_dofs_from_reference_motion 等)
        reference_state_initialization = False
        reference_state_initialization_prob = 0.5  # 50% 概率从专家帧初始化

class QuadCfgPPO_HIM_AMP(QuadCfgPPO_HIM):
    # 切换到 AMP 训练 runner (重写 _init_agent_and_algo / learn,处理 ActorCritic_HIM + estimator + disc)
    runner_class_name = 'AMPRunner_HIM'

    class runner(QuadCfgPPO_HIM.runner):
        # 策略类沿用 HIM (teacher-student 非对称 actor-critic + 历史推理)
        policy_class_name = 'ActorCritic_HIM'
        # PPO_AMP_HIM 实为 PPO_AMP 别名 (ppo_amp_him.py: PPO_AMP_HIM = PPO_AMP)
        algorithm_class_name = 'PPO_AMP_HIM'

        # style reward 系数: 把判别器输出缩放到与 task reward 同量级
        amp_reward_coef = 4.0 * QuadCfg_HIM_AMP.sim.dt
        # 专家动捕文件 (复用 amp_d1)
        amp_motion_files = QuadCfg_HIM_AMP.env.amp_motion_files
        # 预采样 (s,s') 对数量,训练时直接索引采样,省插值开销
        amp_num_preload_transitions = QuadCfg_HIM_AMP.env.num_envs * QuadCfgPPO_HIM.runner.num_steps_per_env * 10
        # 判别器 trunk 隐藏层维度
        amp_discr_hidden_dims = [1024, 512]

        # 按当前 command 类型(stand/forward/backward/strafe/yaw/turn)采对应专家样本。
        # False 时保持原 AMP: policy 样本和全专家混合分布对抗。
        amp_use_bucketed_experts = False
        # expert 侧按 motion 文件名/trajectory 分桶，避免 ramp 低速段被误分到 stand_other。
        # policy 侧仍按 command(vx, vy, wz) 分桶。
        amp_expert_label_source = "filename"
        amp_bucket_lin_vel_threshold = 0.10
        amp_bucket_yaw_threshold = 0.20
        amp_bucket_yaw_lin_vel_threshold = 0.20

        # AMP 动态权重（style curriculum）
        amp_style_curriculum = False
        amp_task_reward_lerp = 0.55 #任务
        amp_task_reward_lerp_min = 0.25
        amp_task_reward_lerp_max = 0.95
        amp_style_curriculum_reward_key = "tracking_ang_vel"
        amp_style_curriculum_success_threshold = 0.8
        amp_style_curriculum_fail_threshold = 0.65
        amp_style_curriculum_style_step = 0.005
        amp_style_curriculum_task_step = 0.01
        amp_style_curriculum_ema_alpha = 0.1
        amp_style_curriculum_update_interval = 10

    class policy(QuadCfgPPO_HIM.policy):

        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(QuadCfgPPO_HIM.algorithm):

        # Symmetry loss config
        symmetry_cfg = {
            "use_data_augmentation" : True,
            "data_augmentation_func": compute_symmetric_states_d1,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.05,
        }
        # AMP 判别器损失: LSGAN 式 (专家+1/策略-1), 与 quad_mapping (peak at d=1) 配套
        amp_loss = "MSELoss"
        # style reward 映射: 在 d=1 (专家目标) 处取峰值
        style_reward_function = "quad_mapping"
        # 是否对 style reward 做经验归一化 (仅 wasserstein 映射用)
        normalize_style_reward = False
        # policy 侧 replay buffer 容量 (存 policy 的 (s,s') 供判别器训练)
        amp_replay_buffer_size = QuadCfg_HIM_AMP.env.num_envs * QuadCfgPPO_HIM.runner.num_steps_per_env * 10
        # 判别器学习率 (远小于 policy lr,防止 disc 过快压倒 policy)
        disc_lr = 1e-4
