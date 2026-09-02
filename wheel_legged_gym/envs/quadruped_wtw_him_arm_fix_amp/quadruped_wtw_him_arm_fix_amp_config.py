"""wtw + AMP 融合配置：继承 wtw 全部参数，叠加 AMP 判别器。

继承链: QuadWtwAmpCfg_HIM → QuadWtwCfg_HIM → LeggedRobotCfg
          QuadWtwAmpCfgPPO_HIM → QuadWtwCfgPPO_HIM → LeggedRobotCfgPPO

本文件只覆盖 AMP 相关字段：
  - asset 加 key_body_names（判别器追踪的脚）
  - env 加 amp_motion_files（专家数据）
  - runner 换 AMPRunner_HIM（叠加 style reward）
  - algorithm 换 PPO_AMP_HIM（判别器训练）
  - 判别器 obs 维度：3+3+12+12+4*3 = 42（与专家数据一致，因为 default_dof_pos=0，相对偏移=绝对值）
"""
import glob
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.quadruped_wtw_him_arm_fix.quadruped_wtw_him_arm_fix_config import (
    QuadWtwCfg_HIM, QuadWtwCfgPPO_HIM
)


class QuadWtwAmpCfg_HIM(QuadWtwCfg_HIM):
    """wtw + AMP 配置：保留 wtw 全部姿态调整，叠加 AMP 判别器。"""

    class asset(QuadWtwCfg_HIM.asset):
        # AMP 判别器/参考姿态初始化需要追踪的关键 body：四个脚
        key_body_names = "foot"

    class init_state(QuadWtwCfg_HIM.init_state):
        # AMP 参考姿态初始化：一部分 reset 把机器人摆成专家动捕帧的姿态
        reference_state_initialization = True
        reference_state_initialization_prob = 0.5  # 50% 概率从专家帧初始化

    class env(QuadWtwCfg_HIM.env):
        # 专家动捕文件（与 amp env 共用同一套数据）
        amp_motion_files = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/d1_data_v1/*.pkl")

        # ★ 关掉时钟：步态由 AMP 判别器管（专家先验），不用强制步频/相位。
        # 这样步频自然对齐专家（~1.0Hz），无时钟-专家冲突。
        # 姿态调整（body_height/pitch/roll）不依赖时钟，仍然有效。
        enable_gait_clock = False
        observe_timing_parameter = False    # 不观测 gait_index
        observe_clock_inputs = False        # 不观测 clock_inputs
        observe_gait_commands = False       # 不观测步态命令（freq/phase/offset 等）
        timing_obs_dim = 0
        # ★ 重新算 num_single_obs（父类在 class body 里已算成 59，这里覆盖成 54）
        # 59 - 1(gait_index) - 4(clock_inputs) = 54
        num_single_obs = 54
        num_observations = int(54 * 30)   # frame_stack=30
        # privileged_obs 也同步减 5（去掉 clock 部分）
        # 父类算成 93（=59+特权），这里 = 93-5 = 88
        single_num_privileged_obs = 88
        num_privileged_obs = 88   # c_frame_stack=1

    class rewards(QuadWtwCfg_HIM.rewards):
        """关时钟后，依赖时钟相位的 reward 会失效（foot_indices/desired_contact 全 0），
        把这些 reward 权重设 0，避免贡献无意义 reward 扰乱训练。
        AMP 判别器接管步态引导（抬腿/柔和落地），不需要这些显式 reward。"""
        class scales(QuadWtwCfg_HIM.rewards.scales):
            # 时钟依赖 reward → 关（AMP 管步态）
            feet_clearance_cmd_exp = 0.0          # 依赖 foot_indices_tensor（关时钟后全0）
            tracking_contacts_shaped_force = 0.0  # 依赖 desired_contact_states（全0）
            tracking_contacts_shaped_vel = 0.0    # 同上
            trot_diagonal_symmetry_positive = 0.0 # 依赖相位（时钟关了无意义）
            # 其余 reward 继承 wtw（速度跟踪/姿态/正则等，不依赖时钟）


class QuadWtwAmpCfgPPO_HIM(QuadWtwCfgPPO_HIM):
    """wtw + AMP 的 PPO 配置：切换到 AMP runner/algorithm。"""

    # 切换到 AMP 训练 runner（处理 ActorCritic_HIM + estimator + disc）
    runner_class_name = 'AMPRunner_HIM'

    class runner(QuadWtwCfgPPO_HIM.runner):
        # 策略类沿用 HIM（保留 wtw 的历史推理 + 姿态调整）
        policy_class_name = 'ActorCritic_HIM'
        # PPO_AMP_HIM 在 HIM 基础上加判别器训练
        algorithm_class_name = 'PPO_AMP_HIM'

        # ===== AMP 判别器参数（从 quadruped_arm_him_amp 抄）=====
        # style reward 系数：把判别器输出缩放到与 task reward 同量级
        amp_reward_coef = 4.0 * QuadWtwAmpCfg_HIM.sim.dt
        # 专家动捕文件
        amp_motion_files = QuadWtwAmpCfg_HIM.env.amp_motion_files
        # 预采样 (s,s') 对数量
        amp_num_preload_transitions = QuadWtwCfg_HIM.env.num_envs * QuadWtwCfgPPO_HIM.runner.num_steps_per_env * 10
        # 判别器 trunk 隐藏层维度
        amp_discr_hidden_dims = [1024, 512]

        # 不按命令分桶（wtw 命令复杂，全专家混合对抗）
        amp_use_bucketed_experts = False
        amp_expert_label_source = "filename"
        amp_bucket_lin_vel_threshold = 0.10
        amp_bucket_yaw_threshold = 0.20
        amp_bucket_yaw_lin_vel_threshold = 0.20

        # AMP 动态权重（style curriculum）——先关，用固定 lerp
        amp_style_curriculum = True
        amp_style_curriculum_reward_key = "tracking_lin_vel"   
        amp_task_reward_lerp = 0.70      # 任务（含姿态调整）占 55%，style 占 45%
        amp_task_reward_lerp_min = 0.25
        amp_task_reward_lerp_max = 0.95

    class policy(QuadWtwCfgPPO_HIM.policy):
        # 网络结构沿用 wtw（512-256-128）
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(QuadWtwCfgPPO_HIM.algorithm):
        # AMP 判别器损失
        amp_loss = "MSELoss"
        # style reward 映射：在 d=1（专家目标）处取峰值
        style_reward_function = "quad_mapping"
        normalize_style_reward = False
        # policy 侧 replay buffer
        amp_replay_buffer_size = QuadWtwCfg_HIM.env.num_envs * QuadWtwCfgPPO_HIM.runner.num_steps_per_env * 10
        # 判别器学习率（远小于 policy lr）
        disc_lr = 1e-4
