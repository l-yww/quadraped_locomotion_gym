# quadruped_arm_him_amp + actor 前视高程图观测。
# 继承链: QuadCfg_HIM_AMP_Heightmap → QuadCfg_HIM_AMP → QuadCfg_HIM → LeggedRobotCfg
#          QuadCfgPPO_HIM_AMP_Heightmap → QuadCfgPPO_HIM_AMP → QuadCfgPPO_HIM
#
# 本文件只在 HIM-AMP 基础上增加 actor 前视高程图；机器人、地形、控制、
# 随机化和奖励均从 HIM-AMP 继承。高程图采样位置沿用 WTW 的标定范围。
#
# 注册名: 'quadruped_arm_him_amp_heightmap' (envs/__init__.py)

from ..quadruped_arm_him_amp.quadruped_arm_him_amp_config import (
    QuadCfg_HIM_AMP,
    QuadCfgPPO_HIM_AMP,
)

# Short aliases keep the nested configuration readable without changing the
# inheritance or any effective values.
_HIM_AMP = QuadCfg_HIM_AMP
_HIM_AMP_PPO = QuadCfgPPO_HIM_AMP

# WTW-aligned actor scan: base_link origin, forward x=0.5..1.0 m,
# lateral y=-0.4..0.4 m, 0.1 m resolution (6 x 9 = 54 points).
_HEIGHT_POINTS_X = [round(0.5 + 0.1 * i, 1) for i in range(6)]
_HEIGHT_POINTS_Y = [round(-0.4 + 0.1 * i, 1) for i in range(9)]
_NUM_HEIGHT_POINTS = len(_HEIGHT_POINTS_X) * len(_HEIGHT_POINTS_Y)


class QuadCfg_HIM_AMP_Heightmap(_HIM_AMP):

    class env(_HIM_AMP.env):
        # 范围、分辨率和点数沿用 WTW height-scan 标定，但不继承 WTW 配置。
        actor_measured_points_x = list(_HEIGHT_POINTS_X)
        actor_measured_points_y = list(_HEIGHT_POINTS_Y)
        num_height_scan_points = _NUM_HEIGHT_POINTS
        num_actor_height_points = _NUM_HEIGHT_POINTS

        # 把前视高程图并入 actor 每帧观测。
        # num_single_obs: 本体(45) + 高程图(54) = 99。
        # num_body_dim: 纯本体维度(45), 给 estimator 的 target_encoder 做对比目标用, 不含高程图。
        _parent_single = _HIM_AMP.env.num_single_obs
        num_body_dim = _parent_single
        num_single_obs = num_body_dim + num_actor_height_points
        num_observations = _HIM_AMP.env.frame_stack * num_single_obs

    class domain_rand(_HIM_AMP.domain_rand):

        height_update_hz = 10.0  # 高程图刷新频率

        # 每个 episode 给 Actor 高程图增加固定的测量零偏 [m]
        randomize_height_offset = False
        height_offset_range = [-0.05, 0.05]

        # 绕 base_link 的水平 yaw 测量误差 [rad]
        randomize_height_yaw = False
        height_yaw_noise_range = [-0.2, 0.2]

        # 模拟传感器安装的 pitch/roll 倾斜偏差 [m 等效高度偏差]
        randomize_height_roll_pitch = False
        height_pitch_bias_range = [-0.02, 0.02]
        height_roll_bias_range = [-0.02, 0.02]

        # 扫描值额外噪声；Gaussian 和 spike 必须同时受 add_height_noise 控制
        add_height_noise = False
        add_height_gaussian_noise = False
        height_gaussian_noise = 0.03            # Gaussian 噪声标准差 [m]
        add_height_spike_noise = False
        height_spike_noise_range = [0.05, 0.2]  # spike 幅值范围 [m]

        # 以该概率重复上一帧扫描，模拟传感器丢帧/延迟
        height_repeat_probability = 0.0

    class noise(_HIM_AMP.noise):
        # Controls Gaussian noise added to the actor's height-map channels
        # after normalization.  0.05 * height_measurements(5.0) is about
        # 5 cm in the unscaled height value.  The environment also clips the
        # resulting noise to this physical bound.
        height_observation_noise = False
        height_observation_noise_max = 0.05

        class noise_scales(_HIM_AMP.noise.noise_scales):
            height_measurements = 0.05 # acitor 5cm的高程图噪声

class QuadCfgPPO_HIM_AMP_Heightmap(_HIM_AMP_PPO):

    class runner(_HIM_AMP_PPO.runner):
        # 换 Heightmap 版策略类: estimator 的 target_encoder 用 num_body_dim, 不被高程图污染。
        # num_body_dim 经 policy_cfg -> ActorCritic_HIM_Heightmap(**kwargs) 透传。
        policy_class_name = 'ActorCritic_HIM_Heightmap'
        experiment_name = 'quadruped_arm_him_amp_heightmap'

    class policy(_HIM_AMP_PPO.policy):
        # num_body_dim 透传给 ActorCritic_HIM_Heightmap -> Estimator_HIM_Heightmap。
        # = 父类本体单帧维度(45), 即不含前视高程图的本体状态维度。
        num_body_dim = _HIM_AMP.env.num_single_obs   # 45

    class algorithm(_HIM_AMP_PPO.algorithm):
        # Actor 单帧包含 54 维高程图，d1 对称函数只支持 45 维本体帧。
        symmetry_cfg = None
        symmetry_scale = 0.0
