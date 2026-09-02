import torch
from isaacgym.torch_utils import quat_rotate_inverse

from ..quadruped_arm_him.quadruped_arm_him_env import QuadHIMEnv
from ..amp_d1.legged_robot_amp import LeggedRobotAMP

class QuadHIMAmpEnv(QuadHIMEnv,LeggedRobotAMP):
    """HIM + AMP 合并环境。

    MRO: QuadHIMAmpEnv → QuadHIMEnv → QuadEnv → LeggedRobotAMP → LeggedRobot → BaseTask

    方法来源 (按 MRO 解析):
    - compute_privileged_observations / compute_observations ← QuadHIMEnv
      (HIM 特权观测: 物理量 + heightmap + estimator GT)
    - step() / reset_idx() / get_amp_observations() ← LeggedRobotAMP
      (AMP 功能: 判别器 obs、终止态 terminal_amp_states、参考姿态初始化)
    - _create_envs() / _init_buffers() ← LeggedRobotAMP
      (初始化 amp_loader 和 key_body_indices)

    关于 reward:
    - env 内部 reward 函数与系数主要继承自 QuadHIMEnv (QuadCfg_HIM.rewards.scales)。
      本环境额外补一个后退/原地转向抬腿项,避免这些命令通过贴地倒脚拿速度奖励。
    - AMP 不修改 env reward,仅在 runner (AMPRunner_HIM) 层叠加 style reward:
        r = (1 - lerp) * style_reward + lerp * task_reward
      其中 task_reward 是 env.step() 返回的原生 reward,style_reward 来自判别器。
    """


