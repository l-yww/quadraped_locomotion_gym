from wheel_legged_gym.envs.amp_d1.walking_d1.walking_d1_config import WalkingD1Cfg, WalkingD1CfgPPO
import glob
from wheel_legged_gym.algo.PPO_AMP.symmetry import compute_symmetric_states_d1
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
# Choose one of these:
# MOTION_FILES = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/amp_dataset/d1_walking/*.pkl")
# MOTION_FILES = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/origin_dataset_pkl/*.pkl")
# MOTION_FILES = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/isaacgym_processed/*.pkl")
# MOTION_FILES = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/d1_data_v1/*.pkl")
# MOTION_FILES = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/d1_data_v2/*.pkl")
# MOTION_FILES = glob.glob(WHEEL_LEGGED_GYM_ROOT_DIR + "/resources/d1_data_v3/*.pkl")

class AmpD1Cfg(WalkingD1Cfg):
    class env(WalkingD1Cfg.env):
        # change the observation dim
        # amp_motion_files = MOTION_FILES
        num_actions = 12
        num_commands = 3
        projected_gravity = True # use projected_gravity or [roll, pitch]

        num_single_obs = num_commands + 3 * num_actions # cmd + dof pos + dof vel + action
        if projected_gravity:
            num_single_obs += 6
        else:
            num_single_obs += 5

        frame_stack = 5        # long history
        actor_input_stack = 5   # 输入给actor的
        c_frame_stack = 5
        num_envs = 4096
        
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + 3 + 1 + 4*3 # vel + base_height

        priv_observe_friction = False
        if priv_observe_friction:
            single_num_privileged_obs += 1

        priv_observe_restitution = False
        if priv_observe_restitution:
            single_num_privileged_obs += 1

        priv_observe_payloads = False
        if priv_observe_payloads:
            single_num_privileged_obs += 1

        priv_observe_inertia = False
        if priv_observe_inertia:
            single_num_privileged_obs += 1

        priv_observe_motor_strength = False
        if priv_observe_motor_strength:
            single_num_privileged_obs += num_actions

        priv_observe_motor_offset = False
        if priv_observe_motor_offset:
            single_num_privileged_obs += num_actions

        priv_observe_com_displacement = False
        if priv_observe_com_displacement:
            single_num_privileged_obs += 3

        priv_observe_heightmap = False
        if priv_observe_heightmap:
            single_num_privileged_obs += 7 * 11

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    class init_state(WalkingD1Cfg.init_state):
        # whether to initialize the robot with the reference motion
        reference_state_initialization = True
        reference_state_initialization_prob = 0.7

    class rewards:
        only_positive_rewards = False 
        tracking_sigma_lin_vel = 0.25
        tracking_sigma_ang_vel = 0.25
        max_contact_force = 700  
        soft_contact_force = 50
        soft_dof_pos_limit = (
            0.9  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 0.8
        soft_dof_vel_limits = 0.9
        class scales:
            ################# termination rewards ##################
            termination = -200
            #################   tracking rewards  ##################
            " 速度跟踪 "
            tracking_lin_vel = 5.0
            tracking_ang_vel = 3.0
            #################    style rewards    ######################
            " 机体姿态 "
            touchdown_impact = 0.0
            feet_contact_forces = -2.5e-5
            contact_momentum = -2.5e-5  #-2.5e-5
            foot_slip = -0.5
            dof_vel_stand_still = -0.5
            dof_pos_stand_still = -0.5
            lin_vel_z = -1.0
            ang_vel_xy = -5e-2
            ############## regulation rewards #####################
            dof_acc = -2.5e-7             # 若有wheel需要注意
            torques = -2e-5
            torque_limits = -0.2
            action_rate = -5e-2
            power = -5e-4
            # action_smoothness = -0.1
            collision = -1.0
            dof_pos_limits = -10.0
            dof_vel_limits = -1.0

    class asset(WalkingD1Cfg.asset):
        key_body_names = "foot"

    class terrain(WalkingD1Cfg.terrain):
        mesh_type = 'trimesh'  #plane
        # terrain_proportions = [0.25, 0.25, 0.1, 0., 0., 0.1, 0, 0, 0.1, 0.2] #服务简单地形的配置（除楼梯） 
        terrain_proportions = [0.2, 0.1, 0.15, 0.2, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0]   #多地形


class AmpD1CfgPPO(WalkingD1CfgPPO):
    seed = 10
    runner_class_name = 'AMPRunner'
    class policy(WalkingD1CfgPPO.policy):
        actor_hidden_dims = [512, 256,128]
        critic_hidden_dims = [1024,512,256]

    class algorithm(WalkingD1CfgPPO.algorithm):
        amp_loss = "MSELoss"  #"MSELoss","WassersteinLoss","BCEWithLogitsLoss"
        style_reward_function = "quad_mapping" # can be "quad_mapping" or "wasserstein_mapping" or "log_mapping"
        normalize_style_reward = False
        amp_replay_buffer_size = AmpD1Cfg.env.num_envs * WalkingD1CfgPPO.runner.num_steps_per_env * 10
        disc_lr = 1.e-4
        # Symmetry loss config
        symmetry_cfg = {
            "use_data_augmentation" : True,
            "data_augmentation_func": compute_symmetric_states_d1,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.5,
        }

    class runner(WalkingD1CfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO_AMP'
        max_iterations = 1000001  # number of policy updates        #  xxw
        amp_reward_coef = 4.0 * AmpD1Cfg.control.dt
        # amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = AmpD1Cfg.env.num_envs * WalkingD1CfgPPO.runner.num_steps_per_env * 10
        amp_task_reward_lerp = 0.5
        amp_style_curriculum = True
        amp_task_reward_lerp_min = 0.2
        amp_task_reward_lerp_max = 0.8    #0.2
        amp_style_curriculum_reward_key = "tracking_ang_vel"
        amp_style_curriculum_success_threshold = 0.9
        amp_style_curriculum_fail_threshold = 0.65
        amp_style_curriculum_style_step = 0.005
        amp_style_curriculum_task_step = 0.08   #0.02
        amp_style_curriculum_ema_alpha = 0.1
        amp_style_curriculum_update_interval = 10
        amp_discr_hidden_dims = [1024, 512]

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'amp_d1'
