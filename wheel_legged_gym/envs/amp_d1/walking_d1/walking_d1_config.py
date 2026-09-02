from wheel_legged_gym.envs.quadruped.quadruped_config import QuadCfg, QuadCfgPPO

class WalkingD1Cfg(QuadCfg):

    class commands(QuadCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        min_curriculum = -1.0
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        zero_cmd_prob = 0.2
        class ranges(QuadCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5] # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # min max [rad/s]
    class control(QuadCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint': 160.0}  # [N*m/rad]
        damping = {'joint': 5.0}     # [N*m*s/rad]

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 50Hz
        action_scale = 0.25   #0.25

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = True
        ratio = 0.9
    class asset(QuadCfg.asset):
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_quadruped_arm_v1/urdf/cowa_quadruped_arm_v1_fix_arm.urdf"
        name = "cowa_quadruped"  # actor name
        foot_name = "foot"
        penalize_contacts_on = ["calf","base_link"]
        terminate_after_contacts_on = ["base_link","thigh","calf"]
        joint_friction = [
            0.0054971277713775635, 6.631016731262207e-06, 0.05982851982116699,
            0.007290467619895935, 0.015472427010536194, 0.07318446040153503,
            0.004257142543792725, 0.006278917193412781, 0.04780678451061249,
            0.005522802472114563, 0.0010340213775634766, 0.02277398109436035,
        ]
        joint_damping = [
            1.2516975402832031e-05, 4.172325134277344e-06, 5.364418029785156e-06,
            3.457069396972656e-05, 7.748603820800781e-06, 1.7881393432617188e-06,
            4.172325134277344e-06, 7.152557373046875e-06, 3.5762786865234375e-06,
            5.364418029785156e-06, 2.384185791015625e-06, 2.384185791015625e-06,
        ]
        joint_armature = [
            0.004417330026626587, 2.086162567138672e-07, 0.07663074135780334,
            1.4007091522216797e-06, 3.2782554626464844e-07, 0.07500061392784119,
            4.559755325317383e-06, 1.4007091522216797e-06, 0.07970243692398071,
            7.450580596923828e-07, 3.5762786865234375e-07, 0.05527627468109131,
        ]
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
            tracking_ang_vel = 5.0
            #################    style rewards    ######################
            " 机体姿态 "
            touchdown_impact = 0.0
            feet_contact_forces = -2.5e-5
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

class WalkingD1CfgPPO(QuadCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner'   # OnPolicyRunnerEstimator

    class policy(QuadCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    class algorithm(QuadCfgPPO.algorithm):
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

    class runner(QuadCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'walking_d1'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt
