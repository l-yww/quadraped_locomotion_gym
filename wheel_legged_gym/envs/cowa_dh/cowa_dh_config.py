from wheel_legged_gym.envs.cowa.cowa_config import CowaCfg, CowaCfgPPO, Cowa_Num
from wheel_legged_gym.utils.helpers import get_joint_names, get_stiffness_damping,\
                                            get_default_joint_friction, get_default_joint_damping, get_default_joint_armature

class CowaCfg_DH(CowaCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(CowaCfg.env):
        # change the observation dim
        frame_stack = 66        # long history的帧数
        short_frame_stack = 5   # short history的帧数
        c_frame_stack = 1       # 输入给critic的帧数
        num_est_prob = 3 + 1    # vel_xyz, height预测的信息的总维度
        
        num_actions = Cowa_Num.DOF
        num_single_obs = 3 + 3*num_actions - 2 + 6 -1  # cmd + dof pos&vel + action + imu
        
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + num_est_prob + 1

        projected_gravity = False   # [True] projected_gravity; [False] Euler Angle

        priv_observe_friction = False
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

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

class CowaCfgPPO_DH(CowaCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_DH_Smooth'  #'OnPolicyRunner_DH'   # OnPolicyRunnerEstimator

    class policy(CowaCfgPPO.policy):
        # short_history estimator
        estimator_hidden_dims=[128, 64]
        # long history type
        long_history_type = 'cnn' # 'cnn' or 'mlp'
        # for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   # long history output dim
        in_channels = CowaCfg_DH.env.frame_stack
        # for long_history mlp only
        long_history_hidden_dims = [1024, 512, 256]

    class algorithm(CowaCfgPPO.algorithm):
        # estimator para
        mlp_learning_rate = 5.e-4
        num_adaptation_module_substeps = 1

        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.02 # 0.01和0.02都可以

    class runner(CowaCfgPPO.runner):
        policy_class_name = 'ActorCritic_DH_Smooth' #'ActorCritic_DH'
        algorithm_class_name = 'PPO_DH_Smooth' #'PPO_DH'
        experiment_name = 'cowa_dh'
        run_name = 'cowa_dh_test'
        # Load and resume
        resume = False
        load_run = ""  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt