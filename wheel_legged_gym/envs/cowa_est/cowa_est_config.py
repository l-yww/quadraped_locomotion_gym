from wheel_legged_gym.envs.cowa.cowa_config import CowaCfg, CowaCfgPPO, Cowa_Num
from wheel_legged_gym.utils.helpers import get_joint_names, get_stiffness_damping,\
                                            get_default_joint_friction, get_default_joint_damping, get_default_joint_armature

class CowaCfg_EST(CowaCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(CowaCfg.env):
        # change the observation dim
        frame_stack = 5        # long history
        short_frame_stack = 5   # short history
        actor_input_stack = 5   # 输入给actor的
        num_est_prob = 3 + 1    # vel_xyz, height, com_shift 预测的信息的总维度
        c_frame_stack = 1
        num_actions = Cowa_Num.DOF
        num_envs = 4096

        num_single_obs = 3 + 3*num_actions - 2 + 6 # cmd + dof pos&vel + action + imu
        projected_gravity = False

        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs + 3 + 1 # vel 和 height

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
    
    class asset(CowaCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_deployed/urdf/cowa_4rad_wo_arm_{DOF}dof.urdf" 

    class domain_rand(CowaCfg.domain_rand):
        use_random = True

        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #       
        WLR_index = Cowa_Num.WLR_index
        DOF = Cowa_Num.DOF

        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        default_joint_friction = get_default_joint_friction(WLR_index, DOF)
        randomize_joint_friction = use_random
        joint_friction_range = [0.2, 1.2]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.2, 1.2]

        default_joint_damping = get_default_joint_damping(WLR_index, DOF)
        randomize_joint_damping = use_random
        joint_damping_range = [0.2, 1.2]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.2, 1.2]

        default_joint_armature = get_default_joint_armature(WLR_index, DOF)
        randomize_joint_armature = use_random 
        joint_armature_range = [0.2, 1.2]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.2, 1.2]

class CowaCfgPPO_EST(CowaCfgPPO):
    runner_class_name = 'OnPolicyRunnerEstimator'   # OnPolicyRunnerEstimator

    class policy(CowaCfgPPO.policy):
        # estimator para
        estimator_hidden_dims=[128, 64]

    class algorithm(CowaCfgPPO.algorithm):
        # estimator para
        mlp_learning_rate = 5.e-4
        num_adaptation_module_substeps = 1

    class runner(CowaCfgPPO.runner):
        policy_class_name = 'ActorCritic_Estimator'
        algorithm_class_name = 'PPO_Estimator'
        experiment_name = 'cowa_est'
        run_name = 'No1-est-plane-stage2-h5-500Hz-[new_joint_params]-[low_pd_params]-[lgx_reward_scales]'
        # Load and resume
        resume = False
        load_run = ""  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt