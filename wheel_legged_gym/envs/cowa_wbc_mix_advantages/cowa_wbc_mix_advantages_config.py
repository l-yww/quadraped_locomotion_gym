import numpy as np
import torch
from wheel_legged_gym.envs.cowa_wbc_mix_advantages.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.utils.helpers import get_joint_names, get_stiffness_damping,\
                                            get_default_joint_friction, get_default_joint_damping, get_default_joint_armature

class Cowa_Num:
    WLR_index = 'v2_16dof'
    DOF = 16

class CowaCfg(LeggedRobotCfg):
    class mode:
        use_net = True

    class env(LeggedRobotCfg.env):

        projected_gravity = True   # [True] projected_gravity; [False] Euler Angle

        # change the observation dim
        frame_stack = 66        # long history的帧数
        short_frame_stack = 5   # short history的帧数
        c_frame_stack = 1       # 输入给critic的帧数
        num_est_prob = 3 + 1 + 3
        num_actions = Cowa_Num.DOF
        episode_length_s = 20

        if projected_gravity:
            num_single_obs = 2 + 3 + 12 + 3*num_actions - 2 + 3 + 3  
        else:
            num_single_obs = 2 + 3 + 12 + 3*num_actions - 2 + 3 + 2  
        
        num_observations = int(frame_stack * num_single_obs)

        if projected_gravity:
            single_num_privileged_obs = num_single_obs + num_est_prob + 4
        else:
            single_num_privileged_obs = num_single_obs + num_est_prob + 4 + 1 

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
        
    class asset(LeggedRobotCfg.asset):
        DOF = Cowa_Num.DOF
        WHEEL_LEGGED_GYM_ROOT_DIR = "{WHEEL_LEGGED_GYM_ROOT_DIR}"
        file = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_v2/cowa_wheel_legged_v2/urdf/wheel_v2_10dof_w_arm.urdf"
        name = "wheel_legged_robotxxxx"  # actor name
        foot_name = "foot"
        wheel_name = "wheel"
        wheel_radius = 0.11
        knee_name = "knee"
        hip_name = "hip"
        penalize_contacts_on = ["hip", "knee", "base", "hand_center","link1", "link2", "link3", "link4", "link5", "link6"]
        terminate_after_contacts_on = ["hip", "knee", "base", "hand_center","link1", "link2", "link3", "link4", "link5", "link6"]
        gripper_name = "link6"
        replace_cylinder_with_capsule = True #False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        fix_base_link = False # fix the base of the robot
        fix_base_link_height = 0  # fix the base of the robot at the height

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # terrain types: [plane ,smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [1, 0.0, 0.0, 0., 0., 0, 0, 0, 0.0, 0.0] #[0.0, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0.0, 0.0]
        slope_treshold = 0.1 # slopes above this threshold will be corrected to vertical surfaces
        measure_heights = False #True
        measured_points_x = [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = True
        rand_init_dof_range = 0.1 # [rad]
        joint_names = get_joint_names(Cowa_Num.DOF, Cowa_Num.WLR_index)
        default_joint_angles = {name: 0.0 for name in joint_names}

    class control(LeggedRobotCfg.control):
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness, damping = get_stiffness_damping(Cowa_Num.DOF, Cowa_Num.WLR_index)

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 100Hz 
        # decimation = 4 # 50Hz
        arm_pos_action_scale = [0.25,0.25,0.25,0.1,0.1,0.1]
        pos_action_scale = 0.25
        vel_action_scale = 2.

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9

        cycle_time = 0.8 #2 #3.0 # ref pos 的 cycle time
        offset = 0.5

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 500Hz 
        # dt = 1/200  # 200Hz 

        body_measure_points = { # transform are related to body frame
            "left_foot": dict(
                x= [i for i in np.arange(-0.11, -0.06, 0.04)],
                y= [-0.12],
                z= [-0.09, -0.045 ,0.0, 0.045, 0.09],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
            "right_foot": dict(
                x= [i for i in np.arange(-0.11, -0.06, 0.04)],
                y= [0.12],
                z= [-0.09, -0.045 ,0.0, 0.045, 0.09],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = False #True  #第一阶段False，第二阶段True

        push_robots = True #use_random
        push_interval_s = 8
        max_push_vel_xy = 0.5  # 0.2
        max_push_ang_vel = 0.2 # 0.1

        action_noise = 0.0 # 0.02
        action_delay = 0. # 0.1

        rand_interval_s = 10    # 每隔rand_interval_s会重置一次随机化
        randomize_rigids_after_start = False     #控制link的friction和restituion的随机化开关
        randomize_friction = use_random             # xxw True
        friction_range = [0.2, 1.6]
        
        randomize_restitution = use_random           
        restitution_range = [0, 1.0]           

        # --------------- 随机化 base_link 质量 & 转动惯量 ----------------- #
        randomize_base_mass = use_random #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-2, 5]                  # 加负载

        randomize_inertia = use_random    
        randomize_inertia_range = [0.8, 1.2]

        # --------------- 随机化 质心位置 ----------------- #
        randomize_com_displacement = use_random      
        com_displacement_range = [-0.08, 0.08]  # base link com的随机化范围
        randomize_each_link = False
        link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)
        
        # --------------- 随机化电机能力 ----------------- #
        randomize_motor_strength = use_random      
        motor_strength_range = [0.8, 1.2]      

        randomize_PD_factor = use_random #             
        Kp_factor_range = [0.8, 1.2]            
        Kd_factor_range = [0.8, 1.2]

        # --------------- randomize_motor_offset与randomize_default_dof_pos含义相同，均模拟关节角度的固定误差 ----------------- #
        randomize_motor_offset = use_random # 目前是使用torque的offset
        motor_offset_range = [-0.05, 0.05] # 仅针对No2，其他的轮足均无该问题

        randomize_default_dof_pos = False # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos_range = [-0.05, 0.05]

        # ------------------- 延迟模拟 -------------------------- #
        '维护Tensor,随机范围延迟'
        '随机延迟中,timesteps是按照PD的频率'
        # action延迟
        add_action_lag = True and use_random
        randomize_lag_timesteps = True and use_random
        randomize_lag_timesteps_perstep = True and use_random
        lag_timesteps_range = [3, 11]       # 2ms * steps
        # 编码器延迟
        add_dof_lag = True and use_random
        randomize_dof_lag_timesteps = True and use_random # [1]False，固定取延迟的最大值; [2]True,随机范围
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 2]    # 2ms * steps
        # IMU延迟
        add_imu_lag = True and use_random
        randomize_imu_lag_timesteps = True and use_random
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [0, 2]    # 2ms * steps

        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #       
        WLR_index = Cowa_Num.WLR_index
        DOF = Cowa_Num.DOF

        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        default_joint_friction = get_default_joint_friction(WLR_index, DOF)
        randomize_joint_friction = use_random
        joint_friction_range = [0.8, 1.2]
        randomize_joint_friction_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_friction_range'] = [0.8, 1.2]

        default_joint_damping = get_default_joint_damping(WLR_index, DOF)
        randomize_joint_damping = use_random
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_damping_range'] = [0.8, 1.2]

        default_joint_armature = get_default_joint_armature(WLR_index, DOF)
        randomize_joint_armature = use_random 
        joint_armature_range = [0.8, 1.2]
        randomize_joint_armature_each_joint = use_random
        for i in range(DOF):
            vars()[f'joint_{i+1}_armature_range'] = [0.8, 1.2]

    class viewer:
        ref_env = 0
        pos = [0, -3, 1]  # [m] [10, 0, 6]   xxw
        lookat = [10, 5, 1.5]  # [m]
        draw_commands = False # for debugger
        draw_base_com = False # for view base com
        debug_viz = True #False # for view scan dot
        class commands:
            color = [0.1, 0.8, 0.1] # rgb
            size = 0.5

        draw_volume_sample_points = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 5  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        num_arm_commands = 12  # 机械臂命令维度（位置3 + 姿态9）
        T_traj = [2.0,2.5]  # 轨迹周期范围（秒）
        num_commands_force = 3  # 力命令维度（x,y,z三个方向）

        push_gripper_stators = False #True  # 是否启用夹爪扰动
        push_gripper_interval_s_cmd = [3.5, 9.0]
        push_gripper_duration_s_cmd = [1.0, 3.0]
        gripper_forced_prob_cmd = 0.8
        push_gripper_interval_s_ext = [3.5, 9.0]
        push_gripper_duration_s_ext = [1.0, 3.0]
        gripper_forced_prob_ext = 0.8

        randomize_gripper_force_gains = False #True
        gripper_force_kp_range = [200., 200.]
        gripper_force_kd_range = [3.0, 3.0]
        gripper_prop_kd = 0.1

        max_push_force_xyz_gripper_cmd = [-5, 5]
        max_push_force_xyz_gripper_ext = [-5, 5]  # 外部扰动力范围（N）
        max_force_cmd_range = 30
        max_force_ext_range = 30  # 外部力最大范围
        
        settling_time_force_gripper_s = 1.0   # 力稳定时间（秒）
        force_start_step = 15000.0  
        tracking_ee_start_step = 3000

        class ranges:
            lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0, 0]  # min max [rad/s]
            height = [0.37, 0.37]
            heading = [-3.14, 3.14]

            pos_l = [0.6, 0.8]
            pos_p = [0.1*torch.pi, 0.3*torch.pi]
            pos_y = [-torch.pi/ 3, torch.pi/3]

            roll = [-torch.pi * 0.3, torch.pi * 0.3]
            pitch = [0.5*torch.pi, torch.pi]
            yaw = [-torch.pi * 0.3, torch.pi * 0.3]

    class rewards:
        only_positive_rewards = False 
        tracking_sigma_lin_vel = 20
        tracking_sigma_ang_vel = 20
        target_feet_height_min = 0.005 #0.02
        target_feet_height_max = 0.03
        clearance_height_target = 0.05
        max_contact_force = 500  
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_torque_limit = 0.95
        soft_dof_vel_limits = 0.95
        clip_single_reward = 1
        min_wheel_dist = 0.637 #0.64
        max_wheel_dist = 0.643 #0.645
        min_feet_dist = 0.44 #0.48
        max_feet_dist = 0.54
        max_collision_xy_force_threshold = 50

        tracking_ee_sigma = 1.0
        tracking_ee_pos_sigma = 1.0 #0.5
        tracking_ee_orn_sigma = 0.8 


        class scales:
            alive = 0.1
            base_height = -1 
            orientation = -2 
            feet_distance = -0.5 #-1
            default_hip_roll = -0.5 #-1
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            dof_vel = -5e-5             
            dof_acc = -5e-7            
            torques = -1e-6 
            wheel_vel = -0.01 #-0.1
            power = -2e-5
            action_rate = -0.1 
            action_smoothness = -0.25 #-0.1 
            collision = -2.0
            dof_pos_limits = -10 #-0.1
            stand_still = -0.2 #-0.4

        class arm_scales:
            
            tracking_ee_combine = 10

    class noise:
        add_noise = True
        noise_level = 1    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.0
            ang_vel = 0.1
            lin_vel = 0.1
            gravity = 0.05
            quat = 0.05
            height_measurements = 0.1

    class normalization:
        class obs_scales:
            lin_vel = 10.0
            ang_vel = 2.
            dof_pos = 1.0
            dof_vel = 0.05  #? 感觉有些太小
            height_measurements = 5.0
            quat = 1
            gravity = 1
            forces = 0.1
            ee_orn_quat = 1.0
            ee_sphe_radius_cmd = 0.5   # 0.2 - 0.7 
            ee_sphe_pitch_cmd = 1.0    # -1.3 , 1.3 
            ee_sphe_yaw_cmd = 1.3
            ee_force = 0.01

        clip_observations = 100.
        clip_actions = 20.

class CowaCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_DH_Smooth_Mix'   # OnPolicyRunnerEstimator

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # short_history estimator
        estimator_hidden_dims=[128, 64]
        # long history type
        long_history_type = 'cnn' # 'cnn' or 'mlp'
        # for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   # long history output dim
        in_channels = CowaCfg.env.frame_stack
        # for long_history mlp only
        long_history_hidden_dims = [1024, 512, 256]

    class algorithm(LeggedRobotCfgPPO.algorithm):
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

        # train net sin 需要注释
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0 #0.1 #对于踏步运动，第一阶段0.05或者0.1均可，效果都还可以

        # use_flip = True
        # symmetry_scale = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_DH_Smooth_Mix' #'ActorCritic_DH'
        algorithm_class_name = 'PPO_DH_Smooth_Mix' #'PPO_DH'
        num_steps_per_env = 48  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'cowa'
        run_name = ''
        # Load and resume
        resume = False
        load_run = ""  # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = '/home/cowa'  # updated from load_run and chkpt