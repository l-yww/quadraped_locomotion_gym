# use height scans estimator to generate latents , get loss with priv-obs
# But how to set the dimesion of latents????
## TODO: [1].new mlp for heights simply
## TODO: [2].use student encoder to get latents
# ---- added by zsy 2025.4.29
# NOTE: One's hard-learning for thousands of years can't be compared with those who born with golden keys forever ---to Miss Dong    2025.5.3
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CowaCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        # height_scan_points num : num_height_scan_input
        num_height_scan_input = 441 # 289
        num_height_scan_output = 16
        # change the observation dim  
        ## est params 
        num_est_prob = 3 + 1 + 6             # vel_xyz + height + feet-mid com-projected distance 
        actor_input_stack = 5

        contact_force_frame = 10

        ## obs frames  
        frame_stack = 5
        c_frame_stack = 3
        ## obs nums
        num_single_obs = 25
        num_observations = int(frame_stack * num_single_obs)  

        single_num_privileged_obs = num_single_obs + 12 + 1 + 3 + 6 + 6 + 6 + (num_height_scan_input + num_est_prob) + 6
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        num_actions = 6
        num_envs = 4096
        episode_length_s = 20  # episode length in seconds
        use_ref_actions = False
        fail_to_terminal_time_s = 0.2


        # privileged obs 
        # priv_observe_friction = False  # 1
        # priv_observe_base_mass = False # 1
        # priv_observe_restitution = False #1
        # priv_observe_com_displacement = False    # 3
        
        # priv_observe_motor_strength = False  # 6
        # priv_observe_motor_offset = False    # 6 
        # priv_observe_gravity = False    # 3
        # priv_observe_measure_heights = False # 187

    class safety:
        # safety factors
        pos_limit = 1
        vel_limit = 1
        torque_limit = 1    #0.85 xxx 1
        # acc_limit = 0.6
        # dof_acc_limits_ratio = 6

    class asset(LeggedRobotCfg.asset):
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged_withoutknee.urdf"
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged.urdf"
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged/urdf/cowa_wheel_legged_rea.urdf"    
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_deployed/urdf/cowa_jiaozhun_4rad_no_collision_No1_new.urdf"
        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_deployed/urdf/cowa_jiaozhun_2rad_no_collision.urdf"     
        name = "cowa_robot"  
        offset = 0.
        l1 = 0.25
        l2 = 0.25
        foot_name = "wheel"
        penalize_contacts_on = ["base", "battery", "hip", "knee"] 
        terminate_after_contacts_on = ["base", "hip", "knee", "battery"]

        disable_gravity = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False #
        replace_cylinder_with_capsule = False
        fix_base_link = False
        fix_base_link_height = 1.8  # fix the base of the robot at the height

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        en_fix_step_height = False
        mesh_type = 'trimesh' 
        curriculum = True  
        track_test = False 
        add_perlin_noise = False 
        # rough terrain only:
        measure_heights = True
        static_friction = 0.8
        dynamic_friction = 0.8
        terrain_length = 8.  
        terrain_width = 8.  
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 5  # number of terrain cols (types)
        max_init_terrain_level = 0  # starting curriculum state 
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down  
        terrain_proportions = [0., 0., 0., 1., 0., 0, 0]
        # terrain_proportions = [0, 0, 0, 1., 0., 0, 0,0]   #
        restitution = 0.5
        measured_points_x = [
            -0.5,
            -0.45,
            -0.4,
            -0.35,
            -0.3,
            -0.25,
            -0.2,
            -0.15,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
        ]  # 0.6m x 1m rectangle (without center line) 
        measured_points_y = [-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        # measured_points_y = [-0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        slope_treshold = 0. # height below this params is change to 
        step_height_buff = None
        draw_scan_dots = False

    # TODO
    class noise:
        add_noise = True
        noise_level = 1    # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5 #0.5
            ang_vel = 0.1
            lin_vel = 0.1
            gravity = 0.05
            quat = 0.1
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = False
        default_joint_angles = {  # target angles when action = 0.0
            "left_hip_pitch_joint": 0,
            "left_knee_pitch_joint": 0,
            "left_wheel_joint": 0,
            "right_hip_pitch_joint": 0,
            "right_knee_pitch_joint": 0,
            "right_wheel_joint": 0                      
        }


    # TODO control_type,pos_action_scale,vel_action_scale
    class control(LeggedRobotCfg.control):
        control_test = False
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {"hip": 110.0, "knee": 273, "wheel": 0}  # [N*m/rad]
        damping = {"hip": 8, "knee": 10.4, "wheel": 3}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5  # 100Hz
        pos_action_scale = 0.5
        vel_action_scale = 10.0

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9
        projected_gravity = False
        # feedforward_force = 6 # 42.46kg / 2

    class sim(LeggedRobotCfg.sim):
        web_vis = False
        port = 6001      
        web_vis_envs = 1 
        keep_default_viewer = False

        dt = 0.002  # 200 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10                                  # xxw
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):

        # TODO randomize_default_dof_pos, randomize_action_delay
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.1  # 0.2
        max_push_ang_vel = 0.1

        action_noise = 0.0 # 0.02
        action_delay = 0.0 # 0.1

        rand_interval_s = 10    # 每隔rand_interval_s会重置一次随机化
        randomize_rigids_after_start = True     #控制link的friction和restituion的随机化开关
        randomize_friction = True             # xxw True
        friction_range = [0.1, 2]
        
        randomize_restitution = True           
        restitution_range = [0, 1.0]           

        # --------------- 随机化 base_link 质量 & 转动惯量 ----------------- #
        randomize_base_mass = True #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-5, 5]                  # 加负载

        randomize_inertia = True    
        randomize_inertia_range = [0.8, 1.2]

        # --------------- 随机化 质心位置 ----------------- #
        randomize_com_displacement = True      
        com_displacement_range = [-0.05, 0.05]  # base link com的随机化范围
        randomize_each_link = False
        link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)
        
        # --------------- 随机化电机能力 ----------------- #
        randomize_motor_strength = True      
        motor_strength_range = [0.9, 1.1]      

        randomize_PD_factor = True #             
        Kp_factor_range = [0.9, 1.1]            
        Kd_factor_range = [0.9, 1.1]

        # --------------- randomize_motor_offset与randomize_default_dof_pos含义相同，均模拟关节角度的固定误差 ----------------- #
        randomize_motor_offset = True # 目前是使用torque的offset
        default_motor_offset = [0, 0.0, 0,\
                                0, 0.0, 0]
        motor_offset_range = [-0.03, 0.03]

        randomize_default_dof_pos = False # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos_range = [-0.1, 0.1]

        # ------------------- 延迟模拟 -------------------------- #
        '维护队列，固定延迟'
        '固定延迟中,timesteps是按照Policy的频率'
        # action传到PD控制器的延迟
        fixed_action_delay = False
        action_delay_steps = 2       # 2~4ms
        # PD控制器到电机扭矩实际达到torque值的延迟
        fixed_torque_delay = False
        torque_delay_steps = 2
        # 编码器和IMU传回Policy的延迟
        fixed_obs_delay = False
        obs_delay_steps = 1

        '维护Tensor,随机范围延迟'
        '随机延迟中,timesteps是按照PD的频率'
        # action延迟
        add_action_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [3, 11]   # 22 # 44
        # 编码器延迟
        add_dof_lag = True
        randomize_dof_lag_timesteps = True # [1]False，固定取延迟的最大值; [2]True,随机范围
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 2] # 1~4ms
        # IMU延迟
        add_imu_lag = True 
        randomize_imu_lag_timesteps = False
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [3, 11] # 实际10~22ms 22

        # ------------- 模拟电机的阻尼/摩擦特性 【在扭矩下发处模拟】 ------------------ #
        '在compute torque,模拟静摩擦和阻尼'
        randomize_motor_friction = False
        joint_stick_friction_range = [0, 2.]
        joint_coulomb_friction_range = [0.0, 0.0]
        
        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #        
        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        # default_joint_friction = [0.01, 0.01, 0.2, 0.01, 0.01, 0.02]
        # default_joint_friction = [0.13, 0.002, 0.000003, \    # old 
        #                          0.13, 0.002, 0.000003, ]
        default_joint_friction = [0., 0., 0., \
                                 0., 0., 0., ]  ## 韩卓
        randomize_joint_friction = False
        joint_friction_range = [0.5, 1.5]
        randomize_joint_friction_each_joint = False
        joint_1_friction_range = [0.5, 1.5]
        joint_2_friction_range = [0.5, 1.5]
        joint_3_friction_range = [0.5, 1.5]
        joint_4_friction_range = [0.5, 1.5]
        joint_5_friction_range = [0.5, 1.5]
        joint_6_friction_range = [0.5, 1.5]

        default_joint_damping = [6, 8, 0.15, 6, 8, 0.15] # wh
        # default_joint_damping = [12, 12, 0.03, 12, 12, 0.03] #  lrk
        # default_joint_damping = [2.15, 4, 0.0003, \   # old
        #                          2.15, 4, 0.0003, ]
        randomize_joint_damping = True
        joint_damping_range = [0.8, 1.2]
        randomize_joint_damping_each_joint = True
        joint_1_damping_range = [0.8, 1.2]
        joint_2_damping_range = [0.8, 1.2] 
        joint_3_damping_range = [0.8, 1.2]
        joint_4_damping_range = [0.8, 1.2]
        joint_5_damping_range = [0.8, 1.2]
        joint_6_damping_range = [0.8, 1.2]

        default_joint_armature = [0, 0, 0, 0, 0, 0]
        # default_joint_armature = [0.5, 0.3, 0.12,\
        #                           0.5, 0.3, 0.12] ## lrk
        randomize_joint_armature = False 
        joint_armature_range = [0.5, 1.5]
        randomize_joint_armature_each_joint = False
        joint_1_armature_range = [0.5, 1.5]
        joint_2_armature_range = [0.5, 1.5]
        joint_3_armature_range = [0.5, 1.5]
        joint_4_armature_range = [0.5, 1.5]
        joint_5_armature_range = [0.5, 1.5]
        joint_6_armature_range = [0.5, 1.5]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        stand_still = False # increse the proportion of command "0"
        stand_still_ratio = 0.8
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            height = [0.33, 0.331]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.36         
        only_positive_rewards = False 
        # tracking_sigma = 4    # vel = 0.5 对应 20; vel = 1 对应 4; vel = 1.5 对应 2; vel = 2 对应 1; vel = 3 对应 0.5
        tracking_sigma = 4  # tracking reward = exp(-error^2 * sigma)
        tracking_sigma_lin_vel = 20
        tracking_sigma_ang_vel = 20
        tracking_vel_enhance = False
        tracking_vel_hard = False   
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        max_contact_force = 100  # Forces above this value are penalized xxx 1400
        clip_single_reward = 1
        
        min_feet_dist = 0.64
        max_feet_dist = 0.645

        base_lin_acc_limit = 0.8  

        ### NOTE: 
        max_feet_z_dist = 0.1
        min_feet_z_dist = -0.1

        # feet_ahead_distance = 0.15


        class scales:
            ###################################
            feet_distance = 0.2
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0
            base_height = 1.0  # low
            # base_height_error = -3
            nominal_state = -0.05
            lin_vel_z = -0.1e-3
            ang_vel_xy = -0.05
            orientation = -10.0
            ###################################
            dof_vel = -5e-4
            dof_acc = -5e-7
            torques = -1e-7
            # wheel_acc = -1e-7
            # torques = -1e-30
            torque_limits = -0.05
            # dof_vel_limits = -0.05
            power = -1e-8
            action_rate = -0.2
            action_smoothness = -0.5 #-0.1  
            # base_lin_acc_limit = -2
            # base_lin_acc = -0.01
            base_acc = -1e-3
            ##################################
            collision = -200.0
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            stand_still_vel_penality = -20.0

            same_foot_z_position = 0.6 # 1.0
            feet_xy_contact_forces = -0.2   # -0.01  
            # feet_air_time = 0.5
            # com_projected_test = 0.4
            # TODO: 0530
            # foot_ahead_body_left = 1.0
            # foot_ahead_body_right = 1.0
            # high_vel_penalize = -1.0



    class normalization:
        class obs_scales:
            lin_vel = 10.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 10.0 # 5.0 方法权重
            torques = 0.05
            quat = 1
            forces = 1.0

        clip_observations = 100.
        clip_actions = 100.




class CowaCfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner_Lidar_Estimator'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 32]
        critic_hidden_dims = [256, 128, 64]
        height_scan_encoder_dims = [128, 64]
        estimator_hidden_dims=[128, 64]

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
        # estimator para
        mlp_learning_rate = 5.0e-4
        num_adaptation_module_substeps = 1

    class runner:
        policy_class_name = 'ActorCritic_Estimator'
        algorithm_class_name = 'PPO_Estimator'
        num_steps_per_env = 48  # per iteration
        max_iterations = 100000  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'wtf'
        run_name = 'wtf'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = 'wtf'  # updated from load_run and chkpt

