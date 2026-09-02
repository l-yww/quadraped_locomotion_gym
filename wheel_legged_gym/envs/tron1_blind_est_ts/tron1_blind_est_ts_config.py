# use height scans estimator to generate latents , get loss with priv-obs
# But how to set the dimesion of latents????
## TODO: [1].new mlp for heights simply
## TODO: [2].use student encoder to get latents
# ---- added by zsy 2025.4.29
# NOTE: One's hard-learning for thousands of years can't be compared with those who born with golden keys forever ---to Miss Dong    2025.5.3
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

"""
S
"""
class Cowa_Num:
    WLR_index = '3'

class Tron1Cfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        """
            三种类型：[1] Teacher [2] normal policy [3] Student Only Dagger [4] Student PPO + Dagger
                    [1] dagger_on = False, teacher = True
                    [2] dagger_on = False, teacher = False
                    [3] dagger_on = True, dagger_only = True
                    [4] dagger_on = True, dagger_only = False
        """
        dagger_only = False # student
        dagger_on = False # False=teacher, True=student
        teacher = False
        load_run_dagger = "Jul30_18-00-51_noise_teacher_stage2_[pre-train]_[w_noise_0.5]"
        checkpoint_dagger = -1

        # change the observation dim  
        ## est params 
        num_est_prob = 3 # + 1 + 2               # vel_xyz + height + contact_mask

        ## obs frames  
        actor_input_stack = 1  # Actor
        frame_stack = 10 # Encoder
        c_frame_stack = 1 # Critic

        ## obs nums
        if teacher:
            num_single_obs = 35
            single_num_privileged_obs = (num_single_obs)
        else:
            num_single_obs = 31
            single_num_privileged_obs = (num_single_obs) + (num_est_prob)           

        num_normal_obs = 31 # 除去priv_obs的normal_obs
        
        num_observations = int(frame_stack * num_single_obs)  # actor
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs) # critic

        num_actions = 8
        num_envs = 4096
        episode_length_s = 20  # episode length in seconds
        fail_to_terminal_time_s = 0.5

        contact_force_frame = 5 # 0.1s

        dof_vel_use_pos_diff = True        
    
    class safety:
        # safety factors
        pos_limit = 0.95
        vel_limit = 0.95
        torque_limit = 0.95

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/cowa_wheel_legged_deployed/urdf/cowa_4rad_wo_arm_8dof.urdf" 
        name = "tron1"  
        foot_name = "wheel"
        foot_radius = 0.127
        penalize_contacts_on = ["hip", "knee", "battery", "base", "hand_center"]
        terminate_after_contacts_on = ["hip", "knee", "battery", "base", "hand_center"]
        disable_gravity = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False #
        replace_cylinder_with_capsule = True
        fix_base_link = False
        fix_base_link_height = 1.8  # fix the base of the robot at the height
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh' 
        curriculum = True  
        track_test = False 
        add_perlin_noise = False 
        # rough terrain only:
        measure_heights = True
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        terrain_length = 8.  
        terrain_width = 8.  
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        max_init_terrain_level = 0  # starting curriculum state 
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down  
        terrain_proportions = [0., 0., 0., 1., 0., 0, 0]
        # terrain_proportions = [0, 0, 0, 1., 0., 0, 0,0]   #
        restitution = 0.
        measured_points_x = [
            -0.6,
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
            0.6,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        slope_treshold = 0.0 # height below this params is change to 
        en_fix_step_height = False # zsy add
        draw_scan_dots = False

    # TODO
    class noise:
        add_noise = True
        noise_level = 1.5    # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.02 # 2cm
            probability = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.72 + 0.1664]  # x,y,z [m] ##NOTE: 修改 原来是 0.8 + 0.1664
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = True
        rand_init_dof_range = 0.2
        default_joint_angles = {  # target angles when action = 0.0
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 0,
            "left_knee_pitch_joint": 0,
            "left_wheel_joint": 0,
            "right_hip_pitch_joint": 0,
            "right_hip_roll_joint": 0,
            "right_knee_pitch_joint": 0,
            "right_wheel_joint": 0                    
        }


    class control(LeggedRobotCfg.control):
        control_test = False
        control_type = "P"  # P: position, V: velocity, T: torques
        WLR_index = Cowa_Num.WLR_index
        # PD Drive parameters:
        if WLR_index == "1":
            """ No1 """
            stiffness = {"hip": 100.0, "knee": 180, "wheel": 0}  # [N*m/rad]
            damping = {"hip": 5, "knee": 7, "wheel": 3}  # [N*m*s/rad]
        elif WLR_index == "2":
            """ No2 """
            stiffness = {"hip": 110.0, "knee": 210, "wheel": 0}  # [N*m/rad]
            damping = {"hip": 8, "knee": 8, "wheel": 3}  # [N*m*s/rad]
        elif WLR_index == "3":
            """ No3 """
            stiffness = {"hip_roll": 110.0, "hip_pitch": 75.0, "knee": 160, "wheel": 0}  # [N*m/rad]
            damping = {"hip_roll": 5.0, "hip_pitch": 4, "knee": 6., "wheel": 3}  # [N*m*s/rad]

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 50 Hz
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]
 
        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9
        projected_gravity = False

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10                                  # xxw
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 1.  # 0.2

        action_noise = 0.0 # 0.02
        action_delay = 0.0 # 0.1

        """ 机器人的惯性参数 """
        randomize_friction = True             # xxw True
        friction_range = [0.2, 1.6]
        randomize_base_mass = True #
        added_mass_range = [-0.5, 2]              
        randomize_restitution = True           # TODO
        restitution_range = [0.2, 1.5]            
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = True    
        randomize_inertia_range = [0.8, 1.2]

        rand_interval_s = 10
        """ 每个10s,刷新一次下面的随机化 【除了delay】 """
        randomize_motor_strength = True      
        motor_strength_range = [0.8, 1.2]      

        randomize_PD_factor = True #             
        Kp_factor_range = [0.8, 1.2]            
        Kd_factor_range = [0.8, 1.2]   

        # --------------- randomize_motor_offset与randomize_default_dof_pos含义相同，均模拟关节角度的固定误差 ----------------- #
        randomize_motor_offset = False # 目前是使用torque的offset
        default_motor_offset = [0,0,0,0,\
                                0,0,0,0]
        motor_offset_range = [-0.05, 0.05]

        randomize_default_dof_pos = True # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos_range = [-0.05, 0.05]

        # ------------------- 延迟模拟 -------------------------- #
        '维护队列，固定延迟'
        '固定延迟中,timesteps是按照Policy的频率'
        # action传到PD控制器的延迟
        randomize_lag_timesteps = False     
        lag_timesteps = 2       #2~4ms 
        # PD控制器到电机扭矩实际达到torque值的延迟
        randomize_torque_delay = False
        torque_delay_steps = 2
        # 编码器和IMU传回Policy的延迟
        randomize_obs_delay = False 
        obs_delay_steps = 1
        '维护Tensor,随机范围延迟'
        '随机延迟中,timesteps是按照PD的频率'
        # action延迟
        add_lag = True
        randomize_lag_timesteps = True # [1]False，固定取延迟的最大值; [2]True,随机范围
        randomize_lag_timesteps_perstep = True
        lag_timesteps_range = [0, 4] # 4*0.005 = 0.02s
        # 编码器延迟    
        add_dof_lag = False
        randomize_dof_lag_timesteps = False
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 2] # 1~4ms
        # IMU延迟
        add_imu_lag = False 
        randomize_imu_lag_timesteps = False
        randomize_imu_lag_timesteps_perstep = False         
        imu_lag_timesteps_range = [3, 11] # 10~22ms

        # <*!->
        add_heights_lag = False 
        randomize_heights_lag_timesteps = False
        randomize_heights_lag_timesteps_perstep = False         
        heights_lag_timesteps_range = [1, 50] # 2 ~ 100ms NOTE: so big big big ...

        # ------------- 模拟电机的阻尼/摩擦特性 【在扭矩下发处模拟】 ------------------ #
        '在compute torque,模拟静摩擦和阻尼'     #! 不建议打开
        randomize_coulomb_friction = False
        joint_stick_friction_range = [0.1, 0.2]
        joint_coulomb_friction_range = [0.0, 0.0]

        # ------------- 模拟电机的阻尼/摩擦特性 【在仿真器中设置参数】 ------------------ #        
        '修改仿真器的关节物理参数，模拟静摩擦和阻尼'
        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        default_joint_friction = [0.0, 0.0, 0.0, 0.0,\
                                 0.0, 0.0, 0.0, 0.0,]  
        joint_friction_range = [0.8, 1.2]
        joint_1_friction_range = [0.9, 1.1]
        joint_2_friction_range = [0.9, 1.1]
        joint_3_friction_range = [0.9, 1.1]
        joint_4_friction_range = [0.9, 1.1]
        joint_5_friction_range = [0.9, 1.1]
        joint_6_friction_range = [0.9, 1.1]

        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        default_joint_damping = [0, 0, 0.0, 0.0,\
                                 0, 0, 0.0, 0.0,]
        joint_damping_range = [0.8, 1.2]
        joint_1_damping_range = [0.8, 1.2]
        joint_2_damping_range = [0.8, 1.2] 
        joint_3_damping_range = [0.8, 1.2]
        joint_4_damping_range = [0.8, 1.2]
        joint_5_damping_range = [0.8, 1.2]
        joint_6_damping_range = [0.8, 1.2]

        randomize_joint_armature = False 
        randomize_joint_armature_each_joint = False
        default_joint_armature = [0.0, 0.0, 0.0, 0.0,\
                                  0.0, 0.0, 0.0, 0.0,]
        joint_armature_range = [0.8, 1.2]     # Factor
        joint_1_armature_range = [0.95, 1.05]
        joint_2_armature_range = [0.95, 1.05]
        joint_3_armature_range = [0.95, 1.05]
        joint_4_armature_range = [0.95, 1.05]
        joint_5_armature_range = [0.9, 1.1]
        joint_6_armature_range = [0.9, 1.1]

    ## 前馈
    class feedforward:
        use_feedforward = True  # if true, the uff is used to load the robot
        ff_action_scale = 2.0
        use_ref_action_only = False # 只用参考轨迹的action
        trigger_len = 3
        """ annealing """
        use_annealing = False
        start_iter = 6000
        duration = 4000
        """
            左腿先swing 0.6s, 隔offset后,
            右腿再swing 0.6s
        """
        swing_duration = 0.6
        phase_offset = 0.6

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 3
        num_commands = 3
        resampling_time = 10.  # time before command are changed[s]
        stand_still = False # increse the proportion of command "0"
        stand_still_ratio = 0.8
        heading_command = False  # if true: compute ang vel command from heading error
        more_go_forward_stand_still = True # 50% 直走, 10% 停下, 其余的 norm < 0.1则停下
        min_norm = 0.1
        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            height = [0.70 + 0.1664, 0.70 + 0.1665]
            heading = [-3.14159, 3.14159]

    class rewards:
        only_positive_rewards = False 
        clip_single_reward = 1
        clip_reward = 100

        tracking_sigma = 0.1  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_vel_x = 10
        tracking_sigma_ang_vel = 10  
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8

        base_lin_acc_limit = 0.8  
        min_feet_dist = 0.25
        max_feet_dist = 0.35
        max_feet_z_dist = 0.05
        min_feet_z_dist = -0.05
        base_height_target = 0.70 + 0.1664 #0.6 + 0.1664
        target_feet_height = 0.08 #0.04
        target_feet_height_max = 0.14 #0.06
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        max_contact_force = 200  # stage2 = 150; stage1 = 200 
        # Forces above this value are penalized

        class scales:
            ################# termination rewards ##################
            # keep_balance = 1.0 
            ################# task rewards ##################
            " 速度跟踪 "
            tracking_lin_vel_x = 1.2
            tracking_lin_vel_y = 1.0
            tracking_ang_vel = 1.0
            tracking_lin_vel_x_pbrs = 1.0
            tracking_lin_vel_y_pbrs = 0.8
            tracking_ang_vel_pbrs = 0.5
            " 抬脚踏步 "
            feet_contact_number = 2 
            feet_air_time = 2 
            feet_clearance = 2 
            ############## style rewards ######################
            " 脚部姿态 "
            tracking_target_joint_pos = 0.8
            hip_roll_default_pose = -1.0
            nominal_foot_position = 1.0  # 速度小的时候脚尽量落在指定高度上
            feet_distance = -10
            feet_height_limit = -50.0 
            # leg_symmetry = 0.5  # 两只脚尽可能保持对称
            # same_foot_x_position = -5.0 #? stage2 需要开启
            # same_foot_z_position = -100
            " 机体姿态 "
            # base_acc = -0.3 # stage2
            base_height = -20
            orientation = -12.0 # stage2 20
            ############## normalized rewards #####################
            feet_z_contact_forces = -5.0
            lin_vel_z = -0.3 # lin_vel_yz
            ang_vel_xy = -0.01
            # power = -1e-8
            dof_vel = -1e-5
            dof_acc = -2.5e-7
            dof_pos_limits = -2
            # dof_vel_limits = -2
            torques = -1.e-5
            # torque_limits = -2
            action_rate = -0.01
            action_smooth = -0.005 #-0.1  
            collision = -50
            opposite_vel = -40.0
            opposite_wheel_vel = -2.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0 ## NOTE: change big make it obvious in heights
            torques = 0.05
            quat = 1
            forces = 0.1

        clip_observations = 100.
        clip_actions = 100.

class Tron1CfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner_Blind_TS'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        estimator_hidden_dims=[256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        # estimator para
        mlp_learning_rate = 1.e-3
        num_adaptation_module_substeps = 1
        # dagger

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 100000  # number of policy updates        #  xxw

        # logging
        save_interval = 500  # Please check for potential savings every `save_interval` iterations.
        experiment_name = ''
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = ''  # updated from load_run and chkpt

    