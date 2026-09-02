import os
import numpy as np
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import time
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from wheel_legged_gym.utils import Logger_quad_wtw

# import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import json
import copy
from wheel_legged_gym.utils.Integrated import Integrated_EST_policy, Integrated_RMA_policy, Integrated_HIM_policy, Integrated_ROA_policy, Integrated_PPO_policy, Integrated_TS_policy, Integrated_EST_Plane_policy, Integrated_DH_policy, Integrated_Prop_policy, Integrated_vallina_policy, Integrated_MoE_CTS_policy

# ==================== Global keyboard command values ====================
COMMAND_VX = 0.0
COMMAND_VY = 0.0
COMMAND_YAW = 0.0
COMMAND_BODY_HEIGHT = 0.4
COMMAND_GAIT_FREQ = 2.0
COMMAND_GAIT_DURATION = 0.5
COMMAND_FOOT_SWING = 0.1
COMMAND_BODY_PITCH = 0.0
COMMAND_BODY_ROLL = 0.0


def setup_keyboard(env):
    """Subscribe keyboard events for command control in play mode."""
    if env.headless or env.viewer is None:
        return

    key_map = {
        gymapi.KEY_W: "CMD_VX_UP",
        gymapi.KEY_S: "CMD_VX_DOWN",
        gymapi.KEY_A: "CMD_VY_LEFT",
        gymapi.KEY_D: "CMD_VY_RIGHT",
        gymapi.KEY_Q: "CMD_YAW_LEFT",
        gymapi.KEY_E: "CMD_YAW_RIGHT",
        gymapi.KEY_R: "CMD_HEIGHT_UP",
        gymapi.KEY_F: "CMD_HEIGHT_DOWN",
        gymapi.KEY_T: "CMD_FREQ_UP",
        gymapi.KEY_G: "CMD_FREQ_DOWN",
        gymapi.KEY_Z: "CMD_RESET",
        gymapi.KEY_X: "CMD_PITCH_UP",
        gymapi.KEY_C: "CMD_PITCH_DOWN",
        gymapi.KEY_1: "CMD_FOOT_SWING_UP",
        gymapi.KEY_2: "CMD_FOOT_SWING_DOWN",
        gymapi.KEY_9: "CMD_GAIT_DURATION_UP",
        gymapi.KEY_0: "CMD_GAIT_DURATION_DOWN",
    }

    for key, action_name in key_map.items():
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action_name)


def handle_keyboard(env):
    """Process keyboard events and update global command values."""
    global COMMAND_VX, COMMAND_VY, COMMAND_YAW
    global COMMAND_BODY_HEIGHT, COMMAND_GAIT_FREQ
    global COMMAND_GAIT_DURATION, COMMAND_FOOT_SWING, COMMAND_BODY_PITCH, COMMAND_BODY_ROLL

    if env.viewer is None:
        return

    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0:
            continue
        action = evt.action

        if action == "CMD_VX_UP":
            COMMAND_VX = min(COMMAND_VX + 0.1, 2.0)
        elif action == "CMD_VX_DOWN":
            COMMAND_VX = max(COMMAND_VX - 0.1, -1.0)
        elif action == "CMD_VY_LEFT":
            COMMAND_VY = max(COMMAND_VY - 0.1, -1.0)
        elif action == "CMD_VY_RIGHT":
            COMMAND_VY = min(COMMAND_VY + 0.1, 1.0)
        elif action == "CMD_YAW_LEFT":
            COMMAND_YAW = max(COMMAND_YAW - 0.1, -2.0)
        elif action == "CMD_YAW_RIGHT":
            COMMAND_YAW = min(COMMAND_YAW + 0.1, 2.0)
        elif action == "CMD_HEIGHT_UP":
            COMMAND_BODY_HEIGHT = min(COMMAND_BODY_HEIGHT + 0.05, 0.50)
        elif action == "CMD_HEIGHT_DOWN":
            COMMAND_BODY_HEIGHT = max(COMMAND_BODY_HEIGHT - 0.05, 0.05)
        elif action == "CMD_FREQ_UP":
            COMMAND_GAIT_FREQ = min(COMMAND_GAIT_FREQ + 0.5, 4.0)
        elif action == "CMD_FREQ_DOWN":
            COMMAND_GAIT_FREQ = max(COMMAND_GAIT_FREQ - 0.5, 1.0)
        elif action == "CMD_RESET":
            COMMAND_VX = 0.0
            COMMAND_VY = 0.0
            COMMAND_YAW = 0.0
            COMMAND_BODY_HEIGHT = 0.39
            COMMAND_GAIT_FREQ = 2.0
            COMMAND_GAIT_DURATION = 0.5
            COMMAND_FOOT_SWING = 0.08
            COMMAND_BODY_PITCH = 0.0
            COMMAND_BODY_ROLL = 0.0
        elif action == "CMD_PITCH_UP":
            COMMAND_BODY_PITCH = min(COMMAND_BODY_PITCH + 0.1, 0.52)
        elif action == "CMD_PITCH_DOWN":
            COMMAND_BODY_PITCH = max(COMMAND_BODY_PITCH - 0.1, -0.52)
        elif action == "CMD_FOOT_SWING_UP":
            COMMAND_FOOT_SWING = min(COMMAND_FOOT_SWING + 0.01, 0.35)
        elif action == "CMD_FOOT_SWING_DOWN":
            COMMAND_FOOT_SWING = max(COMMAND_FOOT_SWING - 0.01, 0.03)
        elif action == "CMD_GAIT_DURATION_UP":
            COMMAND_GAIT_DURATION = min(COMMAND_GAIT_DURATION + 0.05, 0.65)
        elif action == "CMD_GAIT_DURATION_DOWN":
            COMMAND_GAIT_DURATION = max(COMMAND_GAIT_DURATION - 0.05, 0.35)


def apply_keyboard_commands(env):
    """Apply global command values to env.commands tensor."""
    env.commands[:, 0] = COMMAND_VX              # lin_vel_x
    env.commands[:, 1] = COMMAND_VY              # lin_vel_y
    env.commands[:, 2] = COMMAND_YAW             # ang_vel_yaw
    env.commands[:, 3] = COMMAND_BODY_HEIGHT     # body_height
    env.commands[:, 4] = COMMAND_GAIT_FREQ       # gait_frequency
    env.commands[:, 5] = 0.5                     # gait_phase (fixed for trot)
    env.commands[:, 6] = 0.0                     # gait_offset (fixed for trot)
    env.commands[:, 7] = 0.0                     # gait_bound (fixed for trot)
    env.commands[:, 8] = COMMAND_GAIT_DURATION   # gait_duration
    env.commands[:, 9] = COMMAND_FOOT_SWING      # footswing_height
    env.commands[:, 10] = COMMAND_BODY_PITCH     # body_pitch
    env.commands[:, 11] = COMMAND_BODY_ROLL      # body_roll


def print_keyboard_help():
    """Print keyboard control instructions."""
    print("\n=== Keyboard Controls ===")
    print("  W/S      : Forward/Backward velocity (+/- 0.1, range: -1.0 ~ 1.0)")
    print("  A/D      : Lateral Left/Right (+/- 0.1, range: -1.0 ~ 1.0)")
    print("  Q/E      : Turn Left/Right (+/- 0.1, range: -1.0 ~ 1.0)")
    print("  R/F      : Body Height Up/Down (+/- 0.02, range: 0.28 ~ 0.50)")
    print("  T/G      : Gait Frequency Up/Down (+/- 0.1, range: 1.0 ~ 4.0)")
    print("  1/2      : Foot Swing Height Up/Down (+/- 0.01, range: 0.03 ~ 0.35)")
    print("  9/0      : Gait Duration Up/Down (+/- 0.05, range: 0.35 ~ 0.65)")
    print("  X/C      : Body Pitch Up/Down (+/- 0.05, range: -0.52 ~ 0.52)")
    print("  Z        : Reset commands to zero")
    print("  ESC      : Quit")
    print("=========================\n")

# ==================== play =======================
def play(args):
    args.run_name = args.task
    print('args.task-------------------------',args.task)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.env.env_spacing = 1.0

    env_cfg.terrain.en_fix_step_height = False
    env_cfg.terrain.en_fix_slope = True
    # env_cfg.terrain.draw_scan_dots = True

    env_cfg.asset.fix_base_link = HANG_ON 
    env_cfg.mode.use_net = USE_NET

    if args.control_test:
        env_cfg.control.control_test = True   # true for keyboard else for curriculumn
    else:
        env_cfg.control.control_test = False   # true for keyboard else for curriculumn 

    stop_state_log = 2000 # number of steps before plotting states
    env_cfg.env.episode_length_s = 200
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s

    env_cfg.terrain.track_test = False #True
    env_cfg.terrain.track_units =[
                                    # "rough",
                                    # "sloped",
                                    # "pyramid sloped",
                                    # "sloped obstacle",
                                    # "stairs",
                                    "pyramid stairs",
                                    # "obstacles",
                                    # "wave",
                                    # "gap",
                                    # "stone pillars"
                                ]

    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.terrain_proportions =  [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0, 0, 0.1, 0.1]
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 4

    '-----观测噪声----'
    env_cfg.noise.add_noise = True and RANDOM_ON
    '-----扰动、外界摩擦和恢复系数----'
    env_cfg.domain_rand.push_robots = True and RANDOM_ON
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_rigids_after_start = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    '-----质量、质心、转动惯量----'
    env_cfg.domain_rand.randomize_base_mass = True and RANDOM_ON
    env_cfg.domain_rand.randomize_inertia = True and RANDOM_ON
    env_cfg.domain_rand.randomize_com_displacement = True and RANDOM_ON
    '-----电机能力----'
    env_cfg.domain_rand.randomize_motor_strength = True and RANDOM_ON
    env_cfg.domain_rand.randomize_PD_factor = True and RANDOM_ON
    '-----编码器的随机偏置----'
    env_cfg.domain_rand.randomize_motor_offset = True and RANDOM_ON
    env_cfg.domain_rand.randomize_default_dof_pos = False
    '-----固定延迟模拟----'
    env_cfg.domain_rand.fixed_action_delay = False
    env_cfg.domain_rand.fixed_torque_delay = False
    env_cfg.domain_rand.fixed_obs_delay = False
    '-----随机延迟模拟----'
    env_cfg.domain_rand.add_dof_lag = True and RANDOM_ON
    env_cfg.domain_rand.add_imu_lag = True and RANDOM_ON
    env_cfg.domain_rand.add_action_lag = True and RANDOM_ON
    '-----电机的阻尼摩擦特性----'
    env_cfg.domain_rand.randomize_joint_friction = True and RANDOM_ON
    env_cfg.domain_rand.randomize_joint_damping = True and RANDOM_ON
    env_cfg.domain_rand.randomize_joint_armature = True and RANDOM_ON
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # disable automatic command resampling when using keyboard
    if KEYBOARD_ON:
        env_cfg.commands.resampling_time = 999999

    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    obs = env.get_observations()
    commands = env.get_command()

    # setup keyboard control
    if KEYBOARD_ON:
        setup_keyboard(env)
        print_keyboard_help()
    # load policy
    train_cfg.runner.resume = args.resume if hasattr(args, 'resume') else True
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # MoECTS uses 'model', others use 'actor_critic'
    alg_model = getattr(ppo_runner.alg, 'model', None) or getattr(ppo_runner.alg, 'actor_critic', None)
    if alg_model is not None and hasattr(alg_model, 'estimator'):
        encoder = alg_model.estimator
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        if args.load_run is None:
            args.load_run = os.path.basename(train_cfg.runner.load_run) if isinstance(train_cfg.runner.load_run, str) and os.path.isabs(train_cfg.runner.load_run) else str(train_cfg.runner.load_run)
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            'logs', 
            train_cfg.runner.experiment_name, 
            'exported',
            args.load_run, # 在 exported 下创建以 load_run 命名的文件夹
            'polices')
        if not os.path.exists(path):
            os.makedirs(path)
        
        if "him" in args.task and "terrain" in args.task: 
            print("export him policy",env_cfg.env.num_single_obs, env_cfg.env.c_frame_stack, env_cfg.env.frame_stack, train_cfg.policy.enc_hidden_dims[-1])
            integrated_policy = Integrated_HIM_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.estimator.encoder,
                                                env_cfg.env.num_single_obs, 
                                                env_cfg.env.num_est_prob)
        elif "roa" in args.task and "terrain" in args.task:
            print("export roa policy",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_ROA_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.adaptation_encoder, # hist
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
        elif "ppo" in args.task and "terrain" in args.task:
            print("export ppo policy",env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy = Integrated_PPO_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
        elif "est" in args.task and "terrain" in args.task:
            print("export est policy",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_EST_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # hist
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
        elif args.task == 'cowa_est':
            integrated_policy = Integrated_EST_Plane_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # hist
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
        elif "ts" in args.task and "terrain" in args.task:
            print("<><><><><><>< export ts policy",env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy = Integrated_TS_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # hist
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
        elif args.task == 'cowa_dh' or args.task == 'cowa_4dof' or args.task == 'cowa_8dof_plane':
            integrated_policy = Integrated_DH_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # est
                                                ppo_runner.alg.actor_critic.long_history, # long history_encoder
                                                env_cfg.env.num_single_obs, 
                                                env_cfg.env.frame_stack,
                                                env_cfg.env.short_frame_stack,
                                                ppo_runner.alg.actor_critic.in_channels,
                                                ppo_runner.alg.actor_critic.num_proprio_obs,
                                                train_cfg.policy.long_history_type)
        elif args.task == 'cowa_prop':
            integrated_policy = Integrated_Prop_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # est
                                                ppo_runner.alg.actor_critic.long_history, # long history_encoder
                                                env_cfg.env.num_single_obs, 
                                                env_cfg.env.frame_stack,
                                                env_cfg.env.short_frame_stack,
                                                ppo_runner.alg.actor_critic.in_channels,
                                                ppo_runner.alg.actor_critic.num_proprio_obs)
        elif args.task == 'quadruped' or args.task == 'quadruped_wtw':
            integrated_policy = Integrated_vallina_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
        elif 'moe_cts' in args.task:
            num_stacked_obs = env_cfg.env.frame_stack * env_cfg.env.num_single_obs  # 5 * 59 = 295
            integrated_policy = Integrated_MoE_CTS_policy(ppo_runner.alg.model.actor, # actor
                                                ppo_runner.alg.model.student_moe_encoder,
                                                num_stacked_obs,
                                                env_cfg.env.frame_stack)

        integrated_policy.export(path, args.load_run)
        print('Exported policy as jit script to: ', path)

    logger = Logger_quad_wtw(env.dt)
    robot_index = 0 # which robot is used for logging

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.8)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 100.0, (1920, 1080))

    CoM_offset_compensate = False
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    dones = None

    for i in range(stop_state_log * 10):

        # process keyboard input and apply commands
        if KEYBOARD_ON:
            handle_keyboard(env)
            apply_keyboard_commands(env)
        elif FIX_COMMAND:
            env.commands[:, 0] = 1.0   # x 线速度 m/s
            env.commands[:, 1] = 0.0   # y 线速度 m/s
            env.commands[:, 2] = 0.0   # 角速度 rad/s (yaw)
            env.commands[:, 3] = 0.39  # base高度目标 m
        else:
            env.commands[:, 0] = env.command_vel_x
            env.commands[:, 1] = env.command_vel_y
            env.commands[:, 2] = env_cfg.init_state.pos[2] + env.command_height_z

        # print current commands periodically when using keyboard
        if KEYBOARD_ON and i % 50 == 0:
            print(f"\r[Cmd] vx={COMMAND_VX:.2f} vy={COMMAND_VY:.2f} yaw={COMMAND_YAW:.2f} "
                  f"h={COMMAND_BODY_HEIGHT:.2f} freq={COMMAND_GAIT_FREQ:.2f} "
                  f"dur={COMMAND_GAIT_DURATION:.2f} swing={COMMAND_FOOT_SWING:.2f} "
                  f"pitch={COMMAND_BODY_PITCH:.2f}", end="")

        if i > stop_state_log - 3 and i < stop_state_log - 1:
            logger.print_rewards()
            logger.plot_states()  # 绘图
        # NOTE: ----------------------------- add HEIGHT --------------------------------------------
        if "roa" in args.task and "terrain" in args.task:
            # import ipdb;ipdb.set_trace()
            h_s = env_cfg.env.num_height_scan_input
            actions = policy(
                obs.detach()[:,  h_s + (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], #25
                obs.detach()[:,  h_s: ], #25x5
                obs.detach()[:, :h_s  ], #121
                ) 
        elif "ppo" in args.task and "terrain" in args.task:  
            h_s = env_cfg.env.num_height_scan_input
            actions = policy(
                obs.detach()[:,  h_s: ], #125 hist_obs
                obs.detach()[:,  :h_s ], #63 height
                )
        elif "est" in args.task and "terrain" in args.task:  
            h_s = env_cfg.env.num_height_scan_input
            actions = policy(
                obs.detach()[:,  h_s + (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], #25
                obs.detach()[:,  h_s: ], #125 hist_obs
                obs.detach()[:,  :h_s ], #63 height
                )
        elif "ts" in args.task and "terrain" in args.task:  
            h_s = env_cfg.env.num_height_scan_input
            actions = policy(
                obs.detach()[:,  h_s: ], #25
                obs.detach()[:,  h_s: ], #125 hist_obs
                obs.detach()[:,  :h_s ], #63 height
                )
        elif "prop" in args.task:
            actions = policy(obs.detach(), commands)
        elif 'moe_cts' in args.task:
            actions = policy(obs.detach(), dones)
        else:
            actions = policy(obs.detach())

        # # Print actor input (obs) for first 5 frames
        # if i < 5:
        #     obs_input = obs[robot_index].detach().cpu().numpy()
        #     single_dim = env_cfg.env.num_single_obs  # 54
        #     n_frames = env_cfg.env.actor_input_stack  # 5
        #     print(f"\n[Frame {i}] Actor input obs (env {robot_index}), total_shape={obs_input.shape}, single_frame={single_dim}, stacked={n_frames}")
        #     # Print each stacked frame
        #     for f in range(n_frames):
        #         start = f * single_dim
        #         end = (f + 1) * single_dim
        #         frame = obs_input[start:end]
        #         frame_label = f"latest" if f == n_frames - 1 else f"-{(n_frames-1-f)}step"
        #         print(f"  Frame [{start}:{end}] ({frame_label}):")
        #         print(f"    [  0: 12] cmd   = {frame[0:12]}")
        #         print(f"    [ 12: 24] dof_pos= {frame[12:24]}")
        #         print(f"    [ 24: 36] dof_vel= {frame[24:36]}")
        #         print(f"    [ 36: 48] action = {frame[36:48]}")
        #         print(f"    [ 48: 51] ang_vel= {frame[48:51]}")
        #         print(f"    [ 51: 54] proj_grv= {frame[51:54]}")

        #     if i == 0:
        #         print(f"\n  Meaning of each segment (single frame, {single_dim} dims):")
        #         print(f"    [ 0:12] Commands: lin_vel_x, lin_vel_y, ang_vel_yaw, body_height, gait_freq, gait_phase, gait_offset, gait_bound, gait_dur, footswing, body_pitch, body_roll")
        #         print(f"    [12:24] DOF positions (q - default_q) * scale  | order: FL_hip,RL_hip,FR_hip,RR_hip, FL_thigh,RL_thigh,FR_thigh,RR_thigh, FL_calf,RL_calf,FR_calf,RR_calf")
        #         print(f"    [24:36] DOF velocities dq * scale              | same joint order")
        #         print(f"    [36:48] Previous actions from policy            | same joint order")
        #         print(f"    [48:51] Base angular velocity (body frame)      | x, y, z")
        #         print(f"    [51:54] Projected gravity vector               | x, y, z")
        #         print(f"  Note: DOF_pos/dof_vel/ang_vel are lagged (delayed encoder/IMU simulation)")
        #         print(f"  Note: Noise is added if add_noise=True")
        # else:
        #     actions = policy(obs.detach()) 
            
        if "dh" in args.task:
            est = encoder(obs.detach()[:, -env_cfg.env.num_single_obs * env_cfg.env.short_frame_stack :])

            if CoM_offset_compensate:
                if i > 200 and i < 600:
                    vel_cmd[:] = env.commands[:, 0] #* np.clip((i - 200) / 400.0, 0, 1)
                else:
                    vel_cmd[:] = 0
                vel_err_intergral += (
                    (vel_cmd - env.base_lin_vel[:, 0])
                    * env.dt
                    * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
                )
                vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
                env.commands[:, 0] = vel_cmd + vel_err_intergral
        if "him" in args.task:
            obs, critic_obs, rews, dones, infos, _, _, = env.step(actions.detach())
        elif "prop" in args.task:
            obs, critic_obs, rews, dones, infos, commands = env.step(actions.detach())
        else:
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_pos[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)
            
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        logger.log_states(
            {
                # ===== NOTE base =====
                "base_height": env.base_height[robot_index].item(),
                "base_height_cmd": env.commands[robot_index, 3].item(),
                "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                "cmd_vel_x": env.commands[robot_index, 0].item(),
                "cmd_vel_y": env.commands[robot_index, 1].item(),
                "cmd_ang_vel_z": env.commands[robot_index, 2].item(),

                # ===== NOTE base orientation =====
                "base_roll": env.base_euler_rpy[robot_index, 0].item(),
                "base_pitch": env.base_euler_rpy[robot_index, 1].item(),
                "base_yaw": env.base_euler_rpy[robot_index, 2].item(),
                "ang_vel_x": env.base_ang_vel[robot_index, 0].item(),
                "ang_vel_y": env.base_ang_vel[robot_index, 1].item(),
                "ang_vel_z": env.base_ang_vel[robot_index, 2].item(),

                # ===== NOTE base pos =====
                "base_pos_x": env.base_pos[robot_index, 0].item(),
                "base_pos_y": env.base_pos[robot_index, 1].item(),
                "base_pos_z": env.base_pos[robot_index, 2].item(),

                # ===== NOTE noised IMU =====
                "noised_base_roll": env.noised_euler_rpy[robot_index, 0].item(),
                "noised_base_pitch": env.noised_euler_rpy[robot_index, 1].item(),
                "noised_ang_vel_x": env.noised_ang_vel[robot_index, 0].item(),
                "noised_ang_vel_y": env.noised_ang_vel[robot_index, 1].item(),
                "noised_ang_vel_z": env.noised_ang_vel[robot_index, 2].item(),

                # ===== NOTE contact force (4 feet) =====
                "FL_contact_force": torch.norm(env.contact_forces[robot_index, env.feet_indices[0]]).item(),
                "RL_contact_force": torch.norm(env.contact_forces[robot_index, env.feet_indices[1]]).item(),
                "FR_contact_force": torch.norm(env.contact_forces[robot_index, env.feet_indices[2]]).item(),
                "RR_contact_force": torch.norm(env.contact_forces[robot_index, env.feet_indices[3]]).item(),

                # ===== NOTE torque (12 DOF: FL/RL/FR/RR hip, thigh, calf) =====
                # hip joints (0-3)
                "FL_hip_torque": env.torques[robot_index, 0].item(),
                "RL_hip_torque": env.torques[robot_index, 1].item(),
                "FR_hip_torque": env.torques[robot_index, 2].item(),
                "RR_hip_torque": env.torques[robot_index, 3].item(),
                # thigh joints (4-7)
                "FL_thigh_torque": env.torques[robot_index, 4].item(),
                "RL_thigh_torque": env.torques[robot_index, 5].item(),
                "FR_thigh_torque": env.torques[robot_index, 6].item(),
                "RR_thigh_torque": env.torques[robot_index, 7].item(),
                # calf joints (8-11)
                "FL_calf_torque": env.torques[robot_index, 8].item(),
                "RL_calf_torque": env.torques[robot_index, 9].item(),
                "FR_calf_torque": env.torques[robot_index, 10].item(),
                "RR_calf_torque": env.torques[robot_index, 11].item(),

                # ===== NOTE real vel (12 DOF) =====
                "FL_hip_vel": env.dof_vel[robot_index, 0].item(),
                "RL_hip_vel": env.dof_vel[robot_index, 1].item(),
                "FR_hip_vel": env.dof_vel[robot_index, 2].item(),
                "RR_hip_vel": env.dof_vel[robot_index, 3].item(),
                "FL_thigh_vel": env.dof_vel[robot_index, 4].item(),
                "RL_thigh_vel": env.dof_vel[robot_index, 5].item(),
                "FR_thigh_vel": env.dof_vel[robot_index, 6].item(),
                "RR_thigh_vel": env.dof_vel[robot_index, 7].item(),
                "FL_calf_vel": env.dof_vel[robot_index, 8].item(),
                "RL_calf_vel": env.dof_vel[robot_index, 9].item(),
                "FR_calf_vel": env.dof_vel[robot_index, 10].item(),
                "RR_calf_vel": env.dof_vel[robot_index, 11].item(),

                # ===== NOTE noised vel =====
                "noised_FL_hip_vel": env.noised_dq[robot_index, 0].item(),
                "noised_RL_hip_vel": env.noised_dq[robot_index, 1].item(),
                "noised_FR_hip_vel": env.noised_dq[robot_index, 2].item(),
                "noised_RR_hip_vel": env.noised_dq[robot_index, 3].item(),
                "noised_FL_thigh_vel": env.noised_dq[robot_index, 4].item(),
                "noised_RL_thigh_vel": env.noised_dq[robot_index, 5].item(),
                "noised_FR_thigh_vel": env.noised_dq[robot_index, 6].item(),
                "noised_RR_thigh_vel": env.noised_dq[robot_index, 7].item(),
                "noised_FL_calf_vel": env.noised_dq[robot_index, 8].item(),
                "noised_RL_calf_vel": env.noised_dq[robot_index, 9].item(),
                "noised_FR_calf_vel": env.noised_dq[robot_index, 10].item(),
                "noised_RR_calf_vel": env.noised_dq[robot_index, 11].item(),

                # ===== NOTE real pos (12 DOF) =====
                "FL_hip_pos": env.dof_pos[robot_index, 0].item(),
                "RL_hip_pos": env.dof_pos[robot_index, 1].item(),
                "FR_hip_pos": env.dof_pos[robot_index, 2].item(),
                "RR_hip_pos": env.dof_pos[robot_index, 3].item(),
                "FL_thigh_pos": env.dof_pos[robot_index, 4].item(),
                "RL_thigh_pos": env.dof_pos[robot_index, 5].item(),
                "FR_thigh_pos": env.dof_pos[robot_index, 6].item(),
                "RR_thigh_pos": env.dof_pos[robot_index, 7].item(),
                "FL_calf_pos": env.dof_pos[robot_index, 8].item(),
                "RL_calf_pos": env.dof_pos[robot_index, 9].item(),
                "FR_calf_pos": env.dof_pos[robot_index, 10].item(),
                "RR_calf_pos": env.dof_pos[robot_index, 11].item(),

                # ===== NOTE noised pos =====
                "noised_FL_hip_pos": env.noised_q[robot_index, 0].item(),
                "noised_RL_hip_pos": env.noised_q[robot_index, 1].item(),
                "noised_FR_hip_pos": env.noised_q[robot_index, 2].item(),
                "noised_RR_hip_pos": env.noised_q[robot_index, 3].item(),
                "noised_FL_thigh_pos": env.noised_q[robot_index, 4].item(),
                "noised_RL_thigh_pos": env.noised_q[robot_index, 5].item(),
                "noised_FR_thigh_pos": env.noised_q[robot_index, 6].item(),
                "noised_RR_thigh_pos": env.noised_q[robot_index, 7].item(),
                "noised_FL_calf_pos": env.noised_q[robot_index, 8].item(),
                "noised_RL_calf_pos": env.noised_q[robot_index, 9].item(),
                "noised_FR_calf_pos": env.noised_q[robot_index, 10].item(),
                "noised_RR_calf_pos": env.noised_q[robot_index, 11].item(),

                # ===== NOTE action =====
                "FL_hip_action": env.actions[robot_index, 0].item(),
                "RL_hip_action": env.actions[robot_index, 1].item(),
                "FR_hip_action": env.actions[robot_index, 2].item(),
                "RR_hip_action": env.actions[robot_index, 3].item(),
                "FL_thigh_action": env.actions[robot_index, 4].item(),
                "RL_thigh_action": env.actions[robot_index, 5].item(),
                "FR_thigh_action": env.actions[robot_index, 6].item(),
                "RR_thigh_action": env.actions[robot_index, 7].item(),
                "FL_calf_action": env.actions[robot_index, 8].item(),
                "RL_calf_action": env.actions[robot_index, 9].item(),
                "FR_calf_action": env.actions[robot_index, 10].item(),
                "RR_calf_action": env.actions[robot_index, 11].item(),

                # ===== NOTE dof acc =====
                "FL_hip_acc": env.dof_acc_200hz[robot_index, 0].item(),
                "RL_hip_acc": env.dof_acc_200hz[robot_index, 1].item(),
                "FR_hip_acc": env.dof_acc_200hz[robot_index, 2].item(),
                "RR_hip_acc": env.dof_acc_200hz[robot_index, 3].item(),
                "FL_thigh_acc": env.dof_acc_200hz[robot_index, 4].item(),
                "RL_thigh_acc": env.dof_acc_200hz[robot_index, 5].item(),
                "FR_thigh_acc": env.dof_acc_200hz[robot_index, 6].item(),
                "RR_thigh_acc": env.dof_acc_200hz[robot_index, 7].item(),
                "FL_calf_acc": env.dof_acc_200hz[robot_index, 8].item(),
                "RL_calf_acc": env.dof_acc_200hz[robot_index, 9].item(),
                "FR_calf_acc": env.dof_acc_200hz[robot_index, 10].item(),
                "RR_calf_acc": env.dof_acc_200hz[robot_index, 11].item(),
            }
        )
        # if est != None:
        #     logger.log_states(
        #         {
        #             "est_lin_vel_x": est[robot_index, -4].item()
        #                 / env_cfg.normalization.obs_scales.lin_vel,
        #             "est_lin_vel_y": est[robot_index, -3].item()
        #                 / env_cfg.normalization.obs_scales.lin_vel,
        #             "est_lin_vel_z": est[robot_index, -2].item()
        #                 / env_cfg.normalization.obs_scales.lin_vel,
        #             "est_base_height": est[robot_index, -1].item()
        #                 / env_cfg.normalization.obs_scales.height_measurements,
        #         }
        #     )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
    if RENDER:
        video.release()
    
if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # NOTE 更改play_test用来启用机械臂角度的源，是随机课程轨迹还是键盘识别 env.cfg.control.play_test = True 启用键盘，反之
    USE_NET = True           # False为跟随ref
    EXPORT_POLICY = True    # 导出model — MoECTS 需要新的 wrapper（包含 MoE encoder + gating）
    RENDER = False           # 保存视频
    FIX_COMMAND = False
    KEYBOARD_ON = True       # 键盘控制
    MOVE_CAMERA = False
    args = get_args()
    args.play_flag = True
    HANG_ON =  False         # 机器人挂天上
    RANDOM_ON = False        # 开启随机化
    args.control_test = not FIX_COMMAND and not KEYBOARD_ON

    args.task= 'quadruped_moe_cts'
    # args.task= 'quadruped_wtw'
    args.experiment_name = args.task
    args.load_run = None
    args.checkpoint = -1
    args.resume = True
    args.headless = False #False
    args.num_envs = 8

    cuda = 0
    args.sim_device=f"cuda:{cuda}"
    args.rl_device=f"cuda:{cuda}"
    play(args)
