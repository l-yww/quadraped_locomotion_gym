# logs: 规范代码结构，将导出estimator的部分移到Integrated.py中 ///2025.4.16
import os
import numpy as np
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import time
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from wheel_legged_gym.utils import Logger

# import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import json
import copy
from wheel_legged_gym.utils.Integrated import Integrated_EST_policy, Integrated_RMA_policy, Integrated_HIM_policy, Integrated_ROA_policy, Integrated_PPO_policy, Integrated_TS_policy, Integrated_EST_Plane_policy, Integrated_DH_policy

# ==================== play =======================
def play(args):
    args.run_name = args.task
    print('args.task-------------------------',args.task)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.sim.max_gpu_contact_pairs = 2**10

    env_cfg.terrain.en_fix_step_height = False
    # env_cfg.terrain.draw_scan_dots = True

    env_cfg.asset.fix_base_link = HANG_ON 
    env_cfg.mode.use_net = USE_NET

    if args.control_test:
        env_cfg.control.control_test = True   # true for keyboard else for curriculumn
    else:
        env_cfg.control.control_test = False   # true for keyboard else for curriculumn 

    stop_state_log = 1000 # number of steps before plotting states
    env_cfg.env.episode_length_s = 9
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s

    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5

    '-----观测噪声----'
    env_cfg.noise.add_noise = False and RANDOM_ON
    '-----扰动、外界摩擦和恢复系数----'
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_rigids_after_start = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    '-----质量、质心、转动惯量----'
    env_cfg.domain_rand.randomize_base_mass = False and RANDOM_ON
    env_cfg.domain_rand.randomize_inertia = False and RANDOM_ON
    env_cfg.domain_rand.randomize_com_displacement = False and RANDOM_ON
    env_cfg.domain_rand.com_displacement_range = [-0.0, 0.05]
    '-----电机能力----'
    env_cfg.domain_rand.randomize_motor_strength = False and RANDOM_ON
    env_cfg.domain_rand.randomize_PD_factor = False and RANDOM_ON
    '-----编码器的随机偏置----'
    env_cfg.domain_rand.randomize_motor_offset = False and RANDOM_ON
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
    env_cfg.domain_rand.randomize_joint_friction = False and RANDOM_ON
    env_cfg.domain_rand.randomize_joint_damping = False and RANDOM_ON
    env_cfg.domain_rand.randomize_joint_armature = False and RANDOM_ON
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.alg.actor_critic.estimator.mlp_est
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            'logs', 
            train_cfg.runner.experiment_name, 
            'exported',
            args.load_run, # 在 exported 下创建以 load_run 命名的文件夹
            'polices')
        if not os.path.exists(path):
            os.makedirs(path)
        # task_registry.save_played_cfgs(path, name=args.task)
        if "him" in args.task and "terrain" in args.task: 
            print("export him policy",env_cfg.env.num_single_obs, env_cfg.env.c_frame_stack, env_cfg.env.frame_stack, train_cfg.policy.enc_hidden_dims[-1])
            integrated_policy = Integrated_HIM_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.estimator.encoder,
                                                env_cfg.env.num_single_obs, 
                                                env_cfg.env.num_est_prob)
            integrated_policy.export(path,args.load_run)
        elif "roa" in args.task and "terrain" in args.task:
            print("export roa policy",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_ROA_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.adaptation_encoder, # hist
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            integrated_policy.export(path, args.load_run)   
        elif "ppo" in args.task and "terrain" in args.task:
            print("export ppo policy",env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy = Integrated_PPO_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy.export(path, args.load_run)
        elif "est" in args.task and "terrain" in args.task:
            print("export est policy",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_EST_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # hist
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            integrated_policy.export(path, args.load_run)  
        elif args.task == 'cowa_est':
            integrated_policy = Integrated_EST_Plane_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # hist
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            # integrated_policy.export(path, args.load_run)
        elif "ts" in args.task and "terrain" in args.task:
            print("<><><><><><>< export ts policy",env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy = Integrated_TS_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # hist
                                                ppo_runner.alg.actor_critic.height_scan_encoder, # height
                                                env_cfg.env.num_height_scan_input,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy.export(path, args.load_run)  
        elif args.task == 'cowa_dh' or args.task == 'cowa_4dof':
            integrated_policy = Integrated_DH_policy(ppo_runner.alg.actor_critic.actor, # actor
                                                ppo_runner.alg.actor_critic.estimator, # est
                                                ppo_runner.alg.actor_critic.long_history, # long history_encoder
                                                env_cfg.env.num_single_obs, 
                                                env_cfg.env.frame_stack,
                                                env_cfg.env.short_frame_stack,
                                                ppo_runner.alg.actor_critic.in_channels,
                                                ppo_runner.alg.actor_critic.num_proprio_obs,)
        integrated_policy.export(path, args.load_run)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
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

    CoM_offset_compensate = True
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    
    est = None
    
    for i in range(stop_state_log * 100):
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
        else:
            actions = policy(obs.detach()) 
            
        if "dh" in args.task:
            est = encoder(obs.detach()[:, -env_cfg.env.num_single_obs * env_cfg.env.short_frame_stack :])

        if FIX_COMMAND:
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.33
        else:
            env.commands[:, 0] = env.command_vel_x
            env.commands[:, 1] = env.command_vel_y
            env.commands[:, 2] = env_cfg.init_state.pos[2] + env.command_height_z

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
                # NOTE delta pos x 
                # "delta_foot_pos_x_l": env.feet_obs_l[robot_index,0].item(),
                # "delta_foot_pos_x_r": env.feet_obs_r[robot_index,0].item(),
                # NOTE base height 

                # NOTE contact force
                # "contact_forces_buff_l_x": contact_force_left[robot_index,0].item(),
                # "contact_forces_buff_l_y": contact_force_left[robot_index,1].item(),
                # # "contact_forces_buff_l_z": contact_force_left[robot_index,2].item(),
                # "contact_forces_buff_r_x": contact_force_right[robot_index,0].item(),
                # "contact_forces_buff_r_y": contact_force_right[robot_index,1].item(),
                # # "contact_forces_buff_r_z": contact_force_right[robot_index,2].item(),
                # NOTE torque
                "left_hip_vel": env.dof_vel[robot_index, 0].item(),
                "left_hip_torque": env.torques[robot_index, 0].item(),
                "left_wheel_vel": env.dof_vel[robot_index, 1].item(),
                "left_wheel_torque": env.torques[robot_index, 1].item(),
                "right_hip_vel": env.dof_vel[robot_index, 2].item(),
                "right_hip_torque": env.torques[robot_index, 2].item(),
                "right_wheel_vel": env.dof_vel[robot_index, 3].item(),
                "right_wheel_torque": env.torques[robot_index, 3].item(),

                # NOTE # 加入关节角度显示
                "base_vel_x": env.base_lin_vel[robot_index,0].item(),
                "base_vel_y": env.base_lin_vel[robot_index,1].item(),
                "base_vel_z": env.base_lin_vel[robot_index,2].item(),
                "cmd_vel": env.commands[robot_index,0].item(),
                "cmd_ang_vel": env.commands[robot_index,1].item(),
                "left_hip_pos": env.dof_pos[robot_index, 0].item(),
                "right_hip_pos": env.dof_pos[robot_index, 2].item(),
                # NOTE action
                "left_hip_action": env.actions[robot_index, 0].item() * env_cfg.control.pos_action_scale,
                "left_wheel_action": env.actions[robot_index, 1].item() * env_cfg.control.vel_action_scale,
                "right_hip_action": env.actions[robot_index, 2].item() * env_cfg.control.pos_action_scale,
                "right_wheel_action": env.actions[robot_index, 3].item() * env_cfg.control.vel_action_scale,
                # NOTE pos
                "left_hip_ref_pos": env.ref_dof_pos[robot_index, 0].item(),
                "left_wheel_ref_vel": env.ref_dof_vel[robot_index, 1].item(),
                "right_hip_ref_pos": env.ref_dof_pos[robot_index, 2].item(),
                "right_wheel_ref_vel": env.ref_dof_vel[robot_index, 3].item(),

                # NOTE dof acc
                "left_wheel_acc_100hz": env.dof_acc_100hz[robot_index, 1].item(),
                "left_wheel_acc_500hz": env.dof_acc_500hz[robot_index, 1].item(),
                "right_wheel_acc_100hz": env.dof_acc_100hz[robot_index, 3].item(),
                "right_wheel_acc_500hz": env.dof_acc_500hz[robot_index, 3].item(),

                # NOTE IMU
                "base_roll": env.base_euler_rpy[robot_index, 0].item(),
                "base_pitch": env.base_euler_rpy[robot_index, 1].item(),
                "base_yaw": env.base_euler_rpy[robot_index, 2].item(),

                # NOTE base pos
                "base_pos_x": env.base_pos[robot_index, 0].item(),
                "base_pos_y": env.base_pos[robot_index, 1].item(),
                "base_pos_z": env.base_pos[robot_index, 2].item(),
            }
            )
        if est != None:
            logger.log_states(
                {
                    "est_lin_vel_x": est[robot_index, -4].item()
                        / env_cfg.normalization.obs_scales.lin_vel,
                    "est_lin_vel_y": est[robot_index, -3].item()
                        / env_cfg.normalization.obs_scales.lin_vel,
                    "est_lin_vel_z": est[robot_index, -2].item()
                        / env_cfg.normalization.obs_scales.lin_vel,
                    "est_base_height": est[robot_index, -1].item()
                        / env_cfg.normalization.obs_scales.height_measurements,
                }
            )
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
    USE_NET = True         # False为跟随ref
    EXPORT_POLICY = True    # 导出model
    RENDER = False          # 保存视频
    FIX_COMMAND = True
    MOVE_CAMERA = True
    args = get_args()    
    args.play_flag = True   
    HANG_ON = False          # 机器人挂天上
    RANDOM_ON = True        # 开启随机化
    args.control_test = not FIX_COMMAND

    args.task= 'cowa_4dof'
    args.experiment_name = args.task
    args.load_run = "Sep04_20-51-52_based_0904-[随机化变大]-[实机1w轮效果不错]"
    args.checkpoint = 10000
    args.headless = False
    args.num_envs = 1

    cuda = 0
    args.sim_device=f"cuda:{cuda}"
    args.rl_device=f"cuda:{cuda}"
    play(args)