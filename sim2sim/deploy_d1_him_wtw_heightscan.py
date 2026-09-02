import sys
from pathlib import Path
PATH_PARENT = Path(__file__).parent
PATH_ROOT = PATH_PARENT.parent
sys.path.append(str(PATH_PARENT))
sys.path.append(str(PATH_ROOT))
from utils import MujocoRenderUtils

import os
import time
import mujoco.viewer
import mujoco
import numpy as np
LEGGED_GYM_ROOT_DIR = str(PATH_ROOT)
import torch
import onnxruntime as ort
import yaml
import imageio
from argparse import ArgumentParser
import pygame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def quat_rotate_inverse(q, v):
    q = np.array(q, np.float32)
    v = np.array(v, np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def load_config(config_path):
    """Load a deployment config, optionally inheriting a base YAML config."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    base_config = config.pop("base_config", None)
    if base_config is None:
        return config
    base_path = config_path.parent / base_config
    base = load_config(base_path)
    base.update(config)
    return base


def yaw_from_wxyz(quaternion):
    """Return yaw for MuJoCo free-joint quaternions (w, x, y, z)."""
    w, x, y, z = quaternion
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def sample_height_scan(model, data, base_pos, base_quat, local_points, ray_start_height):
    """Sample static MuJoCo collision geometry below yaw-aligned local points."""
    yaw = yaw_from_wxyz(base_quat)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    world_xy = np.empty_like(local_points)
    world_xy[:, 0] = base_pos[0] + cos_yaw * local_points[:, 0] - sin_yaw * local_points[:, 1]
    world_xy[:, 1] = base_pos[1] + sin_yaw * local_points[:, 0] + cos_yaw * local_points[:, 1]

    geomgroup = np.ones(6, dtype=np.uint8)
    ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    heights = np.zeros(len(local_points), dtype=np.float32)
    ray_origin = np.empty(3, dtype=np.float64)
    ray_origin[2] = base_pos[2] + ray_start_height
    geomid = np.empty(1, dtype=np.int32)
    for point_idx, point_xy in enumerate(world_xy):
        ray_origin[:2] = point_xy
        distance = mujoco.mj_ray(
            model, data, ray_origin, ray_direction, geomgroup, 1, -1, geomid
        )
        if distance < 0.0:
            raise RuntimeError(
                "Height-scan ray missed all static terrain geometry. "
                "The MuJoCo scene must contain a static floor or terrain geom."
            )
        heights[point_idx] = ray_origin[2] - distance
    return heights


CMD_VX, CMD_VY, CMD_YAW = 0.0, 0.0, 0.0
CMD_STEP = 0.25


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--config", default="cowa2_him_amp.yaml", help="YAML in sim2sim/configs or an absolute path.")
    args = parser.parse_args()
    save_video = args.save_video
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PATH_PARENT / "configs" / config_path

    pygame.init()
    screen = pygame.display.set_mode((200, 100))
    pygame.display.set_caption("Keyboard Control - HIM-AMP (3D cmd)")
    print("Keyboard control: W/S=vx(+/-0.2), A/D=vy(+/-0.2), Q/E=yaw(+/-0.2) — persistent, Z to reset")

    with open(config_path, "r") as f:
        config = load_config(config_path)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        # torque-vel 约束参数 (从 yaml 读, 见 cowa2_him.yaml 的 torque_vel_limits)
        tv = config["torque_vel_limits"]
        torque_vel_hip_thigh = {
            "indices": np.array(tv["hip_thigh"]["indices"], dtype=np.int64),
            "max_vel": float(tv["hip_thigh"]["max_vel"]),
            "vel_1": float(tv["hip_thigh"]["vel_1"]),
            "max_torque": float(tv["hip_thigh"]["max_torque"]),
            "directional_brake": bool(tv["hip_thigh"].get("directional_brake", False)),
        }
        torque_vel_calf = {
            "indices": np.array(tv["calf"]["indices"], dtype=np.int64),
            "max_vel": float(tv["calf"]["max_vel"]),
            "vel_1": float(tv["calf"]["vel_1"]),
            "max_torque": float(tv["calf"]["max_torque"]),
            "directional_brake": bool(tv["calf"].get("directional_brake", False)),
        }
        top_clip = np.array(tv["top_clip"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        obs_frame_stack = config.get("frame_stack", 30)

        height_scan_enabled = bool(config.get("height_scan", {}).get("enabled", False))
        if height_scan_enabled:
            height_cfg = config["height_scan"]
            measured_points_x = np.asarray(height_cfg["measured_points_x"], dtype=np.float32)
            measured_points_y = np.asarray(height_cfg["measured_points_y"], dtype=np.float32)
            scan_x, scan_y = np.meshgrid(measured_points_x, measured_points_y, indexing="ij")
            height_local_points = np.column_stack((scan_x.ravel(), scan_y.ravel())).astype(np.float32)
            height_measurement_scale = float(height_cfg["scale"])
            height_base_reference = float(height_cfg["base_reference"])
            height_update_steps = max(1, round(1.0 / (simulation_dt * float(height_cfg["update_hz"]))))
            height_ray_start = float(height_cfg.get("ray_start_height", 3.0))
            num_proprio_obs = 59
            num_obs = num_proprio_obs + len(height_local_points)
            if config.get("num_obs", num_obs) != num_obs:
                raise ValueError(f"height_scan config requires num_obs={num_obs}, got {config['num_obs']}")
            if len(cmd_scale) != 12:
                raise ValueError("height_scan config requires 12 command scales")
            enable_gait_clock = True
        else:
            # 步态时钟开关：yaml 显式 enable_gait_clock 优先；未配置时按 num_obs 推导(50=有时钟, 45=无)
            # 必须与训练 env 的 observe_timing_parameter/observe_clock_inputs 一致，否则 onnx 维度对不上
            enable_gait_clock = config.get("enable_gait_clock", config.get("num_obs", 50) > 45)
            # num_obs 由开关推导(45/50)，避免 yaml 里 num_obs 与 enable_gait_clock 不一致
            num_obs = 50 if enable_gait_clock else 45

        # 步态时钟参数（与训练 env 固定常量一致，缺失时用 env 默认值兜底；无时钟时不生效）
        gait_freq = config.get("gait_freq", 1.5)
        gait_phase = config.get("gait_phase", 0.5)
        gait_offset = config.get("gait_offset", 0.0)
        gait_bound = config.get("gait_bound", 0.0)
        gait_duration = config.get("gait_duration", 0.5)
        zero_cmd_thresh = config.get("zero_cmd_thresh", 0.1)
        if height_scan_enabled:
            gait_freq = config.get("gait_frequency", gait_freq)
            gait_phase = config.get("gait_phase", gait_phase)
            gait_offset = config.get("gait_offset", gait_offset)
            gait_bound = config.get("gait_bound", gait_bound)
            gait_duration = config.get("gait_duration", gait_duration)
            body_height_cmd = config.get("body_height_cmd", 0.4)
            footswing_height = config.get("footswing_height", 0.15)
            body_pitch = config.get("body_pitch", 0.0)
            body_roll = config.get("body_roll", 0.0)

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        if "cmd_range" in config:
            cmd_limits = np.array(config["cmd_range"], dtype=np.float32)
        else:
            max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            cmd_limits = np.stack((-max_cmd, max_cmd), axis=1)

        control_dt = simulation_dt * control_decimation

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    video_save_dir = str(PATH_PARENT / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    if policy_path.startswith("CHANGE_ME"):
        raise ValueError(
            "Set policy_path in the selected sim2sim YAML to the exported "
            "ONNX or TorchScript policy for quadruped_wtw_him_arm_fix_height_scan."
        )

    # Auto-detect ONNX: if .onnx exists next to .pt, prefer ONNX
    use_onnx = policy_path.endswith('.onnx')
    if not use_onnx:
        onnx_candidate = policy_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_candidate):
            policy_path = onnx_candidate
            use_onnx = True
            print(f"Auto-detected ONNX: {policy_path}")

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history = np.zeros(obs_frame_stack * num_obs, dtype=np.float32)  # frame_stack * num_obs
    # 步态时钟状态（初值 0，与训练 reset 一致；settle 后在控制循环内开始累加）
    gait_index = 0.0
    clock_inputs = np.zeros(4, dtype=np.float32)
    smoothed_cmd = np.zeros(12, dtype=np.float32)
    last_height_scan = np.zeros(len(height_local_points), dtype=np.float32) if height_scan_enabled else None

    counter = 0

    # Data recording lists
    log_time = []
    log_qpos = []
    log_action = []
    log_target_dof_pos = []
    log_base_z = []
    log_cmd = []
    log_ang_vel = []
    log_torque_time = []
    log_torque = []
    log_motor_vel_time = []
    log_motor_vel = []

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Initialize joint positions
    d.qpos[2] = 0.42  # 与训练 init_state.pos[2]=0.42 对齐（原 0.39）
    d.qpos[7:] = default_angles.copy()
    target_dof_pos = default_angles.copy()
    mujoco.mj_forward(m, d)

    # Stabilize: let robot settle for a few steps before enabling policy control
    print("Stabilizing robot (base z=%.2f)..." % d.qpos[2])
    settle_steps = 200
    for _ in range(settle_steps):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

    if height_scan_enabled:
        terrain_heights = sample_height_scan(
            m, d, d.qpos[:3], d.qpos[3:7], height_local_points, height_ray_start
        )
        last_height_scan[:] = np.clip(
            d.qpos[2] - height_base_reference - terrain_heights, -1.0, 1.0
        ) * height_measurement_scale
        print(
            f"Height scan enabled: {len(last_height_scan)} points, "
            f"{1.0 / (height_update_steps * simulation_dt):.1f} Hz, "
            f"range=[{last_height_scan.min():.3f}, {last_height_scan.max():.3f}]"
        )

    # load policy
    if use_onnx:
        sess = ort.InferenceSession(policy_path)
        onnx_input_name = sess.get_inputs()[0].name
        # 维度自检：onnx 输入必须 = frame_stack * num_obs，否则 enable_gait_clock 与训练模型不一致
        onnx_input_dim = sess.get_inputs()[0].shape[1]
        expected_dim = obs_frame_stack * num_obs
        if onnx_input_dim != expected_dim:
            mode = "height-scan" if height_scan_enabled else ("gait-clock" if enable_gait_clock else "blind")
            raise ValueError(
                f"ONNX 输入维度 {onnx_input_dim} 与 yaml 配置不符(期望 {expected_dim}={obs_frame_stack}×{num_obs})。\n"
                f"  当前观测模式={mode}, num_obs={num_obs}。\n"
                "  请确认 policy_path 指向与此 YAML 完全匹配的导出策略。"
            )
        print(f"Loaded ONNX model: {policy_path} (input: '{onnx_input_name}', dim={onnx_input_dim}, height_scan={height_scan_enabled})")
    else:
        policy = torch.jit.load(policy_path)
        print(f"Loaded HIM-AMP PT model: {policy_path} (gait_clock={enable_gait_clock}, num_obs={num_obs})")

    video_fps = 50
    if save_video:
        renderer = mujoco.Renderer(m, height=360, width=640)
        video_filename = f"{model_name}_{cmd_str}.mp4"
        video_path = os.path.join(video_save_dir, video_filename)
        print(f"Video recording will be saved to: {video_path}")
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
    mujoco_render_utils = MujocoRenderUtils(video_fps, m.opt.timestep)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

        start_wall_time = time.time()
        start_sim_time = d.time

        print("\n")

        while viewer.is_running() and time.time() - start_wall_time < simulation_duration:
            vel = d.qvel[:3]
            local_vel = quat_rotate_inverse(d.qpos[3:7], vel)
            local_ang_vel = d.qvel[3:6]  # 直接读取局部角速度 (已在机体系)

            show_str = f"[HIM-AMP] Vx={local_vel[0]:.2f}, Vy={local_vel[1]:.2f}, Wz={local_ang_vel[2]:.2f}"

            if counter % control_decimation == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        viewer.close()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        key = event.key
                        if key == pygame.K_w:
                            CMD_VX = min(CMD_VX + CMD_STEP, cmd_limits[0][1])
                        elif key == pygame.K_s:
                            CMD_VX = max(CMD_VX - CMD_STEP, cmd_limits[0][0])
                        elif key == pygame.K_a:
                            CMD_VY = min(CMD_VY + CMD_STEP, cmd_limits[1][1])
                        elif key == pygame.K_d:
                            CMD_VY = max(CMD_VY - CMD_STEP, cmd_limits[1][0])
                        elif key == pygame.K_q:
                            CMD_YAW = min(CMD_YAW + CMD_STEP, cmd_limits[2][1])
                        elif key == pygame.K_e:
                            CMD_YAW = max(CMD_YAW - CMD_STEP, cmd_limits[2][0])
                        elif key == pygame.K_z:
                            CMD_VX = 0.0
                            CMD_VY = 0.0
                            CMD_YAW = 0.0

                show_str += f" | Cmd: Vx={CMD_VX:.2f}, Vy={CMD_VY:.2f}, Wz={CMD_YAW:.2f}"
                act_str = (
                    f"action: FL: h={action[0]:+5.2f} t={action[1]:+5.2f} c={action[2]:+5.2f} | "
                    f"FR: h={action[3]:+5.2f} t={action[4]:+5.2f} c={action[5]:+5.2f} | "
                    f"RL: h={action[6]:+5.2f} t={action[7]:+5.2f} c={action[8]:+5.2f} | "
                    f"RR: h={action[9]:+5.2f} t={action[10]:+5.2f} c={action[11]:+5.2f}"
                )
                sys.stdout.write(f"\033[A\r\033[2K{show_str}\n\r\033[2K{act_str}")
                sys.stdout.flush()

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)

            # ===== torque-vel 约束 (逐行对齐训练父类 LeggedRobot._compute_torques) =====
            # 采用平行四边形包络: upper/lower 为两条带符号斜线 -K*(vel ∓ max_vel),
            # 并在两条斜线交点之外 (|vel| > v_int) 把 limit 归零。不区分 directional_brake。
            # 参数从 yaml 的 torque_vel_limits 读, 见 cowa2_him.yaml
            qvel = d.qvel[6:]  # 12 关节速度

            for group in (torque_vel_hip_thigh, torque_vel_calf):
                idx = group["indices"]
                vel = qvel[idx]
                max_vel = group["max_vel"]
                vel_1 = group["vel_1"]
                max_torque = group["max_torque"]
                # 斜率 K = max_torque / (max_vel - vel_1), 与训练侧一致
                vel_range = max(max_vel - vel_1, 1e-6)
                K = max_torque / vel_range
                # 平行四边形的动态上下限, 硬截断在 ±max_torque
                upper_limit = np.clip(-K * (vel - max_vel), -max_torque, max_torque)
                lower_limit = np.clip(-K * (vel + max_vel), -max_torque, max_torque)
                # 两条斜线交点速度 v_int = max_vel + max_torque/K, 超过则两侧 limit 归零
                v_int = max_vel + max_torque / K
                over_intersection = np.abs(vel) > v_int
                upper_limit = np.where(over_intersection, 0.0, upper_limit)
                lower_limit = np.where(over_intersection, 0.0, lower_limit)
                tau[idx] = np.clip(tau[idx], lower_limit, upper_limit)

            # 顶层静态 clip (兜底, 与训练侧 torque_limits 对齐: 取分组 max_torque)
            tau = np.clip(tau, -top_clip, top_clip)
            d.ctrl[:] = tau
            log_torque_time.append(counter * simulation_dt)
            log_torque.append(tau.copy())
            log_motor_vel_time.append(counter * simulation_dt)
            log_motor_vel.append(qvel.copy())
            mujoco.mj_step(m, d)
            if height_scan_enabled and (counter + 1) % height_update_steps == 0:
                terrain_heights = sample_height_scan(
                    m, d, d.qpos[:3], d.qpos[3:7], height_local_points, height_ray_start
                )
                last_height_scan[:] = np.clip(
                    d.qpos[2] - height_base_reference - terrain_heights, -1.0, 1.0
                ) * height_measurement_scale
            mujoco_render_utils.update(np.array([CMD_VX, CMD_VY, CMD_YAW]), d)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"\nError rendering frame: {e}\n")

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                gravity_orientation = get_gravity_orientation(quat)
                ang_vel = d.qvel[3:6]  # 直接读取局部角速度 (已在机体系)

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale

                cmd_vec = np.array([CMD_VX, CMD_VY, CMD_YAW], dtype=np.float32)
                if height_scan_enabled:
                    target_cmd = np.array([
                        CMD_VX, CMD_VY, CMD_YAW, body_height_cmd, gait_freq,
                        gait_phase, gait_offset, gait_bound, gait_duration,
                        footswing_height, body_pitch, body_roll,
                    ], dtype=np.float32)
                    # Match the training-side command filter exactly.
                    smoothed_cmd = 0.8 * smoothed_cmd + 0.2 * target_cmd
                    cmd_for_clock = smoothed_cmd
                else:
                    cmd_for_clock = cmd_vec

                # 更新步态时钟（与训练 _step_contact_targets 一致：先更新相位再拼 obs）
                # 原地检测: cmd xy 模长 < thresh 且 |yaw| < thresh → 时钟停转、clock_inputs 归零、gait_index 保持
                # enable_gait_clock=False(无时钟模型)时跳过，gait_index/clock_inputs 保持 0
                if enable_gait_clock:
                    zero_cmd = (np.linalg.norm(cmd_for_clock[:2]) < zero_cmd_thresh) and (abs(cmd_for_clock[2]) < zero_cmd_thresh)
                    if zero_cmd:
                        clock_inputs[:] = 0.0
                    else:
                        active_frequency = cmd_for_clock[4] if height_scan_enabled else gait_freq
                        active_phase = cmd_for_clock[5] if height_scan_enabled else gait_phase
                        active_offset = cmd_for_clock[6] if height_scan_enabled else gait_offset
                        active_bound = cmd_for_clock[7] if height_scan_enabled else gait_bound
                        active_duration = cmd_for_clock[8] if height_scan_enabled else gait_duration
                        gait_index = (gait_index + control_dt * active_frequency) % 1.0
                        foot_phases = [
                            gait_index + active_phase + active_offset + active_bound,  # FL
                            gait_index + active_bound,                                  # FR
                            gait_index + active_offset,                                 # RL
                            gait_index + active_phase,                                  # RR
                        ]
                        for i in range(4):
                            fp = foot_phases[i] % 1.0
                            if fp < active_duration:
                                fp = fp * (0.5 / active_duration)
                            else:
                                fp = 0.5 + (fp - active_duration) * (0.5 / (1.0 - active_duration))
                            clock_inputs[i] = np.sin(2 * np.pi * fp)

                # 构建观测 (与 quadruped_arm_him config 的 compute_observations 对齐):
                #   commands[:3] * cmd_scale    [0:3]
                #   dof_pos * dof_pos_scale     [3:15]
                #   dof_vel * dof_vel_scale     [15:27]
                #   actions                     [27:39]
                #   ang_vel * ang_vel_scale     [39:42]
                #   projected_gravity           [42:45]
                #   gait_index                  [45]      ← enable_gait_clock=True 时才有
                #   clock_inputs(4)             [46:50]   ← enable_gait_clock=True 时才有
                if height_scan_enabled:
                    obs[:12] = smoothed_cmd * cmd_scale
                    obs[12:24] = qj
                    obs[24:36] = dqj
                    obs[36:48] = action
                    obs[48:51] = ang_vel * ang_vel_scale
                    obs[51:54] = gravity_orientation
                    obs[54] = gait_index
                    obs[55:59] = clock_inputs
                    obs[59:] = last_height_scan
                else:
                    obs[:3] = cmd_vec * cmd_scale
                    obs[3:15] = qj
                    obs[15:27] = dqj
                    obs[27:39] = action
                    obs[39:42] = ang_vel * ang_vel_scale
                    obs[42:45] = gravity_orientation
                    if enable_gait_clock:
                        obs[45] = gait_index
                        obs[46:50] = clock_inputs

                # 与训练一致：观测裁剪到 ±100 (clip_observations)
                obs = np.clip(obs, -100.0, 100.0)

                # 滑动窗口更新 obs_history
                obs_history[:obs_frame_stack * num_obs - num_obs] = obs_history[num_obs:]
                obs_history[obs_frame_stack * num_obs - num_obs:] = obs
                if counter == 0:
                    for k in range(obs_frame_stack):
                        obs_history[k * num_obs:(k + 1) * num_obs] = obs

                last_action = action
                if use_onnx:
                    obs_input = obs_history.astype(np.float32)[np.newaxis, :]
                    action = sess.run(['actions'], {onnx_input_name: obs_input})[0].squeeze()
                else:
                    obs_tensor = torch.from_numpy(obs_history).unsqueeze(0)
                    result = policy(obs_tensor)
                    if isinstance(result, tuple):
                        action = result[0].detach().numpy().squeeze()
                    else:
                        action = result.detach().cpu().numpy().squeeze()

                action = np.clip(action, -20.0, 20.0)  # 与训练 clip_actions=100 对齐（原 -20,20）
                # 训练 config action_smoothness=False，故 deploy 不做低通平滑（原 0.9*action+0.1*last_action 已删）
                target_dof_pos = action * action_scale + default_angles

                # Record data
                log_time.append(counter * simulation_dt)
                log_qpos.append(d.qpos[7:].copy())
                log_action.append(action.copy())
                log_target_dof_pos.append(target_dof_pos.copy())
                log_base_z.append(d.qpos[2].copy())
                log_cmd.append(np.array([CMD_VX, CMD_VY, CMD_YAW]).copy())
                log_ang_vel.append(ang_vel.copy())

            if counter % control_decimation == 0:
                mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
                viewer.sync()

            target_wall_time = start_wall_time + (d.time - start_sim_time)
            time_to_wait = target_wall_time - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    if save_video:
        print(f"\n\nVideo saved successfully to {video_path}")
        writer.close()

    # Plot curves
    log_path = str(PATH_PARENT / "logs")
    os.makedirs(log_path, exist_ok=True)

    joint_names_config = config.get("mujoco_joint_names", [f"joint_{i}" for i in range(num_actions)])

    log_time = np.array(log_time)
    log_qpos = np.array(log_qpos)
    log_action = np.array(log_action)
    log_target_dof_pos = np.array(log_target_dof_pos)
    log_base_z = np.array(log_base_z)
    log_cmd = np.array(log_cmd)
    log_ang_vel = np.array(log_ang_vel)
    log_torque_time = np.array(log_torque_time)
    log_torque = np.array(log_torque)
    log_motor_vel_time = np.array(log_motor_vel_time)
    log_motor_vel = np.array(log_motor_vel)

    # 打印第 5s 时刻下发的关节力矩 (取最接近 5.0s 的采样点)
    if len(log_torque_time) > 0:
        idx_5s = int(np.argmin(np.abs(log_torque_time - 5.0)))
        t_5s = log_torque_time[idx_5s]
        tau_5s = log_torque[idx_5s]
        print(f"\n[Torque @ ~5s] t={t_5s:.4f}s (sample #{idx_5s})")
        print("  joint   : " + "  ".join(f"{n:>10s}" for n in joint_names_config))
        print("  torque  : " + "  ".join(f"{v:+10.3f}" for v in tau_5s))
        print(f"  abs max : {np.max(np.abs(tau_5s)):.3f} Nm")

    n = len(log_time)
    if n > 0:
        # Plot 1: Joint positions vs time
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('[HIM-AMP] Joint Positions (Qpos) vs Time')
        for i in range(num_actions):
            ax = axes[i // 3][i % 3]
            ax.plot(log_time, log_qpos[:, i], label=f'{joint_names_config[i]} qpos', linewidth=1.5)
            ax.axhline(y=default_angles[i], color='r', linestyle='--', linewidth=1.0, label='default')
            ax.set_ylabel('rad')
            ax.legend(fontsize=7)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'qpos_vs_time.png'), dpi=150)
        plt.close()

        # Plot 2: Actions vs time
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('[HIM-AMP] Policy Actions vs Time')
        for i in range(num_actions):
            ax = axes[i // 3][i % 3]
            ax.plot(log_time, log_action[:, i], label=f'{joint_names_config[i]} action', linewidth=1.5)
            ax.set_ylabel('')
            ax.legend(fontsize=7)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'actions_vs_time.png'), dpi=150)
        plt.close()

        # Plot 3: Body height and command velocity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle('[HIM-AMP] Body Height & Commands')
        ax1.plot(log_time, log_base_z, label='base z', linewidth=1.5, color='b')
        ax1.set_ylabel('m')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(log_time, log_cmd[:, 0], label='cmd_vx', linewidth=1.5)
        ax2.plot(log_time, log_cmd[:, 1], label='cmd_vy', linewidth=1.5)
        ax2.plot(log_time, log_cmd[:, 2], label='cmd_yaw', linewidth=1.5)
        ax2.set_ylabel('cmd')
        ax2.legend()
        ax2.grid(True)
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'height_and_cmd.png'), dpi=150)
        plt.close()

        # Plot 4: Angular velocity
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(log_time, log_ang_vel[:, 0], label='ang_vel_x', linewidth=1.5)
        ax.plot(log_time, log_ang_vel[:, 1], label='ang_vel_y', linewidth=1.5)
        ax.plot(log_time, log_ang_vel[:, 2], label='ang_vel_z', linewidth=1.5)
        ax.set_ylabel('rad/s')
        ax.legend()
        ax.grid(True)
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(log_path, 'ang_vel_vs_time.png'), dpi=150)
        plt.close()

        if len(log_torque_time) > 0:
            # Plot 5: Final torques applied to Mujoco actuators after torque-velocity and top clips.
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle('[HIM-AMP] Applied Joint Torques vs Time')
            for i in range(num_actions):
                ax = axes[i // 3][i % 3]
                ax.plot(log_torque_time, log_torque[:, i], label=f'{joint_names_config[i]} torque', linewidth=1.0)
                ax.axhline(y=top_clip[i], color='r', linestyle='--', linewidth=0.8, label='+top clip')
                ax.axhline(y=-top_clip[i], color='r', linestyle='--', linewidth=0.8, label='-top clip')
                ax.set_ylabel('Nm')
                ax.legend(fontsize=7)
                ax.grid(True)
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'torques_vs_time.png'), dpi=150)
            plt.close()

            torque_abs = np.abs(log_torque)
            torque_max = np.max(torque_abs, axis=0)
            torque_rms = np.sqrt(np.mean(np.square(log_torque), axis=0))

            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            x = np.arange(num_actions)
            ax.bar(x - 0.18, torque_max, width=0.36, label='max |torque|')
            ax.bar(x + 0.18, torque_rms, width=0.36, label='rms torque')
            ax.set_xticks(x)
            ax.set_xticklabels(joint_names_config, rotation=35, ha='right')
            ax.set_ylabel('Nm')
            ax.set_title('[HIM-AMP] Torque Summary')
            ax.grid(True, axis='y')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'torque_summary.png'), dpi=150)
            plt.close()

            print("Torque max |Nm|:", np.array2string(torque_max, precision=2, separator=', '))
            print("Torque RMS Nm:", np.array2string(torque_rms, precision=2, separator=', '))

        if len(log_motor_vel_time) > 0:
            vel_limit = np.zeros(num_actions, dtype=np.float32)
            vel_limit[torque_vel_hip_thigh["indices"]] = torque_vel_hip_thigh["max_vel"]
            vel_limit[torque_vel_calf["indices"]] = torque_vel_calf["max_vel"]

            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle('[HIM-AMP] Motor Velocities vs Time')
            for i in range(num_actions):
                ax = axes[i // 3][i % 3]
                ax.plot(log_motor_vel_time, log_motor_vel[:, i], label=f'{joint_names_config[i]} vel', linewidth=1.0)
                ax.axhline(y=vel_limit[i], color='r', linestyle='--', linewidth=0.8, label='+vel limit')
                ax.axhline(y=-vel_limit[i], color='r', linestyle='--', linewidth=0.8, label='-vel limit')
                ax.set_ylabel('rad/s')
                ax.legend(fontsize=7)
                ax.grid(True)
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'motor_vel_vs_time.png'), dpi=150)
            plt.close()

            motor_vel_abs = np.abs(log_motor_vel)
            motor_vel_max = np.max(motor_vel_abs, axis=0)
            motor_vel_rms = np.sqrt(np.mean(np.square(log_motor_vel), axis=0))

            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
            x = np.arange(num_actions)
            ax.bar(x - 0.18, motor_vel_max, width=0.36, label='max |motor vel|')
            ax.bar(x + 0.18, motor_vel_rms, width=0.36, label='rms motor vel')
            ax.plot(x, vel_limit, color='r', marker='o', linewidth=1.0, label='torque-vel max_vel')
            ax.set_xticks(x)
            ax.set_xticklabels(joint_names_config, rotation=35, ha='right')
            ax.set_ylabel('rad/s')
            ax.set_title('[HIM-AMP] Motor Velocity Summary')
            ax.grid(True, axis='y')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, 'motor_vel_summary.png'), dpi=150)
            plt.close()

            print("Motor vel max |rad/s|:", np.array2string(motor_vel_max, precision=2, separator=', '))
            print("Motor vel RMS rad/s:", np.array2string(motor_vel_rms, precision=2, separator=', '))

        print(f"\nCurves saved to {log_path}/")
