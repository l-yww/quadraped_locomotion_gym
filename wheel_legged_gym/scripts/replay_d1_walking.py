import glob
import os
import pickle
import sys
import time

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR

# Compatibility shim: pickle files saved with NumPy >= 2.0 reference
# numpy._core, which does not exist in NumPy < 2.0.
if not hasattr(np, "_core"):
    import numpy.core as _np_core

    for _attr in dir(_np_core):
        _mod_name = f"numpy._core.{_attr}"
        _submod = getattr(_np_core, _attr, None)
        if isinstance(_submod, type(sys)):
            sys.modules.setdefault(_mod_name, _submod)
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
    del _np_core, _attr, _mod_name, _submod


DEFAULT_MOTION_DIR = os.path.join(
    WHEEL_LEGGED_GYM_ROOT_DIR, "resources", "amp_dataset", "d1_walking"
)
DEFAULT_ASSET_FILE = os.path.join(
    WHEEL_LEGGED_GYM_ROOT_DIR,
    "resources",
    "robots",
    "cowa_quadruped_arm_v1",
    "urdf",
    "cowa_quadruped_arm_v1_fix_arm.urdf",
)


def parse_args():
    custom_args = [
        {
            "name": "--motion-dir",
            "type": str,
            "default": DEFAULT_MOTION_DIR,
            "help": "Directory containing d1_walking *_isaacgym.pkl clips.",
        },
        {
            "name": "--motion-file",
            "type": str,
            "default": None,
            "help": "Replay a single motion pkl instead of all clips in --motion-dir.",
        },
        {
            "name": "--asset-file",
            "type": str,
            "default": DEFAULT_ASSET_FILE,
            "help": "URDF asset to replay on.",
        },
        {"name": "--clip-index", "type": int, "default": 0, "help": "Starting clip index."},
        {"name": "--speed", "type": float, "default": 1.0, "help": "Playback speed scale."},
        {"name": "--loop", "action": "store_true", "default": False, "help": "Loop clips."},
        {
            "name": "--keep-root-xy",
            "action": "store_true",
            "default": False,
            "help": "Use the absolute root x/y stored in the motion file.",
        },
        {
            "name": "--height-offset",
            "type": float,
            "default": 0.0,
            "help": "Additional z offset applied to root_pos.",
        },
        {
            "name": "--show-axis",
            "action": "store_true",
            "default": False,
            "help": "Draw a small XYZ axis at the robot base.",
        },
    ]
    return gymutil.parse_arguments(
        description=(
            "Replay d1_walking motion clips on "
            "cowa_quadruped_arm_v1_fix_arm.urdf in Isaac Gym."
        ),
        custom_parameters=custom_args,
    )


def load_motion_files(motion_dir, motion_file):
    if motion_file:
        motion_files = [os.path.abspath(motion_file)]
    else:
        motion_files = sorted(glob.glob(os.path.join(motion_dir, "*.pkl")))

    if not motion_files:
        raise FileNotFoundError(f"No pkl motion files found in {motion_dir}")
    return motion_files


def load_motion(path):
    with open(path, "rb") as f:
        motion = pickle.load(f)

    required_keys = [
        "fps",
        "root_pos",
        "root_rot",
        "root_lin_vel",
        "root_ang_vel",
        "dof_pos",
        "dof_vel",
    ]
    missing = [key for key in required_keys if key not in motion]
    if missing:
        raise KeyError(f"{path} missing motion fields: {missing}")

    num_frames = motion["root_pos"].shape[0]
    if motion["dof_pos"].shape[0] != num_frames or motion["dof_vel"].shape[0] != num_frames:
        raise ValueError(f"{path} has inconsistent root and dof frame counts")
    return motion


def create_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)
    return sim


def load_robot_asset(gym, sim, asset_file):
    asset_file = os.path.abspath(asset_file)
    asset_root = os.path.dirname(os.path.dirname(asset_file))
    asset_name = os.path.join(os.path.basename(os.path.dirname(asset_file)), os.path.basename(asset_file))

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_options.collapse_fixed_joints = True
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = False

    robot_asset = gym.load_asset(sim, asset_root, asset_name, asset_options)
    if robot_asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_file}")
    return robot_asset


def set_actor_state(
    gym,
    sim,
    root_states,
    dof_states,
    root_pos,
    root_rot,
    root_lin_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
):
    device = root_states.device
    root_states[0, 0:3] = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    root_states[0, 3:7] = torch.as_tensor(root_rot, dtype=torch.float32, device=device)
    root_states[0, 7:10] = torch.as_tensor(root_lin_vel, dtype=torch.float32, device=device)
    root_states[0, 10:13] = torch.as_tensor(root_ang_vel, dtype=torch.float32, device=device)
    dof_states[:, 0] = torch.as_tensor(dof_pos, dtype=torch.float32, device=device)
    dof_states[:, 1] = torch.as_tensor(dof_vel, dtype=torch.float32, device=device)

    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))


def draw_base_axis(gym, viewer, env, pos, rot):
    gym.clear_lines(viewer)
    origin = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
    axes = [
        (gymapi.Vec3(0.25, 0.0, 0.0), gymapi.Vec3(1.0, 0.0, 0.0)),
        (gymapi.Vec3(0.0, 0.25, 0.0), gymapi.Vec3(0.0, 1.0, 0.0)),
        (gymapi.Vec3(0.0, 0.0, 0.25), gymapi.Vec3(0.0, 0.0, 1.0)),
    ]
    transform = gymapi.Transform(origin, gymapi.Quat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])))
    for axis, color in axes:
        endpoint = transform.transform_point(axis)
        gymutil.draw_line(origin, endpoint, color, gym, viewer, env)


def replay_clip(gym, sim, viewer, env, root_states, dof_states, motion_path, args, expected_dofs):
    motion = load_motion(motion_path)
    fps = float(motion["fps"])
    if fps <= 0.0:
        raise ValueError(f"{motion_path} has invalid fps: {fps}")
    if args.speed <= 0.0:
        raise ValueError(f"--speed must be positive, got {args.speed}")
    frame_time = 1.0 / (fps * args.speed)
    num_frames = motion["root_pos"].shape[0]

    dof_pos = motion["dof_pos"]
    dof_vel = motion["dof_vel"]
    if dof_pos.shape[1] != expected_dofs:
        raise ValueError(
            f"{motion_path} has {dof_pos.shape[1]} DOFs, but asset has {expected_dofs} DOFs"
        )

    root_xy0 = motion["root_pos"][0, :2].copy()
    print(f"Replay {os.path.basename(motion_path)}: {num_frames} frames @ {fps:g} Hz")

    for frame_id in range(num_frames):
        if gym.query_viewer_has_closed(viewer):
            return False

        root_pos = motion["root_pos"][frame_id].copy()
        if not args.keep_root_xy:
            root_pos[:2] -= root_xy0
        root_pos[2] += args.height_offset

        set_actor_state(
            gym,
            sim,
            root_states,
            dof_states,
            root_pos,
            motion["root_rot"][frame_id],
            motion["root_lin_vel"][frame_id],
            motion["root_ang_vel"][frame_id],
            dof_pos[frame_id],
            dof_vel[frame_id],
        )

        if args.show_axis:
            draw_base_axis(gym, viewer, env, root_pos, motion["root_rot"][frame_id])

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        time.sleep(max(frame_time - (1.0 / 60.0), 0.0))

    return True


def main():
    script_args = parse_args()
    motion_files = load_motion_files(script_args.motion_dir, script_args.motion_file)
    start_index = max(0, min(script_args.clip_index, len(motion_files) - 1))

    gym = gymapi.acquire_gym()
    sim = create_sim(gym, script_args)

    viewer_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, viewer_props)
    if viewer is None:
        raise RuntimeError("Failed to create Isaac Gym viewer")

    robot_asset = load_robot_asset(gym, sim, script_args.asset_file)
    expected_dofs = gym.get_asset_dof_count(robot_asset)
    dof_names = gym.get_asset_dof_names(robot_asset)
    print(f"Loaded asset: {script_args.asset_file}")
    print(f"Asset DOFs ({expected_dofs}): {dof_names}")

    env = gym.create_env(
        sim,
        gymapi.Vec3(-1.0, -1.0, 0.0),
        gymapi.Vec3(1.0, 1.0, 1.0),
        1,
    )
    actor = gym.create_actor(env, robot_asset, gymapi.Transform(), "cowa_quadruped_arm_v1_fix_arm", 0, 1)

    dof_props = gym.get_asset_dof_properties(robot_asset)
    dof_props["driveMode"].fill(int(gymapi.DOF_MODE_NONE))
    gym.set_actor_dof_properties(env, actor, dof_props)
    gym.prepare_sim(sim)

    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_states = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))

    gym.viewer_camera_look_at(
        viewer,
        None,
        gymapi.Vec3(2.0, -2.0, 1.2),
        gymapi.Vec3(0.0, 0.0, 0.35),
    )

    clip_index = start_index
    while True:
        keep_running = replay_clip(
            gym,
            sim,
            viewer,
            env,
            root_states,
            dof_states,
            motion_files[clip_index],
            script_args,
            expected_dofs,
        )
        if not keep_running or gym.query_viewer_has_closed(viewer):
            break

        clip_index += 1
        if clip_index >= len(motion_files):
            if not script_args.loop:
                break
            clip_index = 0

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

# 查看某条数据
# python wheel_legged_gym/scripts/replay_d1_walking.py --motion-file resources/isaacgym_processed/cowa_spin_inplace_reverse_w1_50hz_isaacgym.pkl --loop

# 查看某个目录下的所有数据
# python wheel_legged_gym/scripts/replay_d1_walking.py --physx --sim_device cuda:0 --pipeline gpu --loop --motion-dir resources/isaacgym_processed