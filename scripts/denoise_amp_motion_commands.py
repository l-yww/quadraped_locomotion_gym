import argparse
import csv
import glob
import math
import os
import pickle
import re

import numpy as np


def number_token_to_float(token):
    negative = token.startswith("n")
    if negative:
        token = token[1:]
    value = float(token.replace("p", "."))
    return -value if negative else value


def parse_motion_name(path):
    name = os.path.basename(path).lower()

    match = re.search(r"forward_v([0-9]+p[0-9]+)_w0p0_", name)
    if match is not None:
        return "forward", number_token_to_float(match.group(1)), 0.0, 0.0

    match = re.search(r"backward_v([0-9]+p[0-9]+)_w0p0_", name)
    if match is not None:
        return "backward", -number_token_to_float(match.group(1)), 0.0, 0.0

    match = re.search(r"strafe_(left|right)_v([0-9]+p[0-9]+)_", name)
    if match is not None:
        speed = number_token_to_float(match.group(2))
        return f"strafe_{match.group(1)}", 0.0, speed if match.group(1) == "left" else -speed, 0.0

    match = re.search(r"yaw_(left|right)_w([0-9]+p[0-9]+)_", name)
    if match is not None:
        speed = number_token_to_float(match.group(2))
        return f"yaw_{match.group(1)}", 0.0, 0.0, speed if match.group(1) == "left" else -speed

    match = re.search(r"turn_(left|right)_vx(n?[0-9]+p[0-9]+)_w([0-9]+p[0-9]+)_", name)
    if match is not None:
        vx = number_token_to_float(match.group(2))
        speed = number_token_to_float(match.group(3))
        return f"turn_{match.group(1)}", vx, 0.0, speed if match.group(1) == "left" else -speed

    return "unknown", math.nan, math.nan, math.nan


def quat_from_euler_xyz(euler):
    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    quat = np.empty((euler.shape[0], 4), dtype=np.float32)
    quat[:, 0] = sr * cp * cy - cr * sp * sy
    quat[:, 1] = cr * sp * cy + sr * cp * sy
    quat[:, 2] = cr * cp * sy - sr * sp * cy
    quat[:, 3] = cr * cp * cy + sr * sp * sy
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return quat


def summarize(motion):
    pos = motion["root_pos"]
    euler = motion["root_euler"]
    lin_vel = motion["root_lin_vel"]
    ang_vel = motion["root_ang_vel"]
    yaw = np.unwrap(euler[:, 2])
    return {
        "frames": int(pos.shape[0]),
        "dx": float(pos[-1, 0] - pos[0, 0]),
        "dy": float(pos[-1, 1] - pos[0, 1]),
        "lat_max": float(np.max(np.abs(pos[:, 1] - pos[0, 1]))),
        "yaw_delta": float(yaw[-1] - yaw[0]),
        "vy_p95": float(np.percentile(np.abs(lin_vel[:, 1]), 95)),
        "wz_p95": float(np.percentile(np.abs(ang_vel[:, 2]), 95)),
        "roll_abs_max": float(np.max(np.abs(euler[:, 0]))),
    }


def denoise_motion(motion, mode, vx, vy, wz, straighten_pose):
    out = {}
    for key, value in motion.items():
        out[key] = value.copy() if isinstance(value, np.ndarray) else value

    pure_x = mode in ("forward", "backward")
    pure_y = mode in ("strafe_left", "strafe_right")
    pure_yaw = mode in ("yaw_left", "yaw_right")
    turn = mode in ("turn_left", "turn_right")

    if pure_x:
        out["root_lin_vel"][:, 1] = 0.0
        out["root_ang_vel"][:, 2] = 0.0
        if straighten_pose:
            out["root_pos"][:, 1] = out["root_pos"][0, 1]
            out["root_euler"][:, 2] = out["root_euler"][0, 2]
            out["root_rot"] = quat_from_euler_xyz(out["root_euler"]).astype(out["root_rot"].dtype, copy=False)

    elif pure_y:
        out["root_lin_vel"][:, 0] = 0.0
        out["root_ang_vel"][:, 2] = 0.0
        if straighten_pose:
            out["root_pos"][:, 0] = out["root_pos"][0, 0]
            out["root_euler"][:, 2] = out["root_euler"][0, 2]
            out["root_rot"] = quat_from_euler_xyz(out["root_euler"]).astype(out["root_rot"].dtype, copy=False)

    elif pure_yaw:
        out["root_lin_vel"][:, 0:2] = 0.0
        if straighten_pose:
            out["root_pos"][:, 0:2] = out["root_pos"][0:1, 0:2]

    elif turn:
        # Keep commanded vx and wz, but remove side-slip from the expert observation.
        out["root_lin_vel"][:, 1] = 0.0
        if straighten_pose:
            out["root_pos"][:, 1] = out["root_pos"][0, 1]

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="resources/d1_data_v2_cleaned40_ok")
    parser.add_argument("--dst", default="resources/d1_data_v2_cleaned40_ok_straight")
    parser.add_argument("--report", default="resources/d1_data_v2_cleaned40_ok_straight_report.csv")
    parser.add_argument("--straighten-pose", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    rows = []
    for src_path in sorted(glob.glob(os.path.join(args.src, "*.pkl"))):
        filename = os.path.basename(src_path)
        mode, vx, vy, wz = parse_motion_name(filename)
        with open(src_path, "rb") as f:
            motion = pickle.load(f)

        before = summarize(motion)
        denoised = denoise_motion(motion, mode, vx, vy, wz, args.straighten_pose)
        after = summarize(denoised)

        dst_path = os.path.join(args.dst, filename)
        with open(dst_path, "wb") as f:
            pickle.dump(denoised, f, protocol=pickle.HIGHEST_PROTOCOL)

        row = {"file": filename, "mode": mode}
        row.update({f"before_{key}": value for key, value in before.items()})
        row.update({f"after_{key}": value for key, value in after.items()})
        rows.append(row)

    if rows:
        keys = list(rows[0].keys())
        with open(args.report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    print(f"wrote {len(rows)} files to {args.dst}")
    print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
