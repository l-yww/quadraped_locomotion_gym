import argparse
import csv
import glob
import os
import pickle
import re
from collections import Counter

import numpy as np


FRAME_KEYS = (
    "root_pos",
    "root_lin_vel",
    "root_rot",
    "root_euler",
    "root_ang_vel",
    "dof_pos",
    "dof_vel",
    "key_body_pos_relative_to_base",
)


def name_number_to_float(token):
    negative = token.startswith("n")
    if negative:
        token = token[1:]
    value = float(token.replace("p", "."))
    return -value if negative else value


def float_to_name_number(value):
    prefix = "n" if value < 0 else ""
    value = abs(float(value))
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return prefix + text.replace(".", "p")


def parse_cowa_command(filename):
    name = os.path.basename(filename).lower()
    match = re.search(
        r"vx(n?[0-9]+p[0-9]+)_vy(n?[0-9]+p[0-9]+)_(w|wn)([0-9]+p[0-9]+)",
        name,
    )
    if match is None:
        return None

    vx = name_number_to_float(match.group(1))
    vy = name_number_to_float(match.group(2))
    wz = name_number_to_float(match.group(4))
    if match.group(3) == "wn":
        wz = -wz
    return vx, vy, wz


def command_to_motion_name(vx, vy, wz):
    if abs(wz) > 1.0e-6:
        w_text = float_to_name_number(abs(wz))
        if abs(vx) <= 1.0e-6 and abs(vy) <= 1.0e-6:
            direction = "left" if wz > 0.0 else "right"
            return f"yaw_{direction}_w{w_text}_50hz_isaacgym.pkl"

        direction = "left" if wz > 0.0 else "right"
        vx_text = float_to_name_number(vx)
        return f"turn_{direction}_vx{vx_text}_w{w_text}_50hz_isaacgym.pkl"

    if abs(vy) > 1.0e-6 and abs(vx) <= 1.0e-6:
        direction = "left" if vy > 0.0 else "right"
        v_text = float_to_name_number(abs(vy))
        return f"strafe_{direction}_v{v_text}_50hz_isaacgym.pkl"

    if abs(vx) > 1.0e-6:
        direction = "forward" if vx > 0.0 else "backward"
        v_text = float_to_name_number(abs(vx))
        return f"{direction}_v{v_text}_w0p0_50hz_isaacgym.pkl"

    return "stand_v0p0_w0p0_50hz_isaacgym.pkl"


def command_to_mode(vx, vy, wz):
    if abs(wz) > 1.0e-6:
        if abs(vx) <= 1.0e-6 and abs(vy) <= 1.0e-6:
            return "yaw_left" if wz > 0.0 else "yaw_right"
        return "turn_left" if wz > 0.0 else "turn_right"
    if abs(vy) > 1.0e-6 and abs(vx) <= 1.0e-6:
        return "strafe_left" if vy > 0.0 else "strafe_right"
    if abs(vx) > 1.0e-6:
        return "forward" if vx > 0.0 else "backward"
    return "stand_other"


def quat_rotate_inverse_np(q, v):
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w * q_w - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * (2.0 * q_w)[..., None]
    c = q_vec * (2.0 * np.sum(q_vec * v, axis=-1, keepdims=True))
    return a - b + c


def crop_frame_arrays(data, start, end, normalize_xy):
    cropped = {}
    for key, value in data.items():
        if key in FRAME_KEYS:
            cropped[key] = value[start:end].copy()
        else:
            cropped[key] = value

    if normalize_xy and "root_pos" in cropped:
        cropped["root_pos"][:, :2] -= cropped["root_pos"][0:1, :2]
    return cropped


def window_stats(data, local_lin_vel, local_ang_vel, targets, start, end):
    vx_t, vy_t, wz_t = targets
    root_euler = np.asarray(data["root_euler"])
    dof_pos = np.asarray(data["dof_pos"])
    dof_vel = np.asarray(data["dof_vel"])
    vx_values = local_lin_vel[start:end, 0]
    vy_values = local_lin_vel[start:end, 1]
    wz_values = local_ang_vel[start:end, 2]

    vx = float(np.mean(vx_values))
    vy = float(np.mean(vy_values))
    wz = float(np.mean(wz_values))

    target_errors = []
    for avg, target in ((vx, vx_t), (vy, vy_t), (wz, wz_t)):
        if abs(target) > 1.0e-6:
            target_errors.append(abs(avg - target) / max(abs(target), 1.0e-6))
    target_error = float(np.mean(target_errors)) if target_errors else 0.0

    extra_error = 0.0
    if abs(vx_t) <= 1.0e-6:
        extra_error += abs(vx) / 0.20
    if abs(vy_t) <= 1.0e-6:
        extra_error += abs(vy) / 0.20
    if abs(wz_t) <= 1.0e-6:
        extra_error += abs(wz) / 0.40

    active_masks = []
    for values, target, floor in (
        (vx_values, vx_t, 0.03),
        (vy_values, vy_t, 0.03),
        (wz_values, wz_t, 0.05),
    ):
        if abs(target) > 1.0e-6:
            threshold = max(abs(target) * 0.35, floor)
            active_masks.append(values * np.sign(target) >= threshold)
    active_ratio = float(np.mean(np.logical_and.reduce(active_masks))) if active_masks else 0.0

    opposite_ratios = []
    for values, target, floor in (
        (vx_values, vx_t, 0.06),
        (vy_values, vy_t, 0.06),
        (wz_values, wz_t, 0.08),
    ):
        if abs(target) > 1.0e-6:
            threshold = max(abs(target) * 0.20, floor)
            opposite_ratios.append(float(np.mean(values * np.sign(target) < -threshold)))
    opposite_ratio = float(np.mean(opposite_ratios)) if opposite_ratios else 0.0

    dof_step_abs_max = 0.0
    if end - start > 1:
        dof_step_abs_max = float(np.max(np.abs(np.diff(dof_pos[start:end], axis=0))))

    return {
        "vx": vx,
        "vy": vy,
        "wz": wz,
        "target_error": target_error,
        "extra_error": extra_error,
        "active_ratio": active_ratio,
        "opposite_ratio": opposite_ratio,
        "vx_std": float(np.std(vx_values)),
        "vy_std": float(np.std(vy_values)),
        "wz_std": float(np.std(wz_values)),
        "roll_abs_max": float(np.max(np.abs(root_euler[start:end, 0]))),
        "pitch_abs_max": float(np.max(np.abs(root_euler[start:end, 1]))),
        "dof_step_abs_max": dof_step_abs_max,
        "dof_vel_abs_max": float(np.max(np.abs(dof_vel[start:end]))),
    }


def choose_stable_window(data, targets, min_frames):
    num_frames = data["root_pos"].shape[0]
    min_frames = min(min_frames, num_frames)
    local_lin_vel = quat_rotate_inverse_np(data["root_rot"], data["root_lin_vel"])
    local_ang_vel = quat_rotate_inverse_np(data["root_rot"], data["root_ang_vel"])

    best_score = None
    best = (0, num_frames, None)
    for length in range(min_frames, num_frames + 1, 5):
        for start in range(0, num_frames - length + 1, 2):
            end = start + length
            stats = window_stats(data, local_lin_vel, local_ang_vel, targets, start, end)
            score = (
                1.60 * stats["target_error"]
                + 0.25 * stats["extra_error"]
                + 1.00 * stats["opposite_ratio"]
                + max(0.0, 0.65 - stats["active_ratio"])
                + 0.15 * (stats["vx_std"] + stats["vy_std"])
                + 0.08 * stats["wz_std"]
                - 0.004 * length / float(num_frames)
            )
            if best_score is None or score < best_score:
                best_score = score
                best = (start, end, {**stats, "score": score})
    return best


def status_from_stats(stats, mode):
    issues = []
    if stats["target_error"] > 0.25:
        issues.append("target_speed_error")
    if stats["extra_error"] > 0.35:
        issues.append("extra_axis_drift")
    if stats["active_ratio"] < 0.65 and mode != "stand_other":
        issues.append("low_active_ratio")
    if stats["opposite_ratio"] > 0.12:
        issues.append("opposite_spikes")
    if stats["roll_abs_max"] > 0.70:
        issues.append("large_roll")
    if stats["dof_step_abs_max"] > 1.20:
        issues.append("joint_discontinuity")
    if stats["dof_vel_abs_max"] > 80.0:
        issues.append("huge_dof_vel")
    return ("WARN" if issues else "OK"), issues


def prepare_dataset(args):
    os.makedirs(args.dst, exist_ok=True)
    rows = []
    used_names = Counter()

    for src_path in sorted(glob.glob(os.path.join(args.src, "*.pkl"))):
        src_name = os.path.basename(src_path)
        command = parse_cowa_command(src_name)
        if command is None:
            raise ValueError(f"Could not parse command from filename: {src_path}")
        vx, vy, wz = command
        mode = command_to_mode(vx, vy, wz)
        dst_name = command_to_motion_name(vx, vy, wz)
        used_names[dst_name] += 1
        if used_names[dst_name] > 1:
            stem, ext = os.path.splitext(dst_name)
            dst_name = f"{stem}_{used_names[dst_name]:02d}{ext}"

        with open(src_path, "rb") as f:
            data = pickle.load(f)

        start, end, stats = choose_stable_window(data, command, args.min_frames)
        cropped = crop_frame_arrays(data, start, end, normalize_xy=args.normalize_xy)
        dst_path = os.path.join(args.dst, dst_name)
        with open(dst_path, "wb") as f:
            pickle.dump(cropped, f)

        status, issues = status_from_stats(stats, mode)
        fps = float(data.get("fps", 50))
        rows.append(
            {
                "src_file": src_name,
                "dst_file": dst_name,
                "mode": mode,
                "cmd_vx": vx,
                "cmd_vy": vy,
                "cmd_wz": wz,
                "status": status,
                "issues": ";".join(issues),
                "start": start,
                "end": end,
                "frames": end - start,
                "duration": (end - start - 1) / fps,
                **stats,
            }
        )

    fieldnames = [
        "src_file",
        "dst_file",
        "mode",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "status",
        "issues",
        "start",
        "end",
        "frames",
        "duration",
        "score",
        "vx",
        "vy",
        "wz",
        "target_error",
        "extra_error",
        "active_ratio",
        "opposite_ratio",
        "vx_std",
        "vy_std",
        "wz_std",
        "roll_abs_max",
        "pitch_abs_max",
        "dof_step_abs_max",
        "dof_vel_abs_max",
    ]
    with open(args.report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.warn_list:
        with open(args.warn_list, "w") as f:
            for row in rows:
                if row["status"] != "OK":
                    f.write(f"{row['dst_file']}\t{row['mode']}\t{row['issues']}\n")

    print(f"Prepared {len(rows)} files -> {args.dst}")
    print("Status:", dict(Counter(row["status"] for row in rows)))
    print("Modes:", dict(sorted(Counter(row["mode"] for row in rows).items())))
    print("Report:", args.report)
    if args.warn_list:
        print("Warn list:", args.warn_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="resources/mocap_motions_batch_processed")
    parser.add_argument("--dst", default="resources/mocap_motions_batch_processed_cleaned")
    parser.add_argument("--report", default="resources/mocap_motions_batch_processed_cleaned_report.csv")
    parser.add_argument("--warn-list", default="resources/mocap_motions_batch_processed_cleaned_warn_files.txt")
    parser.add_argument("--min-frames", type=int, default=40)
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize-xy", dest="normalize_xy", action="store_true")
    normalize_group.add_argument("--no-normalize-xy", dest="normalize_xy", action="store_false")
    parser.set_defaults(normalize_xy=True)
    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
