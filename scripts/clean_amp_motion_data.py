import argparse
import csv
import glob
import math
import os
import pickle
import re
import shutil

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


def parse_motion_name(filename):
    name = os.path.basename(filename).lower()

    match = re.search(r"forward_v([0-9]+p[0-9]+)_w0p0_", name)
    if match is not None:
        return "forward", name_number_to_float(match.group(1)), 0.0, 0.0

    match = re.search(r"backward_v([0-9]+p[0-9]+)_w0p0_", name)
    if match is not None:
        return "backward", -name_number_to_float(match.group(1)), 0.0, 0.0

    match = re.search(r"strafe_(left|right)_v([0-9]+p[0-9]+)_", name)
    if match is not None:
        speed = name_number_to_float(match.group(2))
        vy = speed if match.group(1) == "left" else -speed
        return f"strafe_{match.group(1)}", 0.0, vy, 0.0

    match = re.search(r"yaw_(left|right)_w([0-9]+p[0-9]+)_", name)
    if match is not None:
        speed = name_number_to_float(match.group(2))
        wz = speed if match.group(1) == "left" else -speed
        return f"yaw_{match.group(1)}", 0.0, 0.0, wz

    match = re.search(r"turn_(left|right)_vx(n?[0-9]+p[0-9]+)_w([0-9]+p[0-9]+)_", name)
    if match is not None:
        vx = name_number_to_float(match.group(2))
        speed = name_number_to_float(match.group(3))
        wz = speed if match.group(1) == "left" else -speed
        return f"turn_{match.group(1)}", vx, 0.0, wz

    return "unknown", math.nan, math.nan, math.nan


def signed_active(values, target, frac, min_abs):
    if abs(target) <= 1e-6:
        return np.ones(values.shape, dtype=bool)
    threshold = max(abs(target) * frac, min_abs)
    return values * np.sign(target) >= threshold


def mark_step_bad(dof_pos, max_step):
    if dof_pos.shape[0] < 2:
        return np.zeros(dof_pos.shape[0], dtype=bool)
    step = np.max(np.abs(np.diff(dof_pos, axis=0)), axis=1)
    bad = np.zeros(dof_pos.shape[0], dtype=bool)
    bad[:-1] |= step > max_step
    bad[1:] |= step > max_step
    return bad


def window_stats(data, targets, start, end):
    vx_t, vy_t, wz_t = targets
    fps = float(data["fps"])
    duration = max((end - start - 1) / fps, 1.0 / fps)
    pos = data["root_pos"]
    yaw = np.unwrap(data["root_euler"][:, 2])

    vx_avg = float((pos[end - 1, 0] - pos[start, 0]) / duration)
    vy_avg = float((pos[end - 1, 1] - pos[start, 1]) / duration)
    wz_avg = float((yaw[end - 1] - yaw[start]) / duration)

    expected_errors = []
    for avg, target in ((vx_avg, vx_t), (vy_avg, vy_t), (wz_avg, wz_t)):
        if abs(target) > 1e-6:
            expected_errors.append(abs(avg - target) / max(abs(target), 1e-6))
    target_error = float(np.mean(expected_errors)) if expected_errors else 0.0

    extra_error = 0.0
    if abs(vx_t) <= 1e-6:
        extra_error += abs(vx_avg) / 0.25
    if abs(vy_t) <= 1e-6:
        extra_error += abs(vy_avg) / 0.25
    if abs(wz_t) <= 1e-6:
        extra_error += abs(wz_avg) / 0.50

    vx = data["root_lin_vel"][start:end, 0]
    vy = data["root_lin_vel"][start:end, 1]
    wz = data["root_ang_vel"][start:end, 2]
    active_masks = []
    if abs(vx_t) > 1e-6:
        active_masks.append(signed_active(vx, vx_t, 0.35, 0.03))
    if abs(vy_t) > 1e-6:
        active_masks.append(signed_active(vy, vy_t, 0.35, 0.03))
    if abs(wz_t) > 1e-6:
        active_masks.append(signed_active(wz, wz_t, 0.35, 0.05))
    if active_masks:
        active_ratio = float(np.mean(np.logical_and.reduce(active_masks)))
    else:
        active_ratio = 0.0

    opposite_ratios = []
    for values, target, floor in ((vx, vx_t, 0.06), (vy, vy_t, 0.06), (wz, wz_t, 0.08)):
        if abs(target) > 1e-6:
            threshold = max(abs(target) * 0.2, floor)
            opposite_ratios.append(float(np.mean(values * np.sign(target) < -threshold)))
    opposite_ratio = float(np.mean(opposite_ratios)) if opposite_ratios else 0.0

    return {
        "duration": duration,
        "vx_avg": vx_avg,
        "vy_avg": vy_avg,
        "wz_avg": wz_avg,
        "target_error": target_error,
        "extra_error": extra_error,
        "active_ratio": active_ratio,
        "opposite_ratio": opposite_ratio,
    }


def choose_window(data, targets, min_frames, max_roll, max_step, search_stride, length_stride):
    num_frames = data["root_pos"].shape[0]
    min_frames = min(min_frames, num_frames)
    bad_roll = np.abs(data["root_euler"][:, 0]) > max_roll
    bad_step = mark_step_bad(data["dof_pos"], max_step)
    bad = bad_roll | bad_step

    best_score = None
    best = (0, num_frames, None)
    max_frames = num_frames

    for start in range(0, num_frames - min_frames + 1, search_stride):
        for end in range(start + min_frames, max_frames + 1, length_stride):
            stats = window_stats(data, targets, start, end)
            bad_ratio = float(np.mean(bad[start:end]))
            # Prefer windows that match the command, have few bad frames, and are still long enough
            # to contain several gait cycles.
            length_bonus = 0.015 * (end - start) / float(num_frames)
            score = (
                1.25 * stats["target_error"]
                + 0.20 * stats["extra_error"]
                + 3.00 * bad_ratio
                + 1.00 * stats["opposite_ratio"]
                + max(0.0, 0.55 - stats["active_ratio"])
                - length_bonus
            )
            if best_score is None or score < best_score:
                best_score = score
                best = (start, end, {**stats, "bad_ratio": bad_ratio, "score": score})

    return best


def smooth_edges(values, passes):
    if passes <= 0 or values.shape[0] < 5:
        return values
    out = values.astype(np.float32, copy=True)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32) / 9.0
    pad = len(kernel) // 2
    for _ in range(passes):
        padded = np.pad(out, [(pad, pad)] + [(0, 0)] * (out.ndim - 1), mode="edge")
        smoothed = np.zeros_like(out)
        for i, weight in enumerate(kernel):
            smoothed += weight * padded[i:i + out.shape[0]]
        out = smoothed
    return out.astype(values.dtype, copy=False)


def crop_motion(data, start, end, normalize_xy, smooth_dof_passes):
    cropped = {}
    for key, value in data.items():
        if key in FRAME_KEYS:
            cropped[key] = value[start:end].copy()
        else:
            cropped[key] = value

    if normalize_xy:
        cropped["root_pos"][:, :2] -= cropped["root_pos"][0:1, :2]

    if smooth_dof_passes > 0:
        cropped["dof_pos"] = smooth_edges(cropped["dof_pos"], smooth_dof_passes)
        fps = float(cropped["fps"])
        if cropped["dof_pos"].shape[0] >= 2:
            cropped["dof_vel"] = np.gradient(cropped["dof_pos"], 1.0 / fps, axis=0).astype(
                cropped["dof_vel"].dtype,
                copy=False,
            )

    return cropped


def audit_cleaned(data, targets, max_roll, max_step):
    num_frames = data["root_pos"].shape[0]
    stats = window_stats(data, targets, 0, num_frames)
    roll_abs_max = float(np.max(np.abs(data["root_euler"][:, 0])))
    step_abs_max = float(np.max(np.abs(np.diff(data["dof_pos"], axis=0)))) if num_frames > 1 else 0.0
    dq_abs_max = float(np.max(np.abs(data["dof_vel"])))
    vy_p95 = float(np.percentile(np.abs(data["root_lin_vel"][:, 1]), 95))

    issues = []
    if stats["target_error"] > 0.35:
        issues.append("target_speed_error")
    if stats["active_ratio"] < 0.50:
        issues.append("low_active_ratio")
    if stats["opposite_ratio"] > 0.12:
        issues.append("opposite_spikes")
    if roll_abs_max > max_roll:
        issues.append("large_body_tilt")
    if step_abs_max > max_step:
        issues.append("joint_discontinuity")
    if abs(targets[1]) <= 1e-6 and vy_p95 > 0.25:
        issues.append("sustained_extra_vy")

    return {
        **stats,
        "roll_abs_max": roll_abs_max,
        "dof_step_abs_max": step_abs_max,
        "dof_vel_abs_max": dq_abs_max,
        "abs_vy_p95": vy_p95,
        "issues": ";".join(issues),
        "status": "WARN" if issues else "OK",
    }


def clean_dataset(args):
    os.makedirs(args.dst, exist_ok=True)
    rows = []
    src_files = sorted(glob.glob(os.path.join(args.src, "*.pkl")))

    for src_path in src_files:
        filename = os.path.basename(src_path)
        mode, vx_t, vy_t, wz_t = parse_motion_name(filename)
        dst_path = os.path.join(args.dst, filename)

        with open(src_path, "rb") as f:
            data = pickle.load(f)

        if mode == "unknown":
            shutil.copy2(src_path, dst_path)
            rows.append({"file": filename, "mode": mode, "status": "COPIED_UNKNOWN"})
            continue

        start, end, before_stats = choose_window(
            data,
            (vx_t, vy_t, wz_t),
            args.min_frames,
            args.max_roll,
            args.max_step,
            args.search_stride,
            args.length_stride,
        )
        cleaned = crop_motion(data, start, end, args.normalize_xy, args.smooth_dof_passes)

        with open(dst_path, "wb") as f:
            pickle.dump(cleaned, f, protocol=pickle.HIGHEST_PROTOCOL)

        after_stats = audit_cleaned(cleaned, (vx_t, vy_t, wz_t), args.max_roll, args.max_step)
        rows.append({
            "file": filename,
            "mode": mode,
            "target_vx": vx_t,
            "target_vy": vy_t,
            "target_wz": wz_t,
            "orig_frames": data["root_pos"].shape[0],
            "clean_frames": cleaned["root_pos"].shape[0],
            "start_frame": start,
            "end_frame_exclusive": end,
            "pre_score": before_stats["score"],
            "pre_bad_ratio": before_stats["bad_ratio"],
            "pre_active_ratio": before_stats["active_ratio"],
            "pre_target_error": before_stats["target_error"],
            **{f"clean_{k}": v for k, v in after_stats.items()},
        })

    if rows:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        with open(args.report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    counts = {}
    for row in rows:
        status = row.get("clean_status", row.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    print(f"wrote cleaned motions to {args.dst}")
    print(f"wrote report to {args.report}")
    print("status_counts", counts)

    warn_rows = [row for row in rows if row.get("clean_status") == "WARN"]
    if warn_rows:
        print("remaining WARN files:")
        for row in warn_rows[:80]:
            print(
                row["file"],
                row.get("clean_issues", ""),
                f"frames={row['clean_frames']}",
                f"active={float(row['clean_active_ratio']):.2f}",
                f"target_err={float(row['clean_target_error']):.2f}",
                f"roll={float(row['clean_roll_abs_max']):.3f}",
                f"step={float(row['clean_dof_step_abs_max']):.3f}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="resources/d1_data_v2")
    parser.add_argument("--dst", default="resources/d1_data_v2_cleaned")
    parser.add_argument("--report", default="resources/d1_data_v2_cleaned_report.csv")
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--max-roll", type=float, default=0.18)
    parser.add_argument("--max-step", type=float, default=0.18)
    parser.add_argument("--search-stride", type=int, default=2)
    parser.add_argument("--length-stride", type=int, default=2)
    parser.add_argument("--smooth-dof-passes", type=int, default=0)
    parser.set_defaults(normalize_xy=True)
    parser.add_argument("--normalize-xy", dest="normalize_xy", action="store_true")
    parser.add_argument("--no-normalize-xy", dest="normalize_xy", action="store_false")
    args = parser.parse_args()
    clean_dataset(args)


if __name__ == "__main__":
    main()
