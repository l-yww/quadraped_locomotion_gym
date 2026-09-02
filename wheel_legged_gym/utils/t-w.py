import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def acceleration_limit(speed, v101, T101, v126, T126, v150, T150):
    limit = np.zeros_like(speed)
    # 0 ~ v101
    mask = speed <= v101
    limit[mask] = T101
    # v101 ~ v126
    mask = (speed > v101) & (speed <= v126)
    limit[mask] = T101 + (T126 - T101) * (speed[mask] - v101) / (v126 - v101)
    # v126 ~ v150
    mask = (speed > v126) & (speed <= v150)
    limit[mask] = T126 + (T150 - T126) * (speed[mask] - v126) / (v150 - v126)
    # > v150 -> 0
    return limit

def positive_speed_limits(speed, v101, T101, v126, T126, v150, T150):
    accel = acceleration_limit(speed, v101, T101, v126, T126, v150, T150)
    tail_slope = (T150 - T126) / max(v150 - v126, 1e-6)
    v_int = v150 + (-T101 - T150) / tail_slope
    brake_extension = T150 + tail_slope * (speed - v150)
    brake_extension = np.clip(np.minimum(brake_extension, 0.0), -T101, 0.0)
    upper = np.where(speed <= v150, accel, np.where(speed <= v_int, brake_extension, 0.0))
    lower = np.where(speed <= v_int, -T101, 0.0)
    return lower, upper

# 参数：腿关节
leg_params = {
    'v101': 101.0 * (2.0 * np.pi / 60.0),
    'T101': 200.0,
    'v126': 126.0 * (2.0 * np.pi / 60.0),
    'T126': 162.0,
    'v150': 150.0 * (2.0 * np.pi / 60.0),
    'T150': 98.0,
}
# 参数：轮关节
wheel_params = {
    'v101': 101.0 * (2.0 * np.pi / 60.0) * (30.0 / 12.91),
    'T101': (200.0 / 6.67 * 3.0) / (30.0 / 12.91),
    'v126': 126.0 * (2.0 * np.pi / 60.0) * (30.0 / 12.91),
    'T126': (162.0 / 6.67 * 3.0) / (30.0 / 12.91),
    'v150': 150.0 * (2.0 * np.pi / 60.0) * (30.0 / 12.91),
    'T150': (98.0 / 6.67 * 3.0) / (30.0 / 12.91),
}

def compute_limits(speed, p):
    abs_speed = np.abs(speed)
    lower_pos, upper_pos = positive_speed_limits(abs_speed, **{k: p[k] for k in ['v101','T101','v126','T126','v150','T150']})
    upper = np.where(speed >= 0, upper_pos, -lower_pos)
    lower = np.where(speed >= 0, lower_pos, -upper_pos)
    return upper, lower

def plot_range(params):
    tail_slope = (params['T150'] - params['T126']) / (params['v150'] - params['v126'])
    intersection = params['v150'] + (params['T101'] + params['T150']) / (-tail_slope)
    return intersection * 1.08

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (title, params) in zip(axes, [("Leg joints", leg_params), ("Wheel joints", wheel_params)]):
    max_speed = plot_range(params)
    speeds = np.linspace(-max_speed, max_speed, 1200)
    upper, lower = compute_limits(speeds, params)
    ax.fill_between(speeds, lower, upper, alpha=0.2, label='allowed torque')
    ax.plot(speeds, upper, 'r', label='upper limit')
    ax.plot(speeds, lower, 'b', label='lower limit')
    for v in [params['v101'], params['v126'], params['v150']]:
        ax.axvline(v, color='gray', lw=0.5, ls='--')
        ax.axvline(-v, color='gray', lw=0.5, ls='--')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Joint velocity (rad/s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('wheel_legged_gym/utils/quadwheel_torque_velocity_777constraints.png', dpi=200)
