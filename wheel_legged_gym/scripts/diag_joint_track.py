"""临时诊断：悬空 sin 跟踪环境的 obs / action / ref / reward 数值检查。
跑几步，打印关键量，定位 reward 发散原因。"""
import isaacgym  # noqa
import torch
from wheel_legged_gym.envs import task_registry
from wheel_legged_gym.utils import get_args


def main():
    args = get_args()
    args.task = 'quadruped_joint_track'
    args.num_envs = 4
    args.headless = True
    env, cfg = task_registry.make_env(name=args.task, args=args)

    print("\n========== ENV INFO ==========")
    print("num_dof:", env.num_dof, " num_actions:", env.num_actions)
    print("num_single_obs:", cfg.env.num_single_obs)
    print("default_dof_pos:", env.default_dof_pos[0].cpu().numpy())
    print("ref_amplitude:", env.ref_amplitude[0].cpu().numpy())
    print("cycle_time:", env.cycle_time)
    print("dt:", env.dt, " decimation:", cfg.control.decimation)

    # 用全 0 action 跑几步（看 PD 把腿拉回 default 的过程）
    print("\n========== STEP WITH ZERO ACTION ==========")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for s in range(8):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        out = env.step(actions)
        obs, priv, rew, done, info = out[0], out[1], out[2], out[3], out[4]
        env.compute_ref_state()
        print(f"\n--- step {s} ---")
        print("episode_length_buf:", env.episode_length_buf[:4].cpu().numpy())
        print("phase:", env._get_phase()[:4].flatten().cpu().numpy())
        print("ref_dof_pos[0]:", env.ref_dof_pos[0].cpu().numpy())
        print("dof_pos[0]:    ", env.dof_pos[0].cpu().numpy())
        print("dof_vel[0]:    ", env.dof_vel[0].cpu().numpy())
        print("tracking_reward[0]:", env._reward_tracking_joint_pos()[:4].cpu().numpy())
        print("total rew[:4]:", rew[:4].cpu().numpy())

    # ★ Oracle 测试：直接给 action = ref_dof_pos - default（理想跟踪）
    #   跑长一点（覆盖多个 sin 周期），打印逐关节稳态 RMSE，定位是哪个关节拖后腿
    print("\n========== ORACLE ACTION (action = ref - default) ==========")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    # 预热（消除冷启动暂态）
    for s in range(60):
        env.compute_ref_state()
        actions = (env.ref_dof_pos - env.default_dof_pos).clone()
        out = env.step(actions)
        obs = out[0]
    # 稳态统计：逐关节平方误差
    sq_err_acc = torch.zeros(env.num_actions, device=env.device)
    n = 0
    tr_list = []
    for s in range(200):
        env.compute_ref_state()
        actions = (env.ref_dof_pos - env.default_dof_pos).clone()
        out = env.step(actions)
        obs = out[0]
        env.compute_ref_state()
        diff = env.dof_pos - env.ref_dof_pos                       # (N,12)
        sq_err_acc += torch.mean(torch.square(diff), dim=0)        # (12,) 累加
        err = torch.mean(torch.square(diff), dim=1)                # (N,)
        tr_list.append(torch.exp(-err / env.cfg.rewards.tracking_sigma).mean().item())
        n += 1
    per_joint_mse = (sq_err_acc / n).cpu().numpy()
    per_joint_rmse = per_joint_mse ** 0.5
    joint_names = ['FL_hip','FL_th','FL_cf','FR_hip','FR_th','FR_cf',
                   'RL_hip','RL_th','RL_cf','RR_hip','RR_th','RR_cf']
    print("稳态 oracle tracking reward (200步平均):", sum(tr_list)/len(tr_list))
    print("逐关节稳态 RMSE [rad]（越小越好，<0.1 跟踪良好）：")
    for name, rmse in zip(joint_names, per_joint_rmse):
        flag = "  <-- 拖后腿" if rmse > 0.12 else ""
        print(f"  {name:8s}: {rmse:.4f} rad ({rmse*57.3:.1f} deg){flag}")
    # 幅值 vs 误差：看是不是大动作关节（thigh 0.4）相对误差大
    print("各关节幅值 [rad]:", env.ref_amplitude[0].cpu().numpy())
    print("→ 若 RMSE/幅值 比例大的关节拖后腿，说明 PD 对大幅值/重负载关节跟踪吃力")

    # 检查悬空是否生效（base 是否锁住）
    print("\n========== SUSPENSION CHECK ==========")
    print("base_pos z:", env.root_states[:4, 2].cpu().numpy())
    print("base_lin_vel:", env.root_states[:4, 7:10].cpu().numpy())
    print("feet_indices:", env.feet_indices.cpu().numpy())
    # 脚的世界坐标 z（看是否低于 0 = 碰地）
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    foot_z = env.rigid_state.view(env.num_envs, env.num_bodies, 13)[:, env.feet_indices, 2]
    print("feet_pos z:", foot_z[:4].cpu().numpy())
    # 接触力
    env.gym.refresh_net_contact_force_tensor(env.sim)
    cf = env.contact_forces[:, env.feet_indices]
    print("feet contact force norm:", torch.norm(cf, dim=-1)[:4].cpu().numpy())
    print("dof_pos_limits (URDF lower/upper):", env.dof_pos_limits.cpu().numpy())


if __name__ == '__main__':
    main()
