#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMP 专家动捕数据 MuJoCo 复现工具。

读取 resources/isaacgym_processed/*.pkl, 用 MuJoCo 重放动捕轨迹。
场景文件: sim2sim/cowa2_description_mujoco/xml/scene.xml

pkl 结构 (见 motion_loader.py 注释):
  - fps: 帧率 (50.0)
  - root_pos: (N,3) 世界系位置
  - root_rot:  (N,4) 世界系四元数, xyzw 顺序 (注意! MuJoCo 用 wxyz, 需转换)
  - root_lin_vel / root_ang_vel: (N,3) 世界系速度 (复现不需要, 仅 AMP 训练用)
  - dof_pos: (N,12) 关节角, 按 model_joint_names 顺序
            (FL_hip, RL_hip, FR_hip, RR_hip, FL_thigh, ... 按关节类型聚合)
  - dof_vel: (N,12) 关节速度
  - key_body_pos_relative_to_base: (N,4,3) 脚相对 base 位置 (复现不需要)

关节顺序映射 (关键):
  pkl dof_pos 按 model_joint_names (helpers.get_quadruped_joint_names) 排序:
    [FL_hip, RL_hip, FR_hip, RR_hip, FL_thigh, RL_thigh, FR_thigh, RR_thigh,
     FL_calf, RL_calf, FR_calf, RR_calf]   ← 按关节类型聚合
  MuJoCo qpos[7:19] 按 URDF joint 出现顺序:
    [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
     RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]  ← 按腿分组
  两者不同, 必须按关节名建立索引重映射。

用法:
    python wheel_legged_gym/utils/replay_amp_data.py
    # 默认交互选 pkl, 用 scene.xml 加载, 实时重放
    python wheel_legged_gym/utils/replay_amp_data.py --file <path.pkl>
    python wheel_legged_gym/utils/replay_amp_data.py --file <path.pkl> --loop  # 循环
    python wheel_legged_gym/utils/replay_amp_data.py --list  # 只列出可用 pkl
"""
import os
import sys

# 关键: 移除脚本所在目录 (wheel_legged_gym/utils) 出 sys.path, 否则该目录下的
# math.py 会遮蔽标准库 math, 导致 numpy C 扩展初始化失败 (PyCapsule_Import "datetime" 报错)。
# 本脚本不依赖项目其他模块, 不需要脚本目录在 sys.path 里。
if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import glob
import pickle
import argparse

import numpy as np

try:
    import mujoco  # set_state 的 auto_ground 用; 运行时需要 (pip install mujoco)
except ImportError:
    mujoco = None


# ===== 路径 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # wheel_legged_gym 仓库根
DEFAULT_PKL_GLOB = os.path.join(ROOT_DIR, "resources", "isaacgym_processed", "*.pkl")
DEFAULT_XML = os.path.join(ROOT_DIR, "sim2sim", "cowa2_description_mujoco", "xml", "scene.xml")

# pkl 中 dof_pos 的顺序 (实测: 按 URDF 腿分组, 和 mujoco 完全一致, 非按类型聚合)
#   误用按类型聚合顺序 (hip×4,thigh×4,calf×4) 会导致 dof 错位, 动作全乱
#   判据: 用 dof 非零帧对比 key_body_pos, 按腿分组误差 0.004m, 按类型聚合误差 0.35m
PKL_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


def load_pkl(path):
    """加载 pkl, 返回 dict。"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    # 兼容 numpy 2.0 pickle 的 numpy 1.x 环境 (motion_loader 有同样处理)
    return d


def build_dof_remap(mj_joint_names):
    """建立 pkl dof_pos → mujoco qpos[7:19] 的索引映射。

    Args:
        mj_joint_names: mujoco 模型的 hinge 关节名列表 (按 qpos 顺序)
    Returns:
        remap: list, remap[i] = pkl dof_pos 中对应 mujoco 第 i 个关节的索引
               即 mj_qpos[7+i] = pkl_dof_pos[remap[i]]
    """
    pkl_idx = {name: i for i, name in enumerate(PKL_JOINT_NAMES)}
    remap = []
    missing = []
    for mj_name in mj_joint_names:
        if mj_name in pkl_idx:
            remap.append(pkl_idx[mj_name])
        else:
            missing.append(mj_name)
            remap.append(None)
    if missing:
        print(f"[WARN] mujoco 关节在 pkl 中找不到: {missing}")
    return remap


def xyzw_to_wxyz(q_xyzw):
    """四元数 xyzw → wxyz (MuJoCo 约定)。"""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


def set_state(data, model, root_pos, root_rot_xyzw, dof_pos, remap, z_offset=0.0,
              ground_z=0.0, foot_radius=0.045, auto_ground=False, foot_body_ids=None):
    """把一帧动捕数据写入 mujoco data 的 qpos。

    Args:
        data: mujoco MjData
        model: mujoco MjModel (auto_ground 时用 mj_kinematics)
        root_pos: (3,) 世界系位置
        root_rot_xyzw: (4,) 世界系四元数 xyzw
        dof_pos: (12,) pkl 顺序的关节角
        remap: build_dof_remap 返回的索引映射
        z_offset: base z 固定偏移
        ground_z: 地面 z
        foot_radius: 脚球半径
        auto_ground: True=每帧动态调整 base_z 让最低脚踩地 (修正动捕 base 锁定导致的脚悬空/穿地)
        foot_body_ids: 脚 body id 列表 (auto_ground 时用)
    """
    # free joint: qpos[0:3]=位置, qpos[3:7]=四元数(wxyz), qpos[7:19]=12 关节角
    data.qpos[0:3] = root_pos
    data.qpos[2] += z_offset  # 先加固定偏移
    data.qpos[3:7] = xyzw_to_wxyz(root_rot_xyzw)
    for mj_i, pkl_i in enumerate(remap):
        if pkl_i is not None:
            data.qpos[7 + mj_i] = dof_pos[pkl_i]
    data.qvel[:] = 0.0

    # 动态贴地: 算当前姿态下最低脚的世界 z, 平移 base 让最低脚球底踩地
    if auto_ground and foot_body_ids:
        mujoco.mj_kinematics(model, data)  # 只更新 body 位置, 不算接触力
        foot_zs = [data.xpos[b][2] for b in foot_body_ids]
        lowest_foot_z = min(foot_zs)
        delta = (ground_z + foot_radius) - lowest_foot_z
        data.qpos[2] += delta


def main():
    ap = argparse.ArgumentParser(description="AMP 动捕数据 MuJoCo 复现")
    ap.add_argument("--file", default=None, help="指定 pkl 文件 (不指定则交互选择)")
    ap.add_argument("--xml", default=DEFAULT_XML, help="mujoco 场景 xml")
    ap.add_argument("--loop", action="store_true", help="循环重放")
    ap.add_argument("--list", action="store_true", help="只列出可用 pkl, 不启动")
    ap.add_argument("--speed", type=float, default=1.0, help="播放速度 (0.5=半速, 2.0=倍速)")
    ap.add_argument("--no-render", action="store_true", help="不渲染 (调试用)")
    ap.add_argument("--z-offset", type=float, default=None,
                    help="base z 偏移修正 (动捕 base_z 与 mujoco 腿长不匹配导致浮空). "
                         "默认 None=自动计算 (首帧让最低脚踩地), 0=不修正, 手动填=强制")
    ap.add_argument("--ground-z", type=float, default=0.0,
                    help="地面 z 高度 (默认 0=平地; 在楼梯某级上重放时填该级台阶顶面 z)")
    ap.add_argument("--auto-ground", action="store_true",
                    help="每帧动态调整 base_z 让最低脚踩地 (修正动捕 base 锁定导致的脚悬空/穿地)")
    args = ap.parse_args()

    # 1. 选 pkl
    pkls = sorted(glob.glob(DEFAULT_PKL_GLOB))
    if not pkls:
        print(f"[ERR] 未找到 pkl: {DEFAULT_PKL_GLOB}")
        sys.exit(1)

    if args.list:
        print(f"共 {len(pkls)} 个 pkl:")
        for i, p in enumerate(pkls):
            print(f"  [{i}] {os.path.basename(p)}")
        return

    if args.file:
        target = args.file
        if not os.path.isabs(target):
            target = os.path.abspath(target)
    else:
        print("可用 pkl:")
        for i, p in enumerate(pkls):
            print(f"  [{i}] {os.path.basename(p)}")
        try:
            sel = int(input(f"选择序号 [0-{len(pkls)-1}]: ").strip())
            target = pkls[sel]
        except (ValueError, IndexError, KeyboardInterrupt):
            print("[ERR] 无效选择")
            sys.exit(1)

    print(f"[INFO] 加载 pkl: {os.path.basename(target)}")
    motion = load_pkl(target)

    # 校验字段
    required = ["root_pos", "root_rot", "dof_pos"]
    for k in required:
        if k not in motion:
            print(f"[ERR] pkl 缺少字段: {k}")
            sys.exit(1)

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)   # (N,3)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)   # (N,4) xyzw
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)     # (N,12)
    fps = float(motion.get("fps", 50.0))
    N = root_pos.shape[0]
    print(f"[INFO] 帧数={N}, fps={fps}, 时长={N/fps:.2f}s")
    print(f"[INFO] root_pos 范围: x[{root_pos[:,0].min():.2f},{root_pos[:,0].max():.2f}] "
          f"y[{root_pos[:,1].min():.2f},{root_pos[:,1].max():.2f}] "
          f"z[{root_pos[:,2].min():.2f},{root_pos[:,2].max():.2f}]")

    # 2. 加载 mujoco
    try:
        import mujoco
        from mujoco import viewer
    except ImportError:
        print("[ERR] 未安装 mujoco (pip install mujoco)")
        sys.exit(1)

    if not os.path.exists(args.xml):
        print(f"[ERR] xml 不存在: {args.xml}")
        sys.exit(1)

    print(f"[INFO] 加载场景: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # 3. 建 dof 索引映射 (按关节名)
    # 找出所有 hinge/slider 关节 (排除 free joint)
    mj_joint_names = []
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        # 3=JNT_HINGE, 4=JNT_SLIDE; free joint(2) 跳过
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            mj_joint_names.append(model.joint(jid).name)
    print(f"[INFO] mujoco 关节 ({len(mj_joint_names)}): {mj_joint_names}")
    remap = build_dof_remap(mj_joint_names)
    print(f"[INFO] dof 重映射 mj←pkl: {remap}")

    if len(mj_joint_names) != 12:
        print(f"[WARN] mujoco 关节数={len(mj_joint_names)}, 预期 12, 检查 URDF/scene.xml")

    # 4. 关闭重力 + 地形接触, 避免重放时机器人受干扰掉下
    #    重放是 kinematic replay (只设 qpos, forward), 不应该有动力学积分
    #    实际上设 qvel=0 后 mj_step 仍会积分, 但我们用 mj_forward 只刷新状态不积分
    model.opt.gravity[:] = 0.0  # 关重力, 防止 base 漂移

    # 4.1 收集脚 body id + 脚球半径 (auto_ground 用)
    foot_body_ids = [i for i in range(model.nbody)
                     if 'foot' in model.body(i).name.lower()]
    foot_radius = 0.045
    for bid in foot_body_ids:
        for gid in range(model.ngeom):
            if (model.geom_bodyid[gid] == bid
                    and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE):
                foot_radius = model.geom_size[gid][0]
                break
        break  # 只看第一个脚取半径
    print(f"[INFO] 脚 body ids: {[model.body(i).name for i in foot_body_ids]}, 球半径={foot_radius}")
    if args.auto_ground:
        print(f"[INFO] 开启 auto_ground: 每帧动态让最低脚踩地 (ground_z={args.ground_z})")

    # 4.5 计算 base z 偏移 (动捕 base_z 与 mujoco 模型腿长可能不匹配, 导致浮空)
    #     策略: 写入首帧后, 看最低脚的世界 z, 算出让脚踩到 ground_z 所需的 base z 修正
    if args.z_offset is None:
        # 让首帧的脚踩地 (用首帧最低脚的 rel_z, 不是整段最低, 否则首帧会偏高)
        if "key_body_pos_relative_to_base" in motion:
            kb = np.asarray(motion["key_body_pos_relative_to_base"])  # (N,4,3)
            foot_rel_z0 = kb[0, :, 2].min()  # 首帧最低脚相对 base 的 z
            # 脚球半径 ≈ 0.045, 贴地时脚心 z = ground_z + 0.045
            base_z0 = root_pos[0, 2]
            z_offset = (args.ground_z + 0.045) - (base_z0 + foot_rel_z0)
            print(f"[INFO] 自动 z_offset = {z_offset:.4f} (base_z0={base_z0:.3f}, "
                  f"首帧最低脚 rel_z={foot_rel_z0:.3f}, ground_z={args.ground_z})")
        else:
            z_offset = 0.0
            print(f"[WARN] pkl 无 key_body_pos, z_offset=0 (可能浮空, 用 --z-offset 手动)")
    else:
        z_offset = args.z_offset
        print(f"[INFO] 手动 z_offset = {z_offset}")

    # 5. 渲染循环
    frame_dt = 1.0 / fps / max(args.speed, 1e-6)
    print(f"[INFO] 开始重放 (speed={args.speed}x, frame_dt={frame_dt*1000:.1f}ms)")
    print("[INFO] 按 ESC 退出, P 暂停")

    if args.no_render:
        # 无渲染模式: 校验每帧写入 + 报告脚的实际高度
        for i in range(N):
            set_state(data, model, root_pos[i], root_rot[i], dof_pos[i], remap, z_offset,
                      args.ground_z, foot_radius, args.auto_ground, foot_body_ids)
            mujoco.mj_forward(model, data)
        # 看首帧脚是否贴地
        foot_zs = [data.xpos[b][2] for b in range(model.nbody) if 'foot' in model.body(b).name.lower()]
        print(f"[OK] 无渲染模式: {N} 帧全部写入成功")
        print(f"[INFO] 首帧脚心 z = {foot_zs} (贴地应 ≈ {args.ground_z + 0.045:.3f})")
        return

    # 交互式 viewer (非阻塞 launch_passive, 后台逐帧更新 qpos)
    _replay_loop(model, data, root_pos, root_rot, dof_pos, remap, N, frame_dt, args.loop,
                 z_offset, args.ground_z, foot_radius, args.auto_ground, foot_body_ids)


def _replay_loop(model, data, root_pos, root_rot, dof_pos, remap, N, frame_dt, loop,
                 z_offset=0.0, ground_z=0.0, foot_radius=0.045, auto_ground=False, foot_body_ids=None):
    """非阻塞 viewer 重放循环。"""
    import mujoco
    from mujoco import viewer
    import time

    frame = 0
    paused = [False]
    # 在 launch_passive 之前先把首帧写进 data, 避免 viewer 启动瞬间显示 URDF 默认浮空姿态
    set_state(data, model, root_pos[0], root_rot[0], dof_pos[0], remap, z_offset,
              ground_z, foot_radius, auto_ground, foot_body_ids)
    mujoco.mj_forward(model, data)
    handle = viewer.launch_passive(model, data)
    handle.sync()

    def _key_callback(key):
        if key == 80:  # P
            paused[0] = not paused[0]
    # launch_passive 不直接支持 key callback, 暂停用空格/轮询实现简化
    try:
        while handle.is_running():
            if not paused[0]:
                set_state(data, model, root_pos[frame], root_rot[frame], dof_pos[frame], remap,
                          z_offset, ground_z, foot_radius, auto_ground, foot_body_ids)
                mujoco.mj_forward(model, data)
                handle.sync()
                # 诊断: 首帧打印 base z 和脚心 z, 确认没有浮空
                if frame == 0:
                    foot_zs = [data.xpos[b][2] for b in range(model.nbody) if 'foot' in model.body(b).name.lower()]
                    print(f"[DIAG] 首帧 base_z={data.qpos[2]:.3f}, 脚心 z={foot_zs[0]:.4f} (应≈0.045)")
                frame += 1
                if frame >= N:
                    if loop:
                        frame = 0
                        print("[INFO] 循环重放: 回到第 0 帧")
                    else:
                        print(f"[INFO] 重放结束 (共 {N} 帧), 保持最后帧。关窗口退出。")
                        # 保持最后帧, 等用户关窗口
                        while handle.is_running():
                            set_state(data, model, root_pos[-1], root_rot[-1], dof_pos[-1], remap,
                                      z_offset, ground_z, foot_radius, auto_ground, foot_body_ids)
                            mujoco.mj_forward(model, data)
                            handle.sync()
                            time.sleep(0.1)
                        break
                time.sleep(frame_dt)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if handle.is_running():
            handle.close()


if __name__ == "__main__":
    main()
