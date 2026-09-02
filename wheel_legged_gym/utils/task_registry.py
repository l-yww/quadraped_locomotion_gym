# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import os
from typing import Tuple
from datetime import datetime

from wheel_legged_gym.algo import VecEnv
from wheel_legged_gym.algo import OnPolicyRunner
# from wheel_legged_gym.algo import OnPolicyRunnerVAE
from wheel_legged_gym.algo import OnPolicyRunnerEstimator
from wheel_legged_gym.algo import OnPolicyRunnerRMA

# from wheel_legged_gym.algo import OnPolicyRunner_Transformer
from wheel_legged_gym.algo import OnPolicyRunnerEstimator_Arm
from wheel_legged_gym.algo import OnPolicyRunner_HIM
from wheel_legged_gym.algo import OnPolicyRunner_HIM_HeightScan
from wheel_legged_gym.algo import OnPolicyRunnerROA
from wheel_legged_gym.algo import OnPolicyRunner_DH
from wheel_legged_gym.algo import OnPolicyRunner_DH_Smooth
from wheel_legged_gym.algo import OnPolicyRunner_DH_Smooth_Mix
from wheel_legged_gym.algo import OnPolicyRunner_DH_Smooth_Sym
from wheel_legged_gym.algo import OnPolicyRunner_DH_Smooth_Map
from wheel_legged_gym.algo import OnPolicyRunner_VAE_Smooth
from wheel_legged_gym.algo import OnPolicyRunner_DH_Smooth_P3O
from wheel_legged_gym.algo import OnPolicyRunner_Blind_TS
from wheel_legged_gym.algo import OnPolicyRunner_DH_prop

# MoE CTS
from wheel_legged_gym.algo import OnPolicyRunner_MoE_CTS

# <><><><> LIDar <><><><>
# from wheel_legged_gym.algo import OnPolicyRunner_Mimic_HIM
from wheel_legged_gym.algo import OnPolicyRunner_Lidar_ROA
from wheel_legged_gym.algo import OnPolicyRunner_Lidar_ROA_2
from wheel_legged_gym.algo import OnPolicyRunner_Lidar_PPO
from wheel_legged_gym.algo import OnPolicyRunner_Lidar_Estimator
from wheel_legged_gym.algo import OnPolicyRunner_Lidar_TS
from wheel_legged_gym.algo import AMPRunner
from wheel_legged_gym.algo import AMPRunner_HIM
from wheel_legged_gym.algo import AMPRunner_HIM_HeightScan

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import ntpath
from shutil import copyfile

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}  # VecEnv
        self.env_cfgs = {}  # LeggedRobotCfg
        self.train_cfgs = {}  # LeggedRobotCfgPPO
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def save_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        os.makedirs(self.log_dir, exist_ok=True)
        save_items = []
        
        base_env_path = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, "base/legged_robot.py")
        base_config_path = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, "base/legged_robot_config.py")
        quadruped_env_path = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, "quadruped/legged_robot.py")

        task_env_path1 = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, f"{name}/{name}_env.py")
        task_env_path2 = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, f"{name}/legged_robot.py")

        task_config_path1 = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, f"{name}/{name}_config.py")
        task_config_path2 = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, f"{name}/legged_robot_config.py")

        save_items.append(task_env_path1)

        save_items.append(task_config_path1)

        if os.path.exists(task_env_path2):
            save_items.append(task_env_path2)
        elif name == "quadruped_arm_him_amp" and os.path.exists(quadruped_env_path):
            save_items.append(quadruped_env_path)
        elif name in ("quadruped_wtw_him_arm_fix", "quadruped_wtw_him_arm_fix_amp") and os.path.exists(quadruped_env_path):
            save_items.append(quadruped_env_path)
        else:
            save_items.append(base_env_path)

        if os.path.exists(task_config_path2):
            save_items.append(task_config_path2)
        else:
            save_items.append(base_config_path)

        save_items.extend([
            os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, "cowa/cowa_env.py"),
            os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, f"cowa/cowa_config.py"),
            os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, "wheel_legged_gym/utils/terrain.py"),
        ])

        # 保存本次启动训练用的入口脚本,例如 wheel_legged_gym/scripts/train.py。
        # 这样每个 log 目录都能回溯当时 __main__ 里的 task/run_name/num_envs 等手写启动参数。
        launcher_path = None
        try:
            import __main__
            launcher_path = getattr(__main__, "__file__", None)
        except Exception:
            launcher_path = None
        if launcher_path is None:
            try:
                import sys
                launcher_path = sys.argv[0] if len(sys.argv) > 0 else None
            except Exception:
                launcher_path = None
        if launcher_path:
            launcher_path = os.path.abspath(launcher_path)
            if os.path.exists(launcher_path):
                save_items.append(launcher_path)

        # HIM-AMP 与 heightmap 版都继承 quadruped_arm_him；heightmap 还经过
        # quadruped_arm_him_amp 这一层。把所有父类 env/config 一起快照到 log，
        # 才能完整复现本次训练实际继承到的参数。
        if name in ("quadruped_arm_him_amp", "quadruped_arm_him_amp_heightmap"):
            parent_dirs = ["quadruped_arm_him"]
            if name == "quadruped_arm_him_amp_heightmap":
                parent_dirs.insert(0, "quadruped_arm_him_amp")

            for parent_dir in parent_dirs:
                save_prefix = "parent_" if name == "quadruped_arm_him_amp" else ""
                parent_env = os.path.join(
                    WHEEL_LEGGED_GYM_ENVS_DIR, f"{parent_dir}/{parent_dir}_env.py"
                )
                parent_config = os.path.join(
                    WHEEL_LEGGED_GYM_ENVS_DIR, f"{parent_dir}/{parent_dir}_config.py"
                )
                # 旧 AMP 任务加前缀；heightmap 任务使用基类原文件名，便于查找。
                if os.path.exists(parent_env):
                    save_items.append((parent_env, f"{save_prefix}{parent_dir}_env.py"))
                if os.path.exists(parent_config):
                    save_items.append((parent_config, f"{save_prefix}{parent_dir}_config.py"))

        # 针对 quadruped_wtw_him_arm_fix_amp: 额外保存其父类 quadruped_wtw_him_arm_fix 的 config 和 env
        # (wtw_amp 继承 wtw, 父类的 reward/commands/control 等改动也需要记录)
        if name == "quadruped_wtw_him_arm_fix_amp":
            parent_dir = "quadruped_wtw_him_arm_fix"
            parent_env = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, f"{parent_dir}/{parent_dir}_env.py")
            parent_config = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, f"{parent_dir}/{parent_dir}_config.py")
            if os.path.exists(parent_env):
                save_items.append((parent_env, f"parent_{parent_dir}_env.py"))
            if os.path.exists(parent_config):
                save_items.append((parent_config, f"parent_{parent_dir}_config.py"))
            # 也保存 amp_d1 基类（判别器/专家加载逻辑）
            amp_base_env = os.path.join(WHEEL_LEGGED_GYM_ENVS_DIR, "amp_d1/legged_robot_amp.py")
            if os.path.exists(amp_base_env):
                save_items.append((amp_base_env, "parent_amp_d1_legged_robot_amp.py"))
        
        if save_items is not None:
            for save_item in save_items:
                # 支持 tuple (src, dst_name) 用于自定义保存后的文件名 (避免重名)
                if isinstance(save_item, tuple):
                    src, dst_name = save_item
                    copyfile(src, os.path.join(self.log_dir, dst_name))
                else:
                    base_file_name = ntpath.basename(save_item)
                    copyfile(save_item, os.path.join(self.log_dir, base_file_name))

    def make_env(self, name, args=None, env_cfg=None, train_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()

        # 所给的名字必须是已经注册的   task_registry.register( "humanoid_ppo", XBotLFreeEnv, XBotLCfg(), XBotLCfgPPO() )

        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        if train_cfg is None:
            # load train config files
            _, train_cfg = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            train_cfg=train_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        self.env_cfg_for_wandb = env_cfg
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # log文件的地址和命名
        if log_root=="default":
            log_root = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        self.log_dir = log_dir
        train_cfg_dict = class_to_dict(train_cfg)
        env_cfg_dict = class_to_dict(self.env_cfg_for_wandb)
        all_cfg = {**train_cfg_dict, **env_cfg_dict}
        
        runner_class = eval(train_cfg_dict["runner_class_name"])
        runner = runner_class(env, all_cfg, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            print("--------------------------------")
            print("log_root: ", log_root)
            print("load_run:", train_cfg.runner.load_run )
            print("train_cfg.runner.checkpoint: ", train_cfg.runner.checkpoint)
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path, load_optimizer=False)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
