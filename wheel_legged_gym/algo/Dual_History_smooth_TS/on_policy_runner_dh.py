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
import time
import logging
import torch
import torch.nn as nn
# import wandb
import statistics
from collections import deque
from datetime import datetime
from .ppo_dh import PPO_DH_Smooth_TS
from .actor_critic_dh import ActorCritic_DH_Smooth_TS
from ..vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter
from wheel_legged_gym.algo.Dual_History_smooth import ActorCritic_DH_Smooth
logger = logging.getLogger(__name__)


class OnPolicyRunner_DH_Smooth_TS:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic_DH_Smooth_TS = actor_critic_class(
            self.env.cfg.env.num_single_obs * self.env.cfg.env.short_frame_stack,
            self.env.cfg.env.num_single_obs,
            self.env.cfg.env.num_est_prob, 
            num_critic_obs,
            self.env.num_actions,
            history_len = self.env.cfg.env.frame_stack,
            **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO

        self.dagger_on = self.cfg.get("dagger_on", False)
        self.alg: PPO_DH_Smooth_TS = alg_class(actor_critic, device=self.device,dagger_on=self.dagger_on, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        # self.dagger_on = self.cfg.get("dagger_on", False)
        # forward_depth shape: [1, 48, 64] (C, H, W)
        # forward_depth_shape = [1, 48, 64]
        forward_depth_shape=[1,*self.env.cfg.sensor.forward_camera.output_resolution]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.cfg.env.num_single_obs * self.env.cfg.env.frame_stack],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            dagger_on=self.dagger_on,
            forward_depth_shape=forward_depth_shape,
        )
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def get_load_path(self, root, load_run=-1, checkpoint=-1):
        try:
            runs = os.listdir(root)
            # import ipdb;ipdb.set_trace()
            # TODO sort by date to handle change of month
            runs.sort()
            if "exported" in runs:
                runs.remove("exported")
            last_run = os.path.join(root, runs[-1])
        except:
            raise ValueError("No runs in this directory: " + root)
        if load_run == -1:
            load_run = last_run
        else:
            load_run = os.path.join(root, load_run)
        if checkpoint == -1:
            models = [file for file in os.listdir(load_run) if "model" in file]
            models.sort(key=lambda m: "{0:0>15}".format(m))
            model = models[-1]
        else:
            model = "model_{}.pt".format(checkpoint)
        load_path = os.path.join(load_run, model)
        return load_path

    # load models
    ## zsy add
    def load_mlp(self, loading_keys, checkpoint, actvation_func, model_key="model"):
        loading_keys_linear = [k for k in loading_keys if k.endswith('weight')]
        nn_modules = []
        for idx, key in enumerate(loading_keys_linear):
            if len(checkpoint[model_key][key].shape) == 1: # layernorm
                layer = torch.nn.LayerNorm(*checkpoint[model_key][key].shape[::-1])
                nn_modules.append(layer)
            elif len(checkpoint[model_key][key].shape) == 2: # nn
                layer = nn.Linear(*checkpoint[model_key][key].shape[::-1])
                nn_modules.append(layer)
                if idx < len(loading_keys_linear) - 1:
                    nn_modules.append(actvation_func())
            else:
                raise NotImplementedError
        net = nn.Sequential(*nn_modules)
        state_dict = net.state_dict()
        for idx, key_affix in enumerate(state_dict.keys()):
            state_dict[key_affix].copy_(checkpoint[model_key][loading_keys[idx]])
        for param in net.parameters():
            param.requires_grad = False
        return net

    # 用于加载专家模型
    def load_expert(self):
        ## model path
        WHEEL_LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        log_root = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs')
        
        load_path = os.path.join(log_root, self.cfg["teacher_policy_checkpoint_path"])
        
        ## load expert model via TorchScript JIT
        print(f"Loading expert policy from: {load_path}")
        self.expert_policy = torch.jit.load(load_path, map_location=self.device)
        self.expert_policy.eval()
        print("Expert policy loaded successfully.")
        


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.dagger_on:
            self.load_expert()
            print(" --- Now Student --- ")
            assert self.expert_policy is not None, "DAgger is on but no expert policy loaded."
        else:
            print(" --- Now Teacher --- ")

        # initialize writer
        if self.log_dir is not None and self.writer is None:

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.env.compute_observations()
        infos= self.env.extras
        

        ep_infos = []
        forward_depth_list = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Get forward_depth from infos if available
                    forward_depth = infos.get("forward_depth", None) if 'infos' in locals() else None
                    forward_depth_list.append(forward_depth.cpu() if forward_depth is not None else torch.tensor([]))
                    if self.dagger_on and self.expert_policy is not None:
                        # expert_policy_input = critic_obs.detach().clone()
                        expert_policy_input=infos.get("teacher_obs_buf", None) if 'infos' in locals() else None
                        gt_actions = self.expert_policy(
                            expert_policy_input.detach().clone(),
                        )
                    else:
                        gt_actions = None

                    actions = self.alg.act(
                        obs,
                        critic_obs,
                        forward_depth=forward_depth,
                        gt_actions=gt_actions,
                        dagger_on=self.dagger_on)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    # obs, privileged_obs, rewards, dones, infos = self.env.step(gt_actions)
                    # forward_depth=infos.get("forward_depth", None) if 'infos' in locals() else None
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_est_loss, mean_smooth_loss, mean_dagger_loss = self.alg.update(dagger_on=self.dagger_on)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()
            forward_depth_list.clear()  # 防止内存无限增长
            self.current_learning_iteration = it

        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("MLP/mean_est_loss", locs["mean_est_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/smooth_loss", locs["mean_smooth_loss"], locs["it"])
        if locs.get("mean_dagger_loss") is not None:
            self.writer.add_scalar("Loss/dagger_loss", locs["mean_dagger_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Visualize forward_depth images collected during rollout in TensorBoard
        forward_depth_list = locs.get("forward_depth_list", [])
        if len(forward_depth_list) > 0:
            # Use the last step's depth image from env 0
            depth = forward_depth_list[-1]
            if isinstance(depth, torch.Tensor) and depth.numel() > 0:
                img = depth[0].float()  # first env
                if img.dim() == 2:
                    img = img.unsqueeze(0)  # [1, H, W]
                elif img.dim() == 3:
                    img = img[:1]  # take first channel -> [1, H, W]
                # Normalize to [0, 1]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                self.writer.add_image("Obs/forward_depth", img.cpu(), locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Smooth loss:':>{pad}} {locs['mean_smooth_loss']:.4f}\n"""
            )
            if locs.get("mean_dagger_loss") is not None:
                log_string += f"""{'DAgger loss:':>{pad}} {locs['mean_dagger_loss']:.4f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Smooth loss:':>{pad}} {locs['mean_smooth_loss']:.4f}\n"""
            )
            if locs.get("mean_dagger_loss") is not None:
                log_string += f"""{'DAgger loss:':>{pad}} {locs['mean_dagger_loss']:.4f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )

        print(log_string)

    def save(self, path, infos=None):
        save_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Save dagger_optimizer if it exists
        if hasattr(self.alg, 'dagger_optimizer') and self.alg.dagger_optimizer is not None:
            save_dict["dagger_optimizer_state_dict"] = self.alg.dagger_optimizer.state_dict()
        torch.save(save_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        print(f"load dict key is {loaded_dict.keys()}")
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            # Load dagger_optimizer if it exists in checkpoint
            if "dagger_optimizer_state_dict" in loaded_dict and hasattr(self.alg, 'dagger_optimizer') and self.alg.dagger_optimizer is not None:
                self.alg.dagger_optimizer.load_state_dict(loaded_dict["dagger_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
