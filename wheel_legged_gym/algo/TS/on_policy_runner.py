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
import torch
# import wandb
import statistics
from collections import deque
from datetime import datetime
from .ppo import PPO
from .actor_critic import ActorCritic
from ..vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter
# from utils.helpers import get_load_path
from IPython import embed;eee=embed
import torch.nn as nn
# from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR

class OnPolicyRunner_Lidar_TS:
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
        num_actor_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.env.cfg.dagger.dagger_on,  
            self.env.cfg.env.num_single_obs * self.env.cfg.env.frame_stack,  # hist  
            self.env.cfg.env.num_single_obs * self.env.cfg.env.frame_stack,  # input
            self.env.cfg.env.num_est_prob, num_critic_obs, self.env.num_actions, 
            self.env.cfg.env.num_height_scan_input, self.env.cfg.env.num_height_scan_output,
            **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.cfg.env.num_height_scan_input], # 121
            [self.env.cfg.env.num_single_obs * self.env.cfg.env.frame_stack],
            [self.env.cfg.env.num_single_obs * self.env.cfg.env.frame_stack],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            self.env.cfg.dagger.dagger_on,
        )
        # self.obs_now_start_point = (self.env.cfg.env.frame_stack - self.env.cfg.env.actor_input_stack) * self.env.cfg.env.num_single_obs
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


    # 用于加载专家模型 TODO
    def load_expert(self):
        ## model path
        WHEEL_LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        log_root = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs', self.cfg["experiment_name"])
        load_run = self.env.cfg.dagger.load_run_dagger
        checkpoint = self.env.cfg.dagger.checkpoint_dagger
        load_path = self.get_load_path(log_root, load_run, checkpoint)
        ## load expert model
        model_dict = torch.load(load_path) # map the device now
        actvation_func = nn.ELU
        model_key = "model_state_dict"
        net_key_name = "actor"
        loading_keys = [k for k in model_dict[model_key].keys() if k.startswith(net_key_name)]
        self.expert_policy = self.load_mlp(loading_keys, model_dict, actvation_func, model_key = model_key)
        self.expert_policy.to(self.device)
        # self.expert_policy = self.get_inference_policy(device=self.device) # expert policy


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.env.cfg.dagger.dagger_on:
            self.load_expert()
            print(" --- Now Student --- ")
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

        ep_infos = []
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
                    if self.env.cfg.dagger.dagger_on:
                        gt_actions = self.expert_policy(
                            critic_obs.detach().clone(), #184 x 3 input
                        )
                    else:
                        gt_actions = None
                    
                    actions = self.alg.act(
                        obs[:, :self.alg.actor_critic.num_height_scan_input ], # HS
                        obs[:, self.alg.actor_critic.num_height_scan_input: ], # input
                        obs[:, self.alg.actor_critic.num_height_scan_input: ], # history
                        critic_obs, # critic
                        gt_actions, # gt_acions
                        dagger_on=self.env.cfg.dagger.dagger_on) 

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
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

            mean_value_loss, mean_surrogate_loss, v_avg_diff_x ,v_avg_diff_y, v_avg_diff_z, base_height_diff, mean_est_loss, mean_dagger_loss = self.alg.update(self.env.cfg.dagger.dagger_only, self.env.cfg.dagger.dagger_on)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals(), dagger_on=self.env.cfg.dagger.dagger_on)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, dagger_on, width=80, pad=35):
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
        if dagger_on:
            self.writer.add_scalar("MLP/v_avg_diff_x", locs["v_avg_diff_x"] / self.env.cfg.normalization.obs_scales.lin_vel, locs["it"])
            self.writer.add_scalar("MLP/v_avg_diff_y", locs["v_avg_diff_y"] / self.env.cfg.normalization.obs_scales.lin_vel, locs["it"])
            self.writer.add_scalar("MLP/v_avg_diff_z", locs["v_avg_diff_z"] / self.env.cfg.normalization.obs_scales.lin_vel, locs["it"])
            self.writer.add_scalar("MLP/base_height_diff", locs["base_height_diff"] / self.env.cfg.normalization.obs_scales.height_measurements, locs["it"])
            self.writer.add_scalar("MLP/mean_est_loss", locs["mean_est_loss"], locs["it"])
            self.writer.add_scalar("MLP/mean_dagger_loss", locs["mean_dagger_loss"], locs["it"])
        
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
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

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
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
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
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
