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
from .ppo_dh import PPO_DH_Smooth_Stage
from .actor_critic_dh import ActorCritic_DH_Smooth_Stage
from ..vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter


class OnPolicyRunner_DH_Smooth_Stage:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.device = device
        self.env = env
        if self.env.num_privileged_obs_arm is not None:
            num_critic_obs_arm = self.env.num_privileged_obs_arm
        else:
            num_critic_obs_arm = self.env.num_obs_arm

        if self.env.num_privileged_obs_leg is not None:
            num_critic_obs_leg = self.env.num_privileged_obs_leg
        else:
            num_critic_obs_leg = self.env.num_obs_leg

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        
        actor_critic_arm: ActorCritic = actor_critic_class( 
            self.env.cfg.env.num_single_obs_arm * self.env.cfg.env.short_frame_stack, 
            self.env.cfg.env.num_single_obs_arm,
            self.env.cfg.env.num_est_prob_arm, 
            num_critic_obs_arm,
            self.env.num_actions_arm,
            history_len = self.env.cfg.env.frame_stack,
            **self.policy_cfg
        ).to(self.device)

        actor_critic_leg: ActorCritic = actor_critic_class( 
            self.env.cfg.env.num_single_obs_leg * self.env.cfg.env.short_frame_stack, 
            self.env.cfg.env.num_single_obs_leg,
            self.env.cfg.env.num_est_prob_leg, 
            num_critic_obs_leg,
            self.env.num_actions_leg,
            history_len = self.env.cfg.env.frame_stack,
            **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO_DH_Smooth_Stage = alg_class(actor_critic_arm, actor_critic_leg, self.env.cfg.env.frame_stack, self.env.cfg.env.num_single_obs, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_arm_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.cfg.env.num_single_obs_arm * self.env.cfg.env.frame_stack],
            [self.env.num_privileged_obs_arm],
            [self.env.num_actions_arm],
        )
        self.alg.init_leg_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.cfg.env.num_single_obs_leg * self.env.cfg.env.frame_stack],
            [self.env.num_privileged_obs_leg],
            [self.env.num_actions_leg],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
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

        # Ensure arm policy is in eval mode and frozen
        self.alg.actor_critic_arm.eval()
        for param in self.alg.actor_critic_arm.parameters():
            param.requires_grad = False
        self.alg.actor_critic_leg.train()

        ep_infos = []
        arm_rewbuffer = deque(maxlen=100)
        leg_rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        
        cur_arm_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        cur_leg_reward_sum = torch.zeros(
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

                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, leg_rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, leg_rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        leg_rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(leg_rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_leg_reward_sum += leg_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        arm_rewbuffer.extend(cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        leg_rewbuffer.extend(cur_leg_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_arm_reward_sum[new_ids] = 0
                        cur_leg_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss_leg, mean_surrogate_loss_leg, mean_est_loss_leg, mean_smooth_loss_leg = self.alg.update()
            # mean_lpy_error = self.env.mean_lpy_error
            # mean_rpy_error = self.env.mean_rpy_error
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()
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
        mean_std = self.alg.actor_critic_leg.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )
        # self.writer.add_scalar('mean_lpy_error', locs['mean_lpy_error'], locs['it'])
        # self.writer.add_scalar('mean_rpy_error', locs['mean_rpy_error'], locs['it'])
        self.writer.add_scalar("MLP/mean_est_loss_leg", locs["mean_est_loss_leg"], locs["it"])
        self.writer.add_scalar(
            "Loss/value_function_leg", locs["mean_value_loss_leg"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate_leg", locs["mean_surrogate_loss_leg"], locs["it"]
        )
        self.writer.add_scalar("Loss/smooth_loss_leg", locs["mean_smooth_loss_leg"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std_leg", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["leg_rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_leg_reward", statistics.mean(locs["leg_rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["leg_rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["leg_rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss_leg:':>{pad}} {locs['mean_value_loss_leg']:.4f}\n"""
                f"""{'Surrogate loss_leg:':>{pad}} {locs['mean_surrogate_loss_leg']:.4f}\n"""
                f"""{'Smooth loss_leg:':>{pad}} {locs['mean_smooth_loss_leg']:.4f}\n"""
                f"""{'Mean action noise std_leg:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward_leg:':>{pad}} {statistics.mean(locs['leg_rewbuffer']):.2f}\n"""
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
                f"""{'Value function loss_leg:':>{pad}} {locs['mean_value_loss_leg']:.4f}\n"""
                f"""{'Surrogate loss_leg:':>{pad}} {locs['mean_surrogate_loss_leg']:.4f}\n"""
                f"""{'Smooth loss_leg:':>{pad}} {locs['mean_smooth_loss_leg']:.4f}\n""" 
                f"""{'Mean action noise std_leg:':>{pad}} {mean_std.item():.2f}\n"""
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
                'model_state_dict': self.alg.actor_critic_leg.state_dict(),
                'optimizer_state_dict': self.alg.optimizer_leg.state_dict(),
                "estimator_optimizer_state_dict": self.alg.estimator_optimizer_leg.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        
        # Load leg policy
        self.alg.actor_critic_leg.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer_leg.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def load_arm_policy(self, path):
        """Load only the arm policy from a pre-trained model"""
        loaded_dict = torch.load(path, map_location=self.device)

        self.alg.actor_critic_arm.load_state_dict(loaded_dict['model_state_dict'])
        
        # Set arm policy to eval mode and disable gradient computation
        self.alg.actor_critic_arm.eval()
        for param in self.alg.actor_critic_arm.parameters():
            param.requires_grad = False
        
        print(f"Loaded arm policy from {path}. Parameters frozen.")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic_arm.eval()
        self.alg.actor_critic_leg.eval() # switch to evaluation mode (dropout for example)

        if device is not None:
            self.alg.actor_critic_arm.to(device)
            self.alg.actor_critic_leg.to(device)
        return self.alg.actor_critic_arm.act_inference, self.alg.actor_critic_leg.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic_arm.eval()
        self.alg.actor_critic_leg.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic_arm.to(device)
            self.alg.actor_critic_leg.to(device)
        return self.alg.actor_critic_arm.evaluate, self.alg.actor_critic_leg.evaluate