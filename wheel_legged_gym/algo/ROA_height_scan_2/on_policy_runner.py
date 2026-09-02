# NOTE: This 'ROA_height_scan_2' is for 
## height-encoder-->l1
## history-encoder-->l2
## l1+l2+prop --> actor
## expert-encoder only includes v and h

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
from IPython import embed;eee=embed

class OnPolicyRunner_Lidar_ROA_2:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]   
        self.all_cfg = train_cfg
        self.device = device
        self.env = env

        # num_obs是乘过frame_stack的，对应config中的num_observations
        # 如果想要直接访问config中的参数，用self.env.cfg.env.num_observations
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.cfg.env.num_single_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        # actor_critic: ActorCritic_MLP2 = actor_critic_class(
        #     self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        # ).to(self.device)
        actor_critic: ActorCritic = actor_critic_class(
            self.env.cfg.env.num_single_obs, self.env.cfg.env.single_num_privileged_obs, 
            self.env.num_actions, self.env.cfg.env.num_expert_input, self.env.cfg.env.num_latent, self.env.cfg.env.frame_stack,
            self.env.cfg.env.c_frame_stack, self.env.cfg.env.actor_input_stack, 
            self.env.cfg.env.num_height_scan_input, self.env.cfg.env.num_height_scan_output, 
            **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        # xxx 更改num_history
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.cfg.env.num_height_scan_input], # 77
            [self.env.cfg.env.num_single_obs * self.env.cfg.env.actor_input_stack],     #actor  给actor
            [self.env.cfg.env.num_expert_input], # 给expert encoder
            [self.env.cfg.env.frame_stack * self.env.cfg.env.num_single_obs],       #actor_history ， 给adaptation encoder
            [self.env.cfg.env.c_frame_stack * self.env.cfg.env.single_num_privileged_obs],  #critic_history 给critic
            [self.env.num_actions],
        )
        
        # xxx用于获取obs
        self.obs_now_start_point = (self.env.cfg.env.frame_stack - self.env.cfg.env.actor_input_stack)*self.env.cfg.env.num_single_obs
        self.obs_now_start_point_c = (self.env.cfg.env.c_frame_stack - 1)*self.env.cfg.env.single_num_privileged_obs
        self.num_expert_input = self.env.cfg.env.num_expert_input
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # dagger refresh freq
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # init metrics
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.

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
            hist_encoding = (it % self.dagger_update_freq == 0) #20 
            # Rollout  
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 传进去了framestack中的最后n个作为obs，前framestack-1个作为history 
                    # actions = self.alg.act(obs[:, self.obs_now_start_point:], critic_obs[:,self.obs_now_start_point_c:], obs, critic_obs)
                    # NOTE: owing to deque add new ones in the rightest , so  this way below to get the newest one.... --zsy
                    actions = self.alg.act( obs[:, :self.alg.actor_critic.num_height_scan_input], #63
                                            obs[:, self.alg.actor_critic.num_height_scan_input + self.obs_now_start_point:], #25
                                            critic_obs[:, -self.num_expert_input: ], #67  
                                            obs[:, self.alg.actor_critic.num_height_scan_input: ],    # 25*5 
                                            critic_obs,
                                            hist_encoding)  # 126x2  
                    # import ipdb;ipdb.set_trace()    
                    # step更新了obs, obs_buf入队出队.下一次obs[:, self.obs_now_start_point:]索引到的就是上一此更新完后环境的obs  
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

            # 更新dagger
            if hist_encoding:
                  # studemt's dagger
                mean_hist_latent_loss = self.alg.update_dagger()
            else: # teacher's dagger
                mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef = self.alg.update()


            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
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
        # self.writer.add_scalar("Estimator/mean_adaptation_encoder_loss",locs["mean_adaptation_encoder_loss"], locs["it"])
        # self.writer.add_scalar("MLP/v_avg_diff_x", locs["v_avg_diff_x"], locs["it"])
        # self.writer.add_scalar("MLP/v_avg_diff_y", locs["v_avg_diff_y"], locs["it"])
        # self.writer.add_scalar("MLP/v_avg_diff_z", locs["v_avg_diff_z"], locs["it"])
        # self.writer.add_scalar("MLP/mean_vel_loss", locs["mean_vel_loss"], locs["it"])
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
        self.writer.add_scalar( #teacher loss
            "Loss/priv_reg_loss", locs["mean_priv_reg_loss"], locs["it"]
        )
        self.writer.add_scalar( #student loss
            "Loss/hist_latent_loss", locs["mean_hist_latent_loss"], locs["it"]
        )
        self.writer.add_scalar( #priv-coef
            "Loss/priv_ref_lambda", locs["priv_reg_coef"], locs["it"]
        )
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
                f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                f"""{'Privileged info regularizer loss:':>{pad}} {locs['mean_priv_reg_loss']:.4f}\n"""
                f"""{'Privileged info regularizer lambda:':>{pad}} {locs['priv_reg_coef']:.4f}\n"""
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
                f"""{'History latent supervision loss:':>{pad}} {locs['mean_hist_latent_loss']:.4f}\n"""
                f"""{'Privileged info regularizer loss:':>{pad}} {locs['mean_priv_reg_loss']:.4f}\n"""
                f"""{'Privileged info regularizer lambda:':>{pad}} {locs['priv_reg_coef']:.4f}\n"""
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