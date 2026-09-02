from __future__ import annotations


import time
import os
from collections import deque
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from .ppo_amp import PPO_AMP_HeightScan
from .actor_critic import ActorCritic_AMP_HeightScan
from .amp_discriminator import AMPDiscriminator
from .utils import Normalizer
from .normalize import WGAN_Normalizer
from wheel_legged_gym.utils.motion_loader import classify_amp_motion_modes


TrainConfig = Dict[str, Any]
class AMPRunner_HIM_HeightScan:
    """Runner for the direct-height-scan PPO+AMP policy."""

    def __init__(
        self,
        env: Any,
        train_cfg: TrainConfig,
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.cfg = train_cfg['runner']
        self.alg_cfg = train_cfg['algorithm']
        self.policy_cfg = train_cfg['policy']
        self.all_cfg = train_cfg
        self.device = device
        self.env = env
        self._init_agent_and_algo()
        self.num_steps_per_env = self.cfg['num_steps_per_env']
        self.save_interval = self.cfg['save_interval']
        self._init_storage()
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.env.reset()
        # Keep the task/run timestamp prefix aligned with the other AMP runners.
        # 从 log_dir 最后一段解析时间戳 (和 task_registry.make_alg_runner 生成的目录名一致),
        # 不在这里再 strftime (会和 log 目录名的时间差几秒)
        import os as _os
        if log_dir:
            _dir_name = _os.path.basename(log_dir.rstrip('/'))
            # 格式: "Jul06_17-49-06_no_track" -> 取前两段 "Jul06_17-49-06"
            _parts = _dir_name.split('_')
            self.start_time = '_'.join(_parts[:2]) if len(_parts) >= 2 else _dir_name
        else:
            import time as _time
            self.start_time = _time.strftime('%b%d_%H-%M-%S')
        _exp_name = self.cfg.get("experiment_name", "")
        _run_name = self.cfg.get("run_name", "")
        self.exp_str = f"{_exp_name}" + (f"/{_run_name}" if _run_name else "")

    def _init_storage(self):
        num_steps_per_env = self.num_steps_per_env
        self.alg.init_storage(
            self.env.num_envs,
            num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

    def _pre_learn(self, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))


    def _init_agent_and_algo(self) -> None:
        """Initialize the AMP actor-critic and PPO_AMP algorithm."""
        from wheel_legged_gym.utils.motion_loader import AMPLoader

        if self.env.num_privileged_obs is not None:
            num_critic_obs: int = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if self.cfg['policy_class_name'] != 'ActorCritic_AMP_HeightScan':
            raise ValueError('AMPRunner_HIM_HeightScan requires ActorCritic_AMP_HeightScan')
        num_short_obs = self.env.cfg.env.num_single_obs * self.env.cfg.env.short_frame_stack
        num_single_obs = self.env.cfg.env.num_single_obs
        num_est_prob = self.env.cfg.env.num_est_prob
        actor_critic = ActorCritic_AMP_HeightScan(
            num_short_obs,
            num_single_obs,
            num_est_prob,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        amp_use_bucketed_experts = bool(self.cfg.get("amp_use_bucketed_experts", False))
        amp_data = AMPLoader(
            self.device,
            num_dof=self.env.num_actions,
            num_key_bodies=len(self.env.key_body_indices),  # type: ignore[attr-defined]
            time_between_frames=self.env.dt,  # type: ignore[attr-defined]
            preload_transitions=True,
            num_preload_transitions=self.cfg['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"],
            use_bucketed_experts=amp_use_bucketed_experts,
            amp_expert_label_source=self.cfg.get("amp_expert_label_source", "filename"),
            amp_bucket_lin_vel_threshold=self.cfg.get("amp_bucket_lin_vel_threshold", 0.20),
            amp_bucket_yaw_threshold=self.cfg.get("amp_bucket_yaw_threshold", 0.20),
            amp_bucket_yaw_lin_vel_threshold=self.cfg.get("amp_bucket_yaw_lin_vel_threshold", 0.20),
        )
        env_amp_obs_dim = self.env.get_amp_observations().shape[1]  # type: ignore[attr-defined]
        if env_amp_obs_dim != amp_data.observation_dim:
            raise ValueError(
                "AMP observation dimension mismatch: "
                f"env produces {env_amp_obs_dim}, motion data produces {amp_data.observation_dim}. "
                "Check num_dof, key_body_names, and amp_motion_files."
            )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        style_reward_function = self.cfg.get(
            "style_reward_function",
            self.alg_cfg.get("style_reward_function", "quad_mapping"),
        )
        normalize_style_reward = self.cfg.get(
            "normalize_style_reward",
            self.alg_cfg.get("normalize_style_reward", False),
        )
        amp_loss = self.cfg.get("amp_loss", self.alg_cfg.get("amp_loss", "MSELoss"))
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.cfg['amp_reward_coef'],
            self.cfg['amp_discr_hidden_dims'],
            self.device,
            self.cfg['amp_task_reward_lerp'],
            style_reward_function
        ).to(self.device)
        if self.cfg['algorithm_class_name'] != 'PPO_AMP_HeightScan':
            raise ValueError('AMPRunner_HIM_HeightScan requires PPO_AMP_HeightScan')
        if normalize_style_reward:
            style_reward_normalizer = WGAN_Normalizer(1, self.device)
        else:
            style_reward_normalizer = None
        alg_cfg = dict(self.alg_cfg)
        # Height-scan frames have a different layout; this package keeps
        # symmetry disabled unless a compatible transform is added later.
        alg_cfg.pop("symmetry_cfg", None)
        alg_cfg.pop("symmetry_scale", None)
        symmetry_cfg = None
        alg_cfg["amp_use_bucketed_experts"] = amp_use_bucketed_experts
        for amp_runner_key in ("amp_loss", "style_reward_function", "normalize_style_reward"):
            alg_cfg.pop(amp_runner_key, None)
        self.alg = PPO_AMP_HeightScan(
            actor_critic,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            discriminator_loss_function = amp_loss,
            style_reward_normalizer = style_reward_normalizer,
            symmetry_cfg = symmetry_cfg,
            **alg_cfg
        )
        self._init_amp_style_curriculum()

    def _get_amp_command_labels(self) -> torch.Tensor:
        commands = self.env.commands.to(self.device)  # type: ignore[attr-defined]
        vx = commands[:, 0]
        vy = commands[:, 1]
        wz = commands[:, 2]

        lin_threshold = float(self.cfg.get("amp_bucket_lin_vel_threshold", 0.20))
        yaw_threshold = float(self.cfg.get("amp_bucket_yaw_threshold", 0.20))
        yaw_lin_threshold = float(self.cfg.get("amp_bucket_yaw_lin_vel_threshold", 0.20))

        return classify_amp_motion_modes(
            vx,
            vy,
            wz,
            lin_threshold,
            yaw_threshold,
            yaw_lin_threshold,
        )

    def _init_amp_style_curriculum(self) -> None:
        """Initialize task/style reward mixing curriculum."""
        self.amp_style_curriculum = bool(self.cfg.get("amp_style_curriculum", False))
        initial_lerp = float(self.alg.discriminator.task_reward_lerp)
        self.amp_task_reward_lerp_min = float(self.cfg.get("amp_task_reward_lerp_min", initial_lerp))
        self.amp_task_reward_lerp_max = float(self.cfg.get("amp_task_reward_lerp_max", initial_lerp))
        self.amp_style_curriculum_reward_key = self.cfg.get(
            "amp_style_curriculum_reward_key", "tracking_ang_vel"
        )
        self.amp_style_curriculum_success_threshold = float(
            self.cfg.get("amp_style_curriculum_success_threshold", 0.8)
        )
        self.amp_style_curriculum_fail_threshold = float(
            self.cfg.get("amp_style_curriculum_fail_threshold", 0.6)
        )
        default_step = float(self.cfg.get("amp_style_curriculum_step", 0.02))
        self.amp_style_curriculum_style_step = float(
            self.cfg.get("amp_style_curriculum_style_step", default_step)
        )
        self.amp_style_curriculum_task_step = float(
            self.cfg.get("amp_style_curriculum_task_step", default_step)
        )
        self.amp_style_curriculum_ema_alpha = float(
            np.clip(self.cfg.get("amp_style_curriculum_ema_alpha", 1.0), 0.0, 1.0)
        )
        self.amp_style_curriculum_update_interval = max(
            1,
            int(self.cfg.get("amp_style_curriculum_update_interval", 1)),
        )
        self.amp_style_curriculum_update_counter = 0
        self.amp_style_curriculum_last_score: Optional[float] = None
        self.amp_style_curriculum_ema_score: Optional[float] = None
        self.amp_style_curriculum_last_reward: Optional[float] = None
        self.alg.discriminator.task_reward_lerp = float(
            np.clip(
                initial_lerp,
                self.amp_task_reward_lerp_min,
                self.amp_task_reward_lerp_max,
            )
        )

    def _episode_reward_mean(
        self,
        ep_infos: List[Dict[str, Any]],
        reward_name: str,
    ) -> Optional[float]:
        info_key = "rew_" + reward_name
        values = []
        for ep_info in ep_infos:
            if info_key not in ep_info:
                continue
            value = ep_info[info_key]
            if isinstance(value, torch.Tensor):
                values.append(value.detach().float().mean().to(self.device))
            else:
                values.append(torch.tensor(float(value), device=self.device))
        if not values:
            return None
        return torch.stack(values).mean().item()

    def _episode_reward_scale(self, reward_name: str) -> Optional[float]:
        reward_scales = getattr(self.env, "reward_scales", None)
        env_dt = getattr(self.env, "dt", None)
        if reward_scales is None or env_dt is None or reward_name not in reward_scales:
            return None
        scale = reward_scales[reward_name]
        if isinstance(scale, torch.Tensor):
            scale = scale.item()
        if env_dt == 0:
            return None
        return abs(float(scale) / float(env_dt))

    def _update_amp_style_curriculum(self, ep_infos: List[Dict[str, Any]]) -> None:
        if not self.amp_style_curriculum:
            return

        reward = self._episode_reward_mean(ep_infos, self.amp_style_curriculum_reward_key)
        reward_scale = self._episode_reward_scale(self.amp_style_curriculum_reward_key)
        if reward is None or reward_scale is None or reward_scale <= 0:
            return

        score = float(np.clip(reward / reward_scale, 0.0, 1.0))
        if self.amp_style_curriculum_ema_score is None:
            self.amp_style_curriculum_ema_score = score
        else:
            alpha = self.amp_style_curriculum_ema_alpha
            self.amp_style_curriculum_ema_score = (
                (1.0 - alpha) * self.amp_style_curriculum_ema_score + alpha * score
            )
        self.amp_style_curriculum_update_counter += 1
        self.amp_style_curriculum_last_score = score
        self.amp_style_curriculum_last_reward = reward
        if self.amp_style_curriculum_update_counter % self.amp_style_curriculum_update_interval != 0:
            return

        decision_score = self.amp_style_curriculum_ema_score
        old_lerp = float(self.alg.discriminator.task_reward_lerp)
        new_lerp = old_lerp

        if decision_score >= self.amp_style_curriculum_success_threshold:
            new_lerp -= self.amp_style_curriculum_style_step
        elif decision_score <= self.amp_style_curriculum_fail_threshold:
            new_lerp += self.amp_style_curriculum_task_step

        self.alg.discriminator.task_reward_lerp = float(
            np.clip(
                new_lerp,
                self.amp_task_reward_lerp_min,
                self.amp_task_reward_lerp_max,
            )
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run AMP training loop for a specified number of iterations.


        Args:
            num_learning_iterations: Number of learning iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        self._pre_learn(init_at_random_ep_len)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()  # type: ignore[attr-defined]
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = (
            obs.to(self.device),
            critic_obs.to(self.device),
            amp_obs.to(self.device),
        )
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        # Update the supervised state estimator when the policy provides one.
        self._has_estimator = hasattr(self.alg.actor_critic, 'estimator')

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                # Estimator 训练数据收集
                est_obs_history_list: List[torch.Tensor] = []
                est_target_list: List[torch.Tensor] = []

                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    amp_labels = self._get_amp_command_labels() if self.alg.amp_use_bucketed_experts else None

                    # Estimator 数据：保存 step 之前的 obs_history（出动作时的观测）
                    if self._has_estimator:
                        obs_history_before_step = obs.clone()

                    obs, privileged_obs, rewards, dones, infos, reset_env_ids,termination_privileged_obs,terminal_amp_states = self.env.step(actions)  # type: ignore[misc]
                    next_amp_obs = self.env.get_amp_observations()  # type: ignore[attr-defined]

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        next_amp_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Estimator 训练数据：step前的obs_history → step后的特权观测
                    if self._has_estimator:
                        next_critic_obs = critic_obs.clone()
                        if reset_env_ids.numel() > 0:
                            termination_privileged_obs = termination_privileged_obs.to(self.device)
                            next_critic_obs[reset_env_ids] = termination_privileged_obs
                        est_obs_history_list.append(obs_history_before_step)
                        est_target_list.append(next_critic_obs.clone())

                    # Account for terminal states. Use amp states before reset_idx.
                    if reset_env_ids.numel() > 0:
                        next_amp_obs_with_term = torch.clone(next_amp_obs)
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
                    else:
                        next_amp_obs_with_term = next_amp_obs

                    rewards, amp_reward = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        rewards,
                        normalizer=self.alg.amp_normalizer,
                        style_reward_normalizer=self.alg.style_reward_normalizer
                    )
                    amp_obs = next_amp_obs
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term, amp_labels)

                    if 'episode' in infos:
                        # add amp reward to episode info for terminal logging
                        infos['episode']['rew_amp'] = amp_reward / self.env.dt  # type: ignore[attr-defined]
                        ep_infos.append(infos['episode'])

                    if self.log_dir is not None:
                        # Book keeping
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, \
                mean_amp_loss, mean_grad_pen_loss, \
                    mean_policy_pred, mean_expert_pred, mean_symmetry_loss = self.alg.update()

            # Estimator 训练（用 rollout 收集的数据）
            mean_estimation_loss = 0.0
            if self._has_estimator and len(est_obs_history_list) > 0:
                est_obs_history = torch.cat(est_obs_history_list, dim=0)
                est_target = torch.cat(est_target_list, dim=0)
                num_est_steps = len(est_obs_history_list)
                batch_size = est_obs_history.shape[0] // max(1, num_est_steps // 4)  # 分几个 mini-batch
                est_total_loss = 0.0
                est_count = 0
                for _ in range(self.alg.num_learning_epochs):
                    perm = torch.randperm(est_obs_history.shape[0], device=self.device)
                    for start_idx in range(0, est_obs_history.shape[0], batch_size):
                        idx = perm[start_idx:start_idx + batch_size]
                        if idx.numel() == 0:
                            continue
                        est_loss, _ = self.alg.actor_critic.estimator.update(
                            est_obs_history[idx], est_target[idx], lr=self.alg.learning_rate
                        )
                        est_total_loss += est_loss
                        est_count += 1
                mean_estimation_loss = est_total_loss / max(est_count, 1)

            self._update_amp_style_curriculum(ep_infos)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                assert self.log_dir is not None
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        assert self.log_dir is not None
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))


    def log(
        self,
        locs: Dict[str, Any],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Log AMP training metrics to tensorboard and console.


        Args:
            locs: Dictionary containing iteration metrics and buffers.
            width: Width of the log output.
            pad: Padding for log formatting.
        """
        assert self.writer is not None
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('AMP/policy_pred', locs['mean_policy_pred'], locs['it'])
        self.writer.add_scalar('AMP/expert_pred', locs['mean_expert_pred'], locs['it'])
        amp_task_reward_lerp = float(self.alg.discriminator.task_reward_lerp)
        self.writer.add_scalar('AMP/task_reward_lerp', amp_task_reward_lerp, locs['it'])
        self.writer.add_scalar('AMP/style_reward_weight', 1.0 - amp_task_reward_lerp, locs['it'])
        if self.amp_style_curriculum_last_score is not None:
            self.writer.add_scalar(
                'AMP/style_curriculum_score',
                self.amp_style_curriculum_last_score,
                locs['it'],
            )
        if self.amp_style_curriculum_ema_score is not None:
            self.writer.add_scalar(
                'AMP/style_curriculum_ema_score',
                self.amp_style_curriculum_ema_score,
                locs['it'],
            )
        if self.amp_style_curriculum_last_reward is not None:
            self.writer.add_scalar(
                'AMP/style_curriculum_reward',
                self.amp_style_curriculum_last_reward,
                locs['it'],
            )
        if locs['mean_symmetry_loss'] is not None:
            self.writer.add_scalar('Loss/symmetry_loss', locs['mean_symmetry_loss'], locs['it'])
        if self._has_estimator:
            self.writer.add_scalar('Loss/estimator', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str_iter = f" \033[1m [{self.exp_str}] [{self.start_time}] Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        symmetry_string = ''
        if locs['mean_symmetry_loss'] is not None:
            symmetry_string = f"""{'Mean symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""
        estimator_string = ''
        if self._has_estimator:
            estimator_string = f"""{'Estimator loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{symmetry_string}"""
                          f"""{estimator_string}"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{symmetry_string}"""
                          f"""{estimator_string}"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += f"""{'AMP task reward lerp:':>{pad}} {amp_task_reward_lerp:.3f}\n"""
        log_string += f"""{'AMP style reward weight:':>{pad}} {1.0 - amp_task_reward_lerp:.3f}\n"""
        if self.amp_style_curriculum_last_score is not None:
            log_string += f"""{'AMP curriculum score:':>{pad}} {self.amp_style_curriculum_last_score:.3f}\n"""
        if self.amp_style_curriculum_ema_score is not None:
            log_string += f"""{'AMP curriculum ema:':>{pad}} {self.amp_style_curriculum_ema_score:.3f}\n"""
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(
        self,
        path: str,
        infos: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the AMP model checkpoint to disk.

        Args:
            path: File path to save the checkpoint.
            infos: Optional additional information to save with the checkpoint.
        """
        save_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            "amp_style_reward_normalizer": self.alg.style_reward_normalizer,
            "amp_task_reward_lerp": float(self.alg.discriminator.task_reward_lerp),
            "amp_style_curriculum_last_score": self.amp_style_curriculum_last_score,
            "amp_style_curriculum_ema_score": self.amp_style_curriculum_ema_score,
            "amp_style_curriculum_last_reward": self.amp_style_curriculum_last_reward,
            "amp_style_curriculum_update_counter": self.amp_style_curriculum_update_counter,
            'iter': self.current_learning_iteration,
            'infos': infos
        }
        # 保存 estimator optimizer（如果有）
        if hasattr(self.alg.actor_critic, 'estimator'):
            save_dict['estimator_optimizer_state_dict'] = self.alg.actor_critic.estimator.optimizer.state_dict()
        torch.save(save_dict, path)

    def load(
        self,
        path: str,
        load_optimizer: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load an AMP model checkpoint from disk.

        Args:
            path: File path to load the checkpoint from.
            load_optimizer: Whether to load the optimizer state.

        Returns:
            Optional infos dict stored in the checkpoint.
        """
        loaded_dict = torch.load(path, map_location=torch.device('cuda:0'),weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        self.alg.style_reward_normalizer = loaded_dict.get(
            "amp_style_reward_normalizer",
            self.alg.style_reward_normalizer,
        )
        self.alg.discriminator.task_reward_lerp = float(
            loaded_dict.get(
                "amp_task_reward_lerp",
                self.alg.discriminator.task_reward_lerp,
            )
        )
        self.amp_style_curriculum_last_score = loaded_dict.get(
            "amp_style_curriculum_last_score",
            self.amp_style_curriculum_last_score,
        )
        self.amp_style_curriculum_ema_score = loaded_dict.get(
            "amp_style_curriculum_ema_score",
            self.amp_style_curriculum_ema_score,
        )
        self.amp_style_curriculum_last_reward = loaded_dict.get(
            "amp_style_curriculum_last_reward",
            self.amp_style_curriculum_last_reward,
        )
        self.amp_style_curriculum_update_counter = loaded_dict.get(
            "amp_style_curriculum_update_counter",
            self.amp_style_curriculum_update_counter,
        )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            # 加载 estimator optimizer（如果有）
            if hasattr(self.alg.actor_critic, 'estimator') and 'estimator_optimizer_state_dict' in loaded_dict:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
