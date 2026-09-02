## not used

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage
from IPython import embed;eee=embed
class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 adaptation_module_learning_rate=5e-4,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 num_adaptation_module_substeps =1 ,
                 device='cpu',

                 min_policy_std=None,
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # print calc logs
        # for name, param in self.actor_critic.named_parameters():
        #     if param.requires_grad:
        #         print(f"Name: {name}, Shape: {param.shape}")
        # <><><> zsy added <><><>  
        self.adaptation_encoder_optimizer = optim.Adam(self.actor_critic.adaptation_encoder.parameters(), lr=adaptation_module_learning_rate)
        self.expert_encoder_optimizer = optim.Adam(self.actor_critic.expert_encoder.parameters(), lr=adaptation_module_learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual

        # xxx
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        # self.estimator_optimizer = optim.Adam(self.actor_critic.estimator.parameters(), lr=mlp_learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.counter = 0

    def init_storage(self, num_envs, num_transitions_per_env, height_obs_shape, actor_obs_shape, critic_obs_shape, obs_history_shape, obs_history_c_shape, action_shape):
        self.storage = RolloutStorage(      num_envs, 
                                            num_transitions_per_env,
                                            height_obs_shape, 
                                            actor_obs_shape, 
                                            critic_obs_shape, 
                                            obs_history_shape, 
                                            obs_history_c_shape, 
                                            action_shape, 
                                            self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, height_obs, obs, expert_input, obs_history, obs_history_c, hist_encoding=False):
        # Compute the actions and values
        # 这里应该使用expert输出action进行更新
        # history  # NOTE: change student & teacher's update
        if hist_encoding:
            self.transition.actions = self.actor_critic.act_student(obs, obs_history, height_obs).detach()
        else:
            self.transition.actions = self.actor_critic.act_expert(obs, expert_input, height_obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs_history_c).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs  #actor
        self.transition.critic_observations = expert_input    #expert
        self.transition.observations_history = obs_history  #adaptation
        self.transition.observations_history_c = obs_history_c #critic
        self.transition.height_obs = height_obs # height scan
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_priv_reg_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for height_obs_batch, obs_batch, critic_obs_batch, obs_history_batch, obs_history_c_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act_expert(obs_batch, critic_obs_batch, height_obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # eee()
                value_batch = self.actor_critic.evaluate(obs_history_c_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss \
                        + self.value_loss_coef * value_loss \
                        - self.entropy_coef * entropy_batch.mean() \
                        # + 1.0 * priv_reg_loss
                        # + priv_reg_coef * priv_reg_loss

                # Gradient step
                ## TODO only do not update student networks
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                # mean_priv_reg_loss += priv_reg_loss.item()


                ## NOTE: <><><> TEACHER UPDATE <><><>
                for _ in range(self.num_adaptation_module_substeps):
                    adaptation_target = self.actor_critic.get_expert_latent(critic_obs_batch)
                    with torch.inference_mode():
                        adaptation_pred = self.actor_critic.get_adaptation_latent(height_obs_batch, obs_history_batch)
                    ### compute_loss 
                    priv_reg_loss = (adaptation_target - adaptation_pred.detach()).norm(p=2, dim=1).mean()
                    ### backwards
                    self.expert_encoder_optimizer.zero_grad()
                    priv_reg_loss.backward()  
                    nn.utils.clip_grad_norm_(self.actor_critic.expert_encoder.parameters(), self.max_grad_norm)
                    self.expert_encoder_optimizer.step()
                    mean_priv_reg_loss += priv_reg_loss.item()
                
                # 调整损失函数前面的系数，原作者
                # not used
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]



        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_priv_reg_loss /= (num_updates * self.num_adaptation_module_substeps)
        # mean_adaptation_encoder_loss /= (num_updates * self.num_adaptation_module_substeps)
        self.storage.clear()
        self.update_counter()
        return mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef




    # From ROA
    def update_dagger(self):
        mean_hist_latent_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for height_obs_batch, obs_batch, critic_obs_batch, obs_history_batch, obs_history_c_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                ## NOTE: Why to calculate once again???
                # with torch.inference_mode():  
                #     self.actor_critic.act_expert(obs_batch, critic_obs_batch, height_obs_batch)
                # 计算损失函数的地方，priv->adaption的loss  
                # Adaptation module update  
                adaptation_pred = self.actor_critic.get_adaptation_latent(height_obs_batch, obs_history_batch)
                with torch.inference_mode():
                    adaptation_target = self.actor_critic.get_expert_latent(critic_obs_batch)
                # compute loss
                hist_latent_loss = (adaptation_target.detach() - adaptation_pred).norm(p=2, dim=1).mean()
                # backwards
                self.adaptation_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.adaptation_encoder.parameters(), self.max_grad_norm)
                self.adaptation_encoder_optimizer.step()
                mean_hist_latent_loss += hist_latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss


    # update counter
    def update_counter(self):
        self.counter += 1


