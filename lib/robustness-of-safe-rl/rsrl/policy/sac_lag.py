from copy import deepcopy
import gym
import numpy as np
import torch
import torch.nn as nn
from rsrl.policy.offpolicy_base import OffPolicyBase
from rsrl.policy import LagrangianPIDController
from rsrl.policy.model.mlp_ac import SquashedGaussianMLPActor, EnsembleQCritic
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import (count_vars, get_device_name, to_device,
                                     to_ndarray, to_tensor)
from rsrl.policy.adversary.adv_random import AdvUniform, AdvGaussian
from rsrl.policy.adversary.adv_critic import AdvCritic, AdvCriticPPO
from rsrl.policy.adversary.adv_mad import AdvMad, AdvMadPPO
from rsrl.policy.adversary.adv_amad import AdvAmad, AdvAmadPPO
from rsrl.policy.adversary.adv_base import Adv
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_


class SAC(OffPolicyBase):
    attacker_cls = dict(uniform=AdvUniform,
                        gaussian=AdvGaussian,
                        mad=AdvMad,
                        amad=AdvAmad,
                        max_reward=AdvCritic,
                        max_cost=AdvCritic)
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 env_cfg: dict,
                 adv_cfg: dict,
                 rs_mode="vanilla",
                 attacker="uniform",
                 alpha=0.085,
                 adv_start_epoch=0,
                 adv_incr_epoch=100,
                 **kwargs) -> None:
        r'''
        Soft Actor Critic (SAC)

        Args:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param ac_model: the actor critic model name
        @param alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        @param gamma (float): Discount factor. (Always between 0 and 1.)
        @param polyak (float): Interpolation factor in polyak averaging for target 
        @param num_q (int): number of models in the q-ensemble critic.
        '''
        self.adv_cfg = adv_cfg
        self.mode = rs_mode
        self.attacker = attacker
        self.adv_start_epoch = adv_start_epoch
        self.adv_incr_epoch = adv_incr_epoch

        self.alpha = alpha
        super().__init__(env, logger, env_cfg, **kwargs)

        if self.mode == "random":
            self.attacker = "gaussian"
            self.apply_adv_collect_data = True
        else:
            self.attacker = attacker

        if self.mode == "vanilla_adv":
            self.apply_adv_collect_data = True

        # Set up model saving
        self.save_model()
        self._init_adversary()

        assert self.use_retrace != True, "retrace for sac_lag is not implemented yet"

    def _init_adversary(self):
        self.noise_scale_max = self.adv_cfg["noise_scale"]
        self.noise_scale = 0
        self.attack_freq = self.adv_cfg["attack_freq"]
        cfg = self.adv_cfg[self.attacker + "_cfg"]
        adv_cls = self.attacker_cls[self.attacker]
        self.adversary = adv_cls(self.obs_dim, **cfg)
        self.noise_scale_incr_func = lambda x: np.min([
            self.noise_scale_max / self.adv_incr_epoch *
            (x - self.adv_start_epoch), self.noise_scale_max
        ])

    def get_obs_adv(self, adv: Adv, obs: torch.Tensor):
        epsilon = adv.attack_batch(self, obs, self.noise_scale)
        obs_adv = (epsilon + obs).detach()
        return obs_adv

    def _pre_epoch_process(self, epoch):
        if self.apply_adv_collect_data:
            self.use_adv = False
        else:
            if epoch < self.adv_start_epoch:
                self.use_adv = False
            else:
                self.use_adv = True
                # scheduler for alpha or noise scale
                self.noise_scale = self.noise_scale_incr_func(epoch)
                # self.alpha_adv = self.alpha_adv_incr_func(epoch)

    def _post_epoch_process(self, epoch):
        return

    def _preprocess_data(self, data):
        obs, obs_next = data["obs"], data["obs2"]
        obs_adv = self.get_obs_adv(self.adversary, obs)
        data["obs_adv"] = obs_adv
        return data

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, logp.
        This API is used to interact with the env.

        @param obs (1d ndarray): observation
        @param deterministic (bool): True for evaluation mode, which returns the action with highest pdf (mean).
        @param with_logprob (bool): True to return log probability of the sampled action, False to return None
        @return act, logp, (1d ndarray)
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            mean, cholesky, pi_dist = self.actor_forward(obs)
            a = mean if deterministic else pi_dist.sample()
            logp_a = pi_dist.log_prob(a) if with_logprob else None
        # squeeze them to the right shape
        a, logp_a = np.squeeze(to_ndarray(a),
                               axis=0), np.squeeze(to_ndarray(logp_a))
        return a, logp_a

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, rew, obs_next, done)
        '''
        if self.use_adv:
            self._preprocess_data(data)
        
        self._update_critic(data)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False

        self._update_actor(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self._polyak_update_target(self.critic, self.critic_targ)

    def critic_forward(self, critic, obs, act):
        # return the minimum q values and the list of all q_values
        return critic.predict(obs, act)

    def actor_forward(self, obs, return_pi=True):
        r''' 
        Return action distribution and action log prob [optional].
        @param obs, (tensor), [batch, obs_dim]
        @return mean, (tensor), [batch, act_dim]
        @return cholesky, (tensor), (batch, act_dim, act_dim)
        @return pi_dist, (MultivariateNormal)
        '''
        # [batch, (unroll_length), obs_dim]
        mean, cholesky = self.actor(obs)
        pi_dist = MultivariateNormal(
            mean, scale_tril=cholesky) if return_pi else None
        return mean, cholesky, pi_dist

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        def policy_loss():
            obs = data['obs']
            # act, logp_pi = self.actor_forward(obs, False, True)
            _, _, pi_dist = self.actor_forward(obs)
            act = pi_dist.rsample()
            logp_pi = pi_dist.log_prob(act)
            q_pi, _ = self.critic_forward(self.critic, obs, act)

            # Entropy-regularized policy loss
            loss_pi = (self.alpha * logp_pi - q_pi).mean()

            # Useful info for logging
            pi_info = dict(LogPi=to_ndarray(logp_pi))

            return loss_pi, pi_info

        self.actor_optimizer.zero_grad()
        loss_pi, pi_info = policy_loss()
        loss_pi.backward()
        self.actor_optimizer.step()

        # Log actor update info
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

    def _update_critic(self, data):
        '''
        Update the critic network
        '''
        def critic_loss():
            obs, act, reward, obs_next, done = to_tensor(
                data['obs']), to_tensor(data['act']), to_tensor(
                    data['rew']), to_tensor(data['obs2']), to_tensor(
                        data['done'])

            _, q_list = self.critic_forward(self.critic, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                _, _, pi_dist = self.actor_forward(obs_next)
                act_next = pi_dist.sample()
                logp_a_next = pi_dist.log_prob(act_next)
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.critic_targ, obs_next,
                                                   act_next)
                backup = reward + self.gamma**self.n_step * (1 - done) * (
                    q_pi_targ - self.alpha * logp_a_next)
            # MSE loss against Bellman backup
            loss_q = self.critic.loss(backup, q_list, self.td_error_lim)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        loss_critic, loss_q_info = critic_loss()
        loss_critic.backward()
        if self.grad_norm_lim is not None:
            clip_grad_norm_(self.critic.parameters(), self.grad_norm_lim)
        self.critic_optimizer.step()

        # Log critic update info
        # Record things
        self.logger.store(LossQ=loss_critic.item(), **loss_q_info)

    def _polyak_update_target(self, critic, critic_targ):
        '''
        Update target networks by polyak averaging.
        '''
        with torch.no_grad():
            for p, p_targ in zip(critic.parameters(),
                                 critic_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_risk_estimation(self, obs):
        '''
        Given an obs array (obs_dim), output a risk (qc) value, and the action
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            # mean, cholesky, pi_dist = self.actor_forward(obs)
            mean, _ = self.actor(obs)  # [1, obs_dim]
            qc, _ = self.critic_forward(self.qc, obs, torch.squeeze(mean,
                                                                    dim=1))
        return torch.squeeze(qc).item(), np.squeeze(to_ndarray(mean), axis=0)


class SACLagrangian(SAC):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 env_cfg: dict,
                 adv_cfg: dict,
                 use_cost_decay=False,
                 cost_start=100,
                 cost_end=10,
                 decay_epoch=100,
                 KP=0,
                 KI=0.1,
                 KD=0,
                 per_state=True,
                 **kwargs) -> None:
        r'''
        Soft Actor Critic (SAC) with Lagrangian multiplier
        Args in kwargs:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param ac_model: the actor critic model name
        @param alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        @param gamma (float): Discount factor. (Always between 0 and 1.)
        @param polyak (float): Interpolation factor in polyak averaging for target 
        @param num_q (int): number of models in the q-ensemble critic.
        '''
        self.use_cost_decay = use_cost_decay
        self.cost_start = cost_start
        self.cost_end = cost_end
        self.decay_epoch = decay_epoch

        super().__init__(env, logger, env_cfg, adv_cfg, **kwargs)
        '''
        Notice: The output action are normalized in the range [-1, 1], 
        so please make sure your action space's high and low are suitable
        '''

        self.controller = LagrangianPIDController(KP, KI, KD, self.qc_thres, per_state)

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, rew, obs_next, done)
        '''
        if self.use_adv:
            self._preprocess_data(data)

        self._update_critic(data)
        self._update_qc(data)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.qc.parameters():
            p.requires_grad = False

        self._update_actor(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.qc.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self._polyak_update_target(self.critic, self.critic_targ)
        self._polyak_update_target(self.qc, self.qc_targ)

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        def policy_loss():
            obs = data['obs']
            # act, logp_pi = self.actor_forward(obs, False, True)
            # q_pi, q_list = self.critic_forward(self.critic, obs, act)
            # qc_pi, _ = self.critic_forward(self.qc, obs, act)
            _, _, pi_dist = self.actor_forward(obs)
            act = pi_dist.rsample()
            logp_pi = pi_dist.log_prob(act)
            q_pi, _ = self.critic_forward(self.critic, obs, act)
            qc_pi, _ = self.critic_forward(self.qc, obs, act)

            # detach is very important here!
            # Otherwise the gradient will backprop through the multiplier.
            with torch.no_grad():
                multiplier = self.controller.control(qc_pi).detach()

            qc_penalty = ((qc_pi - self.qc_thres) * multiplier).mean()

            # Entropy-regularized policy loss
            loss_actor = (self.alpha * logp_pi - q_pi).mean()

            loss_pi = loss_actor + qc_penalty

            # Useful info for logging
            pi_info = dict(LogPi=to_ndarray(logp_pi),
                           Lagrangian=to_ndarray(multiplier),
                           LossActor=to_ndarray(loss_actor),
                           QcPenalty=to_ndarray(qc_penalty))

            return loss_pi, pi_info

        self.actor_optimizer.zero_grad()
        loss_pi, pi_info = policy_loss()
        loss_pi.backward()
        self.actor_optimizer.step()

        # Log actor update info
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

    def _update_qc(self, data):
        '''
        Update the qc network
        '''
        def critic_loss():
            obs, act, reward, obs_next, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['cost']), to_tensor(
                    data['obs2']), to_tensor(data['done'])

            _, q_list = self.critic_forward(self.qc, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                _, _, pi_dist = self.actor_forward(obs_next)
                act_next = pi_dist.sample()
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.qc_targ, obs_next, act_next)
                backup = reward + self.gamma**self.n_step * q_pi_targ
            # MSE loss against Bellman backup
            loss_q = self.qc.loss(backup, q_list, self.td_error_lim)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QCVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.qc_optimizer.zero_grad()
        loss_qc, loss_qc_info = critic_loss()
        loss_qc.backward()
        if self.grad_norm_lim is not None:
            clip_grad_norm_(self.qc.parameters(), self.grad_norm_lim)
        self.qc_optimizer.step()

        # Log critic update info
        # Record things
        self.logger.store(LossQC=loss_qc.item(), **loss_qc_info, QcThres=self.qc_thres)
