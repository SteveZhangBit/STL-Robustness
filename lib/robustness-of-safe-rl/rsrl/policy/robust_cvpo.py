from copy import deepcopy

import gym
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from cpprb import ReplayBuffer
from rsrl.policy.offpolicy_base import OffPolicyBase
from rsrl.policy.model.mlp_ac import (EnsembleQCritic, CholeskyGaussianActor)
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import (count_vars, get_device_name, to_device,
                                  to_ndarray, to_tensor)
from rsrl.policy.adversary.adv_random import AdvUniform, AdvGaussian
from rsrl.policy.adversary.adv_critic import AdvCritic, AdvCriticPPO
from rsrl.policy.adversary.adv_mad import AdvMad, AdvMadPPO
from rsrl.policy.adversary.adv_amad import AdvAmad, AdvAmadPPO
from rsrl.policy.adversary.adv_base import Adv
from rsrl.policy.solver import ScipySolver, TorchSolver

from torch.distributions.uniform import Uniform
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_


def bt(m: torch.tensor):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m: torch.tensor):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def safe_inverse(A, det):
    indices = torch.where(det <= 1e-6)
    # pseudoinverse
    if len(indices[0]) > 0:
        return torch.linalg.pinv(A)
    return A.inverse()


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of Σi, Σ
    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_det = Σi.det()  # (B,)
    Σ_det = Σ.det()  # (B,)
    Σi_inv = safe_inverse(Σi, Σi_det)  # (B, n, n)
    Σ_inv = safe_inverse(Σ, Σ_det)  # (B, n, n)
    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    Σi_det = torch.clamp_min(Σi_det, 1e-6)
    Σ_det = torch.clamp_min(Σ_det, 1e-6)
    inner_μ = (
        (μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ_det / Σi_det) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ, torch.mean(Σi_det), torch.mean(Σ_det)


class RobustCVPO(OffPolicyBase):
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
                 rs_mode=None,
                 attacker="max_cost",
                 dual_constraint=0.1,
                 sample_action_num=64,
                 lam_max=500,
                 solver_lr=0.02,
                 solver_max_iter=10,
                 solver_use_prev_init=True,
                 kl_mean_constraint=0.01,
                 kl_var_constraint=0.0001,
                 alpha_mean_scale=1.0,
                 alpha_var_scale=100.0,
                 alpha_mean_max=0.1,
                 alpha_var_max=10.0,
                 adv_start_epoch=0,
                 adv_incr_epoch=100,
                 mstep_iteration_num=5,
                 use_torch_solver=True,
                 use_polyak_update_policy=False,
                 **kwargs) -> None:
        r'''
        Constrained Variational Policy Optimization

        Args:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        :param dual_constraint:
        (float) hard constraint of the dual formulation in the E-step
        correspond to [2] p.4 ε
        @param kl_mean_constraint:
            (float) hard constraint of the mean in the M-step
            correspond to [2] p.6 ε_μ for continuous action space
        @param kl_var_constraint:
            (float) hard constraint of the covariance in the M-step
            correspond to [2] p.6 ε_Σ for continuous action space
        @param kl_constraint:
            (float) hard constraint in the M-step
            correspond to [2] p.6 ε_π for discrete action space
        @param discount_factor: (float) discount factor used in Policy Evaluation
        @param alpha_scale: (float) scaling factor of the lagrangian multiplier in the M-step, only used in Discrete action space
        @param sample_episode_num: the number of sampled episodes
        @param sample_episode_maxstep: maximum sample steps of an episode
        @param sample_action_num:
        @param batch_size: (int) size of the sampled mini-batch
        @param episode_rerun_num:
        @param mstep_iteration_num: (int) the number of iterations of the M-step
        @param evaluate_episode_maxstep: maximum evaluate steps of an episode
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param ac_model: the actor critic model name
        @param gamma (float): Discount factor. (Always between 0 and 1.)
        @param polyak (float): Interpolation factor in polyak averaging for target 
        @param num_q (int): number of models in the q-ensemble critic.
        @param use_torch_solver (bool): whether use PyTorch version of solver or SciPy version.
        '''
        super().__init__(env, logger, env_cfg, **kwargs)

        self.adv_cfg = adv_cfg

        self.dual_constraint = dual_constraint
        self.kl_mean_constraint = kl_mean_constraint
        self.kl_var_constraint = kl_var_constraint
        self.alpha_mean_scale = alpha_mean_scale
        self.alpha_var_scale = alpha_var_scale
        self.alpha_mean_max = alpha_mean_max
        self.alpha_var_max = alpha_var_max
        self.sample_action_num = sample_action_num
        self.mstep_iteration_num = mstep_iteration_num
        self.use_torch_solver = use_torch_solver
        self.use_polyak_update_policy = use_polyak_update_policy

        self.adv_start_epoch = adv_start_epoch
        self.adv_incr_epoch = adv_incr_epoch

        self.alpha_mean = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.alpha_var = 0.0  # lagrangian multiplier for continuous action space in the M-step

        self.mode = rs_mode
        if self.mode == "random":
            self.attacker = "gaussian"
            self.apply_adv_collect_data = True
        else:
            self.attacker = attacker

        if self.mode == "vanilla_adv":
            self.apply_adv_collect_data = True

        if self.use_torch_solver:
            self.solver = TorchSolver(self.dual_constraint,
                                      self.qc_thres,
                                      eta_init=1,
                                      lam_init=1,
                                      lr=solver_lr,
                                      bounds=[1e-6, lam_max],
                                      tol=1e-3,
                                      max_iterations=solver_max_iter,
                                      use_prev_init=solver_use_prev_init)
        else:
            self.solver = ScipySolver(self.dual_constraint, self.qc_thres, 1,
                                      1)

        # Set up model saving
        self.save_model()
        self._init_adversary()

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

        # self.alpha_adv_incr_func = lambda x: np.min([(int(x/self.adv_incr_epoch)+1)*0.02, self.alpha_adv_max])

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
        # update target policy network
        if not self.use_polyak_update_policy:
            with torch.no_grad():
                self.actor_targ.load_state_dict(self.actor.state_dict())

    def _preprocess_data(self, data):
        obs, obs_next = data["obs"], data["obs2"]
        if self.use_retrace:
            obs_adv = self.get_obs_adv(self.adversary, obs[:, 0, :])
        else:
            obs_adv = self.get_obs_adv(self.adversary, obs)
        data["obs_adv"] = obs_adv
        return data

    def _init_actor(self,
                    actor_state_dict=None,
                    actor_optimizer_state_dict=None):
        actor = CholeskyGaussianActor(self.obs_dim, self.act_dim,
                                      -self.act_lim, self.act_lim,
                                      self.hidden_sizes, nn.ReLU)
        if actor_state_dict is not None:
            actor.load_state_dict(actor_state_dict)
        self.actor = to_device(actor)
        self.actor_targ = deepcopy(self.actor)
        for p in self.actor_targ.parameters():
            p.requires_grad = False
        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        if actor_optimizer_state_dict is not None:
            self.actor_optimizer.load_state_dict(actor_optimizer_state_dict)

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

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, rew, obs_next, done)
        '''
        # data keys: (obs, obs_adv, act, rew, obs_next, done)
        if self.use_adv:
            self._preprocess_data(data)

            if self.mode == "em_all_adv":
                data["obs"] = data["obs_adv"]

        if self.use_retrace:
            self._update_critic_retrace(data)
            self._update_qc_retrace(data)
        else:
            self._update_critic(data)
            self._update_qc(data)

        self._update_actor(data)

        # Finally, update target networks by polyak averaging.
        self._polyak_update_target(self.critic, self.critic_targ)
        self._polyak_update_target(self.qc, self.qc_targ)
        if self.use_polyak_update_policy:
            self._polyak_update_target(self.actor, self.actor_targ)

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
        if self.use_retrace:
            obs = data['obs'][:, 0, :]  # [batch, obs_dim]
        else:
            obs = data['obs']  # [batch, obs_dim]

        if self.use_adv:
            obs_adv = data['obs_adv']  # [batch, obs_dim]

        N = self.sample_action_num
        K = obs.shape[0]
        da = self.act_dim
        ds = self.obs_dim

        with torch.no_grad():
            # sample N actions per state
            b_mean, b_A = self.actor_targ.forward(obs)  # (K, da)
            b = MultivariateNormal(b_mean, scale_tril=b_A)  # (K, da)
            sampled_actions = b.sample((N, ))  # (N, K, da)
            expanded_states = obs[None, ...].expand(N, -1, -1)  # (N, K, ds)
            target_q, _ = self.critic_forward(self.critic_targ,
                                              expanded_states.reshape(-1, ds),
                                              sampled_actions.reshape(-1, da))
            target_q = target_q.reshape(N, K)  # (N, K)
            target_qc, _ = self.critic_forward(self.qc_targ,
                                               expanded_states.reshape(-1, ds),
                                               sampled_actions.reshape(-1, da))
            target_qc = target_qc.reshape(N, K)  # (N, K)

            # if obs_adv is not None:
            #     b_mean_adv, b_A_adv = self.actor_targ.forward(obs_adv)  # (K,)

        t = time.time()
        eta, lam, iterations = self.solver.solve(target_q, target_qc)
        solver_time = time.time() - t
        qij = torch.softmax((target_q - lam * target_qc) / eta,
                            dim=0)  # (N, K) or (da, K)

        # M-Step of Policy Improvement
        # [2] 4.2 Fitting an improved policy (Step 3)
        for _ in range(self.mstep_iteration_num):

            ############################################################################
            ########################### normal observation  ############################
            ############################################################################
            mean, A = self.actor.forward(obs)
            # First term of last eq of [2] p.5
            # see also [2] 4.2.1 Fitting an improved Gaussian policy
            π1 = MultivariateNormal(loc=mean, scale_tril=b_A)  # (K,)
            π2 = MultivariateNormal(loc=b_mean, scale_tril=A)  # (K,)

            loss_p = -torch.mean(qij * (
                π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                + π2.expand((N, K)).log_prob(sampled_actions)))  # (N, K)

            kl_μ, kl_Σ, Σi_det, Σ_det = gaussian_kl(μi=b_mean,
                                                    μ=mean,
                                                    Ai=b_A,
                                                    A=A)

            if np.isnan(kl_μ.item()):  # This should not happen
                raise RuntimeError('kl_μ is nan')
            if np.isnan(kl_Σ.item()):  # This should not happen
                raise RuntimeError('kl_Σ is nan')

            # Update lagrange multipliers by gradient descent
            self.alpha_mean -= self.alpha_mean_scale * (
                self.kl_mean_constraint - kl_μ).detach().item()
            self.alpha_var -= self.alpha_var_scale * (self.kl_var_constraint -
                                                      kl_Σ).detach().item()
            self.alpha_mean = np.clip(self.alpha_mean, 0.0,
                                      self.alpha_mean_max)
            self.alpha_var = np.clip(self.alpha_var, 0.0, self.alpha_var_max)

            loss_kl = -(self.alpha_mean *
                        (self.kl_mean_constraint - kl_μ) + self.alpha_var *
                        (self.kl_var_constraint - kl_Σ))

            loss_all = loss_p + loss_kl

            ############################################################################
            ############################     optimize     ##############################
            ############################################################################

            self.actor_optimizer.zero_grad()

            loss_all.backward()
            if self.grad_norm_lim is not None:
                clip_grad_norm_(self.actor.parameters(), self.grad_norm_lim)
            self.actor_optimizer.step()

            # Log actor update info
            self.logger.store(
                QcThres=self.qc_thres,
                QcValue=to_ndarray(torch.mean(target_qc)),
                iterations=iterations,
                solver_time=solver_time,
                eta=eta,
                lam=lam,
                actionVar=torch.mean(A).item(),
            )

            self.logger.store(Σ_det=Σ_det.item(),
                              kl_Σ=kl_Σ.item(),
                              kl_μ=kl_μ.item(),
                              alpha_mean=self.alpha_mean,
                              alpha_var=self.alpha_var,
                              tab="learner_kl")
            self.logger.store(LossAll=loss_all.item(),
                              LossMle=loss_p.item(),
                              loss_kl=loss_kl.item(),
                              NoiseScale=self.noise_scale,
                              tab="learner_losses")

    def _update_critic(self, data):
        '''
        Update the critic network
        '''
        def critic_loss():
            obs, act, reward, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['rew']), to_tensor(data['done'])

            obs_next = to_tensor(data['obs2'])

            _, q_list = self.critic_forward(self.critic, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                _, logp_a_next, pi_dist = self.actor_forward(obs_next)
                act_next = pi_dist.sample()
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.critic_targ, obs_next,
                                                   act_next)
                backup = reward + self.gamma**self.n_step * (1 -
                                                             done) * q_pi_targ
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

    def _update_critic_retrace(self, data):
        '''
        Update the critic network
        '''
        def critic_loss():
            # shape: [K, T, obs_dim/act_dim/None]
            obs, act, reward, done, obs_next, logp_a = to_tensor(data['obs']), \
                to_tensor(data['act']), to_tensor(data['rew']), \
                to_tensor(data['done']), to_tensor(data['obs2']), to_tensor(data['logp_a'])

            T = obs.shape[1]
            K = obs.shape[0]
            N = self.sample_action_num
            da = self.act_dim
            ds = self.obs_dim

            _, q_list = self.critic_forward(self.critic, obs, act)  # [K, T]
            for i in range(len(q_list)):
                q_list[i] = q_list[i][:, :-1]  # {K, T-1}

            with torch.no_grad():
                # target q value
                q_t_targ, _ = self.critic_forward(self.critic_targ, obs, act)

                # target_policy_probs
                _, _, pi_dist = self.actor_forward(obs)
                logp_a_targ = pi_dist.log_prob(act)

                # retrace weights
                log_c_ret = 1 / da * (logp_a_targ - logp_a).clamp(max=0)
                c_ret = log_c_ret.exp() * self.retrace_lambda

                # target state value
                sampled_actions = pi_dist.sample((N, ))  # [N, K, T, da]
                expanded_states = obs[None, ...].expand(N, -1, -1,
                                                        -1)  # [N, K, T, ds]
                q, _ = self.critic_forward(self.critic_targ,
                                           expanded_states.reshape(-1, ds),
                                           sampled_actions.reshape(-1, da))
                q = q.reshape(N, K, T)
                v_t = torch.mean(q, dim=0)  # [K, T]

                # Q_ret
                backup = torch.zeros_like(q_t_targ[:, :-1],
                                          dtype=torch.float)  # [K, T-1]
                Q_ret = v_t[:, -1]
                Q_ret[done[:, -1] == 1] = 0  # Q_ret = 0, if done

                # Computes the retrace loss recursively according to
                # L = 𝔼_τ[(Q_t - Q_ret_t)^2]
                # Q_ret_t = r_t + γ * (𝔼_π_target [Q(s_t+1,•)] + c_t+1 * Q_π_target(s_t+1,a_t+1)) + γ * c_t+1 * Q_ret_t+1
                # with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}
                for t in reversed(range(T - 1)):
                    Q_ret = reward[:, t] + self.gamma * Q_ret
                    backup[:, t] = Q_ret
                    Q_ret = c_ret[:, t] * (Q_ret - q_t_targ[:, t]) + v_t[:, t]

            # MSE loss against Bellman backup
            loss_q = self.critic.loss(backup, q_list, self.td_error_lim)
            # Useful info for logging
            q_info = dict()
            q_info["Q_retrace_weight"] = to_ndarray(c_ret)
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

    def _update_qc(self, data):
        '''
        Update the qc network
        '''
        def critic_loss():
            obs, act, reward, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['cost']), to_tensor(data['done'])

            obs_next = to_tensor(data['obs2'])

            _, q_list = self.critic_forward(self.qc, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                _, logp_a_next, pi_dist = self.actor_forward(obs_next)
                act_next = pi_dist.sample()
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.qc_targ, obs_next,
                                                   act_next)
                # backup = reward + self.gamma * (1 - done) * q_pi_targ
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
        self.logger.store(LossQC=loss_qc.item(), **loss_qc_info)

    def _update_qc_retrace(self, data):
        '''
        Update the qc network
        '''
        def critic_loss():
            # shape: [B, T, obs_dim/act_dim/None]
            obs, act, reward, obs_next, logp_a = to_tensor(data['obs']), \
                to_tensor(data['act']), to_tensor(data['cost']), \
                to_tensor(data['obs2']), to_tensor(data['logp_a'])

            T = obs.shape[1]
            K = obs.shape[0]
            N = self.sample_action_num
            da = self.act_dim
            ds = self.obs_dim

            _, q_list = self.critic_forward(self.qc, obs, act)
            for i in range(len(q_list)):
                q_list[i] = q_list[i][:, :-1]

            with torch.no_grad():
                # target q value
                q_t_targ, _ = self.critic_forward(self.qc_targ, obs, act)

                # target_policy_probs
                _, _, pi_dist = self.actor_forward(obs)
                logp_a_targ = pi_dist.log_prob(act)

                # retrace weights
                log_c_ret = 1 / da * (logp_a_targ - logp_a).clamp(max=0)
                c_ret = log_c_ret.exp() * self.retrace_lambda

                sampled_actions = pi_dist.sample((N, ))  # [N, K, T, da]
                expanded_states = obs[None, ...].expand(N, -1, -1,
                                                        -1)  # [N, K, T, ds]
                q, _ = self.critic_forward(self.qc_targ,
                                           expanded_states.reshape(-1, ds),
                                           sampled_actions.reshape(-1, da))
                q = q.reshape(N, K, T)
                v_t = torch.mean(q, dim=0)  # [K, T]

                # Q_ret
                backup = torch.zeros_like(q_t_targ[:, :-1],
                                          dtype=torch.float)  # [K, T]
                Q_ret = v_t[:, -1]

                for t in reversed(range(T - 1)):
                    Q_ret = reward[:, t] + self.gamma * Q_ret
                    backup[:, t] = Q_ret
                    Q_ret = c_ret[:, t] * (Q_ret - q_t_targ[:, t]) + v_t[:, t]

            # MSE loss against Bellman backup
            loss_q = self.qc.loss(backup, q_list, self.td_error_lim)
            # Useful info for logging
            q_info = dict()
            q_info["QC_retrace_weight"] = to_ndarray(c_ret)
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
        self.logger.store(LossQC=loss_qc.item(), **loss_qc_info)
