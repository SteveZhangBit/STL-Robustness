from copy import deepcopy
from tqdm import tqdm
import gym
import numpy as np
import torch
import torch.nn as nn
from rsrl.policy import LagrangianPIDController
from rsrl.policy.onpolicy_base import OnPolicyBase
from rsrl.policy.model.mlp_ac import (MLPCategoricalActor, MLPGaussianActor,
                                      EnsembleQCritic, mlp)
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import (count_vars, get_device_name, to_device,
                                  to_ndarray, to_tensor)
from rsrl.policy.adversary.adv_random import AdvUniform, AdvGaussian
from rsrl.policy.adversary.adv_critic import AdvCritic, AdvCriticPPO
from rsrl.policy.adversary.adv_mad import AdvMad, AdvMadPPO
from rsrl.policy.adversary.adv_amad import AdvAmad, AdvAmadPPO
from rsrl.policy.adversary.adv_base import Adv
from torch.optim import Adam
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_


class RobustPPOLagrangian(OnPolicyBase):
    attacker_cls = dict(uniform=AdvUniform,
                        gaussian=AdvGaussian,
                        mad=AdvMadPPO,
                        amad=AdvAmadPPO,
                        max_reward=AdvCriticPPO,
                        min_reward=AdvCriticPPO,
                        max_cost=AdvCriticPPO)

    def __init__(self, env: gym.Env, logger: EpochLogger, env_cfg: dict,
                 adv_cfg: dict, rs_mode, episode_rerun_num, KP, KI, KD,
                 per_state, use_adv_multiplier, clip_ratio, clip_ratio_adv,
                 weight_adv, target_kl, train_actor_iters, train_critic_iters,
                 update_adv_freq, kl_coef, decay_epoch, attacker_names,
                 start_epoch, **kwargs) -> None:
        r'''
        Proximal Policy Optimization (PPO) with Lagrangian multiplier
        '''
        super().__init__(env, logger, env_cfg, **kwargs)

        self.adv_cfg = adv_cfg
        self.clip_ratio = clip_ratio
        self.clip_ratio_adv = clip_ratio_adv
        self.target_kl = target_kl
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.update_adv_freq = update_adv_freq
        self.kl_coef = kl_coef
        self.weight_adv = weight_adv
        self.episode_rerun_num = episode_rerun_num
        self.use_adv_multiplier = use_adv_multiplier

        self.controller = LagrangianPIDController(KP, KI, KD, self.cost_limit,
                                                  per_state)

        self.controller_adv = LagrangianPIDController(KP, KI, KD,
                                                      self.cost_limit,
                                                      per_state)

        #############################################################################
        ############################ for attacker usage #############################
        #############################################################################
        self._init_adversary_critic()
        self.mode = rs_mode
        self.attacker_names = attacker_names
        self.adv_training_modes = [
            "mc",
            "mr",
            "mcv",
            "mrv",
            "mad",
            "madv",
            "uniform",
            "gaussian",
        ]

        if self.mode in self.adv_training_modes:
            self.apply_adv_in_training = True
            self._init_adversary()
        else:
            self.apply_adv_in_training = False
            self._init_adversary_baseline()

        self.mode_to_loss = {
            "vanilla": self.vanilla_policy_loss,
            "kl": self.kl_regularized_policy_loss,
            "klmc": self.kl_regularized_policy_loss,
            "klmr": self.kl_regularized_policy_loss,
        }
        for m in self.adv_training_modes:
            self.mode_to_loss[m] = self.vanilla_policy_loss

        self.noise_scale_schedule = 0
        self.decay_epoch = decay_epoch
        self.start_epoch = start_epoch
        self.decay_func = lambda x: self.noise_scale - self.noise_scale * np.exp(
            -5. * x / self.decay_epoch)

        # Set up model saving
        self.save_model()

    def _post_epoch_process(self, epoch):
        self.epoch = epoch
        if self.epoch < self.start_epoch:
            self.noise_scale_schedule = 0
        else:
            self.noise_scale_schedule = self.decay_func(epoch -
                                                        self.start_epoch)

    def _init_adversary(self):
        self.noise_scale = self.adv_cfg["noise_scale"]
        self.attack_freq = self.adv_cfg["attack_freq"]
        attacker_names_dict = {
            "mc": ["max_cost"],
            "mcv": ["max_cost", "vanilla"],
            "mr": ["max_reward"],
            "mrv": ["max_reward", "vanilla"],
            "mad": ["mad"],
            "madv": ["mad", "vanilla"],
            "uniform": ["uniform"],
            "gaussian": ["gaussian"],
        }
        self.attacker_names = attacker_names_dict[self.mode]
        self.adversary_dict = {}
        for attacker_name in self.attacker_names:
            if attacker_name in self.attacker_cls:
                cfg = self.adv_cfg[attacker_name + "_cfg"]
                adv_cls = self.attacker_cls[attacker_name]
                adversary = adv_cls(self.obs_dim, **cfg)
            else:
                adversary = None
            self.adversary_dict[attacker_name] = adversary
        self.adversary_num = len(self.attacker_names)

    def _init_adversary_baseline(self):
        self.noise_scale = self.adv_cfg["noise_scale"]
        self.attack_freq = self.adv_cfg["attack_freq"]

        ############## for seperate mode #################
        if self.mode == "klmc":
            cfg = self.adv_cfg["max_cost_cfg"]
            self.adversary = self.attacker_cls["max_cost"](self.obs_dim, **cfg)
        elif self.mode == "klmr":
            cfg = self.adv_cfg["max_reward_cfg"]
            self.adversary = self.attacker_cls["max_reward"](self.obs_dim,
                                                             **cfg)
        elif self.mode == "kl":
            cfg = self.adv_cfg["mad_cfg"]
            self.adversary = self.attacker_cls["mad"](self.obs_dim, **cfg)
        elif self.mode == "vanilla":
            pass
        else:
            raise NotImplementedError

    def _init_adversary_critic(self,
                               critic2_state_dict=None,
                               critic2_optimizer_state_dict=None,
                               qc2_state_dict=None,
                               qc2_optimizer_state_dict=None):
        # off-policy Q network; for attacker usage
        critic2 = EnsembleQCritic(self.obs_dim,
                                  self.act_dim,
                                  self.hidden_sizes,
                                  nn.ReLU,
                                  num_q=2)
        qc2 = EnsembleQCritic(self.obs_dim,
                              self.act_dim,
                              self.hidden_sizes,
                              nn.ReLU,
                              num_q=1)
        if critic2_state_dict is not None:
            critic2.load_state_dict(critic2_state_dict)
        if qc2_state_dict is not None:
            qc2.load_state_dict(qc2_state_dict)

        self.critic2 = to_device(critic2)
        self.critic2_targ = deepcopy(self.critic2)

        self.qc2 = to_device(qc2)
        self.qc2_targ = deepcopy(self.qc2)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic2_targ.parameters():
            p.requires_grad = False
        for p in self.qc2_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for safety critic
        self.qc2_optimizer = Adam(self.qc2.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(),
                                      lr=self.critic_lr)
        if critic2_optimizer_state_dict is not None:
            self.critic2_optimizer.load_state_dict(
                critic2_optimizer_state_dict)
        if qc2_optimizer_state_dict is not None:
            self.qc2_optimizer.load_state_dict(qc2_optimizer_state_dict)

    def get_obs_adv(self, adv: Adv, obs: torch.Tensor):
        epsilon = adv.attack_batch(self, obs, self.noise_scale)
        obs_adv = (epsilon + obs).detach()
        return obs_adv

    def collect_data(self, warmup=False):
        '''
        Interact with the environment to collect data
        '''
        self.cost_list = []
        qc_list = []
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        epoch_num = 0
        for i in range(self.interact_steps):
            epsilon = 0
            if self.apply_adv_in_training and self.noise_scale_schedule > 0:
                attacker_name = self.attacker_names[epoch_num %
                                                    self.adversary_num]
                adversary = self.adversary_dict[attacker_name]
                if adversary is not None:
                    epsilon = adversary.attack_at_eval(
                        self, obs, self.noise_scale_schedule)

            obs = obs + epsilon

            action, value, log_prob = self.act(obs)
            obs_next, reward, done, info = self.env.step(action)

            cost_value = self.get_qc_v(obs)

            if done and "TimeLimit.truncated" in info:
                done = False
                timeout_env = True
            else:
                timeout_env = False

            cost = info["cost"] if "cost" in info else 0
            self.buffer.store(obs, np.squeeze(action), reward, value, log_prob,
                              done, cost, cost_value)
            self.cpp_buffer.add(obs=obs,
                                act=np.squeeze(action),
                                rew=reward,
                                obs2=obs_next,
                                done=done,
                                cost=cost / self.cost_normalizer)
            self.logger.store(VVals=value, CostVVals=cost_value, tab="worker")
            ep_reward += reward
            ep_cost += cost
            ep_len += 1
            obs = obs_next

            timeout = ep_len == self.timeout_steps - 1 or i == self.interact_steps - 1 or timeout_env and not done
            terminal = done or timeout
            if terminal:
                if timeout:
                    _, value, _ = self.act(obs)
                    cost_value = self.get_qc_v(obs)
                else:
                    value = 0
                    cost_value = 0
                self.buffer.finish_path(value, cost_value)
                if i < self.interact_steps - 1:
                    self.logger.store(EpRet=ep_reward,
                                      EpLen=ep_len,
                                      EpCost=ep_cost,
                                      tab="worker")

                self.cumulative_cost_adv += ep_cost
                if ep_reward > self.maximum_reward_adv:
                    self.maximum_reward_adv = ep_reward
                self.logger.store(CumulativeCostAdv=self.cumulative_cost_adv,
                                  MaximumRewardAdv=self.maximum_reward_adv,
                                  tab="worker")

                obs = self.env.reset()
                self.cost_list.append(ep_cost)
                ep_reward = 0
                ep_cost = 0
                ep_len = 0
                epoch_num += 1

        return self.interact_steps

    def train_one_epoch(self, warmup=False, verbose=False):
        '''
        Train one epoch, interact with the runner
        '''
        self.logger.store(CostLimit=self.cost_limit, tab="worker")
        epoch_steps = 0

        if warmup and verbose:
            print("*** Warming up begin ***")

        steps = self.collect_data(warmup=warmup)
        epoch_steps += steps

        training_range = range(self.episode_rerun_num)
        if verbose:
            training_range = tqdm(training_range,
                                  desc='Training steps: ',
                                  position=1,
                                  leave=False)
        for i in training_range:
            off_policy_data = self.get_sample2()
            self.train_Q_network(off_policy_data)

        data = self.get_sample()
        self.learn_on_batch(data)
        return epoch_steps

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, value, logp.
        This API is used to interact with the env.

        @param obs, 1d ndarray
        @param eval, evaluation mode
        @return act, value, logp, 1d ndarray
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            _, a, logp_a = self.actor_forward(obs, deterministic=deterministic)
            v = self.critic_forward(self.critic, obs)
        # squeeze them to the right shape
        a, v, logp_a = np.squeeze(to_ndarray(a), axis=0), np.squeeze(
            to_ndarray(v)), np.squeeze(to_ndarray(logp_a))
        return a, v, logp_a

    def get_risk_estimation(self, obs):
        '''
        Given an obs array (obs_dim), output a risk (qc) value, and the action
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            _, a, logp_a = self.actor_forward(obs, deterministic=True)
            # vc = self.critic_forward(self.qc, obs)
            qc, _ = self.critic_forward2(self.qc2, obs, a)
        return torch.squeeze(qc).item(), np.squeeze(to_ndarray(a), axis=0)

    def train_Q_network(self, data: dict):
        self._update_critic2(data)
        self._update_qc2(data)
        self._polyak_update_target(self.critic2, self.critic2_targ)
        self._polyak_update_target(self.qc2, self.qc2_targ)

    def learn_on_batch(self, data: dict):
        self._update_actor(data)

        LossV, DeltaLossV = self._update_critic(self.critic, data["obs"],
                                                data["ret"],
                                                self.critic_optimizer)
        # Log critic update info
        self.logger.store(LossV=LossV, DeltaLossV=DeltaLossV)

        LossVQC, DeltaLossVQC = self._update_critic(self.qc, data["obs"],
                                                    data["cost_ret"],
                                                    self.qc_optimizer)
        # Log safety critic update info
        self.logger.store(LossVQC=LossVQC, DeltaLossVQC=DeltaLossVQC)

    def critic_forward2(self, critic, obs, act):
        # return the minimum q values and the list of all q_values
        return critic.predict(obs, act)

    def critic_forward(self, critic, obs):
        # Critical to ensure value has the right shape.
        # Without this, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        return torch.squeeze(critic(obs), -1)

    def actor_forward(self, obs, act=None, deterministic=False):
        r''' 
        Return action distribution and action log prob [optional].
        @param obs, [tensor], (batch, obs_dim)
        @param act, [tensor], (batch, act_dim). If None, log prob is None
        @return pi, [torch distribution], (batch,)
        @return a, [torch distribution], (batch, act_dim)
        @return logp, [tensor], (batch,)
        '''
        pi, a, logp = self.actor(obs, act, deterministic)
        return pi, a, logp

    def get_qc_v(self, obs):
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            v = self.critic_forward(self.qc, obs)
        return np.squeeze(to_ndarray(v))

    def get_adv_epcost(self, obs, obs_adv, ep_cost):
        with torch.no_grad():
            pi, a_adv, _ = self.actor_forward(
                obs_adv, deterministic=True)  # (batch, action space)
            qc_adv, _ = self.critic_forward2(self.qc2, obs, a_adv)
            pi, a, _ = self.actor_forward(
                obs, deterministic=True)  # (batch, action space)
            qc, _ = self.critic_forward2(self.qc2, obs, a)
            ratio = torch.mean(qc_adv) / torch.mean(qc)

        # with torch.no_grad(): doesn't work!
        #     v = self.critic_forward(self.qc, obs)
        #     v_adv = self.critic_forward(self.qc, obs_adv)
        #     ratio = torch.mean(v_adv) / torch.mean(v)
        return ratio * ep_cost

    def vanilla_policy_loss(self, obs, act, logp_old, advantage,
                            cost_advantage, multiplier, *args, **kwargs):
        pi, _, logp = self.actor_forward(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                               1 + self.clip_ratio) * advantage

        qc_penalty = (ratio * cost_advantage * multiplier).mean()
        loss_vallina = -(torch.min(ratio * advantage, clip_adv)).mean()
        loss_pi = loss_vallina + qc_penalty
        loss_pi /= 1 + multiplier
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()

        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(KL=approx_kl,
                       Entropy=ent,
                       ClipFrac=clipfrac,
                       LossQcPenalty=to_ndarray(qc_penalty),
                       LossVallina=to_ndarray(loss_vallina))

        return loss_pi, pi_info, pi

    def kl_regularized_policy_loss(self, obs, act, logp_old, advantage,
                                   cost_advantage, multiplier, *args,
                                   **kwargs):
        loss_pi, pi_info, pi = self.vanilla_policy_loss(
            obs, act, logp_old, advantage, cost_advantage, multiplier)
        with torch.no_grad():
            _, a_targ, _ = self.actor_forward(obs, deterministic=True)

        pi_adv, a_adv, _ = self.actor_forward(self.obs_adv, deterministic=True)
        kl_adv = ((a_targ.detach() - a_adv)**2).sum(axis=-1)
        kl_regularizer = torch.mean(kl_adv) * self.kl_coef

        loss_pi += kl_regularizer
        pi_info["LossKLAdv"] = to_ndarray(kl_regularizer)
        return loss_pi, pi_info, pi

    def _update_obs_adv(self, obs):
        if "kl" in self.mode:
            self.obs_adv = self.get_obs_adv(self.adversary, obs)

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        obs, act, advantage, logp_old = data['obs'], data['act'], data[
            'adv'], data['logp']

        # detach is very important here!
        # Otherwise the gradient will backprop through the multiplier.
        cost_ret = data["cost_ret"]
        cost_advantage = data["cost_adv"]
        ep_cost = data["ep_cost"]

        policy_loss = self.mode_to_loss[self.mode]

        multiplier = self.controller.control(ep_cost).detach()

        self._update_obs_adv(obs)

        ep_cost_adv = ep_cost
        multiplier_adv = multiplier

        pi_l_old, pi_info_old, _ = policy_loss(obs, act, logp_old, advantage,
                                               cost_advantage, multiplier,
                                               multiplier_adv)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            # update adversarial obs for a fixed frequency
            if (i + 1) % self.update_adv_freq == 0:
                self._update_obs_adv(obs)

            loss_pi, pi_info, pi = policy_loss(obs, act, logp_old, advantage,
                                               cost_advantage, multiplier,
                                               multiplier_adv)
            if i == 0 and pi_info['KL'] >= 1e-7:
                print("**" * 20)
                print("1st kl: ", pi_info['KL'])
            if pi_info['KL'] > 1.5 * self.target_kl:
                self.logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break

            loss_pi.backward()
            clip_grad_norm_(self.actor.parameters(), 0.02)
            self.actor_optimizer.step()

        self.logger.store(StopIter=i,
                          LossPi=to_ndarray(pi_l_old),
                          ObservedCost=to_ndarray(ep_cost),
                          CostLimit=self.cost_limit,
                          ObservedCostAdv=to_ndarray(ep_cost_adv),
                          Lagrangian=to_ndarray(multiplier),
                          LagrangianAdv=to_ndarray(multiplier_adv),
                          DeltaLossPi=(to_ndarray(loss_pi) -
                                       to_ndarray(pi_l_old)),
                          QcThres=self.qc_thres,
                          QcRet=torch.mean(data["cost_ret"]).item(),
                          **pi_info)

    def _update_critic(self, critic, obs, ret, critic_optimizer):
        '''
        Update the critic network
        '''
        obs, ret = to_tensor(obs), to_tensor(ret)

        def critic_loss():
            ret_pred = self.critic_forward(critic, obs)
            return ((ret_pred - ret)**2).mean()

        loss_old = critic_loss().item()

        # Value function learning
        for i in range(self.train_critic_iters):
            critic_optimizer.zero_grad()
            loss_critic = critic_loss()
            loss_critic.backward()
            critic_optimizer.step()

        return loss_old, to_ndarray(loss_critic) - loss_old

    def _update_critic2(self, data):
        '''
        Update the critic network
        '''
        def critic_loss():
            obs, act, reward, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['rew']), to_tensor(data['done'])

            obs_next = to_tensor(data['obs2'])

            _, q_list = self.critic_forward2(self.critic2, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                pi_dist, act_next, _ = self.actor_forward(obs_next,
                                                          deterministic=False)

                # Target Q-values
                q_pi_targ, _ = self.critic_forward2(self.critic2_targ,
                                                    obs_next, act_next)
                backup = reward + self.gamma * (1 - done) * q_pi_targ
            # MSE loss against Bellman backup
            loss_q = self.critic2.loss(backup, q_list)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.critic2_optimizer.zero_grad()
        loss_critic, loss_q_info = critic_loss()
        loss_critic.backward()
        self.critic2_optimizer.step()

        # Log critic update info
        # Record things
        self.logger.store(LossQ=loss_critic.item(), **loss_q_info)

    def _update_qc2(self, data):
        '''
        Update the qc network
        '''
        def critic_loss():
            obs, act, reward, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['cost']), to_tensor(data['done'])

            obs_next = to_tensor(data['obs2'])

            _, q_list = self.critic_forward2(self.qc2, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                pi_dist, act_next, _ = self.actor_forward(obs_next,
                                                          deterministic=False)
                # Target Q-values
                q_pi_targ, _ = self.critic_forward2(self.qc2_targ, obs_next,
                                                    act_next)
                backup = reward + self.gamma * (1 - done) * q_pi_targ
                # backup = reward + self.gamma * q_pi_targ
            # MSE loss against Bellman backup
            loss_q = self.qc2.loss(backup, q_list)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QCVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.qc2_optimizer.zero_grad()
        loss_qc, loss_qc_info = critic_loss()
        loss_qc.backward()
        self.qc2_optimizer.step()

        # Log critic update info
        # Record things
        self.logger.store(LossQC=loss_qc.item(), **loss_qc_info)

    def save_model(self):
        actor, actor_optimizer = self.actor.state_dict(
        ), self.actor_optimizer.state_dict()
        critic, critic_optimizer = self.critic.state_dict(
        ), self.critic_optimizer.state_dict()
        critic2, critic2_optimizer = self.critic2.state_dict(
        ), self.critic2_optimizer.state_dict()
        qc2, qc2_optimizer = self.qc2.state_dict(
        ), self.qc2_optimizer.state_dict()
        model = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "critic2": critic2,
            "critic2_optimizer": critic2_optimizer,
            "qc2": qc2,
            "qc2_optimizer": qc2_optimizer
        }
        if self.safe_rl:
            qc, qc_optimizer = self.qc.state_dict(
            ), self.qc_optimizer.state_dict()
            model["qc"] = qc
            model["qc_optimizer"] = qc_optimizer
        self.logger.setup_pytorch_saver(model)

    def load_model(self, path):
        model = torch.load(path)
        assert type(model) is dict, "The loaded model type can not be parsed."
        actor, actor_optimizer = model["actor"], model["actor_optimizer"]
        critic, critic_optimizer = model["critic"], model["critic_optimizer"]
        critic2, critic2_optimizer = model["critic2"], model[
            "critic2_optimizer"]
        qc2, qc2_optimizer = model["qc2"], model["qc2_optimizer"]
        self._init_actor(actor, actor_optimizer)
        self._init_critic(critic, critic_optimizer)
        self._init_adversary_critic(critic2, critic2_optimizer, qc2,
                                    qc2_optimizer)
        if self.safe_rl:
            qc, qc_optimizer = model["qc"], model["qc_optimizer"]
            self._init_qc(qc, qc_optimizer)