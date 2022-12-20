from abc import ABC, abstractclassmethod
from copy import deepcopy

import gym
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
from rsrl.policy.model.mlp_ac import (EnsembleQCritic, CholeskyGaussianActor,
                                      MLPGaussianActor, MLPCategoricalActor,
                                      mlp)
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import (to_device, to_ndarray, to_tensor)
from torch.optim import Adam
from collections import deque


class OffPolicyBase(ABC):
    def __init__(self, env: gym.Env, logger: EpochLogger, env_cfg: dict,
                 actor_lr, critic_lr, gamma, polyak, batch_size, hidden_sizes,
                 num_q, buffer_size, sample_episode_num, episode_rerun_num,
                 safe_rl, num_qc, n_step, use_retrace, unroll_length,
                 retrace_lambda, td_error_lim, grad_norm_lim) -> None:
        super().__init__()
        self.env = env
        self.logger = logger
        self.env_cfg = env_cfg
        self.gamma = gamma
        self.polyak = polyak
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.hidden_sizes = hidden_sizes
        self.num_q = num_q
        self.num_qc = num_qc
        self.safe_rl = safe_rl
        self.n_step = n_step
        self.use_retrace = use_retrace
        self.unroll_length = unroll_length
        self.td_error_lim = td_error_lim
        self.grad_norm_lim = grad_norm_lim
        self.retrace_lambda = retrace_lambda
        self.cumulative_cost = 0
        self.maximum_reward = -99
        self.cumulative_cost_adv = 0  # cumulative cost after robust training
        self.maximum_reward_adv = -99
        self.noise_scale = 0
        self.use_adv = False
        self.alpha_mean_adv = 0
        self.alpha_var_adv = 0
        self.alpha_adv = 0

        # for ADV-CVPO
        self.apply_adv_collect_data = False
        self.adversary = None

        # env_cfg
        self.cost_limit = self.env_cfg["cost_limit"]
        self.cost_normalizer = self.env_cfg["cost_normalizer"]
        # self.reward_normalizer = self.env_cfg["reward_normalizer"]
        self.timeout_steps = self.env_cfg["timeout_steps"]

        # worker_cfg
        self.sample_episode_num = sample_episode_num
        self.episode_rerun_num = episode_rerun_num

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        # Action limit for normalization: critically, assumes all dimensions share the same bound!
        self.act_lim = self.env.action_space.high[0]

        self._init_actor()
        self._init_critic()
        self._init_worker()

        if self.safe_rl:
            self._init_qc()

    def _init_actor(self,
                    actor_state_dict=None,
                    actor_optimizer_state_dict=None):
        actor = CholeskyGaussianActor(self.obs_dim, self.act_dim,
                                      -self.act_lim, self.act_lim,
                                      self.hidden_sizes, nn.ReLU)
        if actor_state_dict is not None:
            actor.load_state_dict(actor_state_dict)
        self.actor = to_device(actor)

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        if actor_optimizer_state_dict is not None:
            self.actor_optimizer.load_state_dict(actor_optimizer_state_dict)

    def _init_critic(self,
                     critic_state_dict=None,
                     critic_optimzer_state_dict=None):
        critic = EnsembleQCritic(self.obs_dim,
                                 self.act_dim,
                                 self.hidden_sizes,
                                 nn.ReLU,
                                 num_q=self.num_q)
        if critic_state_dict is not None:
            critic.load_state_dict(critic_state_dict)
        self.critic = to_device(critic)
        self.critic_targ = deepcopy(self.critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_targ.parameters():
            p.requires_grad = False

        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=self.critic_lr)
        if critic_optimzer_state_dict is not None:
            self.critic_optimizer.load_state_dict(critic_optimzer_state_dict)

    def _init_qc(self, qc_state_dict=None, qc_optimizer_state_dict=None):
        # init safety critic
        qc = EnsembleQCritic(self.obs_dim,
                             self.act_dim,
                             self.hidden_sizes,
                             nn.ReLU,
                             num_q=self.num_qc)
        if qc_state_dict is not None:
            qc.load_state_dict(qc_state_dict)
        self.qc = to_device(qc)
        self.qc_targ = deepcopy(self.qc)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.qc_targ.parameters():
            p.requires_grad = False

        self.qc_optimizer = Adam(self.qc.parameters(), lr=self.critic_lr)
        if qc_optimizer_state_dict is not None:
            self.qc_optimizer.load_state_dict(qc_optimizer_state_dict)

        # compute qc threshold
        self.qc_thres = self.cost_limit * (
            1 - self.gamma**self.timeout_steps) / (
                1 - self.gamma) / self.timeout_steps / self.cost_normalizer

    def _init_worker(self):
        env_dict = {
            'act': {
                'dtype': np.float32,
                'shape': self.act_dim
            },
            'done': {
                'dtype': np.float32,
            },
            'obs': {
                'dtype': np.float32,
                'shape': self.obs_dim
            },
            'obs2': {
                'dtype': np.float32,
                'shape': self.obs_dim
            },
            'rew': {
                'dtype': np.float32,
            }
        }
        Nstep = {
            "size": self.n_step,
            "gamma": self.gamma,
            "rew": "rew",
            "next": "obs2"
        }
        if self.safe_rl:
            env_dict["cost"] = {'dtype': np.float32}
            Nstep["rew"] = ["rew", "cost"]

        if self.use_retrace:
            self.n_step = 1
            env_dict = {
                'act': {
                    'dtype': np.float32,
                    'shape': (self.unroll_length, self.act_dim)
                },
                'done': {
                    'dtype': np.float32,
                    'shape': self.unroll_length
                },
                'obs': {
                    'dtype': np.float32,
                    'shape': (self.unroll_length, self.obs_dim)
                },
                'obs2': {
                    'dtype': np.float32,
                    'shape': (self.unroll_length, self.obs_dim)
                },
                'rew': {
                    'dtype': np.float32,
                    'shape': self.unroll_length
                },
                'logp_a': {
                    'dtype': np.float32,
                    'shape': self.unroll_length
                }
            }
            self.q_obs = deque(maxlen=self.unroll_length)
            self.q_act = deque(maxlen=self.unroll_length)
            self.q_obs2 = deque(maxlen=self.unroll_length)
            self.q_rew = deque(maxlen=self.unroll_length)
            self.q_done = deque(maxlen=self.unroll_length)
            self.q_logp_a = deque(maxlen=self.unroll_length)
            if self.safe_rl:
                env_dict["cost"] = {
                    'dtype': np.float32,
                    'shape': self.unroll_length
                }
                self.q_cost = deque(maxlen=self.unroll_length)

        if self.n_step == 1:
            self.cpp_buffer = ReplayBuffer(self.buffer_size, env_dict)
        else:
            self.cpp_buffer = ReplayBuffer(self.buffer_size,
                                           env_dict,
                                           Nstep=Nstep)

    def _polyak_update_target(self, model, model_targ):
        '''
        Update target networks by polyak averaging.
        '''
        with torch.no_grad():
            for p, p_targ in zip(model.parameters(), model_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def collect_data(self, warmup=False):
        '''
        Interact with the environment to collect data
        '''
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        epoch_steps = 0
        terminal_freq = 0
        done_freq = 0
        logp_a = None
        for i in range(self.timeout_steps):
            if warmup:
                action = self.env.action_space.sample()
            else:
                if self.apply_adv_collect_data:
                    obs = to_tensor(obs)
                    epsilon = self.adversary.attack_batch(
                        self, obs[None], self.noise_scale_max)
                    obs_adv = (epsilon + obs).detach()
                    obs = to_ndarray(obs_adv)[0]
                action, logp_a = self.act(obs,
                                          deterministic=False,
                                          with_logprob=True)
            obs_next, reward, done, info = self.env.step(action)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if i == self.timeout_steps - 1 or "TimeLimit.truncated" in info else done
            # ignore the goal met terminal condition
            terminal = done
            done = True if "goal_met" in info and info["goal_met"] else done

            if done:
                done_freq += 1
            if self.use_retrace:
                self.q_obs.append(obs)
                self.q_act.append(np.squeeze(action))
                self.q_rew.append(reward)
                self.q_obs2.append(obs_next)
                self.q_done.append(done)
                self.q_logp_a.append(logp_a)
                if "cost" in info:
                    cost = info["cost"]
                    ep_cost += cost
                    self.q_cost.append(cost)
                    if len(self.q_obs) == self.unroll_length:
                        self.cpp_buffer.add(obs=np.array(self.q_obs),
                                            act=np.array(self.q_act),
                                            rew=np.array(self.q_rew),
                                            obs2=np.array(self.q_obs2),
                                            done=np.array(self.q_done),
                                            logp_a=np.array(self.q_logp_a),
                                            cost=np.array(self.q_cost) /
                                            self.cost_normalizer)
                else:
                    if len(self.obs) == self.unroll_length:
                        self.cpp_buffer.add(obs=np.array(self.q_obs),
                                            act=np.array(self.q_act),
                                            rew=np.array(self.q_rew),
                                            obs2=np.array(self.q_obs2),
                                            done=np.array(self.q_done),
                                            logp_a=np.array(self.q_logp_a))
            else:
                if "cost" in info:
                    cost = info["cost"]
                    ep_cost += cost
                    self.cpp_buffer.add(obs=obs,
                                        act=np.squeeze(action),
                                        rew=reward,
                                        obs2=obs_next,
                                        done=done,
                                        cost=cost / self.cost_normalizer)
                else:
                    self.cpp_buffer.add(obs=obs,
                                        act=np.squeeze(action),
                                        rew=reward,
                                        obs2=obs_next,
                                        done=done)
            ep_reward += reward
            ep_len += 1
            epoch_steps += 1
            obs = obs_next

            if terminal or i == self.timeout_steps - 1:
                self.cumulative_cost += ep_cost
                if self.use_adv:
                    self.cumulative_cost_adv += ep_cost
                    if ep_reward > self.maximum_reward:
                        self.maximum_reward_adv = ep_reward
                if ep_reward > self.maximum_reward:
                    self.maximum_reward = ep_reward
                self.logger.store(EpRet=ep_reward,
                                  EpCost=ep_cost,
                                  EpLen=ep_len,
                                  tab="worker")

            if terminal:
                terminal_freq += 1
                obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
                self.cpp_buffer.on_episode_end()
                if self.use_retrace:
                    self.q_obs.clear()
                    self.q_act.clear()
                    self.q_obs2.clear()
                    self.q_rew.clear()
                    self.q_done.clear()
                    self.q_logp_a.clear()
                    if "cost" in info:
                        self.q_cost.clear()
                # break

        self.logger.store(Terminal=terminal_freq, Done=done_freq, tab="worker")
        self.logger.store(CumulativeCost=self.cumulative_cost,
                          MaximumReward=self.maximum_reward,
                          CumulativeCostAdv=self.cumulative_cost_adv,
                          MaximumRewardAdv=self.maximum_reward_adv,
                          tab="Efficiency")
        return epoch_steps

    def _post_epoch_process(self):
        pass

    def _pre_epoch_process(self):
        pass

    def get_obs_adv(self, adv, obs):
        pass

    def get_sample(self):
        data = to_tensor(self.cpp_buffer.sample(self.batch_size))
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        if "cost" in data:
            data["cost"] = torch.squeeze(data["cost"])
        return data

    def clear_buffer(self):
        self.cpp_buffer.clear()

    def train_one_epoch(self, warmup=False, verbose=False):
        '''
        Train one epoch, interact with the runner
        '''
        self.logger.store(CostLimit=self.cost_limit, tab="misc")
        epoch_steps = 0

        if warmup and verbose:
            print("*** Warming up begin ***")

        collect_data_range = range(self.sample_episode_num)
        if verbose:
            collect_data_range = tqdm(collect_data_range,
                                      desc='Collecting data: ',
                                      position=1,
                                      leave=False)
        epoch_steps = 0
        for i in collect_data_range:
            steps = self.collect_data(warmup=warmup)
            epoch_steps += steps

        train_steps = self.episode_rerun_num * epoch_steps // self.batch_size

        training_range = range(train_steps)
        if verbose:
            training_range = tqdm(training_range,
                                  desc='Training steps: ',
                                  position=1,
                                  leave=False)

        for i in training_range:
            data = self.get_sample()
            self.learn_on_batch(data)

        if warmup and verbose:
            print("*** Warming up finished ***")

        return epoch_steps

    @abstractclassmethod
    def act(self, state, deterministic):
        '''
        Given a single state, return the action, value, logp.
        This API is used to interact with the env.
        '''
        raise NotImplementedError

    @abstractclassmethod
    def learn_on_batch(self, data):
        '''
        Train all models with a batch of data
        '''
        raise NotImplementedError

    def save_model(self):
        '''
        Save the model to dir
        '''
        actor, critic = self.actor.state_dict(), self.critic.state_dict()
        actor_optimzer, critic_optimzer = self.actor_optimizer.state_dict(), \
                                          self.critic_optimizer.state_dict()
        model = {
            "actor": actor,
            "critic": critic,
            "actor_optimzer": actor_optimzer,
            "critic_optimzer": critic_optimzer
        }
        if self.safe_rl:
            qc = self.qc.state_dict()
            qc_optimizer = self.qc_optimizer.state_dict()
            model["qc"] = qc
            model["qc_optimizer"] = qc_optimizer
        self.logger.setup_pytorch_saver(model)

    def load_model(self, path):
        '''
        Load the model from dir
        '''
        model = torch.load(path)
        actor, actor_optimizer = model["actor"], model["actor_optimzer"]
        critic, critic_optimizer = model["critic"], model["critic_optimzer"]
        self._init_actor(actor, actor_optimizer)
        self._init_critic(critic, critic_optimizer)
        if self.safe_rl:
            qc, qc_optimizer = model["qc"], model["qc_optimizer"]
            self._init_qc(qc, qc_optimizer)
