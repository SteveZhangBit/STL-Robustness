from abc import ABC, abstractclassmethod
from copy import deepcopy
from collections import deque

import gym
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam

from cpprb import ReplayBuffer
from rsrl.policy.model.mlp_ac import (EnsembleQCritic, CholeskyGaussianActor,
                                      MLPGaussianActor, MLPCategoricalActor,
                                      mlp)
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import to_device, to_ndarray, to_tensor, combined_shape, discount_cumsum, to_tensor


class OnPolicyBuffer:
    r"""
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.max_size = size
        self.clear()

    def clear(self):
        self.obs_buf = np.zeros(combined_shape(self.max_size, self.obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.max_size, self.act_dim),
                                dtype=np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.val_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_val_buf = np.zeros(self.max_size, dtype=np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)
        self.cost_episode = []
        self.ptr, self.path_start_idx = 0, 0

    def store(self, obs, act, rew, val, logp, done, cost=0, cost_val=0):
        r"""
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.cost_rew_buf[self.ptr] = cost
        self.cost_val_buf[self.ptr] = cost_val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_val=0):
        r"""
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        cost_rews = np.append(self.cost_rew_buf[path_slice], last_cost_val)
        cost_vals = np.append(self.cost_val_buf[path_slice], last_cost_val)

        self.cost_episode.append(np.sum(self.cost_rew_buf[path_slice]))

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas,
                                                   self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        # the next two lines implement GAE-Lambda advantage calculation
        cost_deltas = cost_rews[:-1] + self.gamma * cost_vals[
            1:] - cost_vals[:-1]
        self.cost_adv_buf[path_slice] = discount_cumsum(
            cost_deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.cost_ret_buf[path_slice] = discount_cumsum(cost_rews,
                                                        self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        r"""
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr > 0  # buffer has to have something before you can get
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf[:self.ptr]), np.std(
            self.adv_buf[:self.ptr])
        self.adv_buf[:self.ptr] = (self.adv_buf[:self.ptr] -
                                   adv_mean) / adv_std
        cost_mean = np.mean(self.cost_episode)
        n_traj = len(self.cost_episode)
        data = dict(
            obs=self.obs_buf[:self.ptr],
            act=self.act_buf[:self.ptr],
            ret=self.ret_buf[:self.ptr],
            adv=self.adv_buf[:self.ptr],
            cost_ret=self.cost_ret_buf[:self.ptr],
            cost_adv=self.cost_adv_buf[:self.ptr],
            cost_mean=cost_mean,
            n_traj=n_traj,
            logp=self.logp_buf[:self.ptr],
            done=self.done_buf[:self.ptr],
        )
        tensor_dict = to_tensor(data, dtype=torch.float32)
        return tensor_dict


class OnPolicyBase(ABC):
    def __init__(self, env: gym.Env, logger: EpochLogger, env_cfg: dict,
                 actor_lr, critic_lr, gamma, polyak, batch_size, hidden_sizes,
                 buffer_size, safe_rl, interact_steps, lam,
                 eval_attack_freq) -> None:

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

        self.safe_rl = safe_rl
        self.cost_limit = self.env_cfg["cost_limit"]
        self.cost_normalizer = self.env_cfg["cost_normalizer"]
        self.interact_steps = interact_steps
        self.lam = lam
        self.eval_attack_freq = eval_attack_freq

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        # Action limit for normalization: critically, assumes all dimensions share the same bound!
        self.act_lim = self.env.action_space.high[0]

        self.timeout_steps = self.env_cfg["timeout_steps"]

        self.cumulative_cost_adv = 0  # cumulative cost after robust training
        self.maximum_reward_adv = -99

        self._init_actor()
        self._init_critic()
        self._init_worker()

        if self.safe_rl:
            self._init_qc()

    def _init_actor(self,
                    actor_state_dict=None,
                    actor_optimizer_state_dict=None):
        if isinstance(self.env.action_space, gym.spaces.Box):
            actor = MLPGaussianActor(self.obs_dim, self.act_dim, -self.act_lim,
                                     self.act_lim, self.hidden_sizes, nn.ReLU)
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            actor = MLPCategoricalActor(self.obs_dim, self.env.action_space.n,
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
        critic = mlp([self.obs_dim] + list(self.hidden_sizes) + [1], nn.ReLU)
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
        qc = mlp([self.obs_dim] + list(self.hidden_sizes) + [1], nn.ReLU)
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
                'dtype': np.float32
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
                'dtype': np.float32
            }
        }
        if self.safe_rl:
            env_dict["cost"] = {'dtype': np.float32}

        self.cpp_buffer = ReplayBuffer(self.buffer_size, env_dict)
        self.buffer = OnPolicyBuffer(self.obs_dim, self.act_dim,
                                     self.interact_steps + 1, self.gamma,
                                     self.lam)

    def get_sample(self):
        data = self.buffer.get()
        self.buffer.clear()
        data["ep_cost"] = to_tensor(np.mean(self.cost_list))
        return data

    def get_sample2(self):
        data = to_tensor(self.cpp_buffer.sample(self.batch_size))
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        if "cost" in data:
            data["cost"] = torch.squeeze(data["cost"])
        return data

    def clear_buffer(self):
        self.cpp_buffer.clear()

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

    def _post_epoch_process(self, epoch):
        pass

    def _pre_epoch_process(self, epoch):
        pass

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
