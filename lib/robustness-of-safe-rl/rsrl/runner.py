import time
from copy import deepcopy
import gym

import torch
from tqdm import tqdm
import numpy as np

from rsrl.policy import RobustCVPO, RobustPPOLagrangian, RobustFOCOPS, SACLagrangian
from rsrl.policy.adversary import adv_manager
from rsrl.util.logger import EpochLogger, setup_logger_kwargs
from rsrl.util.run_util import load_config, setup_eval_configs
from rsrl.util.torch_util import export_device_env_variable, seed_torch, to_ndarray, to_tensor
from rsrl.policy.adversary.adv_manager import AdvManager

try:
    import bullet_safety_gym
except ImportError:
    print("can not find bullet gym...")

try:
    import safety_gym
except ImportError:
    print("can not find safety gym...")


class Runner:
    '''
    Main entry that coodrinate learner and worker
    '''
    def __init__(self,
                 seed=0,
                 device="cpu",
                 device_id=0,
                 threads=2,
                 env_cfg=None,
                 adv_cfg=None,
                 policy_cfg=None,
                 policy_name="robust_cvpo",
                 epochs=10,
                 save_best_epoch=0,
                 save_every_epoch=10,
                 warmup=False,
                 evaluate_episode_num=1,
                 evaluate_episode_num_after_save_best=1,
                 reward_normalizer=20,
                 eval_attackers=["uniform"],
                 exp_name="exp",
                 load_dir=None,
                 data_dir=None,
                 verbose=True,
                 config_dict=None,
                 **kwarg) -> None:
        seed_torch(seed)
        torch.set_num_threads(threads)
        export_device_env_variable(device, id=device_id)

        self.seed = seed
        self.env_cfg = env_cfg
        self.adv_cfg = adv_cfg
        self.policy_cfg = policy_cfg
        self.policy_name = policy_name
        self.exp_name = exp_name
        self.load_dir = load_dir
        self.data_dir = data_dir
        self.verbose = verbose
        self.config_dict = config_dict

        self.epochs = epochs
        self.save_best_epoch = save_best_epoch
        self.save_every_epoch = save_every_epoch
        self.warmup = warmup
        self.evaluate_episode_num = evaluate_episode_num
        self.evaluate_episode_num_after_save_best = evaluate_episode_num_after_save_best
        self.reward_normalizer = reward_normalizer
        self.eval_attackers = eval_attackers

        self.maximum_reward_eval = -99
        self.maximum_reward_eval_average = -99

    def _init_env(self):
        # Instantiate environment
        self.env_name = self.env_cfg["env_name"]
        self.timeout_steps = self.env_cfg["timeout_steps"]
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        self.timeout_steps = self.env._max_episode_steps if self.timeout_steps == -1 else self.timeout_steps
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

    def _init_policy(self):
        if "cvpo" in self.policy_name:
            self.policy = RobustCVPO(self.env, self.logger, self.env_cfg, self.adv_cfg,
                                     **self.policy_cfg)
        elif "ppo" in self.policy_name:
            self.policy = RobustPPOLagrangian(self.env, self.logger, self.env_cfg,
                                              self.adv_cfg, **self.policy_cfg)
        elif "focops" in self.policy_name:
            self.policy = RobustFOCOPS(self.env, self.logger, self.env_cfg,
                                       self.adv_cfg, **self.policy_cfg)
        elif "sac_lag" in self.policy_name:
            self.policy = SACLagrangian(self.env, self.logger, self.env_cfg,
                                        self.adv_cfg, **self.policy_cfg)
        else:
            raise NotImplementedError

    def _init_train_mode(self, model_path=None):
        self._init_env()
        # Set up logger and save configuration
        logger_kwargs = setup_logger_kwargs(self.exp_name,
                                            self.seed,
                                            data_dir=self.data_dir,
                                            datestamp=True,
                                            use_tensor_board=True)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(self.config_dict)
        self._init_policy()
        if model_path is not None:
            self.policy.load_model(model_path)
        self._init_adversary()

    def _init_adversary(self):
        self.noise_scale = self.adv_cfg["noise_scale"]
        self.attack_freq = self.adv_cfg["attack_freq"]
        ppo = True if "ppo" in self.policy_name or "focops" in self.policy_name else False
        self.adv_manager = AdvManager(self.obs_dim, self.adv_cfg, ppo=ppo)

    def train(self, model_path=None):
        self._init_train_mode(model_path)
        if self.warmup:
            self.policy.train_one_epoch(warmup=True, verbose=self.verbose)
        total_steps = 0
        score_best = -np.inf
        start_time = time.time()
        epoch_range = range(self.epochs)
        if self.verbose:
            epoch_range = tqdm(epoch_range, desc="Training epochs: ", position=0)
        for epoch in epoch_range:

            if hasattr(self.policy, "_pre_epoch_process"):
                self.policy._pre_epoch_process(epoch)
            epoch_steps = self.policy.train_one_epoch(verbose=self.verbose)

            if hasattr(self.policy, "_post_epoch_process"):
                self.policy._post_epoch_process(epoch)

            total_steps += epoch_steps

            if epoch >= self.save_best_epoch:
                score = self._training_evaluation(self.evaluate_episode_num_after_save_best)
            else:
                score = self._training_evaluation(self.evaluate_episode_num)

            # Save model
            if epoch >= self.save_best_epoch and score >= score_best:
                self.logger.save_state({'env': None}, None)
                score_best = score
            # Save every save_every_epoch or the final results
            if epoch % self.save_every_epoch == 0 or epoch == self.epochs-1:
                self.logger.save_state({'env': None}, itr=epoch)
            # Log info about epoch
            self.logger.store(Epoch=epoch, tab="misc")
            data_dict = self._log_metrics(epoch, total_steps,
                                          time.time() - start_time, self.verbose)

    def _training_evaluation(self, evaluate_episode_num):
        r, c, ep_len, qc_list = self.evaluate_natural(epochs=evaluate_episode_num)
        self.adv_manager.set_amad_thres(qc_list)
        r_list, c_list, ep_len_list = [r], [c], [ep_len]

        for adv_name in self.eval_attackers:
            adv = self.adv_manager.get_adv(adv_name)
            r, c, ep_len = self.evaluate_one_adv(adv,
                                         epochs=evaluate_episode_num,
                                         noise_scale=self.noise_scale)
            r_list.append(r)
            c_list.append(c)
            ep_len_list.append(ep_len)
            if r > self.maximum_reward_eval:
                self.maximum_reward_eval = r

        average_reward = np.mean(r_list)
        average_cost = np.mean(c_list)
        average_ep_len = np.mean(ep_len_list)
        if average_reward > self.maximum_reward_eval_average:
            self.maximum_reward_eval_average = average_reward

        score = -average_cost + average_reward / self.reward_normalizer
        self.logger.store(Score=score,
                          AverageReward=average_reward,
                          AverageCost=average_cost,
                          AverageEpisodeLength=average_ep_len,
                          NoiseScale=self.noise_scale,
                          tab="Eval")
        self.logger.store(MaximumRewardEval=self.maximum_reward_eval, 
                          MaximumRewardEvalAverage=self.maximum_reward_eval_average, 
                          tab="Efficiency")
        return score

    def evaluate_one_adv(self, adversary, epochs=4, noise_scale=0.05):
        ep_reward, ep_len, ep_cost = 0, 0, 0
        for _ in range(epochs):
            obs = self.env.reset()
            for i in range(self.timeout_steps):
                if i % self.attack_freq == 0 and noise_scale > 0:
                    epsilon = adversary.attack_at_eval(self.policy, obs, noise_scale)
                else:
                    epsilon = 0
                obs = obs + epsilon
                action = self.policy.act(obs, deterministic=True, with_logprob=False)[0]
                obs_next, reward, done, info = self.env.step(action)
                if "cost" in info:
                    ep_cost += info["cost"]
                ep_reward += reward
                ep_len += 1
                obs = obs_next
                if done:
                    break
        ep_reward /= epochs
        ep_len /= epochs
        ep_cost /= epochs
        adv_id = adversary.id
        logger_info = {
            adv_id + "Reward": ep_reward,
            adv_id + "Cost": ep_cost,
            # adv_id + "EpisodeLength": ep_len
        }
        self.logger.store(**logger_info, tab="Eval")
        return ep_reward, ep_cost, ep_len

    def evaluate_natural(self, epochs=4):
        ep_reward, ep_len, ep_cost = 0, 0, 0
        # for adaptive MAD usage
        qc_list = []
        for _ in range(epochs):
            obs = self.env.reset()
            for i in range(self.timeout_steps):
                qc, action = self.policy.get_risk_estimation(obs)
                qc_list.append(qc)
                obs_next, reward, done, info = self.env.step(action)
                if "cost" in info:
                    ep_cost += info["cost"]
                ep_reward += reward
                ep_len += 1
                obs = obs_next
                if done:
                    break
        ep_reward /= epochs
        ep_len /= epochs
        ep_cost /= epochs

        self.logger.store(NaturalReward=ep_reward,
                          NaturalCost=ep_cost,
                        #   NaturalEpLen=ep_len,
                          tab="Eval")
        return ep_reward, ep_cost, ep_len, qc_list

    def _log_metrics(self, epoch, total_steps=None, time=None, verbose=True):
        if time is not None:
            self.logger.store(Time=time, tab="misc")
        self.logger.log_tabular('Epoch', epoch)
        if total_steps is not None:
            self.logger.log_tabular('TotalEnvInteracts', total_steps)
        for key in self.logger.logger_keys:
            self.logger.log_tabular(key, average_only=True)

        # data_dict contains all the keys except Epoch and TotalEnvInteracts
        data_dict = self.logger.dump_tabular(
            x_axis="TotalEnvInteracts",
            verbose=verbose,
            env=self.env_name,
        )
        return data_dict
