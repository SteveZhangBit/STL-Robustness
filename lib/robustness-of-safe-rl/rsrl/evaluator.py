import time
from copy import deepcopy
import gym

import torch
from tqdm import tqdm
import numpy as np

from rsrl.policy import RobustCVPO, RobustPPOLagrangian, RobustFOCOPS, SACLagrangian
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


class Evaluator:
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
                 warmup=False,
                 evaluate_episode_num=1,
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
        self.warmup = warmup
        self.evaluate_episode_num = evaluate_episode_num
        self.reward_normalizer = reward_normalizer
        self.eval_attackers = eval_attackers

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

    def _init_eval_mode(self, model_path):
        self._init_env()
        # Set up logger but don't save anything
        logger_kwargs = setup_logger_kwargs(self.exp_name,
                                            self.seed,
                                            data_dir=self.data_dir,
                                            datestamp=True,
                                            use_tensor_board=True)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(self.config_dict)
        self._init_policy()
        print("*"*50)
        print("model path: ", model_path)
        self.policy.load_model(model_path)
        self._init_adversary()

    def _init_adversary(self):
        self.noise_scale = self.adv_cfg["noise_scale"]
        self.attack_freq = self.adv_cfg["attack_freq"]
        ppo = True if "ppo" in self.policy_name or "focops" in self.policy_name else False
        self.adv_manager = AdvManager(self.obs_dim, self.adv_cfg, ppo=ppo)

    def eval(self, model_path):
        self._init_eval_mode(model_path=model_path)

        start_time = time.time()
        epoch_range = range(self.epochs)
        if self.verbose:
            epoch_range = tqdm(epoch_range, desc="Evaluation epochs: ", position=0)
        for epoch in epoch_range:
            self.evaluate_all()
            # Log info about epoch
            data_dict = self._log_metrics(epoch, epoch,
                                          time.time() - start_time, self.verbose)

    def evaluate_all(self):
        r, c, qc_list = self.evaluate_natural(epochs=self.evaluate_episode_num)
        self.adv_manager.set_amad_thres(qc_list)
        r_list, c_list = [r], [c]

        for adv_name in self.eval_attackers:
            adv = self.adv_manager.get_adv(adv_name)
            r, c = self.evaluate_one_adv(adv,
                                         epochs=self.evaluate_episode_num,
                                         noise_scale=self.noise_scale)
            r_list.append(r)
            c_list.append(c)

        average_reward = np.mean(r_list)
        average_cost = np.mean(c_list)
        score = -average_cost + average_reward / self.reward_normalizer
        self.logger.store(Score=score,
                          AverageReward=average_reward,
                          AverageCost=average_cost,
                          NoiseScale=self.noise_scale,
                          tab="Eval")
        return score

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
                # if done:
                #     break
        ep_reward /= epochs
        ep_len /= epochs
        ep_cost /= epochs

        self.logger.store(NaturalReward=ep_reward, NaturalCost=ep_cost, tab="Eval")
        return ep_reward, ep_cost, qc_list

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
                # if done:
                #     break
        ep_reward /= epochs
        ep_len /= epochs
        ep_cost /= epochs
        adv_id = adversary.id
        logger_info = {
            adv_id + "Reward": ep_reward,
            adv_id + "Cost": ep_cost,
        }

        self.logger.store(**logger_info, tab="Eval")
        return ep_reward, ep_cost

    def _log_metrics(self, epoch, total_steps=None, time=None, verbose=True):
        if time is not None:
            self.logger.store(Time=time, tab="worker")
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
