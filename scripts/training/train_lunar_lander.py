# import os

# import matplotlib.pyplot as plt
# import numpy as np

# from robustness.agents.lunar_lander import PPO
# from robustness.analysis import Problem
# from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
#                                             RandomSolver, ExpectationSysEvaluator)
# from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.car_run import DevCarRun, SafetyProp, SafetyProp2
# from robustness.evaluation import Evaluator, Experiment
# from robustness.evaluation.utils import boxplot



winds = [0.0, 20.0]
turbulences = [0.0, 2.0]
env = DevLunarLander(winds, turbulences, (0.0, 0.0))
en, x0bounds = env.instantiate(env.delta_0)
# agent = PPO('models/lunar-lander/ppo.zip')
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Create and wrap the environment
# env = make_vec_env('LunarLander-v2', n_envs=1)

# Instantiate the agent
model = PPO("MlpPolicy", en, verbose=1)

# # Train the agent
# model.learn(total_timesteps=100000)

# # Save the trained model
# model.save("ppo_lunarlander")

# Load the trained model (optional)
model = PPO.load("ppo_lunarlander")

# Test the trained model
obs = en.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = en.step(action)
    en.render()
# phi = SafetyProp()

# prob = Problem(env, agent, phi, L2Norm(env))
# sys_eval = CMASystemEvaluator(
#     0.4, phi, 
#     {'timeout': 1, 'restarts': 0, 'episode_len': 300, 'evals': 40}
# )

# # from datetime import datetime
# # start = datetime.now()
# # print(sys_eval.eval_sys(env.delta_0, prob))
# # print(datetime.now() - start)

# samples = np.arange(1, 6) * 20

# # Use CMA
# solver = CMASolver(0.2, sys_eval)
# evaluator = Evaluator(prob, solver)
# experiment = Experiment(evaluator)
# data1 = experiment.run_diff_max_samples('CMA', samples, out_dir='old_data/lunar-lander-ppo/cma')
# idx = np.argmin(data1['min_dist'])
# plt.figure()
# evaluator.heatmap(
#     winds, turbulences, 25, 25,
#     x_name="Wind", y_name="Turbulence", z_name="System Evaluation $\Gamma$",
#     out_dir='old_data/lunar-lander-ppo',
#     boundary=data1['min_dist'].iat[idx],
#     vmax=0.1, vmin=-0.4
# )
# min_delta = normalize(data1['min_delta'].iat[idx], env.get_dev_bounds())
# plt.scatter(min_delta[0]*25, min_delta[1]*25, color='yellow')
# plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % data1['min_dist'].iat[idx])
# plt.savefig('old_gifs/lunar-lander-ppo/fig-robustness.png', bbox_inches='tight')
# # plt.show()

# # Use Random Solver
# random_solver = RandomSolver(sys_eval)
# evaluator2 = Evaluator(prob, random_solver)
# experiment2 = Experiment(evaluator2)
# data2 = experiment2.run_diff_max_samples('Random', samples, out_dir='old_data/lunar-lander-ppo/random')

# # Use STL2 Evaluator
# phi = SafetyProp2()
# prob = Problem(env, agent, phi, L2Norm(env))
# sys_eval = CMASystemEvaluator(
#     0.4, phi, 
#     {'timeout': 1, 'restarts': 0, 'episode_len': 300, 'evals': 40}
# )
# solver = CMASolver(0.2, sys_eval)
# evaluator = Evaluator(prob, solver)
# experiment = Experiment(evaluator)
# data3 = experiment.run_diff_max_samples('STL2', samples, out_dir='old_data/lunar-lander-ppo/stl2')
# idx = np.argmin(data3['min_dist'])
# plt.figure()
# evaluator.heatmap(
#     winds, turbulences, 25, 25,
#     x_name="Wind", y_name="Turbulence", z_name="System Evaluation $\Gamma$",
#     out_dir='old_data/lunar-lander-ppo/stl2',
#     boundary=data3['min_dist'].iat[idx],
#     # vmax=0.1, vmin=-0.4
# )
# min_delta = normalize(data3['min_delta'].iat[idx], env.get_dev_bounds())
# plt.scatter(min_delta[0]*25, min_delta[1]*25, color='yellow')
# plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % data3['min_dist'].iat[idx])
# plt.savefig('old_gifs/lunar-lander-ppo/fig-robustness-stl2.png', bbox_inches='tight')

# plt.figure()
# plt.xlabel('Number of samples')
# plt.ylabel('Minimum distance')
# boxplot(
#     [
#         [data1['min_dist'].loc[x] for x in samples],
#         [data2['min_dist'].loc[x] for x in samples],
#         [data3['min_dist'].loc[x] for x in samples],
#     ],
#     ['red', 'blue', 'green'],
#     samples * (1 + solver.options()['restarts']),
#     ['CMA', 'Random', 'CMA2']
# )
# plt.savefig('old_gifs/lunar-lander-ppo/fig-boxplot.png', bbox_inches='tight')

