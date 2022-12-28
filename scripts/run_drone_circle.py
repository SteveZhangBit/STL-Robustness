import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver, ExpectationSysEvaluator)
from robustness.analysis.utils import L2Norm, scale
from robustness.envs.drone_circle import DevDroneCircle, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.logger import Logger
from robustness.evaluation.utils import boxplot

load_dir = 'models/drone_circle_ppo_vanilla/model_save/model.pt'
air_density = [1.164, 1.422] # air density between 30°C and -25 °C at 1 atmosphere pressure
mass = [1.34, 1.5]
env = DevDroneCircle(load_dir, air_density, mass)
agent = PPOVanilla(load_dir)
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'timeout': 1, 'restarts': 1, 'episode_len': 300, 'evals': 40}
)
solver = CMASolver(0.1, sys_eval)
evaluator = Evaluator(prob, solver)

os.makedirs('gifs/drone-circle-ppo', exist_ok=True)
experiment = Experiment(evaluator)
# data1, _ = experiment.run_diff_max_samples('CMA', np.arange(25, 126, 25), out_dir='data/drone-circle-ppo/cma')
plt.rc('axes', labelsize=12, titlesize=13)
plt.figure()
evaluator.heatmap(
    air_density, mass, 25, 25,
    x_name="Air Density", y_name="Mass", z_name="System Evaluation $\Gamma$",
    out_dir='data/drone-circle-ppo',
    # boundary=np.min(data1),
)
# plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % np.min(data1))
plt.savefig('gifs/drone-circle-ppo/robustness.png', bbox_inches='tight')

# Use Random Solver
# random_solver = RandomSolver(sys_eval)
# evaluator2 = Evaluator(prob, random_solver)
# experiment2 = Experiment(evaluator2)
# data2, _ = experiment2.run_diff_max_samples('Random', np.arange(25, 126, 25), out_dir='data/drone-circle-ppo/random')

# plt.figure()
# plt.xlabel('Number of samples')
# plt.ylabel('Minimum distance')
# boxplot([data1, data2], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
#         ['CMA', 'Random'])
# plt.savefig('gifs/drone-circle-ppo/sample-boxplot.png', bbox_inches='tight')

# sys_eval3 = ExpectationSysEvaluator(
#     phi,
#     {'timeout': 1, 'restarts': 1, 'episode_len': 300, 'evals': 40}
# )
# solver3 = CMASolver(0.1, sys_eval3)
# evaluator3 = Evaluator(prob, solver3)
# experiment3 = Experiment(evaluator3)
# data3, _ = experiment3.run_diff_max_samples('Expc', np.arange(25, 126, 25), out_dir='data/drone-circle-ppo/expc')
# plt.figure()
# evaluator3.heatmap(
#     air_density, mass, 25, 25,
#     x_name="Air Density", y_name="Mass", z_name="System Evaluation $\Gamma$",
#     out_dir='data/drone-circle-ppo/expc',
#     boundary=np.min(data3),
# )
# plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % np.min(data3))
# plt.savefig('gifs/drone-circle-ppo/robustness-expc.png', bbox_inches='tight')

# plt.figure()
# plt.xlabel('Number of samples')
# plt.ylabel('Minimum distance')
# boxplot([data1, data3], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
#         ['CMA', 'CMA-Expc'])
# plt.savefig('gifs/drone-circle-ppo/sample-boxplot-expc.png', bbox_inches='tight')
