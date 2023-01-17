import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import DQN
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm
from robustness.envs.cartpole import DevCartPole, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/cartpole-dqn', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

# Initialize environment and controller
masses = [0.1, 2.0]
forces = [1.0, 20.0]
env = DevCartPole(masses, forces, (1.0, 10.0))
agent = DQN('models/cartpole/best_dqn.zip')
phi = SafetyProp()

# Create problem and solver
prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 200})

samples = np.arange(1, 6) * 20

# Use CMA
solver = CMASolver(0.2, sys_eval)
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)
data1 = experiment.run_diff_max_samples('CMA', samples, out_dir='data/cartpole-dqn/cma')
min_dist = np.min([d['min_dist'] for d in data1])
plt.figure()
evaluator.heatmap(
    masses, forces, 25, 25,
    x_name="Mass", y_name="Force", z_name="System Evaluation $\Gamma$",
    out_dir='data/cartpole-dqn',
    boundary=min_dist,
    vmax=0.2
)
plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % min_dist)
plt.savefig('gifs/cartpole-dqn/fig-robustness.png', bbox_inches='tight')
# plt.show()

# Use Random Solver
random_solver = RandomSolver(sys_eval)
evaluator2 = Evaluator(prob, random_solver)
experiment2 = Experiment(evaluator2)
data2 = experiment2.run_diff_max_samples('Random', samples, out_dir='data/cartpole-dqn/random')

plt.figure()
plt.xlabel('Number of samples')
plt.ylabel('Minimum distance')
boxplot([data1, data2], ['red', 'blue'], samples * (1 + solver.options()['restarts']),
        ['CMA', 'Random'])
plt.savefig('gifs/cartpole-dqn/fig-boxplot.png', bbox_inches='tight')
# plt.show()

# Use expection evaluator
# sys_eval3 = ExpectationSysEvaluator(
#     phi,
#     {'timeout': 1, 'episode_len': 200}
# )
# solver3 = CMASolver(0.2, sys_eval3)
# evaluator3 = Evaluator(prob, solver3)
# experiment3 = Experiment(evaluator3)
# data3 = experiment3.run_diff_max_samples('Expc', samples, out_dir='data/cartpole-dqn/expc')
# min_dist = np.min([d['min_dist'] for d in data3])
# plt.figure()
# evaluator3.heatmap(
#     masses, forces, 25, 25,
#     x_name="Mass", y_name="Force", z_name="System Evaluation $\Gamma$",
#     out_dir='data/cartpole-dqn/expc',
#     boundary=min_dist,
# )
# plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % min_dist)
# plt.savefig('gifs/cartpole-dqn/fig-robustness-expc.png', bbox_inches='tight')

# plt.figure()
# plt.xlabel('Number of samples')
# plt.ylabel('Minimum distance')
# boxplot([data1, data3], ['red', 'blue'], samples * (1 + solver.options()['restarts']),
#         ['CMA', 'CMA-Expc'])
# plt.savefig('gifs/cartpole-dqn/fig-boxplot-expc.png', bbox_inches='tight')
