import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.lunar_lander import PPO
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm
from robustness.envs.lunar_lander import DevLunarLander, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot


winds = [0.0, 20.0]
turbulences = [0.0, 2.0]
env = DevLunarLander(winds, turbulences, (0.0, 0.0))
agent = PPO('models/lunar-lander/ppo.zip')
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi, 
    {'timeout': 1, 'restarts': 1, 'episode_len': 300, 'evals': 40}
)

# Use CMA
solver = CMASolver(0.2, sys_eval)
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)
data1, _ = experiment.run_diff_max_samples('CMA', np.arange(25, 126, 25), out_dir='data/lunar-lander-ppo/cma')

# FIXME: use boxplot to remove outlier
b = plt.boxplot(np.ndarray.flatten(np.asarray(data1)))
boundary = [l.get_ydata()[1] for l in b['whiskers']][0]

plt.rc('axes', labelsize=12, titlesize=13)
evaluator.heatmap(
    winds, turbulences, 25, 25,
    x_name="Winds", y_name="Turbulences", z_name="System Evaluation $\Gamma$",
    out_dir='data/lunar-lander-ppo',
    boundary=boundary,
    vmax=0.1, vmin=-0.4
)
plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % boundary)
plt.savefig('gifs/lunar-lander-ppo/robustness.png', bbox_inches='tight')
# plt.show()

# Use Random Solver
random_solver = RandomSolver(sys_eval)
evaluator2 = Evaluator(prob, random_solver)
experiment2 = Experiment(evaluator2)
data2, _ = experiment2.run_diff_max_samples('Random', np.arange(25, 126, 25), out_dir='data/lunar-lander-ppo/random')

plt.xlabel('Number of samples')
plt.ylabel('Minimum distance')
boxplot([data1, data2], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
        ['CMA', 'Random'])
plt.savefig('gifs/lunar-lander-ppo/sample-boxplot.png', bbox_inches='tight')
# plt.show()
