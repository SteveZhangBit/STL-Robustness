import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.lunar_lander import LQR
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver, ExpectationSysEvaluator)
from robustness.analysis.utils import L2Norm
from robustness.envs.lunar_lander import (FPS, SCALE, VIEWPORT_H, VIEWPORT_W,
                                          DevLunarLander, SafetyProp)
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot


winds = [0.0, 20.0]
turbulences = [0.0, 2.0]
env = DevLunarLander(winds, turbulences, (0.0, 0.0))
agent = LQR(FPS, VIEWPORT_H, VIEWPORT_W, SCALE)
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
data1, _ = experiment.run_diff_max_samples('CMA', np.arange(25, 126, 25), out_dir='data/lunar-lander-lqr/cma')

# FIXME: use boxplot to remove outlier
b = plt.boxplot(np.ndarray.flatten(np.asarray(data1)))
boundary = [l.get_ydata()[1] for l in b['whiskers']][0]

plt.rc('axes', labelsize=12, titlesize=13)
plt.figure()
evaluator.heatmap(
    winds, turbulences, 25, 25,
    x_name="Winds", y_name="Turbulences", z_name="System Evaluation $\Gamma$",
    out_dir='data/lunar-lander-lqr',
    boundary=boundary,
    vmax=0.1, vmin=-0.4
)
plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % boundary)
plt.savefig('gifs/lunar-lander-lqr/robustness.png', bbox_inches='tight')
# plt.show()

# Use Random Solver
random_solver = RandomSolver(sys_eval)
evaluator2 = Evaluator(prob, random_solver)
experiment2 = Experiment(evaluator2)
data2, _ = experiment2.run_diff_max_samples('Random', np.arange(25, 126, 25), out_dir='data/lunar-lander-lqr/random')

plt.figure()
plt.xlabel('Number of samples')
plt.ylabel('Minimum distance')
boxplot([data1, data2], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
        ['CMA', 'Random'])
plt.savefig('gifs/lunar-lander-lqr/sample-boxplot.png', bbox_inches='tight')
# plt.show()


sys_eval3 = ExpectationSysEvaluator(
    phi,
    {'timeout': 1, 'restarts': 1, 'episode_len': 300, 'evals': 40}
)

# from datetime import datetime
# start = datetime.now()
# print(sys_eval3.eval_sys(env.delta_0, prob))
# print(datetime.now() - start)

solver3 = CMASolver(0.2, sys_eval3)
evaluator3 = Evaluator(prob, solver3)
experiment3 = Experiment(evaluator3)
# data3, _ = experiment3.run_diff_max_samples('Expc', np.arange(25, 126, 25), out_dir='data/lunar-lander-lqr/expc')
plt.figure()
evaluator3.heatmap(
    winds, turbulences, 25, 25,
    x_name="Winds", y_name="Turbulences", z_name="System Evaluation $\Gamma$",
    out_dir='data/lunar-lander-lqr/expc',
    # boundary=np.min(data3),
)
# plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % np.min(data3))
plt.savefig('gifs/lunar-lander-lqr/robustness-expc.png', bbox_inches='tight')

# plt.figure()
# plt.xlabel('Number of samples')
# plt.ylabel('Minimum distance')
# boxplot([data1, data3], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
#         ['CMA', 'CMA-Expc'])
# plt.savefig('gifs/lunar-lander-lqr/sample-boxplot-expc.png', bbox_inches='tight')
