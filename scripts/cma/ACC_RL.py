import os

import matplotlib.pyplot as plt
import numpy as np
from matlab import engine

from robustness.agents.matlab import RLAgent
from robustness.analysis.algorithms.breach import BreachSystemEvaluator
from robustness.analysis.algorithms.cma import CMASolver
from robustness.analysis.algorithms.random import RandomSolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.matlab import DevACC
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/ACC/RL', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

eng = engine.start_matlab()
eng.addpath('lib/breach', nargout=0)
eng.addpath('models/ACC', nargout=0)
eng.addpath('models/ACC/RL', nargout=0)
eng.InitBreach(nargout=0)

amax_lead = [0.01, 1.0]
amin_lead = [-1.0, -0.01]
env = DevACC(eng, amax_lead, amin_lead, (0.5, -0.5))
agent = RLAgent('ACC_SAC_Agent_9_11.mat')
phi = BreachSTL('alw (d_rel[t] - t_gap * v_ego[t] >= D_default - 0.5)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 30})

samples = np.arange(1, 6) * 20

# Use CMA
solver = CMASolver(0.1, sys_eval)
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)
data1 = experiment.run_diff_max_samples('CMA', samples, out_dir='data/ACC/RL/cma')
idx = np.argmin(data1['min_dist'])
plt.figure()
evaluator.heatmap(
    amax_lead, amin_lead, 25, 25,
    x_name="Max Acceleration", y_name="Min Acceleration", z_name="System Evaluation $\Gamma$",
    out_dir='data/ACC/RL',
    boundary=data1['min_dist'].iat[idx],
)
min_delta = normalize(data1['min_delta'].iat[idx], env.get_dev_bounds())
plt.scatter(min_delta[0]*25, min_delta[1]*25, color='yellow')
plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % data1['min_dist'].iat[idx])
plt.savefig('gifs/ACC/RL/fig-robustness.png', bbox_inches='tight')

# Use Random Solver
random_solver = RandomSolver(sys_eval)
evaluator2 = Evaluator(prob, random_solver)
experiment2 = Experiment(evaluator2)
data2 = experiment2.run_diff_max_samples('Random', samples, out_dir='data/ACC/RL/random')

# Plot
plt.figure()
plt.xlabel('Number of Samples')
plt.ylabel('Minimum Distance')
boxplot(
    [
        [data1['min_dist'].loc[x] for x in samples],
        [data2['min_dist'].loc[x] for x in samples],
    ],
    ['red', 'blue'],
    samples * (1 + solver.options()['restarts']),
    ['CMA', 'Random'],
)
plt.savefig('gifs/ACC/RL/fig-boxplot.png', bbox_inches='tight')
