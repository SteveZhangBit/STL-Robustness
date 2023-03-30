'''
The run script for the LKA example with the traditional agent.
'''

import os
import matplotlib.pyplot as plt

from matlab import engine

from robustness.agents.matlab import Traditional
from robustness.analysis.algorithms.breach import BreachSystemEvaluator
from robustness.analysis.algorithms.cma import CMASolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevLKA
from robustness.evaluation.evaluator import Evaluator


os.makedirs('gifs/LKA/traditional', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

eng = engine.start_matlab()
eng.addpath('lib/breach', nargout=0)
eng.addpath(eng.genpath('models/LKA'), nargout=0)
eng.InitBreach(nargout=0)

turn_pos1 = [10, 35]
turn_pos2 = [40, 70]
env = DevLKA(eng, turn_pos1, turn_pos2, (27.19, 56.46))
agent = Traditional()
phi = BreachSTL('alw (abs(lateral_deviation[t]) < 0.85)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 25})
solver = CMASolver(0.1, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

# print(solver.sys_evaluator.eval_sys([27.19, 56.46], prob))

radius = evaluator.smooth_boundary(0.1, 1000, 0.05, 0.9, 'data/LKA/traditional')
# radius = 0

plt.figure()
evaluator.heatmap(
    turn_pos1, turn_pos2, 25, 25,
    x_name="Turn Position 1", y_name="Turn Position 2", z_name="System Evaluation $\Gamma$",
    out_dir='data/LKA/traditional',
    boundary=radius,
)
plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.savefig('gifs/LKA/traditional/fig-smooth-robustness.png', bbox_inches='tight')

eng.quit()
