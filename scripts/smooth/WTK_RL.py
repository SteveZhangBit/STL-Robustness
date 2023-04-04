'''
The run script for the WTK example with the RL agent.
'''

import os

import matplotlib.pyplot as plt
from matlab import engine

from robustness.agents.matlab import RLAgent
from robustness.analysis.algorithms.breach import BreachSystemEvaluator
from robustness.analysis.algorithms.cma import CMASolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevWTK
from robustness.evaluation.evaluator import Evaluator

os.makedirs('gifs/WTK/RL', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

eng = engine.start_matlab()
eng.addpath('lib/breach', nargout=0)
eng.addpath(eng.genpath('models/WTK'), nargout=0)
eng.InitBreach(nargout=0)

in_rates = [0.01, 0.5]
out_rates = [0.01, 0.2]
env = DevWTK(eng, in_rates, out_rates, (0.25, 0.1))
agent = RLAgent('WTK_TD3_Agent_9_20.mat')
phi = BreachSTL('alw_[5,5.9](abs(h_error[t]) < 1) and ' + \
                'alw_[11,11.9](abs(h_error[t]) < 1) and ' + \
                'alw_[17,17.9](abs(h_error[t]) < 1) and ' + \
                'alw_[23,23.9](abs(h_error[t]) < 1)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 30})
solver = CMASolver(0.1, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

radius = evaluator.smooth_boundary(0.1, 1000, 0.05, 0.9, 'data/WTK/RL')
# radius = 0.0

plt.figure()
evaluator.heatmap(
    in_rates, out_rates, 25, 25,
    x_name="In Rate", y_name="Out Rate", z_name="System Evaluation $\Gamma$",
    out_dir='data/WTK/RL',
    boundary=radius,
)
plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.savefig('gifs/WTK/RL/fig-smooth-robustness.png', bbox_inches='tight')

eng.quit()
