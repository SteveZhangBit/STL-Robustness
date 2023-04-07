'''
The run script for the AFC example with the traditional agent.
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
from robustness.envs.matlab import DevAFC
from robustness.evaluation.evaluator import Evaluator

os.makedirs('gifs/AFC/traditional', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

eng = engine.start_matlab()
eng.addpath('lib/breach', nargout=0)
eng.addpath(eng.genpath('models/AFC'), nargout=0)
eng.InitBreach(nargout=0)

MAF_sensor_tols = [0.95, 1.05]
AF_sensor_tols = [0.99, 1.01]
env = DevAFC(eng, MAF_sensor_tols, AF_sensor_tols, (1.0, 1.0))
agent = Traditional()
phi = BreachSTL('alw (AF[t] < 1.2*14.7 and AF[t] > 0.8*14.7)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 30})
solver = CMASolver(0.1, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

# print(solver.sys_evaluator.eval_sys(env.get_delta_0(), prob))

radius = evaluator.smooth_boundary(0.1, 1000, 0.05, 0.9, 'data/AFC/traditional')
# radius = 0.0

plt.figure()
evaluator.heatmap(
    MAF_sensor_tols, AF_sensor_tols, 25, 25,
    x_name="MAF Sensor Tolerance", y_name="AF Sensor Tolerance", z_name="System Evaluation $\Gamma$",
    out_dir='data/AFC/traditional',
    boundary=radius,
)
plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.savefig('gifs/AFC/traditional/fig-smooth-robustness.png', bbox_inches='tight')

eng.quit()
