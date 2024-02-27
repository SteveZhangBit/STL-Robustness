'''
The run script for the ACC example with the RL agent.
'''

import os
import matplotlib.pyplot as plt

from matlab import engine
from robustness.agents.matlab import RLAgent, Traditional
from robustness.analysis.algorithms.breach import BreachSystemEvaluator
from robustness.analysis.algorithms.cma import CMASolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevACC
from robustness.evaluation.evaluator import Evaluator

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
solver = CMASolver(0.1, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

# print(solver.sys_evaluator.eval_sys(env.get_delta_0(), prob))

# radius = evaluator.smooth_boundary(0.1, 1000, 0.05, 0.9, 'data/ACC/RL')
# radius = 0.0

delta, _, _ = evaluator.any_violation()
if delta is None:
    print('No violation found')
    exit()

radius = evaluator.unsafe_region(delta, 0.1, 0.05, 'data/ACC/RL', n=10_000)

plt.figure()
evaluator.heatmap(
    amax_lead, amin_lead, 25, 25,
    x_name="Max Acceleration", y_name="Min Acceleration", z_name="System Evaluation $\Gamma$",
    out_dir='data/ACC/RL',
    boundary=radius,
    center=delta,
)
# plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.title('Unsafe Region: $||\delta - \delta_v||_2 < %.3f$' % radius)
os.makedirs('gifs/ACC/RL/unsafe-region', exist_ok=True)

delta_str = '-'.join([f'{d:.3f}' for d in delta])
# plt.savefig('gifs/ACC/RL/fig-smooth-robustness.png', bbox_inches='tight')
plt.savefig(f'gifs/ACC/RL/unsafe-region/fig-{delta_str}.png', bbox_inches='tight')

eng.quit()
