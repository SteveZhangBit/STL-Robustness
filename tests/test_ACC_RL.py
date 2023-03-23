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
eng.InitBreach(nargout=0)

x0_lead = [50, 90]
v0_lead = [30, 50]
env = DevACC(eng, x0_lead, v0_lead, (70, 40))
agent = RLAgent('ACC_SAC_Agent_9_11.mat')
phi = BreachSTL('alw_[0,50](d_rel[t] - t_gap * v_ego[t] >= D_default)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 2, 'evals': 30})
solver = CMASolver(0.1, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

# print(solver.sys_evaluator.eval_sys(env.get_delta_0(), prob))

radius = evaluator.smooth_boundary(0.1, 100, 0.05, 0.9, 'data/ACC/RL')

plt.figure()
evaluator.heatmap(
    x0_lead, v0_lead, 25, 25,
    x_name="Lead Position", y_name="Lead Velocity", z_name="System Evaluation $\Gamma$",
    out_dir='data/ACC/RL',
    boundary=radius,
)
plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.savefig('gifs/ACC/RL/fig-smooth-robustness.png', bbox_inches='tight')

eng.quit()
