import os
import matplotlib.pyplot as plt

from matlab import engine
from robustness.agents.matlab import Traditional
from robustness.analysis.algorithms.breach import BreachSystemEvaluator
from robustness.analysis.algorithms.cma import CMASolver
from robustness.analysis.algorithms.random import RandomSolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevACC
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment

os.makedirs('gifs/ACC/traditional', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

eng = engine.start_matlab()
eng.addpath('lib/breach', nargout=0)
eng.addpath('models/ACC', nargout=0)
eng.InitBreach(nargout=0)

amax_lead = [0.01, 1.0]
amin_lead = [-1.0, -0.01]
env = DevACC(eng, amax_lead, amin_lead, (0.5, -0.5))
agent = Traditional()
phi = BreachSTL('alw (d_rel[t] - t_gap * v_ego[t] >= D_default - 0.5)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 50})
solver = CMASolver(0.1, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

experiment.record_min_violations(out_dir='data/ACC/traditional/cma')


# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

experiment.record_min_violations(out_dir='data/ACC/traditional/random')
