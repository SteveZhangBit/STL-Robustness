import os
import matplotlib.pyplot as plt

from matlab import engine
from robustness.agents.matlab import RLAgent
from robustness.analysis.algorithms.breach import BreachSystemEvaluator, BreachSystemEvaluatorWithHeuristic
from robustness.analysis.algorithms.cma import CMASolver
from robustness.analysis.algorithms.random import RandomSolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevACC
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment

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
sys_eval_0 = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 50})
sys_eval_0.eval_sys(env.get_delta_0(), prob)
delta_0_signals = sys_eval_0.obj_best_signal_values
sys_eval = BreachSystemEvaluatorWithHeuristic(
    eng, phi,
    ['a_lead', 'd_rel', 'v_ego', 'v_lead', 'in_lead'],
    delta_0_signals,
    {'restarts': 1, 'evals': 50}
)
solver = CMASolver(0.1, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/ACC/RL/cma_heuristic')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/ACC/RL/cma_heuristic')
# Plot the samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/ACC/RL/fig-violations-cma-heuristic-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
# plt.title('ACC-SAC with CMA')
plt.savefig(f'gifs/ACC/RL/fig-violations-cma-heuristic-all.png', bbox_inches='tight')
