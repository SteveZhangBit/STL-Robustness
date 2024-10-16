import os
import matplotlib.pyplot as plt

from matlab import engine
from robustness.agents.matlab import RLAgent
from robustness.analysis.algorithms.breach import BreachSystemEvaluator
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
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 50})
solver = CMASolver(0.1, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/ACC/RL/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/ACC/RL/cma')
# Plot the samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/ACC/RL/fig-violations-cma-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
# plt.title('ACC-SAC with CMA')
plt.savefig(f'gifs/ACC/RL/fig-violations-cma-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_cma)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/ACC/RL', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
#     plt.title('Unsafe region found by CMA')
#     plt.savefig('gifs/ACC/RL/fig-unsafe-region-cma.png', bbox_inches='tight')

# Use random search
# solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
# evaluator = Evaluator(prob, solver)
# experiment = Experiment(evaluator)

# print('Find violations by random search...')
# records_random = experiment.record_min_violations(out_dir='data/ACC/RL/random')
# records_random, violations_random = experiment.summarize_violations(records_random, 'data/ACC/RL/random')
# # Plot the samples
# for i in range(len(records_random)):
#     samples = [(X, Y) for (X, Y, _) in records_random[i]]
#     experiment.plot_samples(samples, 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
#     plt.title('Violations found by Random')
#     plt.savefig(f'gifs/ACC/RL/fig-violations-random-{i}.png', bbox_inches='tight')

# experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_random], 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
# # plt.title('ACC-SAC with Random')
# plt.savefig(f'gifs/ACC/RL/fig-violations-random-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_random)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/ACC/RL', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Max acceleration', 'Min acceleration', 'data/ACC/RL', n=20)
#     plt.title('Unsafe region found by Random')
#     plt.savefig('gifs/ACC/RL/fig-unsafe-region-random.png', bbox_inches='tight')
