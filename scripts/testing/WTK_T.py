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
from robustness.envs.matlab import DevWTK
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment

os.makedirs('gifs/WTK/traditional', exist_ok=True)
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
agent = Traditional()
phi = BreachSTL('alw_[5,5.9](abs(h_error[t]) < 1) and ' + \
                'alw_[11,11.9](abs(h_error[t]) < 1) and ' + \
                'alw_[17,17.9](abs(h_error[t]) < 1) and ' + \
                'alw_[23,23.9](abs(h_error[t]) < 1)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 50})

# Use CMA
solver = CMASolver(0.1, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/WTK/traditional/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/WTK/traditional/cma')
# Plot the samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/WTK/traditional/fig-violations-cma-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20)
# plt.title('WTK-PID with CMA')
plt.savefig(f'gifs/WTK/traditional/fig-violations-cma-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_cma)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/WTK/traditional', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20, vmin=-2.5)
#     plt.title('Unsafe region found by CMA')
#     plt.savefig('gifs/WTK/traditional/fig-unsafe-region-cma.png', bbox_inches='tight')

# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by random search...')
records_random = experiment.record_min_violations(out_dir='data/WTK/traditional/random')
records_random, violations_random = experiment.summarize_violations(records_random, 'data/WTK/traditional/random')
# Plot the samples
for i in range(len(records_random)):
    samples = [(X, Y) for (X, Y, _) in records_random[i]]
    experiment.plot_samples(samples, 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20)
    plt.title('Violations found by Random')
    plt.savefig(f'gifs/WTK/traditional/fig-violations-random-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_random], 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20)
# plt.title('WTK-PID with Random')
plt.savefig(f'gifs/WTK/traditional/fig-violations-random-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_random)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/WTK/traditional', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20, vmin=-2.5)
#     plt.title('Unsafe region found by Random')
#     plt.savefig('gifs/WTK/traditional/fig-unsafe-region-random.png', bbox_inches='tight')
