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
from robustness.envs.matlab import DevAFC
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment

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
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 50})

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/AFC/traditional/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/AFC/traditional/cma')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/traditional', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/AFC/traditional/fig-violations-cma-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/traditional', n=20)
# plt.title('AFC-PID with CMA')
plt.savefig(f'gifs/AFC/traditional/fig-violations-cma-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_cma)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/AFC/traditional', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/traditional', n=20)
#     plt.title('Unsafe region found by CMA')
#     plt.savefig('gifs/AFC/traditional/fig-unsafe-region-cma.png', bbox_inches='tight')

# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by random search...')
records_random = experiment.record_min_violations(out_dir='data/AFC/traditional/random')
records_random, violations_random = experiment.summarize_violations(records_random, 'data/AFC/traditional/random')
# Plot all samples
for i in range(len(records_random)):
    samples = [(X, Y) for (X, Y, _) in records_random[i]]
    experiment.plot_samples(samples, 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/traditional', n=20)
    plt.title('Violations found by random search')
    plt.savefig(f'gifs/AFC/traditional/fig-violations-random-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_random], 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/traditional', n=20)
# plt.title('AFC-PID with Random')
plt.savefig(f'gifs/AFC/traditional/fig-violations-random-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_random)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/AFC/traditional', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/traditional', n=20)
#     plt.title('Unsafe region found by random search')
#     plt.savefig('gifs/AFC/traditional/fig-unsafe-region-random.png', bbox_inches='tight')
