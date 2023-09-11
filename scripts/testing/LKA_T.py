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
from robustness.envs.matlab import DevLKA
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment


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
sys_eval = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 50})

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/LKA/traditional/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/LKA/traditional/cma')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Turn 1', 'Turn 2', 'data/LKA/traditional', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/LKA/traditional/fig-violations-cma-{i}.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
min_violation = experiment.min_violation_of_all(violations_cma)
if min_violation is not None:
    radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/LKA/traditional', n=1000)
    experiment.plot_unsafe_region(min_violation, radius, 'Turn 1', 'Turn 2', 'data/LKA/traditional', n=20)
    plt.title('Unsafe region found by CMA')
    plt.savefig('gifs/LKA/traditional/fig-unsafe-region-cma.png', bbox_inches='tight')

# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by random search...')
records_random = experiment.record_min_violations(out_dir='data/LKA/traditional/random')
records_random, violations_random = experiment.summarize_violations(records_random, 'data/LKA/traditional/random')
# Plot all samples
for i in range(len(records_random)):
    samples = [(X, Y) for (X, Y, _) in records_random[i]]
    experiment.plot_samples(samples, 'Turn 1', 'Turn 2', 'data/LKA/traditional', n=20)
    plt.title('Violations found by random search')
    plt.savefig(f'gifs/LKA/traditional/fig-violations-random-{i}.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
min_violation = experiment.min_violation_of_all(violations_random)
if min_violation is not None:
    radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/LKA/traditional', n=1000)
    experiment.plot_unsafe_region(min_violation, radius, 'Turn 1', 'Turn 2', 'data/LKA/traditional', n=20)
    plt.title('Unsafe region found by random search')
    plt.savefig('gifs/LKA/traditional/fig-unsafe-region-random.png', bbox_inches='tight')
