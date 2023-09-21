import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import DQN
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator, RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.cartpole import DevCartPole, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/cartpole-dqn', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

# Initialize environment and controller
masses = [0.1, 2.0]
forces = [1.0, 20.0]
env = DevCartPole(masses, forces, (1.0, 10.0))
agent = DQN('models/cartpole/best_dqn.zip')
phi = SafetyProp()

# Create problem and solver
prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'restarts': 1, 'evals': 50, 'episode_len': 200}
)

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/cartpole-dqn/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/cartpole-dqn/cma')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples,  'Mass', 'Force', 'data/cartpole-dqn', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/cartpole-dqn/fig-violations-cma-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma],  'Mass', 'Force', 'data/cartpole-dqn', n=20)
plt.title('Violations found by CMA')
plt.savefig(f'gifs/cartpole-dqn/fig-violations-cma-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_cma)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/cartpole-dqn', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Mass', 'Force', 'data/cartpole-dqn', n=20)
#     plt.title('Unsafe region found by CMA')
#     plt.savefig('gifs/cartpole-dqn/fig-unsafe-region-cma.png', bbox_inches='tight')


# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by random search...')
records_random = experiment.record_min_violations(out_dir='data/cartpole-dqn/random')
records_random, violations_random = experiment.summarize_violations(records_random, 'data/cartpole-dqn/random')
for i in range(len(records_random)):
    samples = [(X, Y) for (X, Y, _) in records_random[i]]
    experiment.plot_samples(samples,  'Mass', 'Force', 'data/cartpole-dqn', n=20)
    plt.title('Violations found by Random')
    plt.savefig(f'gifs/cartpole-dqn/fig-violations-random-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_random],  'Mass', 'Force', 'data/cartpole-dqn', n=20)
plt.title('Violations found by Random')
plt.savefig(f'gifs/cartpole-dqn/fig-violations-random-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_random)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/cartpole-dqn', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Mass', 'Force', 'data/cartpole-dqn', n=20)
#     plt.title('Unsafe region found by Random')
#     plt.savefig('gifs/cartpole-dqn/fig-unsafe-region-random.png', bbox_inches='tight')