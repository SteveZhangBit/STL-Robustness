import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.lunar_lander import LQR
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator, RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.lunar_lander import (FPS, SCALE, VIEWPORT_H, VIEWPORT_W,
                                          DevLunarLander, SafetyProp)
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/lunar-lander-lqr', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

winds = [0.0, 10.0]
turbulences = [0.0, 1.0]
env = DevLunarLander(winds, turbulences, (5.0, 0.5))
agent = LQR(FPS, VIEWPORT_H, VIEWPORT_W, SCALE)
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi, 
    {'restarts': 1, 'episode_len': 300, 'evals': 50}
)

solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/lunar-lander-lqr/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma)
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Wind', 'Turbulence', 'data/lunar-lander-lqr', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/lunar-lander-lqr/fig-violations-cma-{i}', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
min_violation = experiment.min_violation_of_all(violations_cma)
if min_violation is not None:
    radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/lunar-lander-lqr', n=1000)
    experiment.plot_unsafe_region(min_violation, radius, 'Wind', 'Turbulence', 'data/lunar-lander-lqr', n=20)
    plt.title('Unsafe region found by CMA')
    plt.savefig('gifs/lunar-lander-lqr/fig-unsafe-region-cma', bbox_inches='tight')

# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by random search...')
records_random = experiment.record_min_violations(out_dir='data/lunar-lander-lqr/random')
records_random, violations_random = experiment.summarize_violations(records_random)
for i in range(len(records_random)):
    samples = [(X, Y) for (X, Y, _) in records_random[i]]
    experiment.plot_samples(samples, 'Wind', 'Turbulence', 'data/lunar-lander-lqr', n=20)
    plt.title('Violations found by Random')
    plt.savefig(f'gifs/lunar-lander-lqr/fig-violations-random-{i}', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
min_violation = experiment.min_violation_of_all(violations_random)
if min_violation is not None:
    radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/lunar-lander-lqr', n=1000)
    experiment.plot_unsafe_region(min_violation, radius, 'Wind', 'Turbulence', 'data/lunar-lander-lqr', n=20)
    plt.title('Unsafe region found by Random')
    plt.savefig('gifs/lunar-lander-lqr/fig-unsafe-region-random', bbox_inches='tight')
