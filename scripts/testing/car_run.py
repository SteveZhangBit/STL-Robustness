import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, scale, normalize
from robustness.envs.car_run import DevCarRun, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/car-run-ppo', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

load_dir = 'models/car_run_ppo_vanilla/model_save/model.pt'
speed = [5.0, 35.0]
steering = [0.2, 0.8]
env = DevCarRun(load_dir, speed, steering)
agent = PPOVanilla(load_dir)
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'restarts': 1, 'episode_len': 200, 'evals': 50}
)

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/car-run-ppo/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/car-run-ppo/cma')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Speed', 'Steering', 'data/car-run-ppo', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/car-run-ppo/fig-violations-cma-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'Speed', 'Steering', 'data/car-run-ppo', n=20)
# plt.title('Car-Run-PPO with CMA')
plt.savefig(f'gifs/car-run-ppo/fig-violations-cma-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_cma)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/car-run-ppo', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Speed', 'Steering', 'data/car-run-ppo', n=20)
#     plt.title('Unsafe region found by CMA')
#     plt.savefig('gifs/car-run-ppo/fig-unsafe-region-cma.png', bbox_inches='tight')

# Generate a gif
import pickle

with open('data/car-run-ppo/cma/records-min-violations-0.pickle', 'rb') as f:
    data = pickle.load(f)
delta = data[0][np.argmin(data[1])]
evaluator.visualize_violation(delta, gif="gifs/car-run-ppo/counterexample.gif", render=True)

# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by random search...')
records_random = experiment.record_min_violations(out_dir='data/car-run-ppo/random')
records_random, violations_random = experiment.summarize_violations(records_random, 'data/car-run-ppo/random')
# Plot all samples
for i in range(len(records_random)):
    samples = [(X, Y) for (X, Y, _) in records_random[i]]
    experiment.plot_samples(samples, 'Speed', 'Steering', 'data/car-run-ppo', n=20)
    plt.title('Violations found by random search')
    plt.savefig(f'gifs/car-run-ppo/fig-violations-random-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_random], 'Speed', 'Steering', 'data/car-run-ppo', n=20)
# plt.title('Car-Run-PPO with Random')
plt.savefig(f'gifs/car-run-ppo/fig-violations-random-all.png', bbox_inches='tight')

# Find the minimum violation and certify an unsafe region
# min_violation = experiment.min_violation_of_all(violations_random)
# if min_violation is not None:
#     radius = evaluator.unsafe_region(min_violation, 0.1, 0.05, 'data/car-run-ppo', n=1000)
#     experiment.plot_unsafe_region(min_violation, radius, 'Speed', 'Steering', 'data/car-run-ppo', n=20)
#     plt.title('Unsafe region found by random search')
#     plt.savefig('gifs/car-run-ppo/fig-unsafe-region-random.png', bbox_inches='tight')
