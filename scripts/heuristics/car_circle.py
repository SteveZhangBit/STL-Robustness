'''
This script generates figures, the data is generated in another script.
'''
import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.car_circle import DevCarCircle, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/car-circle-ppo', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

load_dir = 'models/car_circle_ppo_vanilla/model_save/model.pt'
speed = [5.0, 35.0]
steering = [0.2, 0.8]
env = DevCarCircle(load_dir, speed, steering)
agent = PPOVanilla(load_dir)
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'restarts': 1, 'episode_len': 300, 'evals': 50}
)

# Use CMA
solver = CMASolver(0.1, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/car-circle-ppo/cma_heuristic')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/car-circle-ppo/cma_heuristic')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Speed', 'Steering', 'data/car-circle-ppo', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/car-circle-ppo/fig-violations-cma-heuristic-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'Speed', 'Steering', 'data/car-circle-ppo', n=20)
# plt.title('Car-Circle-PPO with CMA')
plt.savefig(f'gifs/car-circle-ppo/fig-violations-cma-heuristic-all.png', bbox_inches='tight')
