import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            CMASystemEvaluatorWithHeuristic)
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
phi = SafetyProp(env.instantiate(env.get_delta_0())[0])

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval_0 = CMASystemEvaluator(0.4, phi, {'restarts': 1, 'evals': 50, 'episode_len': 200})
sys_eval_0.eval_sys(env.get_delta_0(), prob)
delta_0_signals = sys_eval_0.obj_best_signal_values
sys_eval = CMASystemEvaluatorWithHeuristic(
    0.4, phi, delta_0_signals,
    {'restarts': 1, 'episode_len': 200, 'evals': 50}
)

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/car-run-ppo/cma_heuristic')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/car-run-ppo/cma_heuristic')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'Speed', 'Steering', 'data/car-run-ppo', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/car-run-ppo/fig-violations-cma-heuristic-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'Speed', 'Steering', 'data/car-run-ppo', n=20)
# plt.title('Car-Run-PPO with CMA')
plt.savefig(f'gifs/car-run-ppo/fig-violations-cma-heuristic-all.png', bbox_inches='tight')
