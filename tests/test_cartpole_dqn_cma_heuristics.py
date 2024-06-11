import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import DQN
from robustness.analysis import Problem
from robustness.analysis.algorithms import CMASolver, CMASystemEvaluator, CMASystemEvaluatorWithHeuristic
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.cartpole import DevCartPole, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

# remove all fils under the data folder
os.system('rm -rf tests/data/cartpole-dqn/cma_heuristic')

os.makedirs('tests/gifs/cartpole-dqn', exist_ok=True)
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
sys_eval_0 = CMASystemEvaluator(0.4, phi, {'restarts': 1, 'evals': 10, 'episode_len': 200})
sys_eval_0.eval_sys(env.get_delta_0(), prob)
delta_0_signals = sys_eval_0.obj_best_signal_values
sys_eval = CMASystemEvaluatorWithHeuristic(
    0.4, phi, delta_0_signals,
    {'restarts': 1, 'evals': 10, 'episode_len': 200}
)

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 0, 'evals': 5})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='tests/data/cartpole-dqn/cma_heuristic', runs=1)
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'tests/data/cartpole-dqn/cma_heuristic')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples,  'Mass', 'Force', 'data/cartpole-dqn', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'tests/gifs/cartpole-dqn/fig-violations-cma-heuristic-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma],  'Mass', 'Force', 'data/cartpole-dqn', n=20)
# plt.title('Cart-Pole-DQN with CMA')
plt.savefig(f'tests/gifs/cartpole-dqn/fig-violations-cma-heuristic-all.png', bbox_inches='tight')
