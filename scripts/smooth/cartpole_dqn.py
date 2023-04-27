import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import DQN
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.cartpole import DevCartPole, SafetyProp, SafetyProp2
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
sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 200})
solver = CMASolver(0.2, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

regions = evaluator.multiple_safe_regions(0.1, 0.05, 'data/cartpole-dqn', n=1000, epsilon=0.01)
if len(regions) == 0:
    print('No unsafe regions found')
    exit()

# print('Certified minimum deviation:', evaluator.certified_min_violation())
# radius = evaluator.smooth_boundary(0.1, 1000, 0.05, 0.9, 'data/cartpole-dqn')

plt.figure()
evaluator.heatmap(
    masses, forces, 25, 25,
    x_name="Mass", y_name="Force", z_name="System Evaluation $\Gamma$",
    out_dir='data/cartpole-dqn',
    boundary=[r[1] for r in regions],
    center=[r[0] for r in regions],
    # vmax=0.2
)
# plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.title('Unsafe Regions')
plt.savefig(f'gifs/cartpole-dqn/fig-unsafe-regions.png', bbox_inches='tight')
