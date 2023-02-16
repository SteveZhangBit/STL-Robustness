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
print('Certified minimum deviation:', evaluator.certified_min_violation())
