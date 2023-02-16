import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import PID
from robustness.analysis import Problem
from robustness.analysis.algorithms import CMASolver, CMASystemEvaluator
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.cartpole import DevCartPole, SafetyProp
from robustness.evaluation import Evaluator

# Initialize environment and controller
masses = [0.1, 2.0]
forces = [1.0, 20.0]
env = DevCartPole(masses, forces, (1.0, 10.0))
agent = PID()
phi = SafetyProp()

# Create problem and solver
prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 200})
solver = CMASolver(0.2, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)
print('Certified minimum deviation:', evaluator.certified_min_violation())
