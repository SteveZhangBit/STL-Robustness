import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import PID
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator, RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.cartpole import DevCartPole2, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot


# Initialize environment and controller
cart_masses = [0.1, 2.0]
pole_masses = [0.05, 0.15]
pole_lengths = [0.25, 0.75]
forces = [1.0, 20.0]
env = DevCartPole2(cart_masses, pole_masses, pole_lengths, forces)
agent = PID()
phi = SafetyProp()

# Create problem and solver
prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'restarts': 1, 'evals': 50, 'episode_len': 200}
)

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 200})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/cartpole4-pid/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/cartpole4-pid/cma')

# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 200})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by random search...')
records_random = experiment.record_min_violations(out_dir='data/cartpole4-pid/random')
records_random, violations_random = experiment.summarize_violations(records_random, 'data/cartpole4-pid/random')
