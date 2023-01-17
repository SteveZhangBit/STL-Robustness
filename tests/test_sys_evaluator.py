import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import PID
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASystemEvaluator,
                                            ExpectationSysEvaluator)
from robustness.analysis.utils import L2Norm
from robustness.envs.cartpole import DevCartPole, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

# Initialize environment and controller
masses = [0.1, 2.0]
forces = [1.0, 20.0]
env = DevCartPole(masses, forces, (1.0, 10.0))
agent = PID()
phi = SafetyProp()

# Create problem and solver
prob = Problem(env, agent, phi, L2Norm(env))

from datetime import datetime

# CMA evalutor
start = datetime.now()
sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 200, 'evals': 50, 'restarts': 2})
sys_eval.eval_sys(env.delta_0, prob)
print("Time:", datetime.now() - start)

# Expectation evaluator
start = datetime.now()
sys_eval = ExpectationSysEvaluator(phi, {'timeout': 1, 'episode_len': 200, 'evals': 50, 'restarts': 2})
sys_eval.eval_sys(env.delta_0, prob)
print("Time:", datetime.now() - start)