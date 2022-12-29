import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.cartpole import PID
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver, GACO, NSGA2Solver)
from robustness.analysis.utils import L2Norm
from robustness.envs.cartpole import DevCartPole, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot


def before_each(pickle_safe=False):
    # Initialize environment and controller
    masses = [0.1, 2.0]
    forces = [1.0, 20.0]
    env = DevCartPole(masses, forces, (1.0, 10.0))
    agent = PID()
    phi = SafetyProp(pickle_safe)

    # Create problem and solver
    prob = Problem(env, agent, phi, L2Norm(env))
    sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 100, 'evals': 5, 'restarts': 1})

    return prob, sys_eval

# Use CMA
prob, sys_eval = before_each()
solver = CMASolver(0.2, sys_eval, {'evals': 10, 'restarts': 1})
evaluator = Evaluator(prob, solver)
evaluator.any_violation()
evaluator.any_violation(0.4)
evaluator.min_violation()
evaluator.min_violation(0.4)

# Use Random Solver
prob, sys_eval = before_each()
random_solver = RandomSolver(sys_eval, {'evals': 10, 'restarts': 1})
evaluator = Evaluator(prob, random_solver)
evaluator.any_violation()
evaluator.any_violation(0.4)
evaluator.min_violation()
evaluator.min_violation(0.4)

# PyGMO
prob, sys_eval = before_each(True)
gaco_solver = GACO(1, 8, sys_eval)
evaluator = Evaluator(prob, gaco_solver)
# evaluator.any_violation()
# evaluator.any_violation(0.4)
evaluator.min_violation()
evaluator.min_violation(0.4)

prob, sys_eval = before_each(True)
nsga_solver = NSGA2Solver(1, 8, sys_eval)
evaluator = Evaluator(prob, nsga_solver)
# evaluator.any_violation()
# evaluator.any_violation(0.4)
evaluator.min_violation()
evaluator.min_violation(0.4)
