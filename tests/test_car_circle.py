import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
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
    {'timeout': 1, 'restarts': 0, 'episode_len': 300, 'evals': 40}
)
solver = CMASolver(0.1, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

delta, delta_dist, x0 = evaluator.any_violation()
evaluator.visualize_violation(delta, x0, gif='gifs/car-circle-ppo/counterexample.gif')
