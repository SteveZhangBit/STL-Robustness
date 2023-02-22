import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
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
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'timeout': 1, 'restarts': 0, 'episode_len': 200, 'evals': 50}
)

solver = CMASolver(0.2, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)
# print('Certified minimum deviation:', evaluator.certified_min_violation())
radius = evaluator.smooth_boundary(0.1, 500, 0.05)

plt.figure()
evaluator.heatmap(
    speed, steering, 25, 25,
    x_name="Speed Multiplier", y_name="Steering Multiplier", z_name="System Evaluation $\Gamma$",
    out_dir='data/car-run-ppo',
    boundary=radius,
)
plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.savefig('gifs/car-run-ppo/fig-smooth-robustness.png', bbox_inches='tight')
