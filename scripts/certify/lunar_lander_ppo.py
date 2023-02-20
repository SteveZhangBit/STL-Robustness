import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.lunar_lander import PPO
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver, ExpectationSysEvaluator)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.lunar_lander import DevLunarLander, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/lunar-lander-ppo', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

winds = [0.0, 20.0]
turbulences = [0.0, 2.0]
env = DevLunarLander(winds, turbulences, (0.0, 0.0))
agent = PPO('models/lunar-lander/ppo.zip')
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi, 
    {'timeout': 1, 'restarts': 0, 'episode_len': 300, 'evals': 40}
)
solver = CMASolver(0.2, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)
# print('Certified minimum deviation:', evaluator.certified_min_violation())
radius = evaluator.smooth_boundary(0.2, 100, 0.05)

plt.figure()
evaluator.heatmap(
    winds, turbulences, 25, 25,
    x_name="Wind", y_name="Turbulence", z_name="System Evaluation $\Gamma$",
    out_dir='data/lunar-lander-ppo',
    boundary=radius,
    vmax=0.1, vmin=-0.4
)
plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.savefig('gifs/lunar-lander-ppo/fig-smooth-robustness.png', bbox_inches='tight')
