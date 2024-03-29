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

winds = [0.0, 10.0]
turbulences = [0.0, 1.0]
env = DevLunarLander(winds, turbulences, (5.0, 0.5))
agent = PPO('models/lunar-lander/ppo.zip')
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi, 
    {'timeout': 1, 'restarts': 0, 'episode_len': 300, 'evals': 40}
)
solver = CMASolver(0.3, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)

# print('Certified minimum deviation:', evaluator.certified_min_violation())
# radius = evaluator.smooth_boundary(0.1, 1000, 0.05, 0.9, 'data/lunar-lander-ppo')

delta, _, _ = evaluator.any_violation()
if delta is None:
    print('No violation found')
    exit()

radius = evaluator.unsafe_region(delta, 0.1, 0.05, 'data/lunar-lander-ppo', n=10_000)

plt.figure()
evaluator.heatmap(
    winds, turbulences, 25, 25,
    x_name="Wind", y_name="Turbulence", z_name="System Evaluation $\Gamma$",
    out_dir='data/lunar-lander-ppo',
    boundary=radius,
    center=delta,
    # vmax=0.1, vmin=-0.4
)
# plt.title('Smooth Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % radius)
plt.title('Unsafe Region: $||\delta - \delta_v||_2 < %.3f$' % radius)
os.makedirs('gifs/lunar-lander-ppo/unsafe-region', exist_ok=True)

delta_str = '-'.join([f'{d:.3f}' for d in delta])
plt.savefig(f'gifs/lunar-lander-ppo/unsafe-region/fig-{delta_str}.png', bbox_inches='tight')
# plt.savefig('gifs/lunar-lander-ppo/fig-smooth-robustness.png', bbox_inches='tight')
