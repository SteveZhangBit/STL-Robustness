import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.lunar_lander import PPO
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator, RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.lunar_lander import DevLunarLander, SafetyProp
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
    {'restarts': 1, 'episode_len': 300, 'evals': 50}
)

solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

experiment.record_min_violations(out_dir='data/lunar-lander-ppo/cma')


# Use random search
solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

experiment.record_min_violations(out_dir='data/lunar-lander-ppo/random')
