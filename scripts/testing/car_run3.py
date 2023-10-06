import os

import matplotlib.pyplot as plt
import numpy as np

from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.car_run import DevCarRun2, SafetyProp, SafetyProp3
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

os.makedirs('gifs/car-run-ppo', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
speed = [5.0, 35.0]
steering = [0.2, 0.8]
force = [5,50]
env = DevCarRun2(load_dir, speed, steering, force)
agent = PPOVanilla(load_dir)
phi = SafetyProp3()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'restarts': 0, 'episode_len': 300, 'evals': 50}
)

# Use CMA
solver = CMASolver(0.1, sys_eval, {'restarts': 0, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/car-run32-ppo/cma')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/car-run32-ppo/cma')

# # Use random search
# solver = RandomSolver(sys_eval, {'restarts': 1, 'evals': 50})
# evaluator = Evaluator(prob, solver)
# experiment = Experiment(evaluator)

# print('Find violations by random search...')
# records_random = experiment.record_min_violations(out_dir='data/car-run32-ppo/random')
# records_random, violations_random = experiment.summarize_violations(records_random, 'data/car-run32-ppo/random')


import pickle

with open('data/car-run32-ppo/cma/records-min-violations-0.pickle', 'rb') as f:
    data = pickle.load(f)
print(data)
delta = data[0][np.argmin(data[1])]
#delta = data[0][10]
delta=(20.0, 0.5, 10)
delt = np.array(delta)
evaluator.visualize_violation(delt, gif="gifs/car-run-ppo/nominal.gif", render=True)

