import os
import matplotlib.pyplot as plt

from matlab import engine
from robustness.agents.matlab import RLAgent
from robustness.analysis.algorithms.breach import BreachOneLayerSystemEvaluator
from robustness.analysis.algorithms.one_layer import OneLayerSolver
from robustness.analysis.algorithms.random import RandomSolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevAFCBaseline
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment

os.makedirs('gifs/AFC/RL', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

eng = engine.start_matlab()
eng.addpath('lib/breach', nargout=0)
eng.addpath(eng.genpath('models/AFC'), nargout=0)
eng.InitBreach(nargout=0)

MAF_sensor_tols = [0.95, 1.05]
AF_sensor_tols = [0.99, 1.01]
env = DevAFCBaseline(eng, MAF_sensor_tols, AF_sensor_tols, (1.0, 1.0))
agent = RLAgent('AFC_DDPG_Agent.mat')
phi = BreachSTL('alw (AF[t] < 1.2*14.7 and AF[t] > 0.8*14.7)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachOneLayerSystemEvaluator(eng, phi, {'restarts': 0, 'evals': 100})

solver = OneLayerSolver(sys_eval, ['MAF_sensor_tol', 'AF_sensor_tol'], {'restarts': 0, 'evals': 100})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by OneLayer...')
records_ol = experiment.record_min_violations(out_dir='data/AFC/RL/ol')
records_ol, violations_ol = experiment.summarize_violations(records_ol, 'data/AFC/RL/ol')
# Plot all samples
for i in range(len(records_ol)):
    samples = [(X, Y) for (X, Y, _) in records_ol[i]]
    experiment.plot_samples(samples, 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/RL', n=20)
    plt.title('Violations found by OneLayer')
    plt.savefig(f'gifs/AFC/RL/fig-violations-ol-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_ol], 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/RL', n=20)
# plt.title('AFC-PID with OneLayer')
plt.savefig(f'gifs/AFC/RL/fig-violations-ol-all.png', bbox_inches='tight')
