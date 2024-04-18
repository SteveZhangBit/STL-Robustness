import os
import matplotlib.pyplot as plt

from matlab import engine
from robustness.agents.matlab import RLAgent
from robustness.analysis.algorithms.breach import BreachSystemEvaluator, BreachSystemEvaluatorWithHeuristic
from robustness.analysis.algorithms.cma import CMASolver
from robustness.analysis.algorithms.random import RandomSolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevAFC
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
env = DevAFC(eng, MAF_sensor_tols, AF_sensor_tols, (1.0, 1.0))
agent = RLAgent('AFC_DDPG_Agent.mat')
phi = BreachSTL('alw (AF[t] < 1.2*14.7 and AF[t] > 0.8*14.7)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval_0 = BreachSystemEvaluator(eng, phi, {'restarts': 1, 'evals': 50})
sys_eval_0.eval_sys(env.get_delta_0(), prob)
delta_0_signals = sys_eval_0.obj_best_signal_values
sys_eval = BreachSystemEvaluatorWithHeuristic(
    eng, phi,
    ['es_radps', 'Pedal_Angle', 'MAF', 'af_meas', 'AF', 'AFref'],
    delta_0_signals,
    {'restarts': 1, 'evals': 50}
)

# Use CMA
solver = CMASolver(0.2, sys_eval, {'restarts': 1, 'evals': 50})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by CMA...')
records_cma = experiment.record_min_violations(out_dir='data/AFC/RL/cma_heuristic')
records_cma, violations_cma = experiment.summarize_violations(records_cma, 'data/AFC/RL/cma_heuristic')
# Plot all samples
for i in range(len(records_cma)):
    samples = [(X, Y) for (X, Y, _) in records_cma[i]]
    experiment.plot_samples(samples, 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/RL', n=20)
    plt.title('Violations found by CMA')
    plt.savefig(f'gifs/AFC/RL/fig-violations-cma-heuristic-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_cma], 'MAF Sensor Tolerance', 'AF Sensor Tolerance', 'data/AFC/RL', n=20)
# plt.title('AFC-DDPG with CMA')
plt.savefig(f'gifs/AFC/RL/fig-violations-cma-heuristic-all.png', bbox_inches='tight')
