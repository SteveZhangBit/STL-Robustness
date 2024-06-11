import os

import matplotlib.pyplot as plt
from matlab import engine

from robustness.agents.matlab import Traditional
from robustness.analysis.algorithms.breach import BreachOneLayerSystemEvaluator
from robustness.analysis.algorithms.one_layer import OneLayerSolver
from robustness.analysis.breach import BreachSTL
from robustness.analysis.core import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.matlab import DevWTKBaseline
from robustness.evaluation.evaluator import Evaluator
from robustness.evaluation.experiment import Experiment

# remove all fils under the data folder
os.system('rm -rf tests/data/WTK/traditional/ol')

os.makedirs('tests/gifs/WTK/traditional', exist_ok=True)
plt.rc('axes', labelsize=17, titlesize=17)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

eng = engine.start_matlab()
eng.addpath('lib/breach', nargout=0)
eng.addpath(eng.genpath('models/WTK'), nargout=0)
eng.InitBreach(nargout=0)

in_rates = [0.01, 0.5]
out_rates = [0.01, 0.2]
env = DevWTKBaseline(eng, in_rates, out_rates, (0.25, 0.1))
agent = Traditional()
phi = BreachSTL('alw_[5,5.9](abs(h_error[t]) < 1) and ' + \
                'alw_[11,11.9](abs(h_error[t]) < 1) and ' + \
                'alw_[17,17.9](abs(h_error[t]) < 1) and ' + \
                'alw_[23,23.9](abs(h_error[t]) < 1)')

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = BreachOneLayerSystemEvaluator(eng, phi, {'restarts': 0, 'evals': 10})

solver = OneLayerSolver(sys_eval, ['in_rate', 'out_rate'], {'restarts': 0, 'evals': 10})
evaluator = Evaluator(prob, solver)
experiment = Experiment(evaluator)

print('Find violations by OneLayer...')
records_ol = experiment.record_min_violations(out_dir='tests/data/WTK/traditional/ol', runs=1)
records_ol, violations_ol = experiment.summarize_violations(records_ol, 'tests/data/WTK/traditional/ol')
# Plot the samples
for i in range(len(records_ol)):
    samples = [(X, Y) for (X, Y, _) in records_ol[i]]
    experiment.plot_samples(samples, 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20)
    plt.title('Violations found by OneLayer')
    plt.savefig(f'tests/gifs/WTK/traditional/fig-violations-ol-{i}.png', bbox_inches='tight')

experiment.plot_samples([[(X, Y) for (X, Y, _) in r] for r in records_ol], 'Inflow rate', 'Outflow rate', 'data/WTK/traditional', n=20)
# plt.title('WTK-PID with OneLayer')
plt.savefig(f'tests/gifs/WTK/traditional/fig-violations-ol-all.png', bbox_inches='tight')
