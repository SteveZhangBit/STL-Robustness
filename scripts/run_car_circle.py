from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.utils import L2Norm, scale
from robustness.envs.car_circle import DevCarCircle, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.logger import Logger
from robustness.evaluation.utils import boxplot
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver, ExpectationSysEvaluator)
import matplotlib.pyplot as plt
import numpy as np

load_dir = 'models/car_circle_ppo_vanilla/model_save/model.pt'
speed = [5.0, 60.0]
steering = [0.2, 0.8]
env = DevCarCircle(load_dir, speed, steering)
agent = PPOVanilla(load_dir)
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'timeout': 1, 'restarts': 1, 'episode_len': 300, 'evals': 40}
)

solver = CMASolver(0.1, sys_eval)
evaluator = Evaluator(prob, solver)

# solver.set_options({'evals': 25})
# logger = Logger()
# for i in range(10):
#     delta, dist = evaluator.min_violation(logger=logger)
#     if dist < 0.1:
#         print(logger.find_dev_result(i, delta))
#         break

# from datetime import datetime
# start = datetime.now()
# print(sys_eval.eval_sys([20, 0.5], prob))
# print(datetime.now() - start)

# delta = scale([0.12102787, 0.44451285], env.get_dev_bounds())
# print('\nDistance:', prob.dist.eval_dist(delta), '\n')
# evaluator.visualize_violation(delta, x0=[-0.10101447,  0.32559958], render=True)

experiment = Experiment(evaluator)
data1, _ = experiment.run_diff_max_samples('CMA', np.arange(25, 126, 25), out_dir='data/car-circle-ppo/cma')
plt.rc('axes', labelsize=12, titlesize=13)
plt.figure()
evaluator.heatmap(
    speed, steering, 25, 25,
    x_name="Speed Multiplier", y_name="Steering Multiplier", z_name="System Evaluation $\Gamma$",
    out_dir='data/car-circle-ppo',
    boundary=np.min(data1),
)
plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % np.min(data1))
plt.savefig('gifs/car-circle-ppo/robustness.png', bbox_inches='tight')

# Use Random Solver
random_solver = RandomSolver(sys_eval)
evaluator2 = Evaluator(prob, random_solver)
experiment2 = Experiment(evaluator2)

plt.figure()
data2, _ = experiment2.run_diff_max_samples('Random', np.arange(25, 126, 25), out_dir='data/car-circle-ppo/random')

plt.xlabel('Number of samples')
plt.ylabel('Minimum distance')
boxplot([data1, data2], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
        ['CMA', 'Random'])
plt.savefig('gifs/car-circle-ppo/sample-boxplot.png', bbox_inches='tight')

# import time
# import numpy as np

# inst, _ = env.instantiate(env.get_delta_0(), render=True)
# rewards = []
# for _ in range(1):
#     total = 0
#     obs = inst.reset_to([-0.10101447,  0.32559958])
#     # time.sleep(2)
#     for _ in range(300):
#         obs, reward, _, _ = inst.step(agent.next_action(obs))
#         total += reward
#         time.sleep(0.01)
#     # print(total)
#     rewards.append(total)
# inst.close()
# print('Mean rewards:', np.mean(rewards))

sys_eval3 = ExpectationSysEvaluator(
    phi,
    {'timeout': 1, 'restarts': 1, 'episode_len': 300, 'evals': 40}
)
solver3 = CMASolver(0.1, sys_eval3)
evaluator3 = Evaluator(prob, solver3)
experiment3 = Experiment(evaluator3)
data3, _ = experiment3.run_diff_max_samples('Expc', np.arange(25, 126, 25), out_dir='data/car-circle-ppo/expc')
plt.figure()
evaluator3.heatmap(
    speed, steering, 25, 25,
    x_name="Speed Multiplier", y_name="Steering Multiplier", z_name="System Evaluation $\Gamma$",
    out_dir='data/car-circle-ppo/expc',
    boundary=np.min(data3),
)
plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % np.min(data3))
plt.savefig('gifs/car-circle-ppo/robustness-expc.png', bbox_inches='tight')

plt.figure()
plt.xlabel('Number of samples')
plt.ylabel('Minimum distance')
boxplot([data1, data3], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
        ['CMA', 'CMA-Expc'])
plt.savefig('gifs/car-circle-ppo/sample-boxplot-expc.png', bbox_inches='tight')
