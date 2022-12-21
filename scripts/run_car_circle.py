from robustness.agents.car_circle import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.utils import L2Norm
from robustness.envs.car_circle import DevCarCircle, SafetyProp
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            RandomSolver)
import matplotlib.pyplot as plt
import numpy as np

load_dir = 'models/car_circle_ppo_vanilla/model_save/model.pt'
speed = [1.0, 100.0]
steering = [0.1, 1]
env = DevCarCircle(load_dir, speed, steering)
agent = PPOVanilla(load_dir)
phi = SafetyProp()

prob = Problem(env, agent, phi, L2Norm(env))
sys_eval = CMASystemEvaluator(
    0.4, phi,
    {'timeout': 1, 'restarts': 1, 'episode_len': 200, 'evals': 50}
)

solver = CMASolver(0.2, sys_eval)
evaluator = Evaluator(prob, solver)

# experiment = Experiment(evaluator)
# data1, _ = experiment.run_diff_max_samples('CMA', np.arange(25, 126, 25), out_dir='data/car-circle-ppo/cma')
# plt.rc('axes', labelsize=12, titlesize=13)
evaluator.heatmap(
    speed, steering, 25, 25,
    x_name="Speed Multiplier", y_name="Steering Multiplier", z_name="System Evaluation $\Gamma$",
    out_dir='data/car-circle-ppo',
    # boundary=np.min(data1),
)
# plt.title('Robustness $\hat{\Delta}: ||\delta - \delta_0||_2 < %.3f$' % np.min(data1))
plt.savefig('gifs/car-circle-ppo/robustness.png', bbox_inches='tight')

# Use Random Solver
# random_solver = RandomSolver(sys_eval)
# evaluator2 = Evaluator(prob, random_solver)
# experiment2 = Experiment(evaluator2)
# data2, _ = experiment2.run_diff_max_samples('Random', np.arange(25, 126, 25), out_dir='data/car-circle-ppo/random')

# plt.xlabel('Number of samples')
# plt.ylabel('Minimum distance')
# boxplot([data1, data2], ['red', 'blue'], np.arange(25, 126, 25) * (1 + solver.options()['restarts']),
#         ['CMA', 'Random'])
# plt.savefig('gifs/car-circle-ppo/sample-boxplot.png', bbox_inches='tight')

# from datetime import datetime
# start = datetime.now()
# print(sys_eval.eval_sys([80, 1], prob))
# print(datetime.now() - start)

# evaluator.visualize_violation(env.get_delta_0(), render=True)

# import time
# import numpy as np
# inst, _ = env.instantiate(env.get_delta_0(), render=True)
# rewards = []
# for _ in range(5):
#     total = 0
#     obs = inst.reset_to([2.99945952, -1.85152463])
#     time.sleep(2)
#     for _ in range(200):
#         obs, reward, _, _ = inst.step(agent.next_action(obs))
#         total += reward
#         time.sleep(0.01)
#     print(total)
#     rewards.append(total)
# inst.close()
# print('Mean rewards:', np.mean(rewards))
