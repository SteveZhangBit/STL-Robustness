# import os

# import matplotlib.pyplot as plt
# import numpy as np

# from robustness.agents.lunar_lander import PPO
# from robustness.analysis import Problem
# from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
#                                             RandomSolver, ExpectationSysEvaluator)
# from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.car_run import DevCarRun, SafetyProp, SafetyProp2
# from robustness.evaluation import Evaluator, Experiment
# from robustness.evaluation.utils import boxplot
from robustness.agents.rsrl import PPOVanilla
import time

load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
agent = PPOVanilla(load_dir)

speed = [5.0, 35.0]
steering = [0.2, 0.8]
env = DevCarRun(load_dir, speed, steering)
episode_len = 200
env, x0bounds = env.instantiate([25.980229951441288,0.5157936564503804], render=True)
# set the initial state after

# # Instantiate the agent
# model = BaselinePPO.load('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo.zip')

# # Train the agent
# model.learn(total_timesteps=100000) 

# # Save the trained model
# model.save("ppo_lunarlander")

# # Load the trained model (optional)
# model = PPO.load("ppo_lunarlander")
obs = env.reset_to([-0.06812768194761451,-0.08959302165224223,3.898391196355971]) 
for _ in range(episode_len):
    action = agent.next_action(obs)
    obs, reward, _, _ = env.step(action)
    time.sleep(0.01) 
    env.render()

