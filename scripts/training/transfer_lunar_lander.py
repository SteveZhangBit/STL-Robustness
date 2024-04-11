# import os

# import matplotlib.pyplot as plt
# import numpy as np

# from robustness.agents.lunar_lander import PPO
# from robustness.analysis import Problem
# from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
#                                             RandomSolver, ExpectationSysEvaluator)
# from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.lunar_lander import DevLunarLander, SafetyProp, SafetyProp2
# from robustness.evaluation import Evaluator, Experiment
# from robustness.evaluation.utils import boxplot
from robustness.agents.lunar_lander import PPO
import time
import imageio
winds = [0.0, 10.0]  
turbulences = [0.0, 1.0]
env = DevLunarLander(winds, turbulences, (0.0, 0.0))
from stable_baselines3 import PPO as BaselinePPO

#agent = PPO('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo.zip')
episode_len = 300
#env, x0bounds = env.instantiate([7.832101630613496,0.4323776925279945])
env, x0bounds = env.instantiate([6.8851463628585545,0.6669314213708963])
#6.518368346608023,0.7562079762757379
#6.8851463628585545,0.6669314213708963
#env, x0bounds = env.instantiate([5.879106789841222, 0.5553259436583973])
agent = BaselinePPO.load('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo.zip', env)  

# set the initial state after
obs = env.reset_to([-1.842157832288649,-1.4106696024537086])  
# set the initial state after

# # Instantiate the agent
# model = BaselinePPO.load('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo.zip')

# # Train the agent
agent.learn(total_timesteps=50000) 
frames = []
# # Save the trained model
# model.save("ppo_lunarlander")
obs = env.reset_to([-1.842157832288649,-1.4106696024537086])  
# # Load the trained model (optional)
for _ in range(episode_len):
    action = agent.predict(obs, deterministic=True)[0]
    obs, reward, _, _ = env.step(action)
    time.sleep(0.01) 
    #env.render()
    frames.append(env.render(mode="rgb_array"))

imageio.mimsave('lunar_lander_sat2.gif', frames, fps=30)