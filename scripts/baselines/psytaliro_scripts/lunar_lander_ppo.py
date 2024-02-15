import logging
import math
from typing import Any
import rtamt
import numpy as np
import os
import logging
import math
from typing import List, Sequence
import plotly.graph_objects as go
from staliro.core import BasicResult, ModelResult, Trace, best_eval, best_run, worst_eval, worst_run
from staliro.models import SignalTimes, SignalValues, blackbox
from staliro.optimizers import DualAnnealing
from staliro.options import Options
from staliro.specifications import TaliroPredicate, TpTaliro
from staliro.staliro import simulate_model, staliro
from staliro.specifications import RTAMTDense
from stable_baselines3 import PPO
import os
from staliro.core import best_eval, best_run
from PIL import Image
import gym
from robustness.agents.lunar_lander import PPO
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.lunar_lander import DevLunarLander, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

LunarDataT = ModelResult[List[float], None]



@blackbox()
def lunarmodel(static: Sequence[float], times: SignalTimes, signals: SignalValues) -> LunarDataT:
   winds = [0.0, 20.0]
   turbulences = [0.0, 2.0]
   env = DevLunarLander(winds, turbulences, (0.0, 0.0))
   agent = PPO('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo.zip')
   phi = SafetyProp()
   episode_len = 300
   # set the deviation params first 
   # TODO: make this extensible to multidimensional
   env, x0bounds = env.instantiate(static[0:2])
   # set the initial state after
   obs = env.reset_to(static[2:4])  
   simulation_state = []
   angle_threshold = normalize(45 * 2 * np.pi / 360, np.asarray([env.observation_space.low[4], env.observation_space.high[4]]))
   print(angle_threshold)
   x_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
   y_signal = normalize(np.abs(0.6*obs[1]), np.asarray([env.observation_space.low[1], env.observation_space.high[1]]))
   angle_signal = normalize(np.abs(obs[4]), np.asarray([env.observation_space.low[4], env.observation_space.high[4]]))
   simulation_state.append(np.array([angle_signal, x_signal-y_signal]))
   for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action)
        x_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
        y_signal = normalize(np.abs(0.6*obs[1]), np.asarray([env.observation_space.low[1], env.observation_space.high[1]]))
        angle_signal = normalize(np.abs(obs[4]), np.asarray([env.observation_space.low[4], env.observation_space.high[4]]))
        simulation_state.append(np.array([ angle_signal, x_signal-y_signal]))
   simulation_state = np.array(simulation_state)
   states = simulation_state.T
   
   simulation_trace = Trace(states=states.tolist(), times= list(range(episode_len+1)))

   return BasicResult(simulation_trace)




if __name__ == "__main__":
    phi = "(always(a < 0.625 and dx < 0.1))" 
    specification = RTAMTDense(phi, {"a": 0, "dx": 1})
    optimizer = DualAnnealing()
    options = Options(runs=1, iterations=50, interval=(0, 1), static_parameters=[(0.0,20.0),(0.0,2.0),(-3.0,3.0),(-3.0,3.0)])
    result = staliro(lunarmodel, specification, optimizer, options)
    for run in result.runs:
        for evaluation in run.history:
           print(f"Sample: {evaluation.sample} -> Cost: {evaluation.cost}")
    best_sample = worst_eval(best_run(result)).sample
    winds = [0.0, 20.0]
    turbulences = [0.0, 2.0]
    env = DevLunarLander(winds, turbulences, (0.0, 0.0))
    agent = PPO('/usr0/home/parvk/cj_project/STL-Robustness/models/lunar-lander/ppo.zip')
    episode_len = 300
    # set the deviation params first
    env, x0bounds = env.instantiate(best_sample[0:2])
    # set the initial state after
    obs = env.reset_to(best_sample[2:4]) 
    for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action) 
        env.render()
