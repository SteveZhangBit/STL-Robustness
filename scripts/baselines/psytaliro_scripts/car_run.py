import logging
import math
import time
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
from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.car_run import DevCarRun, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

CarRunDataT = ModelResult[List[float], None]



@blackbox()
def car_run_model(static: Sequence[float], times: SignalTimes, signals: SignalValues) -> CarRunDataT:
   load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
   speed = [5.0, 60.0]
   steering = [0.2, 0.8]
   env = DevCarRun(load_dir, speed, steering)
   agent = PPOVanilla(load_dir)
   phi = SafetyProp()
   episode_len = 200
   # set the deviation params first (steering and speed)
   env, x0bounds = env.instantiate(static[0:2])
   # set the initial state after
   obs = env.reset_to(static[2:5])  
   simulation_state = []
   y_signal = normalize(np.abs(obs[1]), np.asarray([env.observation_space.low[1], env.observation_space.high[1]]))
   tot_v_signal = np.linalg.norm([obs[2], obs[3]])
   v_signal = normalize(tot_v_signal, np.asarray([-10,10]))
   check_y = normalize(0.25, np.asarray([env.observation_space.low[1], env.observation_space.high[1]]))
   check_v = normalize(1.5, np.asarray([-10, 10]))
   simulation_state.append(np.array([y_signal, v_signal]))
   for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action)
        y_signal = normalize(np.abs(obs[1]), np.asarray([env.observation_space.low[1], env.observation_space.high[1]]))
        tot_v_signal = np.clip(np.linalg.norm([obs[2], obs[3]]), -10, 10)
        v_signal = normalize(tot_v_signal, np.asarray([-10, 10]))
        simulation_state.append(np.array([y_signal, v_signal]))
   simulation_state = np.array(simulation_state)
   states = simulation_state.T
   
   simulation_trace = Trace(states=states.tolist(), times= list(range(episode_len+1)))

   return BasicResult(simulation_trace)





if __name__ == "__main__":
    phi = "(always(x < 0.500125 and y < 0.575))" 
    specification = RTAMTDense(phi, {"x": 0, "y": 1})
    optimizer = DualAnnealing()
    options = Options(runs=1, iterations=100, interval=(0, 1), static_parameters=[(5.0,60.0),(0.2,0.8),(-0.1,0.1),(-0.1,0.1),(2.35619449, 3.92699082)])
    result = staliro(car_run_model, specification, optimizer, options)
    for run in result.runs:
        for evaluation in run.history:
           print(f"Sample: {evaluation.sample} -> Cost: {evaluation.cost}")
    best_sample = worst_eval(best_run(result)).sample
    print(best_sample)
    load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
    speed = [5.0, 60.0]
    steering = [0.2, 0.8]
    env = DevCarRun(load_dir, speed, steering)
    agent = PPOVanilla(load_dir)
    phi = SafetyProp()
    episode_len = 200
    # set the deviation params first (steering and speed)
    env, x0bounds = env.instantiate(best_sample[0:2], render=True)
    # set the initial state after
    obs = env.reset_to(best_sample[2:5]) 
    for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action)
        time.sleep(0.2) 
        env.render()
