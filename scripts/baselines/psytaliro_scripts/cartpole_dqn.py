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
from robustness.agents.cartpole import DQN
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.cartpole import DevCartPole, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot

CartpoleDataT = ModelResult[List[float], None]



@blackbox()
def cartmodel(static: Sequence[float], times: SignalTimes, signals: SignalValues) -> CartpoleDataT:
   masses = [0.1, 2.0]
   forces = [1.0, 20.0]
   env = DevCartPole(masses, forces, (1.0, 10.0))
   agent = DQN('/usr0/home/parvk/cj_project/STL-Robustness/models/cartpole/best_dqn')
   phi = SafetyProp()
   episode_len = 200
   # set the deviation params first 
   # TODO: make this extensible to multidimensional
   env, x0bounds = env.instantiate(static[0:2])
   # set the initial state after
   obs = env.reset_to(static[2:6])  
   simulation_state = []
   posit_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
   angle_signal = normalize(np.abs(obs[2]), np.asarray([env.observation_space.low[2], env.observation_space.high[2]]))
   simulation_state.append(np.array([posit_signal, angle_signal]))
   done = False
   for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action)
        posit_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
        angle_signal = normalize(np.abs(obs[2]), np.asarray([env.observation_space.low[2], env.observation_space.high[2]]))
        simulation_state.append(np.array([posit_signal, angle_signal]))
   simulation_state = np.array(simulation_state)
   states = simulation_state.T
   
   simulation_trace = Trace(states=states.tolist(), times= list(range(episode_len+1)))

   return BasicResult(simulation_trace)




if __name__ == "__main__":
    phi = "(always(x <= 0.75 and y <= 0.75))" 
    specification = RTAMTDense(phi, {"x": 0, "y": 1})
    optimizer = DualAnnealing()
    options = Options(runs=1, iterations=50, interval=(0, 1), static_parameters=[(0.0,2.0),(0.0,20.0),(-0.05,0.05),(-0.05,0.05),(-0.05,0.05), (-0.05,0.05)])
    result = staliro(cartmodel, specification, optimizer, options)
    for run in result.runs:
        for evaluation in run.history:
           print(f"Sample: {evaluation.sample} -> Cost: {evaluation.cost}")
    best_sample = worst_eval(best_run(result)).sample
    masses = [0.1, 2.0]
    forces = [1.0, 20.0]
    env = DevCartPole(masses, forces, (1.0, 10.0))
    agent = DQN('/usr0/home/parvk/cj_project/STL-Robustness/models/cartpole/best_dqn')
    phi = SafetyProp()
    episode_len = 200
    # set the deviation params first
    env, x0bounds = env.instantiate(best_sample[0:2])
    # set the initial state after
    obs = env.reset_to(best_sample[2:6]) 
    for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action) 
        env.render()
