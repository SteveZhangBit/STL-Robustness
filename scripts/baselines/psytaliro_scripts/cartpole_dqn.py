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
from staliro.optimizers import DualAnnealing, Behavior
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
from matplotlib import pyplot as plt
import pandas as pd
import csv
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
   delta = normalize(static[0:2], env.dev_bounds)
   delta_0 = normalize(env.delta_0, env.dev_bounds)
   dist = np.sqrt( np.sum((delta - delta_0) ** 2) )
   env, x0bounds = env.instantiate(static[0:2])
   # set the initial state after
   obs = env.reset_to(static[2:6])  
   simulation_state = []
   posit_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
   angle_signal = normalize(np.abs(obs[2]), np.asarray([env.observation_space.low[2], env.observation_space.high[2]]))
   simulation_state.append(np.array([posit_signal, angle_signal, dist]))
   done = False
   for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action)
        posit_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
        angle_signal = normalize(np.abs(obs[2]), np.asarray([env.observation_space.low[2], env.observation_space.high[2]]))
        simulation_state.append(np.array([posit_signal, angle_signal, dist]))
   simulation_state = np.array(simulation_state)
   states = simulation_state.T
   
   simulation_trace = Trace(states=states.tolist(), times= list(range(episode_len+1)))

   return BasicResult(simulation_trace)


def plot_csv_samples(filename, experiment):
    data = pd.read_csv(filename)
    samples = [([row['d1'], row['d2']], row['Cost']) for index, row in data.iterrows()]
    experiment.plot_samples(samples, 'Mass', 'Force', '/usr0/home/parvk/cj_project/STL-Robustness/data/cartpole-dqn', n=20)
    plt.savefig(f'baseline_results/cartpole_dqn_d2.png', bbox_inches='tight')

def create_new_rob_csv(filename):
        masses = [0.1, 2.0]
        forces = [1.0, 20.0]
        env = DevCartPole(masses, forces, (1.0, 10.0))
        agent = DQN('/usr0/home/parvk/cj_project/STL-Robustness/models/cartpole/best_dqn')
        phi = SafetyProp()
        episode_len = 200

        # Create problem and solver
        prob = Problem(env, agent, phi, L2Norm(env))
        sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 200})
        # Use CMA
        solver = CMASolver(0.2, sys_eval)
        evaluator = Evaluator(prob, solver)
        experiment = Experiment(evaluator)
        
        #best_sample = plot_csv_samples(filename, experiment)
        #print(best_sample)
        final_deltas = []
        data = pd.read_csv(filename)
        #data = np.genfromtxt(file_with_min_robustness, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)
        # samples = [([row['d1'], row['d2']], row['Cost']) for index, row in data.iterrows()]
        for index,row in data.iterrows():
            # set the deviation params first (steering and speed)
            e, x0bounds = env.instantiate([row['d1'], row['d2']])
            space = e.observation_space
            # set the initial state after
            obs = e.reset_to([row['i1'], row['i2'],row['i3'], row['i4']]) 
            obs_record = [obs]
            reward_record = [0]
            for _ in range(episode_len):
                action = agent.next_action(obs)
                obs, reward, _, _ = e.step(action)
                obs_record.append(np.clip(obs, space.low, space.high))
                reward_record.append(reward)
            score =sys_eval.phi.eval_trace(np.array(obs_record), np.array(reward_record))
            if score < 0:
                final_deltas.append([row['d1'],row['d2'], score])
            
        with open('baseline_results/delta_cartpole.csv', mode='w', newline='') as file:
            writer = csv.writer(file)

            # # Write the header
            # writer.writerow(['Robustness', ' Delta', ' States', ' Actions'])

            # Write the data rows
            for row in final_deltas:
                writer.writerow(row)
        # print(final_deltas)

if __name__ == "__main__":
    filename = "baseline_results/cartpole_data.csv"
    if not os.path.isfile(filename):
        print('Running baseline \n')
        phi = "(always(x <= 0.75 and y <= 0.75) or d > 0.35)" 
        specification = RTAMTDense(phi, {"x": 0, "y": 1, 'd':2})
        optimizer = DualAnnealing(behavior = Behavior.MINIMIZATION)
        options = Options(runs=100, iterations=100, interval=(0, 1), static_parameters=[(0.0,2.0),(0.0,20.0),(-0.05,0.05),(-0.05,0.05),(-0.05,0.05), (-0.05,0.05)])
        result = staliro(cartmodel, specification, optimizer, options)
        import csv 

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['d1','d2', 'i1', 'i2','i3', 'i4', 'Cost'])
            for run in result.runs:
                evaluation = worst_eval(run)
                writer.writerow([evaluation.sample[0],evaluation.sample[1], evaluation.sample[2],evaluation.sample[3],evaluation.sample[4],evaluation.sample[5], evaluation.cost])
    else:
        print('Data found, plotting.... \n')
        create_new_rob_csv(filename)
        # lot of extra code for plotting tbh
        # masses = [0.1, 2.0]
        # forces = [1.0, 20.0]
        # env = DevCartPole(masses, forces, (1.0, 10.0))
        # agent = DQN('/usr0/home/parvk/cj_project/STL-Robustness/models/cartpole/best_dqn')
        # phi = SafetyProp()

        # # Create problem and solver
        # prob = Problem(env, agent, phi, L2Norm(env))
        # sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 200})
        # # Use CMA
        # solver = CMASolver(0.2, sys_eval)
        # evaluator = Evaluator(prob, solver)
        # experiment = Experiment(evaluator)
        # plot_csv_samples(filename, experiment)
    
    # the following code is to visualize the deviation
    # best_sample = worst_eval(best_run(result)).sample
    # masses = [0.1, 2.0]
    # forces = [1.0, 20.0]
    # env = DevCartPole(masses, forces, (1.0, 10.0))
    # agent = DQN('/usr0/home/parvk/cj_project/STL-Robustness/models/cartpole/best_dqn')
    # phi = SafetyProp()
    # episode_len = 200
    # # set the deviation params first
    # env, x0bounds = env.instantiate(best_sample[0:2])
    # # set the initial state after
    # obs = env.reset_to(best_sample[2:6]) 
    # for _ in range(episode_len):
    #     action = agent.next_action(obs)
    #     obs, reward, _, _ = env.step(action) 
    #     env.render()
