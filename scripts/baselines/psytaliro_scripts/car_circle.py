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
from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.car_circle import DevCarCircle, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment
from robustness.evaluation.utils import boxplot
import pandas as pd
from matplotlib import pyplot as plt
import csv
CarRunDataT = ModelResult[List[float], None]

load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_circle_ppo_vanilla/model_save/model.pt'
agent = PPOVanilla(load_dir)

@blackbox()
def car_circle_model(static: Sequence[float], times: SignalTimes, signals: SignalValues) -> CarRunDataT:
   speed = [5.0, 35.0]
   steering = [0.2, 0.8]
   env = DevCarCircle(load_dir, speed, steering)
   episode_len = 300

   # this is for dist only
   delta = normalize(static[0:2], env.dev_bounds)
   delta_0 = normalize(env.delta_0, env.dev_bounds)
   dist = np.sqrt( np.sum((delta - delta_0) ** 2))

   # set the deviation params first (steering and speed)
   env, x0bounds = env.instantiate(static[0:2])
   # set the initial state after
   obs = env.reset_to(static[2:4])  
   simulation_state = []
   x_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
   x_check = normalize(0.7, np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
   simulation_state.append(np.array([x_signal, dist]))
   for _ in range(episode_len):
        action = agent.next_action(obs)
        obs, reward, _, _ = env.step(action)
        x_signal = normalize(np.abs(obs[0]), np.asarray([env.observation_space.low[0], env.observation_space.high[0]]))
        simulation_state.append(np.array([x_signal, dist]))
   simulation_state = np.array(simulation_state)
   states = simulation_state.T
   
   simulation_trace = Trace(states=states.tolist(), times= list(range(episode_len+1)))

   return BasicResult(simulation_trace)


def plot_csv_samples(filename, experiment):
    data = pd.read_csv(filename)
    samples = [([row['d1'], row['d2']], row['Cost']) for index, row in data.iterrows()]
    experiment.plot_samples(samples, 'Speed', 'Steering', '/usr0/home/parvk/cj_project/STL-Robustness/data/car-circle-ppo', n=20)
    plt.savefig(f'baseline_results/car_circle_ppo.png', bbox_inches='tight')
    min_cost_row = data.loc[data['Cost'].idxmin()]
    return min_cost_row

def create_new_rob_csv(filename):
        # lot of extra code for plotting tbh
        load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_circle_ppo_vanilla/model_save/model.pt'
        speed = [5.0, 35.0]
        steering = [0.2, 0.8]
        env = DevCarCircle(load_dir, speed, steering)
        agent = PPOVanilla(load_dir)
        phi = SafetyProp()
        episode_len = 300

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
            obs = e.reset_to([row['i1'], row['i2']]) 
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
            
        with open('baseline_results/delta_car_circle.csv', mode='w', newline='') as file:
            writer = csv.writer(file)

            # # Write the header
            # writer.writerow(['Robustness', ' Delta', ' States', ' Actions'])

            # Write the data rows
            for row in final_deltas:
                writer.writerow(row)
        # print(final_deltas)
if __name__ == "__main__":
    filename = "baseline_results/car_circle_data.csv"
    if not os.path.isfile(filename):
        phi = "(always(x < 0.50035) or d > 0.3)" 
        specification = RTAMTDense(phi, {"x": 0, "d": 1})
        optimizer = DualAnnealing(behavior = Behavior.MINIMIZATION)
        options = Options(runs=100, iterations=100, interval=(0, 1), static_parameters=[(5.0,35.0),(0.2,0.8),(-3.0,3.0),(-3.0,3.0)])
        result = staliro(car_circle_model, specification, optimizer, options)
        import csv 
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['d1','d2', 'i1', 'i2', 'Cost'])
            for run in result.runs:
                evaluation = worst_eval(run)
                writer.writerow([evaluation.sample[0],evaluation.sample[1], evaluation.sample[2],evaluation.sample[3], evaluation.cost])
    else:
        print('Data found, plotting.... \n')
<<<<<<< HEAD
        create_new_rob_csv(filename)
=======
        load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_circle_ppo_vanilla/model_save/model.pt'
        speed = [5.0, 35.0]
        steering = [0.2, 0.8]
        env = DevCarCircle(load_dir, speed, steering)
        agent = PPOVanilla(load_dir)
        phi = SafetyProp()
        episode_len = 300
>>>>>>> c71058c6ef612cfae4d68418b9128c9fe4de873e

    
