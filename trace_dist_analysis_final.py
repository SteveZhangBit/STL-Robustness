import numpy as np
import matplotlib.pyplot as plt
from robustness.analysis.utils import normalize, L2Norm
from robustness.envs.lunar_lander import LunarLander,DevLunarLander
# from robustness.envs.car_run import DevCarRun
from robustness.envs.cartpole import DevCartPole
from robustness.envs.car_circle import DevCarCircle
import os
import csv
import pandas as pd
import glob 
from robustness.agents.rsrl import PPOVanilla
from robustness.analysis import Problem
from robustness.analysis.algorithms import (CMASolver, CMASystemEvaluator,
                                            ExpectationSysEvaluator,
                                            RandomSolver)
from robustness.analysis.utils import L2Norm, normalize
from robustness.envs.car_run import DevCarRun, SafetyProp, SafetyProp2
from robustness.evaluation import Evaluator, Experiment

def worst_case_rob(folder_path, env, delta_size):
    # Initialize minimum value and file name
    min_robustness = float('inf')
    file_with_min_robustness = ''
    print(folder_path)
    print('\n\n')
    # Iterate through all CSV files in the folder
    for csv_file in glob.glob(folder_path):
        # Load the CSV file
        df = pd.read_csv(csv_file)
        # Check if 'robustness' column exists in the dataframe
        if 'Robustness' in df.columns:
            # Find the minimum 'robustness' value in the current file
            temp_data = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)
            current_min = temp_data[0,0]
            # Update the minimum and file name if current file has a lower value
            if current_min < min_robustness:
                min_robustness = current_min
                file_with_min_robustness = csv_file

    #after getting the worst case file, load its state action pairs
    data = np.genfromtxt(file_with_min_robustness, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)
    # print(file_with_min_robustness)
    # print(min_robustness)
    # print('we done\n\n\n')
    #NOTE: for v1 only returning the state trace since that is easier, data has action too (dont know how to combine different shape state and action)
    state_trace = data[:, delta_size+1:delta_size+1+env.observation_space.shape[0]]
    delta_0 = data[0, 1:delta_size+1]
    # print(state_trace)
    # with open('delta_nom_trace_carcircle2.csv', mode='w', newline='') as file:
    #         writer = csv.writer(file)

    #         # # Write the header
    #         # writer.writerow(['Robustness', ' Delta', ' States', ' Actions'])

    #         # Write the data rows
    #         for row in state_trace:
    #             writer.writerow(row)
    #v2 begins here
    state_action_trace = data[:, delta_size+1:delta_size+1+env.observation_space.shape[0]+env.action_space.shape[0]]
    delta_0 = data[0, 1:delta_size+1] 
    return file_with_min_robustness, state_trace, delta_0, state_action_trace



def compute_cosine_similarity(trace1, trace2):
    '''
    Computes the cosine similarity between two traces of shape n*t.
    
    Parameters:
        trace1 (np.ndarray): First trace matrix of shape (n, t).
        trace2 (np.ndarray): Second trace matrix of shape (n, t).
        
    Returns:
        float: Cosine similarity between the two traces.
    '''
    # Flatten the traces
    trace1_flat = trace1.flatten()
    trace2_flat = trace2.flatten()
    
    # Compute the dot product
    dot_product = np.dot(trace1_flat, trace2_flat)
    
    # Compute the norms
    norm_trace1 = np.linalg.norm(trace1_flat)
    norm_trace2 = np.linalg.norm(trace2_flat)
    
    # Compute cosine similarity
    cosine_similarity = dot_product / (norm_trace1 * norm_trace2)
    return cosine_similarity

def generate_delta_data(csv_file_path, env, delta_size, dist, pe):
    '''
    gets the worst trajectory from all the saved data files for a given delta and then computes the dist, rho and cosine_similarity first
    '''
    #first load delta 0 trajectory
    d0_file_path = csv_file_path +'/delta0/satisfied/*.csv'
    d0_file, d0_trace,d0, d0_cum_trace = worst_case_rob(d0_file_path, env, delta_size)
    # this part remains the same
    # now go through the deviations and generate trajs
    load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
    agent = PPOVanilla(load_dir)
    phi = SafetyProp()
    episode_len = 200

    # Create problem and solver
    prob = Problem(pe, agent, phi, L2Norm(env))
    sys_eval = CMASystemEvaluator(0.4, phi, {'timeout': 1, 'episode_len': 200})
    # Use CMA
    solver = CMASolver(0.2, sys_eval)
    evaluator = Evaluator(prob, solver)
    experiment = Experiment(evaluator)
    #print(best_sample)
    final_array = []
    filename= '/usr0/home/parvk/cj_project/STL-Robustness/scripts/baselines/psytaliro_scripts/baseline_results/car_run_data.csv'
    data = pd.read_csv(filename)
    #data = np.genfromtxt(file_with_min_robustness, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)
    # samples = [([row['d1'], row['d2']], row['Cost']) for index, row in data.iterrows()]
    for index,row in data.iterrows():
        # set the deviation params first (steering and speed)
        e, x0bounds = pe.instantiate([16.01395176927879,0.35417642736709043])
        space = e.observation_space
        # set the initial state after
        obs = e.reset_to([row['i1'], row['i2'],row['i3']]) 
        obs_record = [obs]
        reward_record = [0]
        for _ in range(episode_len):
            action = agent.next_action(obs)
            obs, reward, _, _ = e.step(action)
            obs_record.append(np.clip(obs, space.low, space.high))
            reward_record.append(reward)
        score =sys_eval.phi.eval_trace(np.array(obs_record), np.array(reward_record))
        # if score < 0:
        dis = dist.eval_dist((row['d1'],row['d2']))
        sim = compute_cosine_similarity(d0_trace, np.array(obs_record))
        print(dis)
        print(score)
        print(sim)
        final_array.append([row['d1'],row['d2'],dis, sim, score])

        
    return np.array(final_array)




if __name__ == '__main__':
    # carrun
    load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
    speed = [5.0, 35.0]
    steering = [0.2, 0.8]
    new_env = DevCarRun(load_dir, speed, steering)
    env,x = new_env.instantiate((20.0, 0.5))
    dist = L2Norm(new_env)
    final_array = generate_delta_data('traces/SafetyCarRun', env, 2, dist, new_env)
    op = 'op_car_run_fin.csv'
    # Open the CSV file in write mode
    with open(op, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['d1','d2','dist','sim','rho'])

        # Write the data rows
        for row in final_array:
                writer.writerow(row)
 



