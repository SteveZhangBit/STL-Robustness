import numpy as np
import matplotlib.pyplot as plt
from robustness.analysis.utils import normalize, L2Norm
from robustness.envs.lunar_lander import LunarLander,DevLunarLander
from robustness.envs.car_run import DevCarRun
from robustness.envs.cartpole import DevCartPole
from robustness.envs.car_circle import DevCarCircle
import os
import csv
import pandas as pd
import glob 


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

def generate_delta_data(csv_file_path, env, delta_size, dist):
    '''
    gets the worst trajectory from all the saved data files for a given delta and then computes the dist, rho and cosine_similarity first
    '''
    #first load delta 0 trajectory
    d0_file_path = csv_file_path +'/delta0/satisfied/*.csv'
    d0_file, d0_trace,d0, d0_cum_trace = worst_case_rob(d0_file_path, env, delta_size)
    
    # now go through all the violated trajs and create a map of all the delta + rob + traces with the minimum value. 
    delta_map = {}
    other_file_path = csv_file_path +'/violated/*.csv'
    for csv_file in glob.glob(other_file_path):
        cur_data = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Identify the 'Robustness' column
        robustness = cur_data[0,0]
        
        # Create a tuple of delta values to use as a key
        delta_key = tuple(cur_data[0, 1:delta_size+1])
        if delta_key in delta_map:
            if robustness < delta_map[delta_key]['Robustness']:
                delta_map[delta_key]['Robustness'] = robustness
                delta_map[delta_key]['Trace'] = cur_data[:, delta_size+1:delta_size+1+env.observation_space.shape[0]]
                delta_map[delta_key]['ATrace'] = cur_data[:, delta_size+1:delta_size+1+env.observation_space.shape[0]+env.action_space.shape[0]]
        else:
            delta_map[delta_key] = {'Robustness':robustness, 
                                    'Trace': cur_data[:, delta_size+1:delta_size+1+env.observation_space.shape[0]],
                                    'ATrace': cur_data[:, delta_size+1:delta_size+1+env.observation_space.shape[0]+env.action_space.shape[0]]
                                    }

    sat_file_path = csv_file_path +'/satisfied/*.csv'
    print('SAT FILE PATH')
    print(sat_file_path)
    for csv_file in glob.glob(sat_file_path):
        cur_data2 = np.genfromtxt(csv_file, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)
        
        
        # Identify the 'Robustness' column
        robustness = cur_data2[0,0]
        
        # Create a tuple of delta values to use as a key
        delta_key = tuple(cur_data2[0, 1:delta_size+1])
        if delta_key in delta_map:
            if robustness < delta_map[delta_key]['Robustness']:
                delta_map[delta_key]['Robustness'] = robustness
                delta_map[delta_key]['Trace'] = cur_data2[:, delta_size+1:delta_size+1+env.observation_space.shape[0]]
                delta_map[delta_key]['ATrace'] = cur_data2[:, delta_size+1:delta_size+1+env.observation_space.shape[0]+env.action_space.shape[0]]
        else:
            delta_map[delta_key] = {'Robustness':robustness, 
                                    'Trace': cur_data2[:, delta_size+1:delta_size+1+env.observation_space.shape[0]],
                                    'ATrace': cur_data2[:, delta_size+1:delta_size+1+env.observation_space.shape[0]+env.action_space.shape[0]]
                                    }

    final_array = []

    for key in delta_map.keys():
        dis = dist.eval_dist(key)
        rho = delta_map[key]['Robustness']
        print(d0_trace.shape)
        print('now sat')
        print(np.array(delta_map[key]['Trace']).shape)
        if np.array(delta_map[key]['Trace']).shape== d0_trace.shape:
            sim = compute_cosine_similarity(d0_trace, np.array(delta_map[key]['Trace']))
            final_array.append([dis, rho, sim])
    #final_array.append([0,0.12 , 1])
    return np.array(final_array)
  
def plot_pairwise(data):
    '''
    Plots pairwise to investigate the trend, if any, by creating side-by-side scatter plots.
    '''
    plt.figure(figsize=(16, 6))  # Set the figure size to be wider

    # Plotting 0th column against the 1st column
    plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot
    plt.scatter(data[:, 0], data[:, 1], c='blue', label='Column 0 vs Column 1')
    plt.xlabel('dist')
    plt.ylabel('rho')
    plt.title('dist vs rho')
    plt.legend()

    # Plotting 0th column against the 2nd column
    plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
    plt.scatter(data[:, 0], data[:, 2], c='green', label='Column 0 vs Column 2')
    plt.xlabel('dist')
    plt.ylabel('sim')
    plt.title('dist vs sim')
    plt.legend()

    # Plotting 1st column against the 2nd column
    plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd subplot
    plt.scatter(data[:, 1], data[:, 2], c='red', label='Column 1 vs Column 2')
    plt.xlabel('rho')
    plt.ylabel('sim')
    plt.title('rho vs sim')
    plt.legend()
    plt.savefig('an.png')
    plt.show()


# def has_valid_num_columns(row, expected_columns):
#     return len(row) == expected_columns

# def plot_data(csv_file_path, save_file_path, offset):
#     expected_columns = 14  #what are these columns??
#     data = np.genfromtxt(csv_file_path, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)

#     # Apply the custom filter function to skip rows with an inconsistent number of columns
#     data = data[[has_valid_num_columns(row, expected_columns) for row in data]]

#     # reshaping for plotting
#     desired_num_rows = data.shape[0] // 301
#     desired_shape = (desired_num_rows, 301, 14)
#     reshaped_array = np.reshape(data, desired_shape)
#     angle_bound = np.tile(np.array([45]), (301, 1))
#     delta_xbound = np.tile(np.array([0.1]), (301, 1))

#     # lunar lander specific info
#     env = LunarLander(enable_wind=True, continuous=True)
#     obs_space = env.observation_space
#     angle_range = np.asarray([obs_space.low[4], obs_space.high[4]])
#     x_range = np.asarray([obs_space.low[0], obs_space.high[0]])
#     y_range = np.asarray([obs_space.low[1], obs_space.high[1]])
#     delta_x = normalize(np.abs(reshaped_array[:,:,4]), x_range) - normalize(np.abs(0.6 * reshaped_array[:,:,5]), y_range)
#     # Timesteps (x-axis)
#     delta_x_ch = delta_xbound.T - delta_x
#     angle_x_ch =  angle_bound.T  -np.degrees(reshaped_array[:, :, 8]) 
#     timesteps = np.arange(301)  


#     # plotting begins here 
#     fig, ax = plt.subplots(2, 1, figsize=(10, 8))
#     for i in range(0, desired_num_rows-offset):
#         ax[0].plot(timesteps, np.degrees(reshaped_array[i, :, 8]), label=f'Row {i+1}')
#     ax[0].plot(timesteps, angle_bound, label=f'Row {i+1}')
#     ax[0].plot(timesteps, -angle_bound, label=f'Row {i+1}')

#     # Add labels and legend for angle plot
#     ax[0].set_xlabel('Timestep')
#     ax[0].set_ylabel('Angle')
#     ax[0].text(0, 50, f'{desired_num_rows} data points', fontsize = 12)
#     for i in range(desired_num_rows-offset):
#         ax[1].plot(timesteps, delta_x[i], label=f'Delta_x Row {i+1}')
#     ax[1].plot(timesteps, delta_xbound, label=f'Row {i+1}')

#     # Add labels and legend for the delta_x plot
#     ax[1].set_xlabel('Timestep')
#     ax[1].set_ylabel(f'|x|- 0.6*|y|')
#     #plt.savefig(save_file_path)
#     plt.show()

if __name__ == '__main__':
    #lunar lander
    winds = [0.0, 10.0]
    turbulences = [0.0, 1.0]
    new_env = DevLunarLander(winds, turbulences, (5.0, 0.5))
    env = LunarLander(enable_wind=True, continuous=True)

    # cartpole
    # masses = [0.1, 2.0]
    # forces = [1.0, 20.0]
    # new_env = DevCartPole(masses, forces, (1.0, 10.0))
    # env,x = new_env.instantiate((1.0, 10.0))
    #env = LunarLander(enable_wind=True, continuous=True)
    
    # carrun
    # load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_run_ppo_vanilla/model_save/model.pt'
    # speed = [5.0, 60.0]
    # steering = [0.2, 0.8]
    # new_env = DevCarRun(load_dir, speed, steering)
    # env,x = new_env.instantiate((20.0, 0.5))
    # dist = L2Norm(new_env)

    #carcircle
    # load_dir = '/usr0/home/parvk/cj_project/STL-Robustness/models/car_circle_ppo_vanilla/model_save/model.pt'
    # speed = [5.0, 35.0]
    # steering = [0.2, 0.8]
    # new_env = DevCarCircle(load_dir, speed, steering)
    # env,x = new_env.instantiate((20.0, 0.5))
    dist = L2Norm(new_env)
    final_array = generate_delta_data('traces/LunarLander', env, 2, dist)
    op = 'op_lunarfin.csv'
    # Open the CSV file in write mode
    with open(op, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['dist','rho','sim'])

        # Write the data rows
        for row in final_array:
                writer.writerow(row)
    print(final_array)
    plot_pairwise(final_array)
 



