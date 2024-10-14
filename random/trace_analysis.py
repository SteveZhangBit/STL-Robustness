import numpy as np
import matplotlib.pyplot as plt
from robustness.analysis.utils import normalize, L2Norm
from robustness.envs.lunar_lander import LunarLander,DevLunarLander2
import os
import csv


def compute_trend(x, signal, degree=1):
    # Fit a polynomial to the signal to compute the trend
    coeffs = np.polyfit(x, signal, degree)
    trend = np.polyval(coeffs, x)
    return trend
def find_zero_crossings(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return zero_crossings

# Path to the CSV file
csv_file_path = 'traces/trace_data.csv'  
def laplace_filter(signal, timesteps):
    # Laplace filter: computes second derivative
    laplace_filtered_signal = np.diff(np.diff(signal))
    x = timesteps
    num =1
    # plt.figure(figsize=(10, 5))
    # plt.subplot(2, 2, 1)
    # plt.plot(x, signal[0], label='Original Signal')
    # plt.legend()
    laplace_filtered_signal = signal
    # plt.subplot(2, 1, 2)
    plt.plot(x[:], laplace_filtered_signal[num], label='Laplace Filtered Signal')
    plt.legend()
    zero_crossings = find_zero_crossings(laplace_filtered_signal[num])
    print(len(zero_crossings))

    plt.scatter(x[:][zero_crossings], laplace_filtered_signal[num][zero_crossings], color='red', label='Zero Crossings')
    # trend = compute_trend(timesteps, signal[0], degree=1)
    # plt.subplot(2, 2, 3)
    # plt.plot(x, trend, label='Trend (Linear Fit)')
    plt.show()
    #plt.savefig('nominal_metrics.png')
    return laplace_filtered_signal




def has_valid_num_columns(row, expected_columns):
    return len(row) == expected_columns

def plot_data(csv_file_path, save_file_path, offset):
    expected_columns = 14  
    data = np.genfromtxt(csv_file_path, delimiter=',', dtype=float, skip_header=1, invalid_raise=False, usemask=True, filling_values=np.nan)

    # Apply the custom filter function to skip rows with an inconsistent number of columns
    data = data[[has_valid_num_columns(row, expected_columns) for row in data]]

    # reshaping for plotting
    desired_num_rows = data.shape[0] // 301
    desired_shape = (desired_num_rows, 301, 14)
    reshaped_array = np.reshape(data, desired_shape)
    angle_bound = np.tile(np.array([45]), (301, 1))
    delta_xbound = np.tile(np.array([0.1]), (301, 1))

    # lunar lander specific info
    env = LunarLander(enable_wind=True, continuous=True)
    obs_space = env.observation_space
    angle_range = np.asarray([obs_space.low[4], obs_space.high[4]])
    x_range = np.asarray([obs_space.low[0], obs_space.high[0]])
    y_range = np.asarray([obs_space.low[1], obs_space.high[1]])
    delta_x = normalize(np.abs(reshaped_array[:,:,4]), x_range) - normalize(np.abs(0.6 * reshaped_array[:,:,5]), y_range)
    # Timesteps (x-axis)
    delta_x_ch = delta_xbound.T - delta_x
    angle_x_ch =  angle_bound.T  -np.degrees(reshaped_array[:, :, 8]) 
    timesteps = np.arange(301)  
    laplace_filter(delta_x_ch, timesteps=timesteps)

    # plotting begins here 
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for i in range(0, desired_num_rows-offset):
        ax[0].plot(timesteps, np.degrees(reshaped_array[i, :, 8]), label=f'Row {i+1}')
    ax[0].plot(timesteps, angle_bound, label=f'Row {i+1}')
    ax[0].plot(timesteps, -angle_bound, label=f'Row {i+1}')

    # Add labels and legend for angle plot
    ax[0].set_xlabel('Timestep')
    ax[0].set_ylabel('Angle')
    ax[0].text(0, 50, f'{desired_num_rows} data points', fontsize = 12)
    for i in range(desired_num_rows-offset):
        ax[1].plot(timesteps, delta_x[i], label=f'Delta_x Row {i+1}')
    ax[1].plot(timesteps, delta_xbound, label=f'Row {i+1}')

    # Add labels and legend for the delta_x plot
    ax[1].set_xlabel('Timestep')
    ax[1].set_ylabel(f'|x|- 0.6*|y|')
    #plt.savefig(save_file_path)
    #plt.show()

if __name__ == '__main__':
    # csv_file_path = 'traces/trace_nominal_data.csv'
    # plot_data(csv_file_path, 'nominal_figure.png', 4000)
    csv_file_path = 'traces/trace_data_new.csv'
    plot_data(csv_file_path, 'violating_figure.png',0)    

#delta_x = normalize(np.abs(reshaped_array[:,:,4]), x_range) - normalize(np.abs(0.6 * reshaped_array[:,:,5]), y_range)

# this code makes another csv from this csv
# T_value =[]
# rob_delta = []
# states = []
# states_x = []
# states_y = []
# for i in range(desired_num_rows):
#     xy_indices_above_threshold = np.argwhere(delta_x[i] > 0.1)
#     angle_index = np.argwhere(np.abs(np.degrees(reshaped_array[i,:,8])) > 45)
#     rob_delta.append(reshaped_array[i,0,0:4])
#     states.append(delta_x[i][xy_indices_above_threshold[0,0]])
#     states_x.append(reshaped_array[i,xy_indices_above_threshold[0,0],4])
#     states_y.append(reshaped_array[i,xy_indices_above_threshold[0,0],5])
#     #print(angle_index)
#     #print(f'xy : {xy_indices_above_threshold[0,0]}')
#     T_value.append(xy_indices_above_threshold[0,0])

# T_value = np.array(T_value).reshape(len(T_value), 1)
# print(np.mean(T_value))
# print(np.std(T_value))
# rob_delta= np.array(rob_delta)
# states = np.array(states).reshape(len(T_value), 1)
# states_x = np.array(states_x).reshape(len(T_value), 1)
# states_y = np.array(states_y).reshape(len(T_value), 1)
# print(np.mean(states_x))
# print(np.std(states_x))
# print(np.mean(states_y))
# print(np.std(states_y))
# # print(f'shapes : {T_value.shape, rob_delta.shape, states.shape}')
# first_column = rob_delta[:, 0]

# # Extract all columns except the first one
# remaining_columns = rob_delta[:, 1:]
# winds = [0.0, 10.0]
# turbulences = [0.0, 1.0]
# grav = [-12.0, 0.0]
# # calculate distance from nominal here
# new_env = DevLunarLander2(winds, turbulences, grav, (5.0, 0.5, -10.0))
# print(env.observation_space)
# dist = L2Norm(new_env)
# dist_nom = []
# for i in remaining_columns:
#     dist_nom.append(dist.eval_dist(i))
# dist_nom = np.array(dist_nom).reshape(len(T_value), 1)
# # print(dist_nom.shape)
# # Concatenate the remaining columns with the first column at the end
# arr_switched = np.concatenate((remaining_columns, first_column[:, np.newaxis]), axis=1)
# os.makedirs('traces/', exist_ok=True)
# file_path = 'traces/processed_data.csv'
# fin = np.concatenate((arr_switched, T_value), axis=1)
# fin_trace = np.concatenate((fin, states), axis=1)
# final_trace = np.concatenate((fin_trace, dist_nom), axis=1)
# final_trace_1 = np.concatenate((final_trace, states_x), axis=1)
# final_trace_2 = np.concatenate((final_trace_1, states_y), axis=1)

# # Open the CSV file in write mode
# with open(file_path, mode='w', newline='') as file:
#     writer = csv.writer(file)

#     # Write the header
#     writer.writerow(['Wind', 'Turbulence', 'Gravity' , 'Rob', ' T', 'pred', 'dist', 'state_x', 'state_y'])

#     # Write the data rows
#     for row in final_trace_2:
#         writer.writerow(row)

# # Plot delta_x for each row




# # # plotting deltas

# # # winds 
# # ax[2].plot(reshaped_array[:, 0, 1], reshaped_array[:, 0, 0], label='Wind', marker='o', linestyle='')
# # # wind_bound = np.tile(np.array(winds[0]), (desired_num_rows, 1))
# # # wind_bound_2 = np.tile(np.array(winds[1]), (desired_num_rows, 1))
# # # ax[2].plot(range(desired_num_rows), wind_bound, label='Wind')
# # # ax[2].plot(range(desired_num_rows), wind_bound_2, label='Wind')
# # ax[2].set_xlabel('data_point')
# # ax[2].set_ylabel(f'wind_delta')

# # # turbulence
# # ax[3].plot(range(desired_num_rows), reshaped_array[:, 0, 2], label='turbulence')
# # turbulence_bound = np.tile(np.array(turbulences[0]), (desired_num_rows, 1))
# # turbulence_bound_2 = np.tile(np.array(turbulences[1]), (desired_num_rows, 1))
# # ax[3].plot(range(desired_num_rows), turbulence_bound, label='turbulence')
# # ax[3].plot(range(desired_num_rows), turbulence_bound_2, label='turbulence')
# # ax[3].set_xlabel('data_point')
# # ax[3].set_ylabel(f'turbulence_delta')

# # # gravity
# # ax[4].plot(range(desired_num_rows), reshaped_array[:, 0, 3], label='grav')
# # grav_bound = np.tile(np.array(grav[0]), (desired_num_rows, 1))
# # grav_bound_2 = np.tile(np.array(grav[1]), (desired_num_rows, 1))
# # ax[4].plot(range(desired_num_rows), grav_bound, label='grav')
# # ax[4].plot(range(desired_num_rows), grav_bound_2, label='grav')
# # ax[4].set_xlabel('data_point')
# # ax[4].set_ylabel(f'grav_delta')

# plt.tight_layout()
# plt.savefig('trace_analysis.png')
# plt.figure(figsize=(8, 6))

# # Plot the new data
# plt.plot(reshaped_array[:, 0, 1],reshaped_array[:, 0, 0] , label='Windplot', marker='o', linestyle='')


# # Add labels and legend for the standalone plot
# plt.xlabel('Wind delta')
# plt.ylabel('robustness')
# plt.title('Wind Analysis')
# plt.legend()
# plt.savefig('wind.png')

# plt.figure(figsize=(8, 6))

# # Plot the new data
# plt.plot(reshaped_array[:, 0, 2],reshaped_array[:, 0, 0] , label='Turbplot', marker='o', linestyle='')


# # Add labels and legend for the standalone plot
# plt.xlabel('Turbulence delta')
# plt.ylabel('robustness')
# plt.title('Turbulence Analysis')
# plt.legend()
# plt.savefig('turb.png')
# plt.figure(figsize=(8, 6))

# # Plot the new data
# plt.plot(reshaped_array[:, 0, 3],reshaped_array[:, 0, 0] , label='gravplot', marker='o', linestyle='')


# # Add labels and legend for the standalone plot
# plt.xlabel('Grav delta')
# plt.ylabel('robustness')
# plt.title('Grav Analysis')
# plt.legend()



# # Display the plot
# plt.show(block=False)
# plt.savefig('grav.png')

# nominal_env_delta = (5.0, 0.5, -10.0)


