import numpy as np
import pandas as pd


data_dirs = [
    'cartpole-pid',
    'cartpole-dqn',
    'cartpole4-pid',
    'lunar-lander-lqr',
    'lunar-lander-ppo',
    'car-circle-ppo',
    'car-run-ppo',
    'ACC/RL',
    'ACC/traditional',
    'AFC/RL',
    'AFC/traditional',
    'WTK/RL',
    'WTK/traditional',
]

names = [
    'cartpole-pid',
    'cartpole-dqn',
    'cartpole4-pid',
    'lunar-lander-lqr',
    'lunar-lander-ppo',
    'car-circle-ppo',
    'car-run-ppo',
    'ACC-RL',
    'ACC-traditional',
    'AFC-RL',
    'AFC-traditional',
    'WTK-RL',
    'WTK-traditional',
]

def read_data(data_dir):
    data = pd.read_csv(data_dir)
    data['Violations/Total'] = data.apply(lambda row: f"{row['Num of violations']:.0f}/{row['Num of samples']:.0f}", axis=1)
    data['Violation Distance'] = data.apply(
        lambda row: f"{row['Mean distance']:.2f}$\\pm${row['Std distance']:.2f} ({row['Min distance']:.2f}-{row['Max distance']:.2f})",
        axis=1)
    return data

cma_datas = [read_data(f'data/{data_dir}/cma/summary.csv') for data_dir in data_dirs]
cma_data = pd.concat(cma_datas, keys=names)
cma_data.index.names = ['Problem', 'Trial']
print(cma_data)
cma_data[['Violations/Total', 'Violation distance', 'Total time (s)']].to_csv('data/cma-summary.csv', float_format='%.2f')

random_datas = [read_data(f'data/{data_dir}/random/summary.csv') for data_dir in data_dirs]
random_data = pd.concat(random_datas, keys=names)
random_data.index.names = ['Problem', 'Trial']
print(random_data)
random_data[['Violations/Total', 'Violation distance', 'Total time (s)']].to_csv('data/random-summary.csv', float_format='%.2f')
