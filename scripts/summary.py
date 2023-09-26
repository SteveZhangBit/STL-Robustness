import numpy as np
import pandas as pd


def read_data(data_dir):
    data = pd.read_csv(data_dir)
    data['Violations/Total'] = data.apply(lambda row: f"{row['Num of violations']:.0f}/{row['Num of samples']:.0f}", axis=1)
    data['Violation Distance'] = data.apply(
        lambda row: f"{row['Mean distance']:.2f}$\\pm${row['Std distance']:.2f} ({row['Min distance']:.2f}-{row['Max distance']:.2f})",
        axis=1)
    return data

def summarize(data_dirs, names, kind, out):
    datas = [read_data(f'data/{data_dir}/{kind}/summary.csv') for data_dir in data_dirs]
    data = pd.concat(datas, keys=names)
    data.index.names = ['Problem', 'Trial']
    print(data)
    data[['Violations/Total', 'Violation Distance', 'Total time (s)']].to_csv(out, float_format='%.2f')

data_dirs = [
    'cartpole-pid',
    'cartpole-dqn',
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

summarize(data_dirs, names, 'cma', 'data/cma-summary.csv')
summarize(data_dirs, names, 'random', 'data/random-summary.csv')

data_dirs = [
    'cartpole4-pid',
    'cartpole4-dqn',
    'lunarlander3-lqr',
    'lunarlander3-ppo',
    'car-circle3-ppo',
    'ACC3/traditional',
    'ACC3/RL',
]

names = [
    'cartpole4-pid',
    'cartpole4-dqn',
    'lunarlander3-lqr',
    'lunarlander3-ppo',
    'car-circle3-ppo',
    'ACC3-traditional',
    'ACC3-RL',
]

summarize(data_dirs, names, 'cma', 'data/cma-summary-2.csv')
summarize(data_dirs, names, 'random', 'data/random-summary-2.csv')
