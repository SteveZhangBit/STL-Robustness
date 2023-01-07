import os
import re
import sys
from os import path

import numpy as np
import pandas as pd


def read_csv(p, f):
    data = pd.read_csv(path.join(p, f), header=None, names=['dist'])
    # mean = data.mean()
    # std = data.std()
    # data = data.iloc[(np.abs(data - mean) < 2 * std).values]
    data['sample'] = int(re.findall('^\w+-dists-\d+-(\d+).csv', f)[0])
    return data

def read_data(p):
    files = filter(lambda x: 'dists' in x, os.listdir(p))
    csvs = [read_csv(p, f) for f in files]
    data = pd.concat(csvs, axis=0)
    data = data.reset_index(drop=True)
    return data

def process_data(p, a, b):
    cma = read_data(path.join(p, a))
    cma['group'] = a
    random = read_data(path.join(p, b))
    random['group'] = b
    data = pd.concat([cma, random], axis=0)
    data.to_csv(path.join(p, f'processed_{a}_{b}.csv'), index=None)


if __name__ == '__main__':
    process_data(sys.argv[1], sys.argv[2], sys.argv[3])
