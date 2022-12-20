import os
import re
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

def process_data(p):
    cma = read_data(path.join(p, 'cma'))
    cma['group'] = 'cma'
    random = read_data(path.join(p, 'random'))
    random['group'] = 'random'
    data = pd.concat([cma, random], axis=0)
    data.to_csv(path.join(p, 'processed.csv'), index=None)
