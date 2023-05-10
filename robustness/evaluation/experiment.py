import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from robustness.evaluation import Evaluator


class Experiment:
    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
    
    def run_diff_max_samples(self, name, samples, n=10, out_dir='data'):
        data = []
        for s in samples:
            d = self.run_one_max_sample(name, s, n=n, out_dir=out_dir)
            d.index = [[s]*n, d.index]
            data.append(d)
        return pd.concat(data)
    
    def run_one_max_sample(self, name, sample, n=10, out_dir='data'):
        os.makedirs(out_dir, exist_ok=True)

        data_name = f'{out_dir}/{name}-{sample}-{n}.pickle'
        if os.path.exists(data_name):
            with open(data_name, 'rb') as f:
                data = pickle.load(f)
        else:
            tmp = self.evaluator.solver.options()['evals']

            self.evaluator.solver.set_options({'evals': sample})
            data = []
            for _ in range(n):
                start = datetime.now()
                min_delta, min_dist, x0 = self.evaluator.min_violation()
                data.append([min_delta, min_dist, x0, (datetime.now() - start).total_seconds()])
            data = pd.DataFrame(data, columns=['min_delta', 'min_dist', 'x0', 'time'])
            with open(data_name, 'wb') as f:
                pickle.dump(data, f)
            
            self.evaluator.solver.set_options({'evals': tmp})
        
        return data

    def record_min_violations(self, runs=3, out_dir='data'):
        os.makedirs(out_dir, exist_ok=True)

        records = []
        for i in range(runs):
            record_name = f'{out_dir}/records-min-violations-{i}.pickle'
            if os.path.exists(record_name):
                with open(record_name, 'rb') as f:
                    record = pickle.load(f)
            else:
                record = ([], [])
                self.evaluator.min_violation(sample_logger=record)
                with open(record_name, 'wb') as f:
                    pickle.dump(record, f)
            
            records.append(record)
        
        return records
