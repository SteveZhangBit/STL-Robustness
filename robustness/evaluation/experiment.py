import os
from datetime import datetime

import numpy as np
import pandas as pd

from robustness.evaluation import Evaluator


class Experiment:
    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
    
    def run_diff_max_samples(self, name, samples, n=10, out_dir='data'):
        return [
            self.run_one_max_sample(name, s, n=n, out_dir=out_dir)
            for s in samples
        ]
    
    def run_one_max_sample(self, name, sample, n=10, out_dir='data'):
        os.makedirs(out_dir, exist_ok=True)

        data_name = f'{out_dir}/{name}-{sample}-{n}.csv'
        if os.path.exists(data_name):
            data = pd.read_csv(data_name)
        else:
            tmp = self.evaluator.solver.options()['evals']

            self.evaluator.solver.set_options({'evals': sample})
            data = []
            for _ in range(n):
                start = datetime.now()
                min_delta, min_dist, x0 = self.evaluator.min_violation()
                data.append([min_delta, min_dist, x0, (datetime.now() - start).total_seconds()])
            data = pd.DataFrame(data, columns=['min_delta', 'min_dist', 'x0', 'time'])
            data.to_csv(data_name)
            
            self.evaluator.solver.set_options({'evals': tmp})
        
        return data
