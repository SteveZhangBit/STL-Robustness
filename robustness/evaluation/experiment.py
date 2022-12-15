import os
from datetime import datetime

import numpy as np

from robustness.evaluation import Evaluator


class Experiment:
    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
    
    def run_diff_max_samples(self, name, samples, n=10, out_dir='data'):
        data = []
        data_times = []
        for s in samples:
            dists, times = self.run_one_max_sample(name, s, n=n, out_dir=out_dir)
            data.append(dists)
            data_times.append(times)
        return data, data_times
    
    def run_one_max_sample(self, name, sample, n=10, out_dir='data'):
        dists_name = f"{out_dir}/{name}-dists-{n}-{sample}.csv"
        times_name = f"{out_dir}/{name}-times-{n}-{sample}.csv"
        if os.path.exists(dists_name) and os.path.exists(times_name):
            dists = np.loadtxt(dists_name, delimiter=',')
            times = np.loadtxt(times_name, delimiter=',')
        else:
            tmp = self.evaluator.solver.options()['evals']
            self.evaluator.solver.set_options({'evals': sample})
            dists, times = [], []
            for _ in range(n):
                start = datetime.now()
                _, dist = self.evaluator.min_violation()
                dists.append(dist)
                times.append((datetime.now() - start).total_seconds())

            dists, times = np.array(dists), np.array(times)
            np.savetxt(dists_name, dists, delimiter=',')
            np.savetxt(times_name, times, delimiter=',')
            
            self.evaluator.solver.set_options({'evals': tmp})
        
        return dists, times
