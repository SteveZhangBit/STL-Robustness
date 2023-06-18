import os
import pickle
from datetime import datetime
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from robustness.analysis.utils import normalize

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
                record = [[], [], None]
                start_time = datetime.now()
                self.evaluator.min_violation(sample_logger=record)
                record[2] = (datetime.now() - start_time).total_seconds()
                with open(record_name, 'wb') as f:
                    pickle.dump(record, f)
            
            records.append(record)
        
        return records

    def summarize_violations(self, records):
        print('Total time of sampling: ', [record[2] for record in records])
        records = [
            [
                (X, Y, self.evaluator.problem.dist.eval_dist(X))
                for (X, Y) in zip(record[0], record[1])
            ]
            for record in records
        ]
        violations = [[r for r in record if r[1] < 0.0] for record in records]
        
        print('Number of violations found: ', [len(record) for record in violations])
        print('Minimum violation found: ',
              [np.min([z for (_, _, z) in record]) if len(record) > 0 else None for record in violations])
        print('Maximum violation found: ',
              [np.max([z for (_, _, z) in record]) if len(record) > 0 else None for record in violations])
        print('Average violation found: ',
              [np.mean([z for (_, _, z) in record]) if len(record) > 0 else None for record in violations])

        return records, violations

    def plot_samples(self, samples, x_name, y_name, out_dir, n, **kwargs):
        dev_bounds = self.evaluator.problem.env.get_dev_bounds()

        plt.figure()
        self.evaluator.heatmap(
            dev_bounds[0], dev_bounds[1], n, n,
            x_name=x_name, y_name=y_name, z_name="System Evaluation $\Gamma$",
            out_dir=out_dir,
            **kwargs
        )
        points = np.array([normalize(X, dev_bounds) for (X, Y) in samples if Y >= 0.0])
        plt.scatter(points[:, 0] * (n-1), points[:, 1] * (n-1), c=np.arange(len(points)),
                    cmap='Greys', marker='x', s=50)

        points = np.array([normalize(X, dev_bounds) for (X, Y) in samples if Y < 0.0])
        if len(points) > 0:
            plt.scatter(points[:, 0] * (n-1), points[:, 1] * (n-1), c='yellow', marker='x', s=100)
