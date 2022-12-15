from datetime import datetime

import numpy as np

from robustness.analysis import *


class RandomSolver(Solver):
    def any_unsafe_deviation(self, problem: Problem, boundary=None):
        dist = np.inf
        delta = None

        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        dev_bounds = problem.env.get_dev_bounds()
        for _ in range(1 + restarts):
            start = datetime.now()
            for _ in range(max_evals):
                delta = np.random.uniform(dev_bounds[:, 0], dev_bounds[:, 1])
                dist = problem.dist.eval_dist(delta)
                constraint = self.sys_evaluator.eval_sys(delta)[0]
                if constraint < 0:
                    return delta, dist
                if (datetime.now() - start).total_seconds() > timeout * 60:
                    break
        
        return delta, dist
    
    def min_unsafe_deviation(self, problem: Problem, boundary=None):
        min_dist = np.inf
        min_delta = None

        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        dev_bounds = problem.env.get_dev_bounds()
        for _ in range(1 + restarts):
            start = datetime.now()
            for _ in range(max_evals):
                delta = np.random.uniform(dev_bounds[:, 0], dev_bounds[:, 1])
                dist = problem.dist.eval_dist(delta)
                constraint = self.sys_evaluator.eval_sys(delta)[0]
                if constraint < 0 and dist < min_dist:
                    min_dist = dist
                    min_delta = delta
                if (datetime.now() - start).total_seconds() > timeout * 60:
                    break
        
        return min_delta, min_dist
