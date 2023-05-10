from datetime import datetime

import numpy as np

from robustness.analysis import *


class ExpectationSysEvaluator(SystemEvaluator):
    def eval_sys(self, delta, problem: Problem):
        restarts = self._options['restarts']
        timeout = self._options['timeout'] * (1 + restarts)
        max_evals = self._options['evals'] * (1 + restarts)

        env, x0_bounds = problem.env.instantiate(delta)
        vs = []
        start = datetime.now()
        for _ in range(max_evals):
            v = self._eval_trace(
                np.random.uniform(low=x0_bounds[:, 0], high=x0_bounds[:, 1]),
                env,
                problem.agent
            )
            vs.append(v)
            if (datetime.now() - start).total_seconds() > timeout * 60:
                break
        env.close()
        return np.mean(vs), None


class RandomSolver(Solver):
    def any_unsafe_deviation(self, problem: Problem, boundary=None, constraints=None):
        assert boundary is None, "RandomSolver does not support boundary."
        assert constraints is None, "RandomSolver does not support constraints."

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
                constraint = self.sys_evaluator.eval_sys(delta, problem)[0]
                if constraint < 0:
                    return delta, dist
                if (datetime.now() - start).total_seconds() > timeout * 60:
                    break
        
        return delta, dist, None
    
    def min_unsafe_deviation(self, problem: Problem, boundary=None, sample_logger=None):
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
                constraint = self.sys_evaluator.eval_sys(delta, problem)[0]

                if sample_logger is not None:
                    sample_logger[0].append(delta)
                    sample_logger[1].append(constraint)

                if constraint < 0 and dist < min_dist:
                    min_dist = dist
                    min_delta = delta
                if (datetime.now() - start).total_seconds() > timeout * 60:
                    break
        
        return min_delta, min_dist, None
