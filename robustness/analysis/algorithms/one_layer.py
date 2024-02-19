import numpy as np
from datetime import datetime
from robustness.analysis.core import Problem, Solver


class OneLayerSolver(Solver):
    def __init__(self, sys_evaluator, param_names, opts=None):
        super().__init__(sys_evaluator, opts)
        self.param_names = param_names

    def min_unsafe_deviation(self, problem: Problem, boundary=None, sample_logger=None):
        min_dist = np.inf
        min_delta = None

        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        for _ in range(1 + restarts):
            start = datetime.now()
            for _ in range(max_evals):
                delta, dist, constraint = self._run_one_layer(problem)

                if sample_logger is not None:
                    sample_logger[0].append(delta)
                    sample_logger[1].append(constraint)

                if dist < min_dist:
                    min_dist = dist
                    min_delta = delta

                if (datetime.now() - start).total_seconds() > timeout * 60:
                    break
        
        return min_delta, min_dist, None
    
    def _run_one_layer(self, problem: Problem):
        self.sys_evaluator.eval_sys(problem.env.get_delta_0(), problem)
        obj_values = self.sys_evaluator.get_obj_values()
        deltas = np.array([[self.sys_evaluator.get_params(n)[i] for n in self.param_names] for i in range(len(obj_values))])
        distances = np.array([problem.dist.eval_dist(delta) for delta in deltas])

        obj_values -= distances
        negative_obj_idx = np.where(obj_values < 0)
        obj_with_negative = obj_values[negative_obj_idx]

        if len(obj_with_negative) == 0:
            min_idx = np.argmin(obj_values)
            return deltas[min_idx], distances[min_idx], obj_values[min_idx]
        else:
            deltas_with_negative = deltas[negative_obj_idx]
            distances_with_negative = distances[negative_obj_idx]
            
            min_idx = np.argmin(distances_with_negative)
            return deltas_with_negative[min_idx], distances_with_negative[min_idx], obj_with_negative[min_idx]
