import cma
import numpy as np

from robustness.analysis import *
from robustness.analysis.utils import normalize, scale


class CMASystemEvaluator(SystemEvaluator):
    def __init__(self, sigma, phi: TraceEvaluator, opts=None):
        super().__init__(phi, opts)
        self.sigma = sigma
    
    def eval_sys(self, delta, problem: Problem):
        timeout = self._options['timeout']
        restarts = self._options['restarts']
        max_evals = self._options['evals']

        env, x0_bounds = problem.env.instantiate(delta)
        x, es = cma.fmin2(
            lambda x: self._eval_trace(scale(x, x0_bounds), env, problem.agent),
            lambda: np.random.rand(len(x0_bounds)),
            self.sigma,
            {'bounds': [0.0, 1.0], 'maxfevals': max_evals, 'timeout': timeout * 60, 'verbose': -9},
            restarts=restarts,
        )
        env.close()
        return es.result.fbest, scale(x, x0_bounds)


class CMASolver(Solver):
    def __init__(self, sigma, sys_evaluator: SystemEvaluator, opts=None):
        super().__init__(sys_evaluator, opts)
        self.sigma = sigma

    def any_unsafe_deviation(self, problem: Problem, boundary=None):
        delta = None
        dist = np.inf

        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        dev_bounds = problem.env.get_dev_bounds()
        if boundary is not None:
            constraints = lambda delta: [problem.dist.eval_dist(delta) - boundary]

            def random_x0():
                x0 = normalize(problem.env.get_delta_0(), dev_bounds)
                return np.clip(np.random.normal(x0, boundary / 2), 0.0, 1.0)
            
            cfun = cma.ConstrainedFitnessAL(
                lambda x: self.sys_evaluator.eval_sys(scale(x, dev_bounds))[0],
                lambda x: constraints(scale(x, dev_bounds)),
                find_feasible_first=True
            )
            _, es = cma.fmin2(
                cfun,
                random_x0,
                self.sigma,
                {'bounds': [0.0, 1.0], 'tolstagnation': 0, 'tolx': 1e-4, 'timeout': timeout * 60,
                 'maxfevals': max_evals},
                callback=cfun.update,
                restarts=restarts
            )
            print("=============== CMA Results:=================>")
            print(es.result)

            if cfun.best_feas.info is not None and cfun.best_feas.info['f'] < 0:
                print("=============== CMA Feasible Results: ==============>")
                print(cfun.best_feas.info)
                delta = scale(cfun.best_feas.info['x'], dev_bounds)
                dist = problem.dist.eval_dist(delta)
        else:
            _, es = cma.fmin2(
                lambda x: self.sys_evaluator.eval_sys(scale(x, dev_bounds), problem)[0],
                lambda: np.random.rand(len(problem.env.get_delta_0())),
                self.sigma,
                {'bounds': [0.0, 1.0], 'tolstagnation': 0, 'tolx': 1e-4, 'timeout': timeout * 60,
                 'ftarget': 0.0, 'maxfevals': max_evals},
                restarts=restarts,
            )
            print("=============== CMA Results:=================>")
            print(es.result)
            if es.result.fbest < 0.0:
                delta = scale(es.result.xbest, dev_bounds)
                dist = problem.dist.eval_dist(delta)
        
        return delta, dist
    
    def min_unsafe_deviation(self, problem: Problem, boundary=None):
        min_dist = np.inf
        min_delta = None

        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        dev_bounds = problem.env.get_dev_bounds()

        if boundary is not None:
            constraints = lambda delta: [self.sys_evaluator.eval_sys(delta, problem)[0],
                                         problem.dist.eval_dist(delta) - boundary]
        else:
            constraints = lambda delta: [self.sys_evaluator.eval_sys(delta, problem)[0]]

        def random_x0():
            x0 = normalize(problem.env.get_delta_0(), dev_bounds)
            return np.clip(np.random.normal(x0, self.sigma), 0.0, 1.0)

        for _ in range(1 + restarts):
            cfun = cma.ConstrainedFitnessAL(
                lambda x: problem.dist.eval_dist(scale(x, dev_bounds)),
                lambda x: constraints(scale(x, dev_bounds)),
                find_feasible_first=True,
            )
            es = cma.CMAEvolutionStrategy(
                random_x0(),
                self.sigma,
                {'bounds': [0.0, 1.0], 'tolstagnation': 0, 'tolx': 1e-4, 'timeout': timeout * 60,
                 'maxfevals': max_evals},
            )
            while not es.stop():
                X = es.ask()
                es.tell(X, [cfun(x) for x in X])
                cfun.update(es)
            
            print("=============== CMA Results: ===============>")
            print(es.result)

            if cfun.best_feas.info is not None:
                print("=============== CMA Feasible Results: ==============>")
                print(cfun.best_feas.info)
                delta = scale(cfun.best_feas.info['x'], dev_bounds)
                delta_dist = problem.dist.eval_dist(delta)
                if delta_dist < min_dist:
                    min_dist = delta_dist
                    min_delta = delta
        
        return min_delta, min_dist
