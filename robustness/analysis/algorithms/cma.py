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
        min_v = np.inf
        min_x0 = None
        for _ in range(1 + restarts):
            x, es = cma.fmin2(
                lambda x: self._eval_trace(scale(x, x0_bounds), env, problem.agent),
                lambda: np.random.rand(len(x0_bounds)),
                self.sigma,
                {'bounds': [0.0, 1.0], 'maxfevals': max_evals, 'timeout': timeout * 60, 'verbose': -9},
            )
            v, x0 = es.result.fbest, scale(x, x0_bounds)
            if v < min_v:
                min_v = v
                min_x0 = x0
        env.close()
        return min_v, min_x0


class CMASolver(Solver):
    def __init__(self, sigma, sys_evaluator: SystemEvaluator, opts=None):
        super().__init__(sys_evaluator, opts)
        self.sigma = sigma

    def any_unsafe_deviation(self, problem: Problem, boundary=None, constraints=None):
        delta = None
        dist = np.inf
        logger = {}

        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        dev_bounds = problem.env.get_dev_bounds()

        def eval_sys(delta):
            v, x0 = self.sys_evaluator.eval_sys(delta, problem)
            logger[tuple(delta)] = x0
            return v

        if boundary is not None or constraints is not None:
            all_constraints = lambda delta: (constraints(delta) if constraints is not None else []) + \
                ([problem.dist.eval_dist(delta) - boundary] if boundary is not None else [])

            def random_x0():
                if boundary is None:
                    return np.random.rand(len(problem.env.get_delta_0()))
                else:
                    x0 = normalize(problem.env.get_delta_0(), dev_bounds)
                    return np.clip(np.random.normal(x0, boundary // 2), 0.0, 1.0)
            
            cfun = cma.ConstrainedFitnessAL(
                lambda x: eval_sys(scale(x, dev_bounds)),
                lambda x: all_constraints(scale(x, dev_bounds)),
                find_feasible_first=True
            )
            for _ in range(1 + restarts):
                _, es = cma.fmin2(
                    cfun,
                    random_x0,
                    self.sigma,
                    {'bounds': [0.0, 1.0], 'tolstagnation': 0, 'tolx': 1e-4, 'timeout': timeout * 60,
                    'maxfevals': max_evals},
                    callback=cfun.update,
                )
                print("=============== CMA Results:=================>")
                print(es.result)

                if cfun.best_feas.info is not None and cfun.best_feas.info['f'] < 0:
                    print("=============== CMA Feasible Results: ==============>")
                    print(cfun.best_feas.info)
                    delta = scale(cfun.best_feas.info['x'], dev_bounds)
                    dist = problem.dist.eval_dist(delta)
                    break
        else:
            for _ in range(1 + restarts):
                _, es = cma.fmin2(
                    lambda x: eval_sys(scale(x, dev_bounds)),
                    lambda: np.random.rand(len(problem.env.get_delta_0())),
                    self.sigma,
                    {'bounds': [0.0, 1.0], 'tolstagnation': 0, 'tolx': 1e-4, 'timeout': timeout * 60,
                     'ftarget': 0.0, 'maxfevals': max_evals},
                )
                print("=============== CMA Results:=================>")
                print(es.result)
                if es.result.fbest < 0.0:
                    delta = scale(es.result.xbest, dev_bounds)
                    dist = problem.dist.eval_dist(delta)
                    break
        
        x0 = logger[tuple(delta)] if delta is not None else None
        return delta, dist, x0
    
    def min_unsafe_deviation(self, problem: Problem, boundary=None):
        min_dist = np.inf
        min_delta = None
        logger = {}

        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        dev_bounds = problem.env.get_dev_bounds()

        def eval_sys(delta):
            v, x0 = self.sys_evaluator.eval_sys(delta, problem)
            logger[tuple(delta)] = x0
            return v

        if boundary is not None:
            constraints = lambda delta: [eval_sys(delta),
                                         problem.dist.eval_dist(delta) - boundary]
        else:
            constraints = lambda delta: [eval_sys(delta)]

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
        
        x0 = logger[tuple(min_delta)] if min_delta is not None else None
        return min_delta, min_dist, x0
