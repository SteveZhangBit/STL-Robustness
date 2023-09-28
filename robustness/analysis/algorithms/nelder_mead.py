from scipy.optimize import minimize
import numpy as np

from robustness.analysis import *
from robustness.analysis.utils import normalize, scale



class Nelder_MeadSolver(Solver):
    def __init__(self, sigma, sys_evaluator: SystemEvaluator, opts=None):
        super().__init__(sys_evaluator, opts)
        self.sigma = sigma
    
    # def display_progress(self, x):
    #     global iteration
    #     iteration += 1
    #     # constraints = lambda delta: [eval_sys(delta)]
    #     print(f"Iteration {iteration}")

    def min_unsafe_deviation(self, problem: Problem, boundary=None, sample_logger=None):
        min_dist = np.inf
        min_delta = None
        logger = {}
        # print('\n\n\n\n')
        #iteration = 0
        restarts = self._options['restarts']
        timeout = self._options['timeout']
        max_evals = self._options['evals']

        dev_bounds = problem.env.get_dev_bounds()

        def eval_sys(delta):
            v, x0 = self.sys_evaluator.eval_sys(delta, problem)
            logger[tuple(delta)] = (v, x0)
            return v

        if boundary is not None:
            constraints = lambda delta: [eval_sys(delta),
                                         problem.dist.eval_dist(delta) - boundary]
        else:
            constraints = lambda delta: [eval_sys(delta)]

        def random_x0():
            x0 = normalize(problem.env.get_delta_0(), dev_bounds)
            return np.clip(np.random.normal(x0, self.sigma), 0.0, 1.0)
        # edit this function below to add the two objectives which would now be a joint objective but rest remains the same
        for _ in range(1 + restarts):
            # cfun = cma.ConstrainedFitnessAL(
            #     lambda x: problem.dist.eval_dist(scale(x, dev_bounds)),
            #     lambda x: constraints(scale(x, dev_bounds)),
            #     find_feasible_first=True,
            # )
            objective = lambda x: problem.dist.eval_dist(scale(x, dev_bounds)) + constraints(scale(x, dev_bounds))
            initial_guess = random_x0() 
            tolerance = 1e-6
            options = {'disp': True}

            result = minimize(objective, initial_guess, method='Nelder-Mead', tol=tolerance, options=options)
            # print("Optimal solution:", result.x)
            # print("Optimal objective value:", result.fun)
            # es = cma.CMAEvolutionStrategy(
            #     random_x0(),
            #     self.sigma,
            #     {'bounds': [0.0, 1.0], 'tolstagnation': 0, 'tolx': 1e-4, 'timeout': timeout * 60,
            #      'maxfevals': max_evals},
            # )
            # while not es.stop():
            #     X = es.ask()
            #     es.tell(X, [cfun(x) for x in X])
            #     cfun.update(es)
            
            print("=============== NM Results: ===============>")
            print("Optimal solution:", result.x)
            print("Optimal objective value:", result.fun)
            delta = scale(result.x, dev_bounds)
            delta_dist = problem.dist.eval_dist(delta)
            if delta_dist < min_dist:
                min_dist = delta_dist
                min_delta = delta

            # if cfun.best_feas.info is not None:
            #     print("=============== CMA Feasible Results: ==============>")
            #     print(cfun.best_feas.info)
            #     delta = scale(cfun.best_feas.info['x'], dev_bounds)
            #     delta_dist = problem.dist.eval_dist(delta)
            #     if delta_dist < min_dist:
            #         min_dist = delta_dist
            #         min_delta = delta
        
        if sample_logger is not None:
            for (k, v) in logger.items():
                sample_logger[0].append(k)
                sample_logger[1].append(v[0])
        
        _, x0 = logger[tuple(min_delta)] if min_delta is not None else (None, None)
        return min_delta, min_dist, x0
