import numpy as np
import pygmo as pg

from robustness.analysis import Problem, Solver, SystemEvaluator


class MOMinDev:
    def __init__(self, problem: Problem, sys_evaluator: SystemEvaluator):
        self.problem = problem
        self.sys_evaluator = sys_evaluator
    
    def fitness(self, delta):
        objective = self.problem.dist.eval_dist(delta)
        stl = self.sys_evaluator.eval_sys(delta, self.problem)[0]
        return [objective, stl]

    def get_bounds(self):
        return self.problem.env.get_dev_bounds()
    
    # Return number of objectives
    def get_nobj(self):
        return 2


class NSGA2Solver(Solver):
    def __init__(self, gen, popsize, sys_evaluator: SystemEvaluator, opts=None):
        super().__init__(sys_evaluator, opts)
        self.gen = gen
        self.popsize = popsize
    
    def min_unsafe_deviation(self, problem: Problem, boundary=None, logger=None):
        p = pg.problem(MOMinDev(problem, self.sys_evaluator))
        algo = pg.algorithm(pg.nsga2(gen=self.gen, m=0.1))
        pop = pg.population(p, size=self.popsize)

        pop = algo.evolve(pop)
        fits, vectors = pop.get_f(), pop.get_x()
        feasible = fits[:, 1] < 0
        if feasible.sum() == 0:
            min_dist = np.inf
            min_delta = None
        else:
            fits, vectors = fits[feasible], vectors[feasible]
            min_idx = fits[:, 0].argmin()
            min_dist = fits[min_idx, 0]
            min_delta = vectors[min_idx]
        
        return min_delta, min_dist


class ConstrainedMinDev:
    def __init__(self, problem: Problem, sys_evaluator: SystemEvaluator):
        self.problem = problem
        self.sys_evaluator = sys_evaluator
    
    def fitness(self, delta):
        objective = self.problem.dist.eval_dist(delta)
        stl = self.sys_evaluator.eval_sys(delta, self.problem)[0]
        return [objective, stl]

    def get_bounds(self):
        return self.problem.env.get_dev_bounds()
    
    # Inequality Constraints
    def get_nic(self):
        return 1

    # Equality Constraints
    def get_nec(self):
        return 0


class GACO(Solver):
    def __init__(self, gen, popsize, sys_evaluator: SystemEvaluator, opts=None):
        super().__init__(sys_evaluator, opts)
        self.gen = gen
        self.popsize = popsize
    
    def min_unsafe_deviation(self, problem: Problem, boundary=None, logger=None):
        p = pg.problem(ConstrainedMinDev(problem, self.sys_evaluator))
        algo = pg.algorithm(pg.gaco(gen=self.gen, ker=self.popsize))
        pop = pg.population(p, size=self.popsize)

        pop = algo.evolve(pop)
        return pop.champion_x, pop.champion_f[0]
