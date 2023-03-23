import os

from robustness.analysis import *


class BreachSystemEvaluator(SystemEvaluator):
    def __init__(self, eng, phi: TraceEvaluator, opts=None):
        super().__init__(phi, opts)
        self.eng = eng
    
    def eval_sys(self, delta, problem: Problem):
        trials = self._options['restarts'] + 1
        max_evals = self._options['evals']

        env = problem.env.instantiate(delta, problem.agent)

        self.eng.addpath(os.path.dirname(__file__))
        v = self.eng.breach_falsification(env, str(self.phi), trials, max_evals)
        return v, None
