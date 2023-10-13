import os

from robustness.analysis import *


class BreachSystemEvaluator(SystemEvaluator):
    def __init__(self, eng, phi: TraceEvaluator, opts=None):
        super().__init__(phi, opts)
        self.eng = eng
        self.all_signal_values = None
        self.violation_signal_values = None
    
    def eval_sys(self, delta, problem: Problem):
        trials = self._options['restarts'] + 1
        max_evals = self._options['evals']

        env = problem.env.instantiate(delta, problem.agent)

        self.eng.addpath(os.path.dirname(__file__))
        obj_best, all_signal_values, violation_signal_values = self.eng.breach_falsification(env, str(self.phi), trials, max_evals, nargout=3)
        self.all_signal_values = all_signal_values
        self.violation_signal_values = violation_signal_values
        return obj_best, None
    
    def get_all_traces(self):
        return self.all_signal_values
    
    def get_violating_traces(self):
        return self.violation_signal_values
