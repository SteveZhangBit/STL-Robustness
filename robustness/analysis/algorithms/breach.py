import os
import numpy as np

from robustness.analysis import *


class BreachSystemEvaluator(SystemEvaluator):
    def __init__(self, eng, phi: TraceEvaluator, opts=None):
        super().__init__(phi, opts)
        self.eng = eng
        self.all_signal_values = None
        self.violation_signal_values = None
        self.all_param_values = None
        self.violation_param_values = None
        self.obj_values = None
    
    def eval_sys(self, delta, problem: Problem):
        trials = self._options['restarts'] + 1
        max_evals = self._options['evals']

        env = problem.env.instantiate(delta, problem.agent)

        self.eng.addpath(os.path.dirname(__file__))
        obj_best, obj_values, all_signal_values, violation_signal_values, all_param_values, violation_param_values = \
            self._breach_falsification(env, trials, max_evals)
        
        self.obj_values = np.array(obj_values)[0]
        self.all_signal_values = all_signal_values
        self.violation_signal_values = violation_signal_values
        self.all_param_values = {k: np.array(all_param_values[k])[0] for k in all_param_values.keys()}
        self.violation_param_values = {k: np.array(violation_param_values[k])[0] for k in violation_param_values.keys()}
        return obj_best, None

    def _breach_falsification(self, env, trials, max_evals):
        return self.eng.breach_falsification(env, str(self.phi), trials, max_evals, nargout=6)
    
    def get_obj_values(self):
        return self.obj_values
    
    def get_all_traces(self):
        return self.all_signal_values
    
    def get_violating_traces(self):
        return self.violation_signal_values
    
    def get_all_params(self):
        return self.all_param_values
    
    def get_violating_params(self):
        return self.violation_param_values

    def get_params(self, name):
        return self.all_param_values[name]


class BreachOneLayerSystemEvaluator(BreachSystemEvaluator):
    def _breach_falsification(self, env, trials, max_evals):
        return self.eng.one_layer_breach_falsification(env, str(self.phi), trials, max_evals, nargout=6)
