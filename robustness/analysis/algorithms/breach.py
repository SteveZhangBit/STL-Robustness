import os
import numpy as np

from robustness.analysis import *
from robustness.analysis.utils import compute_cosine_similarity


class BreachSystemEvaluator(SystemEvaluator):
    def __init__(self, eng, phi: TraceEvaluator, opts=None):
        super().__init__(phi, opts)
        self.eng = eng
        self.all_signal_values = None
        # self.violation_signal_values = None
        self.all_param_values = None
        # self.violation_param_values = None
        self.obj_values = None
        self.obj_best_signal_values = None
    
    def eval_sys(self, delta, problem: Problem):
        trials = self._options['restarts'] + 1
        max_evals = self._options['evals']

        env = problem.env.instantiate(delta, problem.agent)

        self.eng.addpath(os.path.dirname(__file__))
        obj_best, obj_values, obj_best_signal_values, all_signal_values, all_param_values = \
            self._breach_falsification(env, trials, max_evals)
        
        self.obj_values = np.array(obj_values)
        self.obj_best_signal_values = {k: np.squeeze(obj_best_signal_values[k]) for k in obj_best_signal_values.keys()}
        self.all_signal_values = {k: np.array(all_signal_values[k]) for k in all_signal_values.keys()}
        # self.violation_signal_values = violation_signal_values
        self.all_param_values = {k: np.array(all_param_values[k]) for k in all_param_values.keys()}
        # self.violation_param_values = {k: np.array(violation_param_values[k]) for k in violation_param_values.keys()}
        return obj_best, None

    def _breach_falsification(self, env, trials, max_evals):
        return self.eng.breach_falsification(env, str(self.phi), trials, max_evals, nargout=5)
    
    def get_obj_values(self):
        return self.obj_values
    
    def get_all_traces(self):
        return self.all_signal_values
    
    # def get_violating_traces(self):
    #     return self.violation_signal_values
    
    def get_all_params(self):
        return self.all_param_values
    
    # def get_violating_params(self):
    #     return self.violation_param_values

    def get_params(self, name):
        return self.all_param_values[name]


class BreachSystemEvaluatorWithHeuristic(BreachSystemEvaluator):
    def __init__(self, eng, phi: TraceEvaluator, signal_names, delta_0_signals, opts=None):
        super().__init__(eng, phi, opts)
        self.signal_names = signal_names
        self.delta_0_signals = np.array([delta_0_signals[s] for s in signal_names]).T
        self.obj_best = None
    
    def eval_sys(self, delta, problem: Problem):
        obj_best, _ = super().eval_sys(delta, problem)
        self.obj_best = obj_best
        delta_signals = np.array([self.obj_best_signal_values[s] for s in self.signal_names])
        delta_signals = delta_signals.T
        cos_similarity = compute_cosine_similarity(self.delta_0_signals, delta_signals)
        return obj_best + cos_similarity, None


class BreachOneLayerSystemEvaluator(BreachSystemEvaluator):
    def _breach_falsification(self, env, trials, max_evals):
        return self.eng.one_layer_breach_falsification(env, str(self.phi), trials, max_evals, nargout=5)
