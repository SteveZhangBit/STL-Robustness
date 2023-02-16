import numpy as np
import signal_tl as stl

from robustness.analysis import TraceEvaluator


class STLEvaluator(TraceEvaluator):
    def __init__(self, pickle_safe=False):
        self._phi = None
        self.pickle_safe = pickle_safe

    def prop(self):
        '''
        Returns the STL property specified by the signal_tl library.
        '''
        raise NotImplementedError()
    
    def build_signal(self, record, time_index):
        '''
        Returns the signal dictionary from the observation records and
        time index.
        '''
        raise NotImplementedError()
    
    def _compute_rob(self, record):
        time_index = np.arange(len(record))
        signal = self.build_signal(record, time_index)
        if self.pickle_safe:
            rob = stl.compute_robustness(self.prop(), signal)
        else:
            if self._phi is None:
                self._phi = self.prop()
            rob = stl.compute_robustness(self._phi, signal)
        return rob.at(0)

    def eval_trace(self, obs_record, reward_record):
        return self._compute_rob(obs_record)
    
    def __str__(self) -> str:
        return 'STL'


class STLEvaluator2(TraceEvaluator):
    '''
    Deprecated. This is used for our custom STL metric that cumulates the positive robustness
    and negative robustness for safety properties only.
    '''
    def eval_trace(self, obs_record, reward_record):
        rob = np.array([self.eval_one_timepoint(x) for x in obs_record])
        positive = rob[rob > 0].sum()
        negative = rob[rob < 0].sum()
        return negative if negative < 0 else positive
    
    def eval_one_timepoint(self, obs):
        raise NotImplementedError()
