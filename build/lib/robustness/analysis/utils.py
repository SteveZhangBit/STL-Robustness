import numpy as np

from robustness.analysis import DistanceEvaluator


def scale(x, bounds):
    """Scale the input numbers in [0, 1] to the range of each variable"""
    if bounds.ndim == 1:
        return bounds[0] + x * (bounds[1] - bounds[0])
    else:
        return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])


def normalize(x_scaled, bounds):
    if bounds.ndim == 1:
        return (x_scaled - bounds[0]) / (bounds[1] - bounds[0])
    else:
        return (x_scaled - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


class L2Norm(DistanceEvaluator):
    def eval_dist(self, delta):
        delta = normalize(delta, self.env.get_dev_bounds())
        delta_0 = normalize(self.env.get_delta_0(), self.env.get_dev_bounds())
        return np.sqrt( np.sum((delta - delta_0) ** 2) )
