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

def compute_cosine_similarity(trace1, trace2):
    '''
    Computes the cosine similarity between two traces of shape n*t.
    
    Parameters:
        trace1 (np.ndarray): First trace matrix of shape (n, t).
        trace2 (np.ndarray): Second trace matrix of shape (n, t).
        
    Returns:
        float: Cosine similarity between the two traces.
    '''
    # Flatten the traces
    trace1_flat = trace1.flatten()
    trace2_flat = trace2.flatten()
    
    # Compute the dot product
    dot_product = np.dot(trace1_flat, trace2_flat)
    
    # Compute the norms
    norm_trace1 = np.linalg.norm(trace1_flat)
    norm_trace2 = np.linalg.norm(trace2_flat)
    
    # Compute cosine similarity
    cosine_similarity = dot_product / (norm_trace1 * norm_trace2)
    return cosine_similarity

class L2Norm(DistanceEvaluator):
    def eval_dist(self, delta):
        delta = normalize(delta, self.env.get_dev_bounds())
        delta_0 = normalize(self.env.get_delta_0(), self.env.get_dev_bounds())
        return np.sqrt( np.sum((delta - delta_0) ** 2) )
