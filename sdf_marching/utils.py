import numpy as np
from scipy.spatial.distance import cdist

"""
A naive distance function
Replace this with Log-GPIS or others for online mapping
"""
def naive_distance_function(
    obstacle_positions,
    test_positions
):
    dist = cdist(np.atleast_2d(test_positions), obstacle_positions)
    return np.min(dist, axis=-1)
