import numpy as np

def __euclidean(vector, **kwargs):
    return np.linalg.norm(vector, **kwargs)

def __angle(vector, **kwargs):
    # add wrap-around here.
    return np.linalg.norm(vector, **kwargs)

# TODO: make this selectable.
norm = __euclidean
print("Using Euclidean distance")