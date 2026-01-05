import numpy as np

# assume vec has positions across last dimensions. normalise
def normalise(vec):
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

