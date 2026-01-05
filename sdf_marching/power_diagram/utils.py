import numpy as np

# --- Misc. geometry code -----------------------------------------------------

'''
Pick N points uniformly from the unit disc
This sampling algorithm does not use rejection sampling.
'''
def disc_uniform_pick(N, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    angle = (2 * np.pi) * rng.uniform(size=N)
    out = np.stack([np.cos(angle), np.sin(angle)], axis = 1)
    out *= np.sqrt(rng.uniform(size=N))[:,None]
    return out

def disc_uniform_pick_nd(N, ndim=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    angle = (2 * np.pi) * rng.uniform(size=(N, ndim-1))
    
    out = np.zeros(
        (N, ndim)
    )
    out[:, 0] = np.sqrt(rng.uniform(size=N))

    for dim in range(ndim-1):
        out[:, dim+1] = out[:, dim] * np.sin(angle[:, dim])
        out[:, dim] *= np.cos(angle[:, dim])
    return out

def norm2(X):
	return np.sqrt(np.sum(X ** 2))

def normalized(X):
	return X / norm2(X)
