import jax
import jax.numpy as jnp
from scipy.io import loadmat

def load_data_jax(
    path = 'data/gazebo1.mat',
    skip_every = 100
):
    data = loadmat(path)

    poses = data['poses'][0:-1:skip_every, :] # N_samples x 3
    ranges = data['ranges'][0:-1:skip_every, :].T # N_ray x N_samples
    thetas = data['thetas'].T # N_ray x 1

    abs_angles = poses[:, -1] + thetas # N_ray x N_samples

    rel_positions = jnp.stack(
        [ranges * jnp.cos(abs_angles), ranges * jnp.sin(abs_angles)],
        axis = -1
    ) # N_ray x N_samples x 2

    abs_positions = poses[:, :2] + rel_positions
    return jnp.array(abs_positions.reshape([-1, 2]))

@jax.jit
def distance_function_jax(observations, test_positions):
    """JAX version of naive distance function"""
    test_positions = jnp.atleast_2d(test_positions)
    
    # Compute pairwise distances using broadcasting
    distances = jnp.sqrt(
        jnp.sum((test_positions[:, None, :] - observations[None, :, :]) ** 2, axis=-1)
    )
    
    # Return minimum distance for each test position
    return jnp.min(distances, axis=-1)[0]  # Take first element since we want scalar

def get_bounds(observations):
    return jnp.min(observations, axis=0), jnp.max(observations, axis=0)

def create_distance_fn(path='data/gazebo1.mat', skip_every=100):
    observations = load_data_jax(path, skip_every)
    return jax.jit(lambda x: distance_function_jax(observations, jnp.atleast_2d(x))), observations 