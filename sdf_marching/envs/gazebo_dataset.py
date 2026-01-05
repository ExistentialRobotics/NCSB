from scipy.io import loadmat
import numpy as np
from sdf_marching.utils import naive_distance_function

"""
Using the Gazebo dataset
Data is presented as an N x 2 array of pointcloud in map coordinates (assuming perfect localisation)
"""
def load_data(
    path = 'data/gazebo1.mat',
    skip_every = 100
):
    data = loadmat(path)

    poses = data['poses'][0:-1:skip_every, :] # N_samples x 3
    ranges = data['ranges'][0:-1:skip_every, :].T # N_ray x N_samples
    thetas = data['thetas'].T # N_ray x 1

    abs_angles = poses[:, -1] + thetas # N_ray x N_samples

    rel_positions = np.stack(
        [ranges * np.cos(abs_angles), ranges * np.sin(abs_angles)],
        axis = -1
    ) # N_ray x N_samples x 2

    abs_positions = poses[:, :2] + rel_positions
    return abs_positions.reshape([-1, 2])

class GazeboDataset:
    def __init__(
            self, 
            path = 'data/gazebo1.mat',
            skip_every = 100
        ):
            self.obs = load_data(path=path, skip_every=skip_every)
    
    def distance_function(self, test_positions):
        return naive_distance_function(
             self.obs,
             test_positions              
        )
    
    def __call__(self, test_positions):
         return self.distance_function(test_positions)

    @property
    def mins(self):
        return np.min(self.obs, axis=0)
    
    @property
    def maxs(self):
        return np.max(self.obs, axis=0)

class GazeboDataset3D(GazeboDataset):
     def __init__(
        self,
        zmin,
        zmax,
        path='data/gazebo1.mat',
        skip_every=100
     ):
        super().__init__(path = path, skip_every=skip_every)
        self.zmin = zmin
        self.zmax = zmax
     
     def distance_function_3d(self, test_positions):

        test_positions_2d_array = np.atleast_2d(test_positions)

        distance_2d = self.distance_function(test_positions_2d_array[:, :2])
        distance_z = np.minimum(
            np.abs(test_positions_2d_array[:, 2] - self.zmin),
            np.abs(test_positions_2d_array[:, 2] - self.zmax)
        )

        return np.minimum(distance_2d, distance_z)
     
     def __call__(self, test_positions):
         return self.distance_function_3d(test_positions)
     
     @property
     def mins(self):
        return np.append(super().mins, self.zmin)
     
     @property
     def maxs(self):
        return np.append(super().maxs, self.zmax)
     

     def points(self, num_z=10):
         z_values = np.linspace(self.zmin, self.zmax, num_z)

         return np.concatenate(
            [
                np.concatenate(
                    [
                        self.obs, 
                        z_value * np.ones( (self.obs.shape[0], 1) )
                    ], 
                    axis=-1
                )
            for z_value in z_values
         ], axis=0)