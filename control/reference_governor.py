import numpy as np
from numpy import linalg as npla

class ReferenceGovernor:
    def __init__(self, initial_state, path_configs, dt, length_ref=8):
        """
        Initialize Reference Governor.
        
        Args:
            initial_state: Initial robot state
            path_configs: numpy array of configurations along the path (N, num_joints)
            dt: Time step
            length_ref: Reference length parameter
        """
        self.g = initial_state
        self.path_configs = path_configs
        self.dt = dt
        self.s = 0.0  # Progress variable s ∈ [0,1]
        self.z = initial_state
        self.zeta = 15  # Power term in dynamics
        
    def get_path_point(self, s):
        """
        Get configuration at path parameter s ∈ [0,1] using linear interpolation
        """
        N = len(self.path_configs) - 1
        idx = min(int(s * N), N-1)
        alpha = (s * N) - idx
        return (1 - alpha) * self.path_configs[idx] + alpha * self.path_configs[idx + 1]

    def update(self, z):
        """
        Update reference governor state using the dynamics:
        ṡ = k/(1 + ||x - γ(s)||) * (1 - s^ζ)
        """
        self.z = z
        
        # Get current point on path
        gamma_s = self.get_path_point(self.s)
        
        # Compute distance to current reference point
        distance_to_ref = npla.norm(z - gamma_s)
        
        # Reference governor dynamics
        k = 0.5  # Gain term
        s_dot = (k / (1 + distance_to_ref)) * (1 - self.s**self.zeta)
        
        # Update progress variable
        self.s += self.dt * s_dot
        self.s = min(1.0, max(0.0, self.s))  # Ensure s stays in [0,1]
        
        # Update governor position
        self.g = self.get_path_point(self.s)
        
        return self.g, self.s

    def update_path(self, new_path):
        """
        Update the reference path.
        """
        self.path_configs = new_path
        # Could add logic here to maintain continuity when switching paths