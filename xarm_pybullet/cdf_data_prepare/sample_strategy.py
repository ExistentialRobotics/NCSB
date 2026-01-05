import matplotlib.pyplot as plt
import torch
from torch.quasirandom import SobolEngine
import numpy as np
from math import pi

# Generate 1000 2D points using both methods
n_points = 1000

# Random uniform
random_points = torch.rand(n_points, 2)

# Sobol sequence
sobol = SobolEngine(dimension=2, scramble=True)
sobol_points = sobol.draw(n_points)




import torch
from torch.quasirandom import SobolEngine
import numpy as np

def analyze_sobol_distances(n_samples=100000, dim=6, q_min=None, q_max=None):
    """
    Analyze distances between neighboring configurations in Sobol sequence
    """
    # Initialize Sobol engine
    sobol = SobolEngine(dimension=dim, scramble=True)
    
    # Generate configurations
    configs = sobol.draw(n_samples)
    
    # Scale to joint limits if provided
    if q_min is not None and q_max is not None:
        configs = configs * (q_max - q_min) + q_min
    
    # Sample random pairs to estimate distances (using subset for efficiency)
    n_pairs = n_samples  # Analyze fewer pairs for efficiency
    indices = np.random.choice(n_samples, size=n_pairs, replace=False)
    selected_configs = configs[indices]
    
    # Compute distances to nearest neighbors (excluding self)
    distances = []
    for i in range(len(selected_configs)):
        # Create mask to exclude current config
        mask = torch.ones(len(configs), dtype=bool)
        mask[indices[i]] = False
        
        # Compute distances to all other configs
        diff = configs[mask] - selected_configs[i]
        dist = torch.norm(diff, dim=1)
        min_dist = torch.min(dist)
        distances.append(min_dist.item())
    
    distances = np.array(distances)
    
    print(f"\nAnalysis of {n_pairs} random configurations from Sobol sequence:")
    print(f"Minimum distance between configs: {np.min(distances):.6f}")
    print(f"Maximum distance between configs: {np.max(distances):.6f}")
    print(f"Mean distance between configs: {np.mean(distances):.6f}")
    print(f"Median distance between configs: {np.median(distances):.6f}")
    print(f"Std dev of distances: {np.std(distances):.6f}")
    
    return distances

# Run analysis with xArm's actual joint limits
q_min = torch.tensor([
    -2*pi,      # Joint 1
    -2.059,     # Joint 2
    -3.927,     # Joint 3
    -1.745,     # Joint 4
    -2.059,     # Joint 5
    -6.283      # Joint 6
])

q_max = torch.tensor([
    2*pi,       # Joint 1
    2.059,      # Joint 2
    0.191,      # Joint 3
    3.927,      # Joint 4
    2.059,      # Joint 5
    6.283       # Joint 6
])

distances = analyze_sobol_distances(n_samples=100000, dim=6, q_min=q_min, q_max=q_max)

# Plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.scatter(random_points[:, 0], random_points[:, 1], alpha=0.5)
# ax1.set_title('Random Uniform')
# ax2.scatter(sobol_points[:, 0], sobol_points[:, 1], alpha=0.5)
# ax2.set_title('Sobol Sequence')
# plt.show()