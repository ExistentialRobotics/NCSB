import torch
import numpy as np
import random
from pathlib import Path
import sys
import matplotlib.pyplot as plt
# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from robot_models import Robot2D

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_cdf_and_gradients(points, configs, contact_configs, link_indices, device='cpu'):
    """
    Compute CDF values and gradients for 2D robot
    Args:
        points: [batch_x, 2] target points
        configs: [batch_q, 2] query configurations
        contact_configs: List of [M_i, 2] contact configurations for each point
        link_indices: List of [M_i] link indices for each point
    Returns:
        cdf_values: [batch_x, batch_q] CDF values
        cdf_gradients: [batch_x, batch_q, 2] CDF gradients
    """
    batch_x = len(points)
    batch_q = len(configs)
    cdf_values = torch.zeros(batch_x, batch_q, device=device)
    cdf_gradients = torch.zeros(batch_x, batch_q, 2, device=device)
    
    for i in range(batch_x):
        contact_configs_i = torch.tensor(contact_configs[i], device=device)
        point_link_indices = torch.tensor(link_indices[i], device=device)
        
        # Compute all distances at once
        distances = torch.zeros(batch_q, len(contact_configs_i), device=device)
        for config_idx, link_idx in enumerate(point_link_indices):
            relevant_joints = slice(0, link_idx + 1)
            diff = configs[:, relevant_joints].unsqueeze(1) - contact_configs_i[config_idx:config_idx+1, relevant_joints]
            distances[:, config_idx] = torch.norm(diff.reshape(batch_q, -1), dim=1)
        
        # Find minimum distances and corresponding indices
        min_distances, min_indices = distances.min(dim=1)
        cdf_values[i] = min_distances
        
        # Compute gradients for minimum configurations
        for j in range(batch_q):
            min_idx = min_indices[j]
            min_link_idx = point_link_indices[min_idx]
            
            # Only compute gradient for relevant joints
            diff = configs[j, :min_link_idx+1] - contact_configs_i[min_idx, :min_link_idx+1]
            dist = min_distances[j]
            
            if dist > 0:  # Avoid division by zero
                grad = torch.zeros(2, device=device)
                grad[:min_link_idx+1] = diff / dist
                cdf_gradients[i, j] = grad
    
    return cdf_values, cdf_gradients

class CDFDataProcessor:
    def __init__(self, contact_db_path, device='cuda', batch_x=100, batch_q=100):
        """
        Initialize CDF data processor
        Args:
            contact_db_path: Path to contact database .npy file
            device: torch device
        """
        # Initialize robot model
        self.robot = Robot2D(device=device)
        
        # Load compressed contact database
        self.db = np.load(contact_db_path, allow_pickle=True).item()
        self.valid_points = torch.tensor(self.db['points'], device=device)
        self.contact_configs = self.db['contact_configs']
        self.link_indices = self.db['link_indices']
        
        # Print database statistics
        print("\nContact Database Statistics:")
        print(f"Total number of points: {len(self.valid_points)}")
        print(f"Number of points with no configurations: {sum(len(configs) == 0 for configs in self.contact_configs)}")
        print(f"Configuration counts per point: min={min(len(configs) for configs in self.contact_configs)}, "
              f"max={max(len(configs) for configs in self.contact_configs)}, "
              f"mean={np.mean([len(configs) for configs in self.contact_configs]):.1f}")
        
        # Device and batch settings
        self.device = device
        self.batch_x = batch_x  # Number of points per batch
        self.batch_q = batch_q  # Number of configurations per batch
        
        # Joint limits
        self.q_min = -np.pi
        self.q_max = np.pi

    def sample_q(self, batch_q=None):
        """Sample random configurations"""
        if batch_q is None:
            batch_q = self.batch_q
        q_sampled = torch.rand(batch_q, 2, device=self.device) * (self.q_max - self.q_min) + self.q_min
        q_sampled.requires_grad = True
        return q_sampled

    def sample_batch(self):
        """Sample batch and compute CDF values and gradients"""
        # Sample random points and configurations
        point_indices = torch.randint(0, len(self.valid_points), (self.batch_x,))
        points = self.valid_points[point_indices]
        configs = self.sample_q()
        
        # Get corresponding contact configurations and link indices
        contact_configs_batch = [self.contact_configs[idx] for idx in point_indices]
        link_indices_batch = [self.link_indices[idx] for idx in point_indices]
        
        # Compute CDF values and gradients
        cdf_values, cdf_gradients = compute_cdf_and_gradients(
            points=points,
            configs=configs,
            contact_configs=contact_configs_batch,
            link_indices=link_indices_batch,
            device=self.device
        )
        
        return points, configs, cdf_values, cdf_gradients

    def visualize_test_points(self, num_points=3):
        """
        Visualize test points and their closest contact configurations
        Args:
            num_points: Number of random points to visualize
        """
        import matplotlib.pyplot as plt
        
        # Sample some random configurations
        test_configs = self.sample_q(num_points)
        
        # Randomly select points
        point_indices = torch.randint(0, len(self.valid_points), (num_points,))
        points = self.valid_points[point_indices]
        
        # Create subplot grid
        fig, axes = plt.subplots(1, num_points, figsize=(5*num_points, 5))
        if num_points == 1:
            axes = [axes]
        
        for i, (point, test_config) in enumerate(zip(points, test_configs)):
            # Get contact configurations for this point
            contact_configs = torch.tensor(self.contact_configs[point_indices[i]], device=self.device)
            
            # Find closest contact configuration
            diffs = test_config.unsqueeze(0) - contact_configs
            distances = torch.norm(diffs, dim=1)
            closest_idx = torch.argmin(distances)
            closest_config = contact_configs[closest_idx]
            
            # Get joint positions for both configurations
            test_joints = self.robot.forward_kinematics(test_config.unsqueeze(0))[0]
            contact_joints = self.robot.forward_kinematics(closest_config.unsqueeze(0))[0]
            
            # Plot on current subplot
            ax = axes[i]
            
            # Plot target point
            ax.scatter(point[0].detach().cpu(), point[1].detach().cpu(), color='red', s=100, label='Target Point')
            
            # Plot test configuration
            ax.plot(test_joints[:, 0].detach().cpu(), test_joints[:, 1].detach().cpu(), 
                   'b-', linewidth=2, label='Test Config')
            
            # Plot closest contact configuration
            ax.plot(contact_joints[:, 0].detach().cpu(), contact_joints[:, 1].detach().cpu(), 
                   'g--', linewidth=2, label='Contact Config')
            
            # Set plot properties
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.grid(True)
            ax.set_aspect('equal')
            ax.legend()
            ax.set_title(f'Point {i+1}')
        
        plt.tight_layout()
        plt.show()

def test_cdf_computation():
    """Test CDF computation with sample data"""
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Initialize processor
    src_dir = Path(__file__).parent
    contact_db_path = src_dir / "data" / "contact_db_2d_refined.npy"
    processor = CDFDataProcessor(contact_db_path)
    
    # Sample batch and compute CDF
    points, configs, cdf_values, cdf_gradients = processor.sample_batch()
    
    # Print statistics
    print("\nCDF Computation Test Results:")
    print(f"CDF values shape: {cdf_values.shape}")
    print(f"CDF gradients shape: {cdf_gradients.shape}")
    print(f"Mean CDF value: {cdf_values.mean().item():.4f}")
    print(f"Mean gradient norm: {torch.norm(cdf_gradients, dim=2).mean().item():.4f}")
    
    # Visualize test points
    print("\nVisualizing test points and their contact configurations...")
    processor.visualize_test_points(num_points=3)

if __name__ == "__main__":
    test_cdf_computation() 