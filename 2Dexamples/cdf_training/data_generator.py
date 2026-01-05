import torch
import numpy as np
import os
import sys
# Add parent directory to path to import xarm modules
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, '..'))

from robot_models import Robot2D, RobotSDF
from utils_env import visualize_sdf
from torchmin import minimize
import time

class Robot2DCDFDataGenerator:
    def __init__(self, device='cpu'):
        # Initialize robot models
        self.robot = Robot2D(device=device)
        self.robot_sdf = RobotSDF(self.robot, device=device)
        
        # Device
        self.device = device
        
        # Joint limits [-pi, pi] for both joints
        self.q_min = torch.tensor([-np.pi, -np.pi], device=device)
        self.q_max = torch.tensor([np.pi, np.pi], device=device)
        
        # Data generation parameters
        self.radius = 4.0              # Workspace radius
        self.n_discrete = 50           # Points per dimension for workspace discretization
        self.batchsize = 2000          # Batch size for configuration sampling
        self.epsilon = 0.05            # Distance threshold to filter data
        
        # Create save directory
        self.save_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _cost_function(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Cost function for optimization
        Args:
            q: Joint configurations (N, 2)
            x: Target point (2,)
        Returns:
            cost: Scalar cost value
        """
        q_reshaped = q.reshape(-1, 2)
        points = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        d, _ = self.robot_sdf.query_sdf(points, q_reshaped)
        cost = (d**2).sum()
        return cost

    def find_valid_configurations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find valid configurations for a given point using optimization"""
        t0 = time.time()
        
        # Initialize random configurations
        q = torch.rand(self.batchsize, 2, device=self.device) * (self.q_max - self.q_min) + self.q_min
        q = q.detach().requires_grad_(True)
        
        try:
            result = minimize(
                lambda q: self._cost_function(q, x),
                q,
                method='l-bfgs',
                options=dict(line_search='strong-wolfe'),
                max_iter=50,
                disp=0
            )
            
            # Get final distances and link indices
            with torch.no_grad():
                q_result = result.x.reshape(-1, 2)
                points = x.unsqueeze(0).unsqueeze(0).expand(len(q_result), 1, -1)
                distances, link_indices = self.robot_sdf.query_sdf(points, q_result)
                
                # Filter valid configurations
                mask = distances.squeeze() < self.epsilon
                valid_q = q_result[mask]
                valid_indices = link_indices[mask]
                
                print(f'Found {len(valid_q)} valid configurations in {time.time()-t0:.2f}s')
                
                return valid_q, valid_indices
                
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return torch.zeros(0, 2, device=self.device), torch.zeros(0, device=self.device)

    def generate_dataset(self, save_path=None):
        """Generate CDF training dataset"""
        print("\n=== Starting Contact Database Generation ===")
        
        # Create grid of points in circular workspace
        r = torch.linspace(0, self.radius, self.n_discrete, device=self.device)
        theta = torch.linspace(0, 2*np.pi, self.n_discrete, device=self.device)
        R, Theta = torch.meshgrid(r, theta, indexing='ij')
        X = R * torch.cos(Theta)
        Y = R * torch.sin(Theta)
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        print(f"\nTotal grid points to process: {len(points)}")
        
        # Process each point
        data = {}
        start_time = time.time()
        
        for i, point in enumerate(points):
            print(f"\nProcessing point {i+1}/{len(points)}: {point.cpu().numpy()}")
            
            q, idx = self.find_valid_configurations(point)
            
            data[i] = {
                'x': point.cpu().numpy(),
                'q': q.cpu().numpy(),
                'idx': idx.cpu().numpy(),
            }
            
            # Print progress
            elapsed_time = time.time() - start_time
            avg_time_per_point = elapsed_time / (i + 1)
            remaining_points = len(points) - (i + 1)
            estimated_remaining_time = remaining_points * avg_time_per_point
            
            print(f'Progress: {i+1}/{len(points)} points ({(i+1)/len(points)*100:.1f}%)')
            print(f'Time elapsed: {elapsed_time/60:.1f} minutes')
            print(f'Estimated remaining time: {estimated_remaining_time/60:.1f} minutes')
        
        # Save data
        if save_path:
            compressed_db = {
                'points': np.array([data[i]['x'] for i in data.keys()]),
                'contact_configs': [data[i]['q'] for i in data.keys()],
                'link_indices': [data[i]['idx'] for i in data.keys()]
            }
            np.save(save_path, compressed_db)
            print(f"\nSaved contact database to: {save_path}")
        
        return data

    def visualize_test_point(self):
        """Visualize a test point and its valid configurations"""
        # Test with a point in workspace
        point = torch.tensor([2.0, 2.0], device=self.device)
        
        print(f"\nTesting with point: {point.cpu().numpy()}")
        valid_q, link_indices = self.find_valid_configurations(point)
        
        if len(valid_q) > 0:
            print(f"\nFound {len(valid_q)} valid configurations")
            print(f"First configuration: {valid_q[0].cpu().numpy()}")
            print(f"Link index: {link_indices[0].cpu().numpy()}")
            
            # Visualize first valid configuration
            visualize_sdf(self.robot_sdf, joint_angles=valid_q[0].unsqueeze(0))
        else:
            print("No valid configurations found")

if __name__ == "__main__":
    generator = Robot2DCDFDataGenerator()
    
    # Generate and save dataset
    data = generator.generate_dataset(os.path.join(generator.save_dir, 'contact_db_2d.npy')) 